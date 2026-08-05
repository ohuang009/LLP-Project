from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
import unicodedata
from typing import Callable, Iterable

from .common import normalized, stable_id
from .lexicon import Lexicon, lexicon_spans
from .span_detection import Span, select_spans

def flatten_sentences(parsed: dict) -> list[dict]:
    rows: list[dict] = []
    for section in parsed.get("sections", []):
        sequence = [
            (sentence, paragraph)
            for paragraph in section.get("paragraphs", [])
            for sentence in paragraph.get("sentences", [])
        ]
        for index, (sentence, paragraph) in enumerate(sequence):
            lo = max(0, index - 1)
            hi = min(len(sequence), index + 2)
            positions = list(range(lo, hi))
            if len(positions) < 2 and len(sequence) >= 2:
                positions = [0, 1] if index == 0 else [len(sequence) - 2, len(sequence) - 1]
            context = [
                {
                    "sentence_id": sequence[position][0]["id"],
                    "paragraph_id": sequence[position][1]["id"],
                    "role": "target" if position == index else ("previous" if position < index else "next"),
                    "text": sequence[position][0]["text"],
                    "pages": sequence[position][0].get("pages", []),
                }
                for position in positions
            ]
            rows.append({
                "sentence": sentence,
                "paragraph": paragraph,
                "section": section,
                "context": context,
            })
    return rows


def trace_source(row: dict, span: Span) -> dict:
    sentence, paragraph, section = row["sentence"], row["paragraph"], row["section"]
    return {
        "kind": "sentence_span",
        "sentence_id": sentence["id"],
        "paragraph_id": paragraph["id"],
        "section_id": section["id"],
        "section_title": section["title"],
        "pages": sentence.get("pages", []),
        "start_char": span.start,
        "end_char": span.end,
        "quote": sentence["text"][span.start:span.end],
        "evidence_quote": sentence["text"],
        "context_sentence_ids": [item["sentence_id"] for item in row["context"]],
        "context_sentences": row["context"],
        "context_policy": "target sentence plus previous and/or next narrative sentence within the same section; 2–3 sentences when available",
        "provenance_precision": "exact sentence characters; page set inherited from parsed paragraph",
    }


def metadata_mentions(parsed: dict) -> list[dict]:
    document = parsed["document"]
    title = document.get("title") or document["filename"]
    title_sentence_id = stable_id("metadata-sentence", document["id"], "document.title")
    result = [{
        "mention_id": stable_id("mention", document["id"], "title", title),
        "document_id": document["id"], "label": "Publication", "surface_text": title,
        "canonical_name": title, "canonical_id": stable_id("publication", normalized(title)),
        "status": "accepted", "confidence": 1.0, "extraction_method": "structural_metadata",
        "source": {
            "kind": "document_metadata", "field": "document.title",
            "sentence_id": title_sentence_id, "evidence_quote": title,
            "context_sentence_ids": [title_sentence_id],
            "context_sentences": [{"sentence_id": title_sentence_id, "role": "target", "text": title, "pages": []}],
        },
        "validation": {"outcome": "accept", "judge": "structural_metadata_rule", "llm_used": False},
    }]
    authors = document.get("authors_text") or ""
    authors = re.sub(r"\s+([\u0300-\u036f])", r"\1", authors)
    authors = unicodedata.normalize("NFKD", authors)
    authors = "".join(char for char in authors if not unicodedata.combining(char))
    authors = re.sub(r"\s+and\s+", ",", authors, flags=re.I)
    detected_names = re.findall(
        r"(?<![A-Za-zÀ-ÖØ-öø-ÿ])"
        r"[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ-]+"
        r"(?:\s+[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ-]+){1,3}",
        authors,
    )
    author_values = detected_names if len(detected_names) >= 2 else [part.strip(" ,;*") for part in authors.split(",")]
    authors_sentence_id = stable_id("metadata-sentence", document["id"], "document.authors_text")
    for position, author in enumerate(author_values):
        if len(author) < 3 or any(char.isdigit() for char in author):
            continue
        result.append({
            "mention_id": stable_id("mention", document["id"], "author", position, author),
            "document_id": document["id"], "label": "Person", "surface_text": author,
            "canonical_name": author, "canonical_id": "", "status": "accepted", "confidence": .98,
            "extraction_method": "structural_metadata",
            "source": {
                "kind": "document_metadata", "field": "document.authors_text",
                "sentence_id": authors_sentence_id, "evidence_quote": authors,
                "context_sentence_ids": [authors_sentence_id],
                "context_sentences": [{"sentence_id": authors_sentence_id, "role": "target", "text": authors, "pages": []}],
            },
            "validation": {"outcome": "accept", "judge": "structural_metadata_rule", "llm_used": False},
        })
    return result


def mention_dedup_key(row: dict) -> tuple:
    """Keep distinct metadata values while deduplicating repeated source spans."""
    source = row.get("source", {})
    if source.get("kind") == "document_metadata":
        return (
            row["document_id"], row["label"], source.get("field", ""),
            normalized(row.get("canonical_name") or row.get("surface_text", "")),
        )
    return (
        row["document_id"], row["label"], source.get("sentence_id", ""),
        source.get("start_char"), source.get("end_char"),
    )


def remove_redundant_review_candidates(
    accepted: list[dict], review: list[dict],
) -> list[dict]:
    """Do not re-adjudicate an exact sentence span already accepted by a trusted channel."""
    trusted_spans = {
        (
            row.get("document_id", ""), row.get("source", {}).get("sentence_id", ""),
            row.get("source", {}).get("start_char"), row.get("source", {}).get("end_char"),
            normalized(row.get("surface_text", "")),
        )
        for row in accepted
    }
    return [
        row for row in review
        if (
            row.get("document_id", ""), row.get("source", {}).get("sentence_id", ""),
            row.get("source", {}).get("start_char"), row.get("source", {}).get("end_char"),
            normalized(row.get("surface_text", "")),
        ) not in trusted_spans
    ]


def generate_candidates(
    parsed: dict, lexicon: Lexicon, grammatical_candidates: dict[str, list[dict]] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Combine trusted extraction with SVO arguments and optional SciBERT types.

    Structural metadata and unambiguous lexicon aliases are trusted. Every
    grammatical subject/object candidate remains reviewable for DeepSeek, even
    when pretrained SciBERT cannot assign a sufficiently distinct type.
    """
    accepted: list[dict] = metadata_mentions(parsed)
    review: list[dict] = []
    rows = flatten_sentences(parsed)
    grammatical_candidates = grammatical_candidates or {}

    for row in rows:
        sentence = row["sentence"]
        grammar_spans = [
            Span(
                int(candidate["start"]), int(candidate["end"]), candidate["label"],
                candidate.get("method", "scibert_svo_argument"), float(candidate["confidence"]),
                candidate.get("canonical_id", ""), candidate.get("canonical_name", ""),
                candidate.get("metadata", {}),
            )
            for candidate in grammatical_candidates.get(sentence["id"], [])
        ]
        for span in select_spans([
            *lexicon_spans(sentence["text"], lexicon),
            *grammar_spans,
        ], sentence["text"]):
            source = trace_source(row, span)
            surface = source["quote"]
            candidate_id = stable_id("candidate", parsed["document"]["id"], sentence["id"], span.start, span.end, span.label)
            base = {
                "candidate_id": candidate_id,
                "document_id": parsed["document"]["id"],
                "label": span.label,
                "surface_text": surface,
                "canonical_name": span.canonical_name or surface,
                "canonical_id": span.canonical_id,
                "confidence": span.confidence,
                "extraction_method": span.method,
                "candidate_generators": [{"name": span.method, "confidence": span.confidence}],
                "candidate_metadata": span.metadata or {},
                "source": source,
            }
            if span.method == "persistent_lexicon":
                accepted.append({
                    **base,
                    "mention_id": stable_id("mention", candidate_id),
                    "status": "accepted",
                    "validation": {
                        "outcome": "accept",
                        "judge": "persistent_lexicon_exact_match",
                        "llm_used": False,
                        "reason": "The exact source span matches one unambiguous reviewed or seeded lexicon alias.",
                    },
                })
            else:
                review.append({
                    **base,
                    "status": "review",
                    "validation": {
                        "outcome": "review", "judge": "high_recall_candidate_generator", "llm_used": False,
                        "reason": "A high-recall generator found this exact span; DeepSeek or a human must adjudicate it.",
                    },
                })
    return accepted, review
