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

from Pipeline.Node_Pipeline.common import stable_id
from .common import CONDITION_RE, COUNT_RESULT_RE, MEASUREMENT_RE, NAMED_SCORE_RE, QUANTITATIVE_RESULT_RE, _is_current_study_sentence, _sentence_rows

def assemble_observation_mentions(parsed: dict, mentions: list[dict]) -> list[dict]:
    """Create occurrence-scoped Observation nodes for explicit quantitative results."""
    document_id = parsed["document"]["id"]
    labels_by_sentence: dict[str, set[str]] = defaultdict(set)
    for mention in mentions:
        sentence_id = mention.get("source", {}).get("sentence_id")
        if sentence_id:
            labels_by_sentence[sentence_id].add(mention["label"])

    additions: list[dict] = []
    for sentence, paragraph, section in _sentence_rows(parsed):
        text = sentence["text"]
        if not _is_current_study_sentence(text, section.get("title", "")):
            continue
        assumption = re.search(
            r"\b(?:assum(?:e|es|ed|ing)|unit price|a price of|is set to|was set to)\b",
            text, re.I,
        )
        reported_change = re.search(
            r"\b(?:achiev|reduc|increas|decreas|chang|sav|lower|higher|outperform|surpass)",
            text, re.I,
        )
        if assumption and not reported_change:
            continue
        measurements = [
            {"value": match.group("value"), "unit": match.group("unit")}
            for pattern in (MEASUREMENT_RE, COUNT_RESULT_RE)
            for match in pattern.finditer(text)
        ]
        measurements.extend({
            "value": match.group("value"), "unit": "", "metric": match.group("metric")
        } for match in NAMED_SCORE_RE.finditer(text))
        if not QUANTITATIVE_RESULT_RE.search(text) or not measurements:
            continue
        source = {
            "kind": "sentence_span",
            "sentence_id": sentence["id"],
            "paragraph_id": paragraph["id"],
            "section_id": section["id"],
            "section_title": section.get("title", ""),
            "pages": sentence.get("pages", []),
            "start_char": 0,
            "end_char": len(text),
            "quote": text,
            "evidence_quote": text,
            "context_sentence_ids": [sentence["id"]],
            "context_sentences": [{
                "sentence_id": sentence["id"],
                "paragraph_id": paragraph["id"],
                "role": "target",
                "text": text,
                "pages": sentence.get("pages", []),
            }],
            "context_policy": "exact quantitative result sentence",
            "provenance_precision": "exact sentence characters; page set inherited from parsed paragraph",
        }
        conditions = [{
            "name": match.group("name"),
            "value": match.group("value"),
            "unit": match.group("unit") or "",
        } for match in CONDITION_RE.finditer(text)]
        if "Observation" not in labels_by_sentence[sentence["id"]]:
            additions.append({
                "mention_id": stable_id("mention", document_id, sentence["id"], "Observation"),
                "document_id": document_id,
                "label": "Observation",
                "surface_text": text,
                "canonical_name": text,
                "canonical_id": stable_id("observation", document_id, sentence["id"]),
                "status": "accepted",
                "confidence": 0.96,
                "extraction_method": "quantitative_observation_assembler_v1",
                "source": source,
                "attributes": {"measurements": measurements, "conditions": conditions},
                "validation": {
                    "outcome": "accept",
                    "judge": "quantitative_observation_structure",
                    "llm_used": False,
                    "reason": "An exact quantitative scientific result sentence is represented as a provenance-local observation.",
                },
            })
    return [*mentions, *additions]
