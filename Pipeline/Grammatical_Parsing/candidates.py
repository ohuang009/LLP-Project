"""Predicate-first grammatical candidate generation for scientific sentences.

The stage has two deliberately separate responsibilities:

1. spaCy supplies complete token/POS/morphology/dependency annotations and
   subject-predicate-object argument spans.
2. A pretrained SciBERT encoder ranks the existing ontology definitions for
   each subject/object span. This is provisional zero-shot typing, not a
   fine-tuned environmental NER model.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


SUBJECT_DEPS = frozenset({"nsubj", "nsubjpass", "csubj", "csubjpass"})
OBJECT_DEPS = frozenset({"dobj", "obj", "iobj", "attr", "oprd", "dative", "xcomp", "ccomp"})
PREDICATE_DEPS = frozenset({"ROOT", "conj", "advcl", "relcl", "ccomp", "xcomp"})
PREDICATE_PART_DEPS = frozenset({"aux", "auxpass", "neg", "prt", "advmod"})
EXCLUDED_ONTOLOGY_LABELS = frozenset({"EvidenceFragment", "Mention"})
REMOVABLE_ARTICLES = frozenset({"a", "an", "the"})


def _stable_id(prefix: str, *values: object) -> str:
    raw = "\u241f".join(str(value) for value in values)
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:20]}"


@dataclass(frozen=True)
class ArgumentSpan:
    start: int
    end: int
    text: str
    head_token: int


@dataclass(frozen=True)
class PredicateSpan:
    start: int
    end: int
    text: str
    source_text: str
    head_token: int
    token_indices: list[int]


class GrammarRuntimeError(RuntimeError):
    pass


def clean_argument_phrase(
    sentence_text: str, start_char: int, end_char: int, tokens: list[dict],
) -> dict:
    """Remove only syntactically certain excess while preserving source truth.

    The original span is immutable provenance. The cleaned span/name is a
    derived value used by SciBERT and later normalization stages.
    """
    argument_tokens = [
        token for token in tokens
        if int(token["start_char"]) < end_char and int(token["end_char"]) > start_char
    ]
    kept = list(argument_tokens)
    transformations: list[dict] = []
    while (
        kept
        and kept[0]["coarse_pos"] == "DET"
        and kept[0]["text"].casefold() in REMOVABLE_ARTICLES
    ):
        token = kept.pop(0)
        transformations.append({
            "operation": "remove_leading_article",
            "text": token["text"],
            "start_char": token["start_char"],
            "end_char": token["end_char"],
            "reason": "Leading a/an/the is a grammatical determiner, not part of the candidate identity.",
        })
    while kept and kept[-1].get("is_punctuation"):
        token = kept.pop()
        transformations.append({
            "operation": "remove_trailing_punctuation",
            "text": token["text"],
            "start_char": token["start_char"],
            "end_char": token["end_char"],
            "reason": "Terminal punctuation is not part of the candidate identity.",
        })
    if kept:
        cleaned_start = int(kept[0]["start_char"])
        cleaned_end = int(kept[-1]["end_char"])
    else:
        cleaned_start, cleaned_end = start_char, end_char
        transformations = []
    return {
        "original_text": sentence_text[start_char:end_char],
        "cleaned_text": sentence_text[cleaned_start:cleaned_end],
        "original_start_char": start_char,
        "original_end_char": end_char,
        "cleaned_start_char": cleaned_start,
        "cleaned_end_char": cleaned_end,
        "transformations": transformations,
        "policy": "remove only leading a/an/the and terminal punctuation; preserve adjectives and scientific qualifiers",
    }


class SpacyPredicateTripleExtractor:
    """Extract grammatical triples from a trained spaCy dependency parse."""

    def __init__(self, nlp) -> None:
        self.nlp = nlp

    @classmethod
    def load(cls, model_name: str):
        try:
            import spacy
        except ImportError as exc:
            raise GrammarRuntimeError(
                "spaCy is required for grammatical candidate generation. Install the repository requirements.txt."
            ) from exc
        try:
            nlp = spacy.load(model_name)
        except OSError as exc:
            raise GrammarRuntimeError(
                f"The configured spaCy pipeline {model_name!r} is unavailable. Install the declared English model."
            ) from exc
        required = {"parser", "tagger"}
        missing = sorted(required - set(nlp.pipe_names))
        if missing:
            raise GrammarRuntimeError(
                f"The spaCy pipeline lacks required components: {', '.join(missing)}."
            )
        return cls(nlp)

    def analyze_records(self, records: list[dict]) -> tuple[list[dict], list[dict]]:
        analyses: list[dict] = []
        triples: list[dict] = []
        texts = [record["text"] for record in records]
        for record, doc in zip(records, self.nlp.pipe(texts, batch_size=32)):
            sentence_analysis, sentence_triples = self._analyze_doc(record, doc)
            analyses.append(sentence_analysis)
            triples.extend(sentence_triples)
        return analyses, triples

    def _analyze_doc(self, record: dict, doc) -> tuple[dict, list[dict]]:
        noun_chunks: dict[int, object] = {}
        for chunk in doc.noun_chunks:
            for token in chunk:
                noun_chunks[token.i] = chunk

        tokens = [{
            "index": token.i,
            "text": token.text,
            "lemma": token.lemma_,
            "coarse_pos": token.pos_,
            "fine_pos": token.tag_,
            "morphology": token.morph.to_dict(),
            "dependency": token.dep_,
            "head_index": token.head.i,
            "head_text": token.head.text,
            "start_char": token.idx,
            "end_char": token.idx + len(token.text),
            "is_stop": bool(token.is_stop),
            "is_punctuation": bool(token.is_punct),
        } for token in doc]

        predicates = [
            token for token in doc
            if token.pos_ in {"VERB", "AUX"}
            and (token.dep_ in PREDICATE_DEPS or any(child.dep_ in SUBJECT_DEPS for child in token.children))
        ]
        result: list[dict] = []
        seen: set[tuple[int, int, int, int, int, int]] = set()
        for predicate in predicates:
            subjects = self._subjects(predicate)
            objects = self._objects(predicate)
            if not subjects or not objects:
                continue
            for subject_root in self._expand_conjunctions(subjects):
                subject = self._argument_span(subject_root, noun_chunks, doc)
                for object_root, preposition in self._expand_objects(objects):
                    object_span = self._argument_span(object_root, noun_chunks, doc)
                    if subject.start < object_span.end and object_span.start < subject.end:
                        continue
                    predicate_span = self._predicate_span(predicate, preposition, doc)
                    key = (
                        subject.start, subject.end, predicate_span.start,
                        predicate_span.end, object_span.start, object_span.end,
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    voice = "passive" if any(child.dep_ in {"nsubjpass", "auxpass"} for child in predicate.children) else "active"
                    result.append({
                        "schema_version": "1.0",
                        "triple_id": _stable_id("grammar-triple", record["sentence_id"], *key),
                        "document_id": record["document_id"],
                        "sentence_id": record["sentence_id"],
                        "section_id": record.get("section_id", ""),
                        "section_title": record.get("section_title", ""),
                        "pages": record.get("pages", []),
                        "sentence_text": record["text"],
                        "voice": voice,
                        "subject": subject.__dict__,
                        "predicate": predicate_span.__dict__,
                        "object": object_span.__dict__,
                    })

        return {
            "schema_version": "1.0",
            "document_id": record["document_id"],
            "sentence_id": record["sentence_id"],
            "section_id": record.get("section_id", ""),
            "section_title": record.get("section_title", ""),
            "pages": record.get("pages", []),
            "text": record["text"],
            "tokens": tokens,
            "predicate_token_indices": [token.i for token in predicates],
            "triple_ids": [triple["triple_id"] for triple in result],
        }, result

    @staticmethod
    def _subjects(predicate) -> list:
        subjects = [child for child in predicate.children if child.dep_ in SUBJECT_DEPS]
        if not subjects and predicate.dep_ == "conj" and predicate.head is not predicate:
            subjects = [child for child in predicate.head.children if child.dep_ in SUBJECT_DEPS]
        return subjects

    @staticmethod
    def _objects(predicate) -> list[tuple[object, object | None]]:
        direct_objects = [child for child in predicate.children if child.dep_ in OBJECT_DEPS]
        objects = [(child, None) for child in direct_objects]
        for preposition in (
            child for child in predicate.children
            if child.dep_ in {"prep", "agent"}
        ):
            objects.extend(
                (child, preposition)
                for child in preposition.children
                if child.dep_ in {"pobj", "obj"}
            )
        # Scientific relations often attach a source/target preposition to the
        # direct object: "removes contaminants from wastewater". Preserve the
        # verb as the predicate head while adding the preposition to its text.
        for direct_object in direct_objects:
            for preposition in (
                child for child in direct_object.children
                if child.dep_ in {"prep", "agent"}
            ):
                objects.extend(
                    (child, preposition)
                    for child in preposition.children
                    if child.dep_ in {"pobj", "obj"}
                )
        if not objects and predicate.dep_ == "conj" and predicate.head is not predicate:
            return SpacyPredicateTripleExtractor._objects(predicate.head)
        return objects

    @staticmethod
    def _expand_conjunctions(tokens: Iterable) -> list:
        result = []
        seen = set()
        for token in tokens:
            for value in (token, *token.conjuncts):
                if value.i not in seen:
                    seen.add(value.i)
                    result.append(value)
        return result

    @staticmethod
    def _expand_objects(objects: Iterable[tuple[object, object | None]]) -> list[tuple[object, object | None]]:
        result = []
        seen = set()
        for token, preposition in objects:
            for value in (token, *token.conjuncts):
                key = (value.i, preposition.i if preposition is not None else -1)
                if key not in seen:
                    seen.add(key)
                    result.append((value, preposition))
        return result

    @staticmethod
    def _argument_span(root, noun_chunks: dict[int, object], doc) -> ArgumentSpan:
        chunk = noun_chunks.get(root.i)
        if chunk is not None:
            start_token, end_token = chunk.start, chunk.end
        else:
            subtree = [token for token in root.subtree if not token.is_punct]
            start_token = min((token.i for token in subtree), default=root.i)
            end_token = max((token.i for token in subtree), default=root.i) + 1
        while start_token < end_token and doc[start_token].is_punct:
            start_token += 1
        while end_token > start_token and doc[end_token - 1].is_punct:
            end_token -= 1
        start = doc[start_token].idx
        end = doc[end_token - 1].idx + len(doc[end_token - 1].text)
        return ArgumentSpan(start, end, doc.text[start:end], root.i)

    @staticmethod
    def _predicate_span(predicate, preposition, doc) -> PredicateSpan:
        parts = [predicate, *(
            child for child in predicate.children if child.dep_ in PREDICATE_PART_DEPS
        )]
        if preposition is not None:
            parts.append(preposition)
        start_token = min(token.i for token in parts)
        end_token = max(token.i for token in parts) + 1
        start = doc[start_token].idx
        end = doc[end_token - 1].idx + len(doc[end_token - 1].text)
        ordered = sorted({token.i: token for token in parts}.values(), key=lambda token: token.i)
        relation_text = " ".join(token.text for token in ordered)
        return PredicateSpan(
            start, end, relation_text, doc.text[start:end], predicate.i,
            [token.i for token in ordered],
        )


class SciBERTOntologyTyper:
    """Rank ontology definitions using an unfine-tuned SciBERT encoder."""

    def __init__(
        self, ontology: dict, model_name: str, *, threshold: float,
        top_k: int, cache_dir: Path, min_margin: float = 0.0, encoder=None,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self.min_margin = min_margin
        self.top_k = top_k
        labels = [
            label for label in ontology["nodes"]
            if label not in EXCLUDED_ONTOLOGY_LABELS
        ]
        self.labels = labels
        self.ontology_texts = [self._ontology_text(label, ontology["nodes"][label]) for label in labels]
        if encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise GrammarRuntimeError(
                    "sentence-transformers is required for pretrained SciBERT typing. Install the repository requirements.txt."
                ) from exc
            encoder = SentenceTransformer(model_name, cache_folder=str(cache_dir))
        self.encoder = encoder
        self.ontology_vectors = np.asarray(
            self.encoder.encode(
                self.ontology_texts, batch_size=32, show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=True,
            )
        )

    @staticmethod
    def _ontology_text(label: str, spec: dict) -> str:
        return " ".join(filter(None, [
            f"Ontology type: {label}.",
            f"Category: {spec.get('category', '')}.",
            spec.get("definition", ""),
            f"Example: {spec.get('example', '')}." if spec.get("example") else "",
        ]))

    def type_arguments(self, arguments: list[dict]) -> list[dict]:
        if not arguments:
            return []
        # Raw SciBERT is a language encoder, not a sentence-similarity model.
        # Keeping this comparison focused on the argument phrase avoids letting
        # shared sentence context dominate every candidate's ontology ranking.
        texts = [row.get("cleaned_text") or row["surface_text"] for row in arguments]
        vectors = np.asarray(self.encoder.encode(
            texts, batch_size=32, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        ))
        similarities = vectors @ self.ontology_vectors.T
        results: list[dict] = []
        for row, scores in zip(arguments, similarities):
            order = np.argsort(scores)[::-1][:self.top_k]
            alternatives = [
                {"label": self.labels[int(index)], "score": round(float(scores[int(index)]), 6)}
                for index in order
            ]
            best = alternatives[0]
            runner_up_score = alternatives[1]["score"] if len(alternatives) > 1 else -1.0
            score_margin = round(best["score"] - runner_up_score, 6)
            applicable = best["score"] >= self.threshold and score_margin >= self.min_margin
            results.append({
                **row,
                "selected_label": best["label"] if applicable else "",
                "selected_score": best["score"],
                "score_margin": score_margin,
                "applicable": applicable,
                "alternatives": alternatives,
                "typing_method": "pretrained_scibert_candidate_ontology_similarity",
                "model": self.model_name,
                "threshold": self.threshold,
                "min_margin": self.min_margin,
            })
        return results


def _sentence_records(parsed: dict) -> list[dict]:
    document_id = parsed["document"]["id"]
    return [{
        "document_id": document_id,
        "sentence_id": sentence["id"],
        "section_id": section["id"],
        "section_title": section.get("title", ""),
        "paragraph_id": paragraph["id"],
        "pages": sentence.get("pages", []),
        "text": sentence["text"],
    }
        for section in parsed.get("sections", [])
        for paragraph in section.get("paragraphs", [])
        for sentence in paragraph.get("sentences", [])
    ]


def generate_grammatical_candidates(
    parsed: dict, ontology_path: Path, config: dict, *, parser=None, typer=None,
) -> dict:
    """Return full grammar artifacts plus typed subject/object candidates."""
    spec = config.get("grammatical_candidate_generation", {})
    if not spec.get("enabled", True):
        return {
            "sentence_analysis": [], "triples": [], "typings": [],
            "candidates_by_sentence": {},
            "status": {"enabled": False, "status": "disabled"},
        }
    records = _sentence_records(parsed)
    try:
        parser = parser or SpacyPredicateTripleExtractor.load(spec.get("parser_model", "en_core_web_sm"))
        if typer is None:
            ontology = json.loads(ontology_path.read_text(encoding="utf-8"))
            typer = SciBERTOntologyTyper(
                ontology, spec.get("scibert_model", "allenai/scibert_scivocab_uncased"),
                threshold=float(spec.get("typing_threshold", 0.25)),
                top_k=int(spec.get("typing_top_k", 5)),
                min_margin=float(spec.get("typing_min_margin", 0.0)),
                cache_dir=Path(spec.get("model_cache_dir", ontology_path.parent.parent / "cache" / "scibert")),
            )
        analyses, triples = parser.analyze_records(records)
        analyses_by_sentence = {row["sentence_id"]: row for row in analyses}
    except Exception as exc:
        if spec.get("required", True):
            if isinstance(exc, GrammarRuntimeError):
                raise
            raise GrammarRuntimeError(f"Grammatical candidate generation failed: {exc}") from exc
        return {
            "sentence_analysis": [], "triples": [], "typings": [],
            "candidates_by_sentence": {},
            "status": {"enabled": True, "status": "unavailable", "reason": str(exc)},
        }

    argument_index: dict[tuple[str, int, int], dict] = {}
    for triple in triples:
        for role in ("subject", "object"):
            span = triple[role]
            key = (triple["sentence_id"], int(span["start"]), int(span["end"]))
            row = argument_index.setdefault(key, {
                "document_id": triple["document_id"],
                "sentence_id": triple["sentence_id"],
                "surface_text": span["text"],
                "start_char": int(span["start"]),
                "end_char": int(span["end"]),
                "sentence_text": triple["sentence_text"],
                "argument_tokens": [
                    token for token in analyses_by_sentence[triple["sentence_id"]]["tokens"]
                    if int(token["start_char"]) < int(span["end"])
                    and int(token["end_char"]) > int(span["start"])
                ],
                "grammatical_roles": set(),
                "triple_ids": set(),
            })
            row["grammatical_roles"].add(role)
            row["triple_ids"].add(triple["triple_id"])

    arguments = []
    for row in argument_index.values():
        normalization = clean_argument_phrase(
            row["sentence_text"], row["start_char"], row["end_char"], row["argument_tokens"],
        )
        arguments.append({
            **row,
            "cleaned_text": normalization["cleaned_text"],
            "deterministic_normalization": normalization,
            "grammatical_roles": sorted(row["grammatical_roles"]),
            "triple_ids": sorted(row["triple_ids"]),
        })
    typings = typer.type_arguments(arguments)
    candidates: dict[str, list[dict]] = defaultdict(list)
    for typing in typings:
        candidates[typing["sentence_id"]].append({
            "start": typing["start_char"],
            "end": typing["end_char"],
            # An empty label is intentional: grammar creates the candidate;
            # SciBERT supplies a type only when its ranking clears both gates.
            "label": typing["selected_label"],
            "canonical_name": typing["cleaned_text"],
            "confidence": typing["selected_score"],
            "method": "scibert_svo_argument",
            "metadata": {
                "grammatical_roles": typing["grammatical_roles"],
                "triple_ids": typing["triple_ids"],
                "deterministic_normalization": typing["deterministic_normalization"],
                "argument_tokens": typing["argument_tokens"],
                "typing_applicable": typing["applicable"],
                "typing_score": typing["selected_score"],
                "typing_score_margin": typing.get("score_margin"),
                "typing_alternatives": typing["alternatives"],
                "typing_model": typing["model"],
                "typing_threshold": typing["threshold"],
                "typing_min_margin": typing.get("min_margin"),
            },
        })

    return {
        "sentence_analysis": analyses,
        "triples": triples,
        "typings": typings,
        "candidates_by_sentence": dict(candidates),
        "status": {
            "enabled": True,
            "required": bool(spec.get("required", True)),
            "status": "complete",
            "parser_model": spec.get("parser_model", "en_core_web_sm"),
            "typing_model": spec.get("scibert_model", "allenai/scibert_scivocab_uncased"),
            "sentences": len(analyses),
            "tokens": sum(len(row["tokens"]) for row in analyses),
            "triples": len(triples),
            "unique_arguments": len(typings),
            "generated_candidates": sum(len(rows) for rows in candidates.values()),
            "typed_candidates": sum(row["applicable"] for row in typings),
            "untyped_arguments": sum(not row["applicable"] for row in typings),
        },
    }
