"""Raw grammatical subject-verb-object extraction and advisory SciBERT typing.

The source spans in this module are immutable. No article removal, lemmatizing,
canonicalization, or other cleanup occurs here; name cleanup belongs to the
first Qwen/Ollama pass in the node pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time

import numpy as np

from Pipeline.core import stable_id


SUBJECT_DEPS = frozenset({"nsubj", "nsubjpass", "csubj", "csubjpass"})
OBJECT_DEPS = frozenset({"dobj", "obj", "iobj", "attr", "oprd", "dative"})
CONTEXT_DEPS = frozenset({"advcl", "relcl", "ccomp", "xcomp", "parataxis", "appos"})
EXCLUDED_ONTOLOGY_LABELS = frozenset({"EvidenceFragment", "Mention"})
_PARSER_CACHE: dict[str, "RawSVOExtractor"] = {}
_TYPER_CACHE: dict[tuple, "SciBERTOntologyTyper"] = {}
_LOCK = threading.Lock()


class GrammarRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class RawSpan:
    text: str
    start: int
    end: int
    head_token: int


class RawSVOExtractor:
    """Extract the main-clause SVO and keep other clause fragments as context."""

    def __init__(self, nlp) -> None:
        self.nlp = nlp

    @classmethod
    def load(cls, model_name: str) -> "RawSVOExtractor":
        try:
            import spacy
            nlp = spacy.load(model_name)
        except (ImportError, OSError) as exc:
            raise GrammarRuntimeError(
                f"spaCy model {model_name!r} is required for raw SVO extraction"
            ) from exc
        if "parser" not in nlp.pipe_names:
            raise GrammarRuntimeError(f"spaCy model {model_name!r} has no dependency parser")
        return cls(nlp)

    def analyze_records(self, records: list[dict]) -> tuple[list[dict], list[dict]]:
        analyses: list[dict] = []
        triples: list[dict] = []
        for record, doc in zip(records, self.nlp.pipe([row["text"] for row in records], batch_size=32)):
            analysis, sentence_triples = self._analyze(record, doc)
            analyses.append(analysis)
            triples.extend(sentence_triples)
        return analyses, triples

    def _analyze(self, record: dict, doc) -> tuple[dict, list[dict]]:
        chunks = {token.i: chunk for chunk in doc.noun_chunks for token in chunk}
        roots = [token for token in doc if token.dep_ == "ROOT" and token.pos_ in {"VERB", "AUX"}]
        main_predicates = list(roots)
        if roots:
            main_predicates.extend(
                token for token in roots[0].conjuncts if token.pos_ in {"VERB", "AUX"}
            )
        result: list[dict] = []
        covered: list[tuple[int, int]] = []
        for predicate in main_predicates:
            subjects = self._subjects(predicate)
            objects = self._objects(predicate)
            for subject_root in subjects:
                subject = self._argument_span(subject_root, chunks, doc)
                for object_root in objects:
                    obj = self._argument_span(object_root, chunks, doc)
                    verb = RawSpan(predicate.text, predicate.idx, predicate.idx + len(predicate.text), predicate.i)
                    if self._overlap(subject, obj):
                        continue
                    triple = {
                        "triple_id": stable_id(
                            "svo", record["sentence_id"], subject.start, verb.start, obj.start
                        ),
                        "document_id": record["document_id"],
                        "section_id": record.get("section_id", ""),
                        "section_title": record.get("section_title", ""),
                        "paragraph_id": record.get("paragraph_id", ""),
                        "sentence_id": record["sentence_id"],
                        "sentence_text": record["text"],
                        "subject": subject.__dict__,
                        "verb": verb.__dict__,
                        "object": obj.__dict__,
                    }
                    result.append(triple)
                    covered.extend([(subject.start, subject.end), (verb.start, verb.end), (obj.start, obj.end)])

        context = self._context_fragments(doc, covered)
        tokens = [{
            "index": token.i,
            "text": token.text,
            "dependency": token.dep_,
            "head_index": token.head.i,
            "start": token.idx,
            "end": token.idx + len(token.text),
        } for token in doc]
        return {
            "document_id": record["document_id"],
            "section_id": record.get("section_id", ""),
            "section_title": record.get("section_title", ""),
            "paragraph_id": record.get("paragraph_id", ""),
            "sentence_id": record["sentence_id"],
            "sentence_text": record["text"],
            "main": result,
            "context": context,
            "tokens": tokens,
        }, result

    @staticmethod
    def _subjects(predicate) -> list:
        rows = [child for child in predicate.children if child.dep_ in SUBJECT_DEPS]
        if not rows and predicate.dep_ == "conj":
            rows = [child for child in predicate.head.children if child.dep_ in SUBJECT_DEPS]
        return [item for row in rows for item in (row, *row.conjuncts)]

    @staticmethod
    def _objects(predicate) -> list:
        rows = [child for child in predicate.children if child.dep_ in OBJECT_DEPS]
        for prep in (child for child in predicate.children if child.dep_ in {"prep", "agent"}):
            rows.extend(child for child in prep.children if child.dep_ in {"pobj", "obj"})
        return [item for row in rows for item in (row, *row.conjuncts)]

    @staticmethod
    def _argument_span(root, chunks: dict[int, object], doc) -> RawSpan:
        chunk = chunks.get(root.i)
        tokens = list(chunk) if chunk is not None else [
            token for token in root.subtree
            if not token.is_punct and token.dep_ not in CONTEXT_DEPS
        ]
        tokens = sorted(tokens or [root], key=lambda token: token.i)
        start = tokens[0].idx
        end = tokens[-1].idx + len(tokens[-1].text)
        return RawSpan(doc.text[start:end], start, end, root.i)

    @staticmethod
    def _overlap(left: RawSpan, right: RawSpan) -> bool:
        return left.start < right.end and right.start < left.end

    @staticmethod
    def _context_fragments(doc, covered: list[tuple[int, int]]) -> list[dict]:
        """Return exact subordinate/non-core fragments, separate from the main SVO."""
        fragments: list[dict] = []
        seen: set[tuple[int, int]] = set()
        for root in (token for token in doc if token.dep_ in CONTEXT_DEPS):
            subtree = sorted(root.subtree, key=lambda token: token.i)
            if not subtree:
                continue
            start = subtree[0].idx
            end = subtree[-1].idx + len(subtree[-1].text)
            if (start, end) in seen or any(start >= a and end <= b for a, b in covered):
                continue
            seen.add((start, end))
            fragments.append({
                "text": doc.text[start:end],
                "start": start,
                "end": end,
                "kind": root.dep_,
            })
        # Keep the largest clause when spaCy marks nested tokens as additional
        # context roots; this avoids repeating the same fragment several times.
        selected: list[dict] = []
        for fragment in sorted(fragments, key=lambda row: (row["start"], -(row["end"] - row["start"]))):
            if any(fragment["start"] >= row["start"] and fragment["end"] <= row["end"] for row in selected):
                continue
            selected.append(fragment)
        return sorted(selected, key=lambda row: (row["start"], row["end"]))


class SciBERTOntologyTyper:
    """Use SciBERT similarity as advisory evidence; low-confidence output is NONE."""

    def __init__(
        self, ontology: dict, model_name: str, *, threshold: float, min_margin: float,
        top_k: int, cache_dir: Path, local_files_only: bool = True, encoder=None,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self.min_margin = min_margin
        self.top_k = top_k
        self.labels = [label for label in ontology["nodes"] if label not in EXCLUDED_ONTOLOGY_LABELS]
        descriptions = [
            f"{label}. {ontology['nodes'][label].get('definition', '')}"
            for label in self.labels
        ]
        if encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                encoder = SentenceTransformer(
                    model_name, cache_folder=str(cache_dir), local_files_only=local_files_only
                )
            except ImportError as exc:
                raise GrammarRuntimeError("sentence-transformers is required for SciBERT typing") from exc
        self.encoder = encoder
        self.ontology_vectors = np.asarray(encoder.encode(
            descriptions, batch_size=32, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        ))

    def type_arguments(self, arguments: list[dict]) -> list[dict]:
        if not arguments:
            return []
        vectors = np.asarray(self.encoder.encode(
            [row["node_name"] for row in arguments], batch_size=32,
            show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True,
        ))
        scores = vectors @ self.ontology_vectors.T
        output: list[dict] = []
        for row, candidate_scores in zip(arguments, scores):
            order = np.argsort(candidate_scores)[::-1][:self.top_k]
            alternatives = [
                {"label": self.labels[int(index)], "score": round(float(candidate_scores[int(index)]), 6)}
                for index in order
            ]
            best = alternatives[0]
            margin = best["score"] - (alternatives[1]["score"] if len(alternatives) > 1 else -1)
            applicable = best["score"] >= self.threshold and margin >= self.min_margin
            output.append({
                **row,
                "scibert_type": best["label"] if applicable else "NONE",
                "scibert_score": best["score"],
                "scibert_margin": round(margin, 6),
                "scibert_alternatives": alternatives,
                "scibert_applicable": applicable,
                "scibert_model": self.model_name,
            })
        return output


def _records(parsed: dict) -> list[dict]:
    document_id = parsed["document"]["id"]
    return [{
        "document_id": document_id,
        "section_id": section["id"],
        "section_title": section.get("title", ""),
        "paragraph_id": paragraph["id"],
        "sentence_id": sentence["id"],
        "text": sentence["text"],
    } for section in parsed.get("sections", [])
      for paragraph in section.get("paragraphs", [])
      for sentence in paragraph.get("sentences", [])]


def _cached_grammatical_models(ontology_path: Path, spec: dict):
    parser_name = spec.get("parser_model", "en_core_web_sm")
    model_name = spec.get("scibert_model", "allenai/scibert_scivocab_uncased")
    cache_dir = Path(spec.get("model_cache_dir", ontology_path.parent.parent / "cache" / "scibert"))
    key = (str(ontology_path), model_name, float(spec.get("typing_threshold", .25)),
           float(spec.get("typing_min_margin", .015)), int(spec.get("typing_top_k", 5)))
    with _LOCK:
        parser = _PARSER_CACHE.get(parser_name)
        parser_loaded = parser is None
        if parser is None:
            parser = _PARSER_CACHE.setdefault(parser_name, RawSVOExtractor.load(parser_name))
        typer = _TYPER_CACHE.get(key)
        typer_loaded = typer is None
        if typer is None:
            typer = SciBERTOntologyTyper(
                json.loads(ontology_path.read_text(encoding="utf-8")), model_name,
                threshold=float(spec.get("typing_threshold", .25)),
                min_margin=float(spec.get("typing_min_margin", .015)),
                top_k=int(spec.get("typing_top_k", 5)), cache_dir=cache_dir,
                local_files_only=bool(spec.get("scibert_local_files_only", True)),
            )
            _TYPER_CACHE[key] = typer
    return parser, typer, {
        "parser_model": parser_name, "scibert_model": model_name,
        "parser_loaded_now": parser_loaded, "scibert_loaded_now": typer_loaded,
    }


def preload_grammatical_models(ontology_path: Path, config: dict) -> dict:
    started = time.perf_counter()
    spec = config.get("grammar", config.get("grammatical_candidate_generation", {}))
    _, _, status = _cached_grammatical_models(ontology_path, spec)
    return {**status, "ready": True, "elapsed_seconds": round(time.perf_counter() - started, 3)}


def extract_raw_svo(parsed: dict, *, parser=None) -> dict:
    records = _records(parsed)
    if parser is None:
        parser = RawSVOExtractor.load("en_core_web_sm")
    analyses, triples = parser.analyze_records(records)
    return {"sentences": analyses, "triples": triples}


def type_with_scibert(arguments: list[dict], ontology_path: Path, config: dict, *, typer=None) -> list[dict]:
    spec = config.get("grammar", config.get("grammatical_candidate_generation", {}))
    if typer is None:
        _, typer, _ = _cached_grammatical_models(ontology_path, spec)
    return typer.type_arguments(arguments)
