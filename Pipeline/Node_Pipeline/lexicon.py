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

from .common import (
    LEXICON_PATH, RUNS_DIR, SCOPED_LABELS, json_read, json_write, jsonl_read,
    normalized, now_iso, stable_id,
)
from .span_detection import Span

class Lexicon:
    def __init__(self, path: Path = LEXICON_PATH) -> None:
        self.path = path
        self.data = json_read(path)
        self.entries = self.data["entries"]
        self.alias_index: dict[str, list[dict]] = defaultdict(list)
        for entry in self.entries:
            if entry.get("status") == "deprecated":
                continue
            for alias in {entry["canonical_name"], *entry.get("aliases", [])}:
                key = normalized(alias)
                if all(existing["canonical_id"] != entry["canonical_id"] for existing in self.alias_index[key]):
                    self.alias_index[key].append(entry)

    def find(self, value: str, label: str | None = None) -> dict | None:
        matches = self.alias_index.get(normalized(value), [])
        if label is not None:
            matches = [entry for entry in matches if entry["label"] == label]
        return matches[0] if len(matches) == 1 else None

    def alias_is_unambiguous(self, value: str, canonical_id: str) -> bool:
        matches = self.alias_index.get(normalized(value), [])
        return len(matches) == 1 and matches[0]["canonical_id"] == canonical_id

    def add_reviewed_aliases(self, label: str, canonical_name: str, aliases: list[str], provenance: dict) -> dict:
        requested = {normalized(canonical_name), *(normalized(alias) for alias in aliases)}
        matching = [
            entry for entry in self.entries
            if entry.get("status") != "deprecated" and entry["label"] == label
            and requested & {normalized(entry["canonical_name"]), *(normalized(alias) for alias in entry.get("aliases", []))}
        ]
        chosen = next((entry for entry in matching if normalized(entry["canonical_name"]) == normalized(canonical_name)), None)
        if chosen is None and matching:
            chosen = matching[0]
        if chosen is None:
            chosen = {
                "canonical_id": stable_id("lex", label, normalized(canonical_name)),
                "label": label,
                "canonical_name": canonical_name,
                "aliases": [],
                "status": "human_reviewed",
                "definition": f"Human-reviewed canonical {label} for Engineered Water Systems extraction.",
                "provenance": [],
            }
            self.entries.append(chosen)
        merged = {
            canonical_name, chosen["canonical_name"], *chosen.get("aliases", []), *aliases,
            *(value for entry in matching for value in [entry["canonical_name"], *entry.get("aliases", [])]),
        }
        chosen["canonical_name"] = canonical_name
        chosen["aliases"] = sorted(merged, key=lambda value: (value.casefold() != canonical_name.casefold(), value.casefold()))
        chosen["status"] = "human_reviewed"
        chosen.setdefault("provenance", []).append(provenance)
        for entry in matching:
            if entry is chosen:
                continue
            entry["status"] = "deprecated"
            entry["replaced_by"] = chosen["canonical_id"]
            entry.setdefault("provenance", []).append(provenance)
        self.data["version"] = int(self.data.get("version", 0)) + 1
        self.data["updated_at"] = now_iso()
        json_write(self.path, self.data)
        self.__init__(self.path)
        return chosen


def load_prior_node_catalog(
    lexicon: Lexicon, *, exclude_run_dir: Path | None = None,
) -> list[dict]:
    """Load reusable nodes from the lexicon and completed earlier runs."""
    catalog: dict[tuple[str, str], dict] = {}

    def add(row: dict) -> None:
        label = str(row.get("label", ""))
        name = str(row.get("canonical_name", "")).strip()
        if not label or not name or label in SCOPED_LABELS:
            return
        key = (label, normalized(name))
        current = catalog.setdefault(key, {
            "node_id": row.get("node_id") or row.get("canonical_id") or row.get("entity_id") or stable_id("prior", label, normalized(name)),
            "label": label,
            "canonical_name": name,
            "aliases": [],
            "definition": row.get("definition", ""),
            "source_document_ids": [],
            "source_sentence_ids": [],
            "sources": [],
        })
        current["aliases"] = sorted({
            *current["aliases"], name, *row.get("aliases", []),
        }, key=str.casefold)
        current["source_document_ids"] = sorted({
            *current["source_document_ids"], *row.get("source_document_ids", []),
        })
        current["source_sentence_ids"] = sorted({
            value for value in [
                *current["source_sentence_ids"], *row.get("source_sentence_ids", []),
            ] if value
        })
        source = row.get("source")
        if source and source not in current["sources"]:
            current["sources"].append(source)

    for entry in lexicon.entries:
        add({**entry, "node_id": entry["canonical_id"], "source": "persistent_lexicon"})

    excluded = exclude_run_dir.resolve() if exclude_run_dir else None
    for run_dir in RUNS_DIR.iterdir() if RUNS_DIR.is_dir() else []:
        if excluded and run_dir.resolve() == excluded:
            continue
        manifest_path = run_dir / "manifest.json"
        if manifest_path.is_file():
            manifest = json_read(manifest_path)
            if str(manifest.get("test_scope", "full_paper")) != "full_paper":
                continue
        entity_path = run_dir / "canonical_entities_merged.jsonl"
        if not entity_path.is_file():
            entity_path = run_dir / "canonical_entities.jsonl"
        if not entity_path.is_file():
            continue
        mentions = {row.get("mention_id"): row for row in jsonl_read(run_dir / "mentions.jsonl")}
        parsed = json_read(run_dir / "parsed.json") if (run_dir / "parsed.json").is_file() else {}
        document_id = parsed.get("document", {}).get("id", "")
        for entity in jsonl_read(entity_path):
            sentence_ids = {str(value) for value in entity.get("sentence_ids", []) if value}
            for mention_id in entity.get("mention_ids", []):
                sentence_id = mentions.get(mention_id, {}).get("source", {}).get("sentence_id")
                if sentence_id:
                    sentence_ids.add(str(sentence_id))
            add({
                **entity,
                "node_id": entity.get("entity_id", ""),
                "source_document_ids": [document_id] if document_id else [],
                "source_sentence_ids": sorted(sentence_ids),
                "source": run_dir.name,
            })
    return sorted(catalog.values(), key=lambda row: (row["label"], row["canonical_name"].casefold()))


def attach_prior_nodes_same_type(
    candidates: list[dict], catalog: list[dict], *, maximum: int = 12,
) -> list[dict]:
    """Attach ranked earlier nodes, never comparing across ontology types."""
    for candidate in candidates:
        metadata = candidate.setdefault("candidate_metadata", {})
        labels = {candidate.get("label", "")}
        labels.update(
            row.get("label", "")
            for row in metadata.get("typing_alternatives", [])
        )
        labels.discard("")
        probe = normalized(candidate.get("canonical_name") or candidate["surface_text"])
        ranked = []
        for prior in catalog:
            if prior["label"] not in labels:
                continue
            alias_scores = [
                SequenceMatcher(None, probe, normalized(value)).ratio()
                for value in [prior["canonical_name"], *prior.get("aliases", [])]
            ]
            ranked.append({**prior, "lexical_rank_score": round(max(alias_scores, default=0.0), 4)})
        metadata["prior_nodes_same_type"] = sorted(
            ranked, key=lambda row: (-row["lexical_rank_score"], row["canonical_name"].casefold()),
        )[:maximum]
        metadata["prior_node_comparison_policy"] = (
            "DeepSeek may compare only against supplied nodes whose ontology label equals its selected label; "
            "returning an exact prior canonical name maps to that persistent node ID."
        )
    return candidates


def lexicon_spans(text: str, lexicon: Lexicon) -> list[Span]:
    spans: list[Span] = []
    entries = sorted(lexicon.entries, key=lambda row: max(map(len, row.get("aliases", [row["canonical_name"]]))), reverse=True)
    for entry in entries:
        if entry.get("status") == "deprecated":
            continue
        for alias in sorted({entry["canonical_name"], *entry.get("aliases", [])}, key=len, reverse=True):
            if len(alias) < 2:
                continue
            pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", re.I)
            for match in pattern.finditer(text):
                if entry["label"] == "Tool" and (
                    (match.start() > 0 and text[match.start() - 1] == "-")
                    or (match.end() < len(text) and text[match.end()] == "-")
                ):
                    continue
                if entry["label"] == "Model" and re.match(r"(?:\.\d+|o\b|\s+Search\b)", text[match.end():], re.I):
                    continue
                method = (
                    "persistent_lexicon"
                    if lexicon.alias_is_unambiguous(alias, entry["canonical_id"])
                    else "persistent_lexicon_ambiguous"
                )
                spans.append(Span(match.start(), match.end(), entry["label"], method, .99, entry["canonical_id"], entry["canonical_name"]))
    return spans
