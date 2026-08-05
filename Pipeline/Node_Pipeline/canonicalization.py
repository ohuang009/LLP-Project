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

from .common import PROVENANCE_ONLY_LABELS, SCOPED_LABELS, jsonl_read, jsonl_write, normalized, stable_id

def canonical_entities(mentions: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for mention in mentions:
        if mention["label"] in PROVENANCE_ONLY_LABELS:
            continue
        canonical_name = mention.get("canonical_name") or mention["surface_text"]
        # Canonical IDs can originate from different discovery channels (for
        # example a reviewed lexicon ID and a deterministic introduction ID).
        # Exact normalized names of reusable entities are nevertheless one
        # identity. Occurrence-scoped statistics retain their occurrence ID.
        key_name = (
            mention.get("canonical_id") or normalized(canonical_name)
            if mention["label"] in SCOPED_LABELS
            else normalized(canonical_name)
        )
        groups[(mention["label"], key_name)].append(mention)
    entities = []
    for (label, key_name), rows in groups.items():
        names = [row.get("canonical_name") or row["surface_text"] for row in rows]
        canonical_name = max(names, key=len)
        # A pronoun or phrase such as "this model" is evidence, never a reusable lexical alias.
        lexical_surfaces = {row["surface_text"] for row in rows if not row.get("reference_resolution")}
        aliases = sorted(lexical_surfaces | set(names), key=str.casefold)
        persistent_id = next(
            (row.get("canonical_id") for row in rows if row.get("extraction_method") == "persistent_lexicon" and row.get("canonical_id")),
            "",
        )
        entity_id = persistent_id or next((row.get("canonical_id") for row in rows if row.get("canonical_id")), "") or stable_id("entity", label, key_name)
        evidence = [
            {"mention_id": row["mention_id"], "quote": row["source"].get("evidence_quote", ""),
             "span": row["source"].get("quote", ""), "pages": row["source"].get("pages", []),
             "sentence_id": row["source"]["sentence_id"],
             "section_title": row["source"].get("section_title", ""),
             "reference_resolution": row.get("reference_resolution")}
            for row in rows[:5]
        ]
        sentence_ids = sorted({row["source"]["sentence_id"] for row in rows})
        entities.append({
            "entity_id": entity_id, "label": label, "canonical_name": canonical_name,
            "aliases": aliases, "mention_ids": [row["mention_id"] for row in rows],
            "mention_count": len(rows), "sentence_ids": sentence_ids,
            "evidence": evidence, "status": "canonical_unmerged",
            "identity_scope": "occurrence" if label in SCOPED_LABELS else "canonical",
            "attributes": [row.get("attributes", {}) for row in rows if row.get("attributes")],
        })
    return sorted(entities, key=lambda row: (row["label"], row["canonical_name"].casefold()))


def similarity_features(left: dict, right: dict) -> tuple[float, list[str], dict]:
    a, b = left["canonical_name"], right["canonical_name"]
    na, nb = normalized(a), normalized(b)
    ta, tb = set(na.split()), set(nb.split())
    jaccard = len(ta & tb) / max(1, len(ta | tb))
    char = SequenceMatcher(None, na, nb).ratio()
    alias_overlap = bool({normalized(x) for x in left["aliases"]} & {normalized(x) for x in right["aliases"]})
    containment = (na in nb or nb in na) and min(len(na), len(nb)) >= 4
    left_versions = re.findall(r"\d+(?:\.\d+)*", a)
    right_versions = re.findall(r"\d+(?:\.\d+)*", b)
    version_conflict = bool(left_versions and right_versions and left_versions != right_versions)
    if version_conflict:
        return 0.0, ["different explicit model or software versions"], {
            "alias_overlap": alias_overlap,
            "containment": containment, "version_conflict": True,
            "token_jaccard": round(jaccard, 4), "character_similarity": round(char, 4),
        }
    score = max(.995 if alias_overlap else 0, .88 if containment else 0, .55*jaccard + .45*char)
    reasons = []
    if alias_overlap:
        reasons.append("normalized alias overlap")
    if containment:
        reasons.append("one specific name contains the other")
    if jaccard >= .5:
        reasons.append("strong token overlap")
    if char >= .78:
        reasons.append("strong spelling similarity")
    return score, reasons or ["moderate combined lexical similarity"], {
        "alias_overlap": alias_overlap,
        "containment": containment, "version_conflict": False,
        "token_jaccard": round(jaccard, 4), "character_similarity": round(char, 4),
    }


def similarity_candidates(entities: list[dict], threshold: float = .64) -> list[dict]:
    pairs: list[dict] = []
    for index, left in enumerate(entities):
        if left["label"] in SCOPED_LABELS:
            continue
        for right in entities[index + 1:]:
            if right["label"] != left["label"] or right["label"] in SCOPED_LABELS:
                continue
            score, reasons, features = similarity_features(left, right)
            if score < threshold:
                continue
            preferred = max((left["canonical_name"], right["canonical_name"]), key=len)
            pairs.append({
                "resolution_id": stable_id("resolution", left["entity_id"], right["entity_id"]),
                "label": left["label"], "left": left, "right": right,
                "score": round(score, 4), "features": features, "reasons": reasons,
                "recommended_canonical_name": preferred,
                "recommended_action": "human_review", "status": "pending",
                "merge_effect": "Aliases and mentions combine, then accepted relationship endpoints are rewired to the merged entity.",
            })
    return sorted(pairs, key=lambda row: (-row["score"], row["left"]["canonical_name"].casefold()))


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def materialize_merges(run_dir: Path) -> list[dict]:
    entities = jsonl_read(run_dir / "canonical_entities.jsonl")
    pairs = jsonl_read(run_dir / "similar_nodes_review.jsonl")
    uf = UnionFind(row["entity_id"] for row in entities)
    chosen_names: dict[str, str] = {}
    for pair in pairs:
        if pair.get("status") == "same":
            left_id, right_id = pair["left"]["entity_id"], pair["right"]["entity_id"]
            uf.union(left_id, right_id)
            chosen_names[left_id] = pair.get("chosen_canonical_name") or pair["recommended_canonical_name"]
            chosen_names[right_id] = chosen_names[left_id]
    groups: dict[str, list[dict]] = defaultdict(list)
    for entity in entities:
        groups[uf.find(entity["entity_id"])].append(entity)
    merged = []
    for root, rows in groups.items():
        names = [row["canonical_name"] for row in rows]
        canonical_name = next((chosen_names[row["entity_id"]] for row in rows if row["entity_id"] in chosen_names), max(names, key=len))
        entity_id = rows[0]["entity_id"] if len(rows) == 1 else stable_id("entity", rows[0]["label"], normalized(canonical_name))
        merged.append({
            "entity_id": entity_id,
            "label": rows[0]["label"], "canonical_name": canonical_name,
            "aliases": sorted({alias for row in rows for alias in row["aliases"]} | set(names), key=str.casefold),
            "mention_ids": [mention for row in rows for mention in row["mention_ids"]],
            "mention_count": sum(row["mention_count"] for row in rows),
            "sentence_ids": sorted({sentence_id for row in rows for sentence_id in row.get("sentence_ids", [])}),
            "merged_entity_ids": [row["entity_id"] for row in rows],
            "evidence": [evidence for row in rows for evidence in row["evidence"]][:10],
            "identity_scope": "occurrence" if rows[0]["label"] in SCOPED_LABELS else "canonical",
            "attributes": [attribute for row in rows for attribute in row.get("attributes", [])],
            "status": "human_merged" if len(rows) > 1 else "canonical_unmerged",
            "relationship_policy": "Accepted assertion endpoints are rewired to this entity ID; duplicate assertion evidence is retained.",
        })
    jsonl_write(run_dir / "canonical_entities_merged.jsonl", merged)
    return merged
