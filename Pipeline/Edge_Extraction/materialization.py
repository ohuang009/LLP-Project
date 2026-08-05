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

from Pipeline.Node_Pipeline.common import jsonl_write, stable_id

def materialize_canonical_relationships(
    run_dir: Path, entities: list[dict], assertions: list[dict]
) -> list[dict]:
    mention_to_entity = {
        mention_id: entity["entity_id"]
        for entity in entities
        for mention_id in entity.get("mention_ids", [])
    }
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for assertion in assertions:
        subject = mention_to_entity.get(assertion["subject_mention_id"])
        target = mention_to_entity.get(assertion["object_mention_id"])
        if not subject or not target or subject == target:
            continue
        grouped[(subject, assertion["predicate"], target)].append(assertion)
    rows = []
    for key, values in sorted(grouped.items()):
        evidence_units: dict[tuple, dict] = {}
        for assertion in values:
            sentence_ids = tuple(assertion.get("evidence_sentence_ids") or [])
            evidence_key = (
                ("sentences", *sentence_ids)
                if sentence_ids else ("assertion", assertion["assertion_id"])
            )
            previous = evidence_units.get(evidence_key)
            if previous is None or assertion.get("confidence", 0) > previous.get("confidence", 0):
                evidence_units[evidence_key] = assertion
        support = sorted(evidence_units.values(), key=lambda row: row["assertion_id"])
        evidence_sentence_ids = sorted({
            sentence_id
            for assertion in support
            for sentence_id in assertion.get("evidence_sentence_ids", [])
            if sentence_id
        })
        rows.append({
            "relationship_id": stable_id("relationship", *key),
            "subject_entity_id": key[0],
            "predicate": key[1],
            "object_entity_id": key[2],
            "support_count": len(support),
            "assertion_ids": [row["assertion_id"] for row in support],
            "evidence_sentence_ids": evidence_sentence_ids,
            "status": "materialized_from_unique_evidence_units",
        })
    jsonl_write(run_dir / "canonical_relationships.jsonl", rows)
    return rows
