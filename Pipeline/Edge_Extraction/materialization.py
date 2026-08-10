"""Aggregate accepted mention-level assertions into canonical graph edges."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from Pipeline.core import jsonl_write, stable_id


def materialize_canonical_relationships(
    run_dir: Path, entities: list[dict], assertions: list[dict],
) -> list[dict]:
    """Aggregate accepted mention-level assertions into canonical graph edges."""
    mention_to_entity = {
        mention_id: entity["entity_id"]
        for entity in entities for mention_id in entity.get("mention_ids", [])
    }
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for assertion in assertions:
        subject = mention_to_entity.get(assertion["subject_mention_id"])
        obj = mention_to_entity.get(assertion["object_mention_id"])
        if subject and obj and subject != obj:
            grouped[(subject, assertion["predicate"], obj)].append(assertion)
    rows = []
    for (subject, predicate, obj), support in sorted(grouped.items()):
        sentence_ids = sorted({
            sentence_id for row in support
            for sentence_id in row.get("evidence_sentence_ids", [])
        })
        rows.append({
            "relationship_id": stable_id("relationship", subject, predicate, obj),
            "subject_entity_id": subject, "predicate": predicate, "object_entity_id": obj,
            "support_count": len(support),
            "assertion_ids": [row["assertion_id"] for row in support],
            "evidence_sentence_ids": sentence_ids, "status": "materialized",
        })
    jsonl_write(run_dir / "canonical_relationships.jsonl", rows)
    return rows
