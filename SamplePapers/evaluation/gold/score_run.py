"""Strictly score one completed extraction run against the EWS-R07 gold set."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


DEFAULT_GOLD = Path(__file__).with_name("EWS-R07_sentence_gold.jsonl")


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _normalized(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def score(gold_path: Path, run_dir: Path) -> dict:
    gold = [row for row in _jsonl(gold_path) if row.get("sentence_id")]
    mentions = _jsonl(run_dir / "mentions.jsonl")
    entities = _jsonl(run_dir / "canonical_entities_merged.jsonl")
    relationships = _jsonl(run_dir / "canonical_relationships.jsonl")

    mention_index = {
        (
            str(row.get("source", {}).get("sentence_id") or ""),
            str(row.get("label") or ""),
            str(row.get("surface_text") or ""),
        )
        for row in mentions
    }
    node_targets = [
        (row["sentence_id"], node[1], node[0])
        for row in gold
        for node in row.get("required_nodes", [])
    ]
    node_hits = sum(target in mention_index for target in node_targets)

    entity_index: dict[tuple[str, str], str] = {}
    for row in entities:
        names = [row.get("canonical_name", ""), *row.get("aliases", [])]
        for name in names:
            entity_index[(str(row.get("label") or ""), _normalized(name))] = row["entity_id"]

    relationship_index = {
        (
            row.get("subject_entity_id"), row.get("predicate"),
            row.get("object_entity_id"), sentence_id,
        )
        for row in relationships
        for sentence_id in row.get("evidence_sentence_ids", [])
    }
    predicate_targets: Counter[str] = Counter()
    predicate_resolved: Counter[str] = Counter()
    predicate_hits: Counter[str] = Counter()
    edge_targets = edge_resolved = edge_hits = 0
    forbidden_targets = forbidden_resolved = forbidden_violations = 0
    violation_details: list[dict] = []
    for row in gold:
        for subject_name, subject_type, predicate, object_name, object_type in row.get(
            "required_relationships", []
        ):
            edge_targets += 1
            predicate_targets[predicate] += 1
            subject_id = entity_index.get((subject_type, _normalized(subject_name)))
            object_id = entity_index.get((object_type, _normalized(object_name)))
            if not subject_id or not object_id:
                continue
            edge_resolved += 1
            predicate_resolved[predicate] += 1
            key = (subject_id, predicate, object_id, row["sentence_id"])
            if key in relationship_index:
                edge_hits += 1
                predicate_hits[predicate] += 1

        for subject_name, subject_type, predicate, object_name, object_type in row.get(
            "forbidden_relationships", []
        ):
            forbidden_targets += 1
            subject_id = entity_index.get((subject_type, _normalized(subject_name)))
            object_id = entity_index.get((object_type, _normalized(object_name)))
            if not subject_id or not object_id:
                continue
            forbidden_resolved += 1
            key = (subject_id, predicate, object_id, row["sentence_id"])
            if key in relationship_index:
                forbidden_violations += 1
                violation_details.append({
                    "sentence_id": row["sentence_id"],
                    "subject": subject_name, "predicate": predicate, "object": object_name,
                })

    predicates = {
        predicate: {
            "targets": predicate_targets[predicate],
            "resolvable_endpoints": predicate_resolved[predicate],
            "hits": predicate_hits[predicate],
        }
        for predicate in sorted(predicate_targets)
    }
    return {
        "schema_version": "1.0",
        "policy": "strict sentence+span+type nodes; strict endpoint+predicate+direction+evidence edges",
        "run": str(run_dir.resolve()),
        "nodes": {
            "hits": node_hits,
            "targets": len(node_targets),
            "recall": round(node_hits / len(node_targets), 4) if node_targets else 1.0,
        },
        "relationships": {
            "hits": edge_hits,
            "targets": edge_targets,
            "resolvable_endpoints": edge_resolved,
            "recall": round(edge_hits / edge_targets, 4) if edge_targets else 1.0,
        },
        "forbidden_relationships": {
            "violations": forbidden_violations,
            "targets": forbidden_targets,
            "resolvable_endpoints": forbidden_resolved,
            "details": violation_details,
        },
        "predicates": predicates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    args = parser.parse_args()
    print(json.dumps(score(args.gold, args.run), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
