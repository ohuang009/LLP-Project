"""Validate gold JSONL against pinned parser output and the runtime ontology."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GOLD = Path(__file__).with_name("EWS-R07_sentence_gold.jsonl")
DEFAULT_PARSED = (
    PROJECT_ROOT / "PipelineAudits" / "runs"
    / "evaluation_07_edge_iter04_20260810" / "parsed.json"
)
DEFAULT_ONTOLOGY = PROJECT_ROOT / "Pipeline" / "GENERAL" / "ontology" / "ontology.json"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate(gold_path: Path, parsed_path: Path, ontology_path: Path) -> dict:
    records = _jsonl(gold_path)
    annotations = [row for row in records if row.get("sentence_id")]
    parsed = _json(parsed_path)
    ontology = _json(ontology_path)
    sentences = {
        sentence["id"]: sentence["text"]
        for section in parsed.get("sections", [])
        for paragraph in section.get("paragraphs", [])
        for sentence in paragraph.get("sentences", [])
    }
    node_types = set(ontology.get("nodes", {}))
    predicates = ontology.get("relationships", {})
    failures: list[str] = []
    seen: set[str] = set()

    for row in annotations:
        sentence_id = str(row["sentence_id"])
        if sentence_id in seen:
            failures.append(f"duplicate sentence: {sentence_id}")
        seen.add(sentence_id)
        if sentence_id not in sentences:
            failures.append(f"unknown sentence: {sentence_id}")
            continue
        if row.get("text") != sentences[sentence_id]:
            failures.append(f"changed sentence text: {sentence_id}")
        for span, label, _canonical_name in row.get("required_nodes", []):
            if span not in row["text"]:
                failures.append(f"non-exact span: {sentence_id}: {span!r}")
            if label not in node_types:
                failures.append(f"unknown node type: {sentence_id}: {label}")
        for _s_name, s_type, predicate, _o_name, o_type in row.get(
            "required_relationships", []
        ):
            spec = predicates.get(predicate)
            if spec is None:
                failures.append(f"unknown predicate: {sentence_id}: {predicate}")
                continue
            if s_type not in spec.get("domain", []):
                failures.append(
                    f"invalid domain: {sentence_id}: {s_type} -{predicate}-> {o_type}"
                )
            if o_type not in spec.get("range", []):
                failures.append(
                    f"invalid range: {sentence_id}: {s_type} -{predicate}-> {o_type}"
                )

    return {
        "status": "PASS" if not failures else "FAIL",
        "annotations": len(annotations),
        "required_nodes": sum(len(row.get("required_nodes", [])) for row in annotations),
        "required_relationships": sum(
            len(row.get("required_relationships", [])) for row in annotations
        ),
        "forbidden_relationships": sum(
            len(row.get("forbidden_relationships", [])) for row in annotations
        ),
        "ontology_gap_notes": sum(len(row.get("ontology_gaps", [])) for row in annotations),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--parsed", type=Path, default=DEFAULT_PARSED)
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY)
    args = parser.parse_args()
    report = validate(args.gold, args.parsed, args.ontology)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
