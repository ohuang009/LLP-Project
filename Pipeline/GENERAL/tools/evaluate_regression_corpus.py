from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT.parents[1] / "PipelineAudits" / "runs"
DEFAULT_CONFIG = ROOT / "config" / "regression_corpus.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalized(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def latest_run(source_filename: str) -> Path | None:
    matches: list[tuple[str, Path]] = []
    for manifest_path in RUNS.glob("*/manifest.json"):
        try:
            manifest = read_json(manifest_path)
        except (OSError, ValueError):
            continue
        if manifest.get("source_filename") == source_filename:
            matches.append((manifest.get("created_at", ""), manifest_path.parent))
    return max(matches, default=("", None), key=lambda item: item[0])[1]


def evaluate_case(spec: dict, run_dir: Path | None) -> dict:
    failures: list[str] = []
    if run_dir is None:
        return {"id": spec["id"], "status": "FAIL", "run_id": None, "failures": ["No completed run found."]}

    manifest = read_json(run_dir / "manifest.json")
    validation = read_json(run_dir / "validation_report.json")
    entities = read_jsonl(run_dir / "canonical_entities_merged.jsonl")
    relationships = read_jsonl(run_dir / "canonical_relationships.jsonl")
    assertions = read_jsonl(run_dir / "assertions.jsonl")
    questions = read_json(run_dir / "question_answerability.json")

    entity_keys = {(row["label"], normalized(row["canonical_name"])) for row in entities}
    for label, name in spec.get("required_entities", []):
        if (label, normalized(name)) not in entity_keys:
            failures.append(f"Missing required entity {label}:{name}.")
    forbidden = {normalized(value) for value in spec.get("forbidden_entity_names", [])}
    present_forbidden = sorted(row["canonical_name"] for row in entities if normalized(row["canonical_name"]) in forbidden)
    if present_forbidden:
        failures.append("Forbidden entities present: " + ", ".join(present_forbidden) + ".")

    duplicate_keys = len(entities) - len(entity_keys)
    if duplicate_keys:
        failures.append(f"Found {duplicate_keys} duplicate canonical label/name keys.")
    if validation.get("status") != "PASS":
        failures.append("Validation report did not pass.")
    if len(relationships) < int(spec.get("minimum_relationships", 0)):
        failures.append(f"Only {len(relationships)} relationships were produced.")
    answerable = int(questions.get("answerable_questions", 0))
    if answerable < int(spec.get("minimum_answerable_questions", 0)):
        failures.append(f"Only {answerable} questions were answerable.")
    if any(
        not row.get("evidence_quote")
        or (not row.get("evidence_sentence_ids") and row.get("extraction_method") != "structural_rule")
        for row in assertions
    ):
        failures.append("At least one accepted assertion lacks exact sentence evidence.")
    neo4j = manifest.get("neo4j", {})
    if neo4j.get("duplicate_canonical_keys", 0) != 0:
        failures.append("Neo4j reported duplicate canonical keys.")
    if neo4j.get("traceable_relationships") != neo4j.get("relationships"):
        failures.append("Not every Neo4j relationship is traceable.")

    return {
        "id": spec["id"],
        "status": "FAIL" if failures else "PASS",
        "run_id": run_dir.name,
        "entities": len(entities),
        "relationships": len(relationships),
        "answerable_questions": answerable,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the latest run for each engineered-water regression paper.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = read_json(args.config)
    results = [evaluate_case(spec, latest_run(spec["source_filename"])) for spec in config["papers"]]
    payload = {"status": "PASS" if all(row["status"] == "PASS" for row in results) else "FAIL", "papers": results}
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
