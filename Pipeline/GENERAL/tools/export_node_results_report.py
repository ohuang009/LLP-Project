from __future__ import annotations

import argparse
from pathlib import Path
import json


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def cell(value: object) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    candidates = read_jsonl(run_dir / "node_candidates.jsonl")
    judgments = {row["candidate_id"]: row for row in read_jsonl(run_dir / "candidate_judgments.jsonl")}
    accepted = read_jsonl(run_dir / "canonical_entities_merged.jsonl")
    review = read_jsonl(run_dir / "review_queue.jsonl")
    rejected = read_jsonl(run_dir / "node_rejections.jsonl")

    possible = []
    for row in candidates:
        judgment = judgments.get(row["candidate_id"], {})
        possible.append({
            "surface_text": row["surface_text"],
            "suggested_label": row.get("suggested_label", ""),
            "trusted": bool(row.get("trusted")),
            "decision": judgment.get("decision", "unjudged"),
            "final_label": judgment.get("ontology_label", ""),
            "canonical_name": judgment.get("canonical_name", row.get("suggested_canonical_name", "")),
            "sentence_id": row.get("sentence_id", ""),
        })

    report = {
        "run_id": run_dir.name,
        "test_scope": manifest.get("test_scope", ""),
        "counts": {
            "possible_candidates": len(possible),
            "accepted_mentions": manifest["counts"]["accepted_mentions"],
            "accepted_canonical_nodes": len(accepted),
            "needs_review": len(review),
            "rejected": len(rejected),
        },
        "possible_candidates": possible,
        "accepted_canonical_nodes": accepted,
        "needs_review": review,
        "rejected_candidates": rejected,
    }
    (run_dir / "waterRAG_node_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# WaterRAG abstract node-pipeline results",
        "",
        f"Run: `{run_dir.name}`  ",
        f"Scope: `{manifest.get('test_scope', '')}`  ",
        f"Possible candidates: **{len(possible)}** · Accepted mentions: **{manifest['counts']['accepted_mentions']}** "
        f"· Accepted canonical nodes: **{len(accepted)}** · Review: **{len(review)}** · Rejected: **{len(rejected)}**",
        "",
        "## Possible candidates and final decisions",
        "",
        "| Surface text | Suggested type | Final decision | Final type | Canonical name | Sentence ID |",
        "|---|---|---|---|---|---|",
    ]
    lines.extend(
        f"| {cell(row['surface_text'])} | {cell(row['suggested_label'])} | {cell(row['decision'])} "
        f"| {cell(row['final_label'])} | {cell(row['canonical_name'])} | `{cell(row['sentence_id'])}` |"
        for row in possible
    )
    lines.extend(["", "## Accepted canonical nodes", "", "| Canonical node | Type | Mentions | Sentence IDs |", "|---|---|---:|---|"])
    lines.extend(
        f"| {cell(row['canonical_name'])} | {cell(row['label'])} | {row.get('mention_count', 0)} "
        f"| {', '.join(f'`{cell(value)}`' for value in row.get('sentence_ids', []))} |"
        for row in accepted
    )
    lines.extend(["", "## Needs human review", "", "| Candidate | Proposed type | Sentence ID | Reason |", "|---|---|---|---|"])
    lines.extend(
        f"| {cell(row.get('canonical_name') or row.get('surface_text'))} | {cell(row.get('label'))} "
        f"| `{cell(row.get('source', {}).get('sentence_id'))}` | {cell(row.get('validation', {}).get('reason'))} |"
        for row in review
    )
    (run_dir / "waterRAG_node_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    outputs = manifest.setdefault("outputs", [])
    for name in ("waterRAG_node_results.md", "waterRAG_node_results.json"):
        if name not in outputs:
            outputs.append(name)
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(run_dir / "waterRAG_node_results.md")


if __name__ == "__main__":
    main()
