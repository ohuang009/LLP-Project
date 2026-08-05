from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Pipeline.Node_Pipeline.contracts import judgment_artifact
from Pipeline.Node_Pipeline import (
    CONFIG_PATH, RUBRIC_PATH, canonical_entities, json_read, json_write, jsonl_read,
    jsonl_write, materialize_merges, mention_dedup_key, now_iso, optional_llm_review,
    remove_redundant_review_candidates, similarity_candidates,
)
from Pipeline.Graph_Persistence.neo4j_writer import upsert_run


def internal_candidate(row: dict) -> dict:
    generators = row.get("candidate_generators", [])
    generator = generators[0] if generators else {"name": "candidate_artifact", "confidence": 0.0}
    return {
        "candidate_id": row["candidate_id"],
        "document_id": row["document_id"],
        "label": row.get("suggested_label", ""),
        "surface_text": row["surface_text"],
        "canonical_name": row.get("suggested_canonical_name") or row["surface_text"],
        "canonical_id": "",
        "confidence": float(generator.get("confidence", 0.0)),
        "extraction_method": generator.get("name", "candidate_artifact"),
        "status": "review",
        "candidate_metadata": row.get("candidate_metadata", {}),
        "source": row["source"],
        "validation": {
            "outcome": "review", "judge": "rematerialized_model_decision",
            "llm_used": True, "reason": "Reapplying validated model output through current deterministic gates.",
        },
    }


class RecordedJudge:
    def __init__(self, judgments: dict[str, dict], calls: list[dict]) -> None:
        self.judgments = judgments
        self.audit_records = calls
        self.audit_by_sentence = {
            sentence_id: call
            for call in calls for sentence_id in call.get("sentence_ids", [])
        }

    def verify_runtime(self) -> None:
        return None

    def judge(self, records: list[dict]) -> dict[str, dict]:
        output = {}
        for record in records:
            sentence_id = record["sentence_id"]
            decisions = []
            for candidate in record.get("candidates", []):
                saved = self.judgments[candidate["candidate_id"]]
                decisions.append({
                    "candidate_id": candidate["candidate_id"],
                    "decision": saved["decision"],
                    "ontology_label": saved.get("ontology_label") or "NONE",
                    "canonical_name": saved.get("canonical_name") or candidate["surface_text"],
                    "definition": saved.get("definition") or "Recorded DeepSeek node decision.",
                    "reason": saved.get("reason") or "Recorded DeepSeek node decision.",
                    "confidence": float(saved.get("confidence", 0.0)),
                })
            output[sentence_id] = {
                "sentence_id": sentence_id,
                "candidate_judgments": decisions,
                "discoveries": [],
                "new_class_candidates": [],
                "_llm_audit": self.audit_by_sentence.get(sentence_id, {
                    "model": "deepseek-r1:7b", "request_hash": "recorded", "prompt_version": "recorded",
                }),
            }
        return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_run", type=Path)
    parser.add_argument("--target-name", default="")
    args = parser.parse_args()
    source = args.source_run.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target_name = args.target_name or f"{source.name}_grounded_{stamp}"
    target = source.parent / target_name
    if target.exists():
        raise FileExistsError(target)
    shutil.copytree(source, target)

    candidates = jsonl_read(target / "node_candidates.jsonl")
    trusted_ids = {row["candidate_id"] for row in candidates if row.get("trusted")}
    review = [internal_candidate(row) for row in candidates if not row.get("trusted")]
    trusted_mentions = [
        row for row in jsonl_read(target / "mentions.jsonl")
        if (row.get("candidate_id") or row.get("mention_id")) in trusted_ids
    ]
    review = remove_redundant_review_candidates(trusted_mentions, review)
    saved_judgments = {
        row["candidate_id"]: row for row in jsonl_read(target / "candidate_judgments.jsonl")
        if row["candidate_id"] not in trusted_ids
    }
    calls = jsonl_read(target / "llm_node_judge_calls.jsonl")
    accepted, review, rejected, judgments, _, status = optional_llm_review(
        review, lambda *_: None, node_judge=RecordedJudge(saved_judgments, calls),
    )

    mentions = []
    seen = set()
    for row in [*trusted_mentions, *accepted]:
        key = mention_dedup_key(row)
        if key not in seen:
            seen.add(key)
            mentions.append(row)
    rubric = json_read(RUBRIC_PATH)
    trusted_judgments = [
        judgment_artifact(row, decision="accept", validation=row["validation"], rubric=rubric)
        for row in trusted_mentions
    ]
    jsonl_write(target / "mentions.jsonl", mentions)
    jsonl_write(target / "review_queue.jsonl", review)
    jsonl_write(target / "node_review_candidates.jsonl", review)
    jsonl_write(target / "node_rejections.jsonl", rejected)
    jsonl_write(target / "candidate_judgments.jsonl", [*trusted_judgments, *judgments])

    entities = canonical_entities(mentions)
    jsonl_write(target / "canonical_entities.jsonl", entities)
    threshold = float(json_read(CONFIG_PATH)["similarity"]["candidate_threshold"])
    pairs = similarity_candidates(entities, threshold)
    jsonl_write(target / "similar_nodes_review.jsonl", pairs)
    merged = materialize_merges(target)
    graph = upsert_run(target, json_read(target / "parsed.json"), merged, [])
    json_write(target / "neo4j_upsert.json", graph)

    manifest = json_read(target / "manifest.json")
    manifest.update({
        "run_id": target.name,
        "created_at": now_iso(),
        "rematerialized_from": source.name,
        "llm": status,
    })
    manifest["counts"].update({
        "accepted_mentions": len(mentions),
        "canonical_entities": len(entities),
        "node_review_candidates": len(review),
        "rejected_node_candidates": len(rejected),
        "similar_node_candidates": len(pairs),
        "merged_entities": len(merged),
    })
    manifest["neo4j"] = graph
    json_write(target / "manifest.json", manifest)
    print(target)


if __name__ == "__main__":
    main()
