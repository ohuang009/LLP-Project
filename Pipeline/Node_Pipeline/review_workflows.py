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

from .adjudication import _prior_node_comparison
from .canonicalization import canonical_entities, materialize_merges, similarity_candidates
from .common import (CONFIG_PATH, DECISIONS_PATH, GENERIC_TERMS, LEXICON_PATH, ONTOLOGY_PATH,
    PROVENANCE_ONLY_LABELS, RUBRIC_PATH, SCOPED_LABELS, append_jsonl, json_read, json_write,
    jsonl_read, jsonl_write, normalized, now_iso, stable_id)
from .lexicon import Lexicon
from .contracts import judgment_artifact

def apply_node_review(
    run_dir: Path,
    candidate_id: str,
    decision: str,
    *,
    ontology_label: str = "",
    canonical_name: str = "",
    reason: str = "",
) -> dict:
    """Apply a human accept/reject decision and rebuild downstream node artifacts."""
    if decision not in {"accept", "reject"}:
        raise ValueError("decision must be 'accept' or 'reject'")
    queue = jsonl_read(run_dir / "review_queue.jsonl")
    item = next((row for row in queue if row.get("candidate_id") == candidate_id), None)
    if item is None:
        raise KeyError("Unknown or already reviewed node candidate")

    sentence_id = str(item.get("source", {}).get("sentence_id", "")).strip()
    reviewed_at = now_iso()
    effective_decision = decision
    if not sentence_id:
        effective_decision = "reject"
        reason = "Rejected by the sentence-traceability gate because no specific sentence ID identifies its evidence."

    mentions = jsonl_read(run_dir / "mentions.jsonl")
    rubric = json_read(RUBRIC_PATH)
    if effective_decision == "accept":
        label = (ontology_label or item.get("label", "")).strip()
        allowed_labels = set(json_read(ONTOLOGY_PATH).get("nodes", {}))
        if label not in allowed_labels:
            raise ValueError("Choose one existing ontology node type before accepting this candidate.")
        name = (canonical_name or item.get("canonical_name") or item.get("surface_text", "")).strip()
        if not name or normalized(name) in GENERIC_TERMS:
            raise ValueError("A vague node cannot be accepted; supply a clear, specific canonical name or reject it.")
        comparison = _prior_node_comparison(item, label, name)
        accepted_name = comparison["matched_canonical_name"] or name
        accepted_id = comparison["matched_node_id"] or item.get("canonical_id", "")
        validation = {
            "outcome": "accept",
            "judge": "human_node_reviewer",
            "llm_used": bool(item.get("validation", {}).get("llm_used")),
            "reason": reason.strip() or "A human accepted this grounded candidate under the approved node-validity rubric.",
            "normalization": {
                **item.get("validation", {}).get("normalization", {}),
                "human_canonical_name": accepted_name,
                "reviewed_at": reviewed_at,
            },
            "prior_node_comparison": comparison,
        }
        mention = {
            **item,
            "mention_id": stable_id("mention", candidate_id, "human_accept"),
            "label": label,
            "canonical_name": accepted_name,
            "canonical_id": accepted_id,
            "confidence": 1.0,
            "extraction_method": "human_reviewed_node_candidate",
            "status": "accepted",
            "validation": validation,
        }
        if all(row.get("mention_id") != mention["mention_id"] for row in mentions):
            mentions.append(mention)
        if label not in SCOPED_LABELS and label not in PROVENANCE_ONLY_LABELS:
            entry = Lexicon().add_reviewed_aliases(
                label,
                accepted_name,
                sorted({item.get("surface_text", ""), accepted_name} - {""}, key=str.casefold),
                {
                    "run_id": run_dir.name,
                    "candidate_id": candidate_id,
                    "document_id": item.get("document_id", ""),
                    "sentence_id": sentence_id,
                    "reviewed_at": reviewed_at,
                },
            )
            mention["canonical_id"] = entry["canonical_id"]
        decided_item = mention
    else:
        validation = {
            **item.get("validation", {}),
            "outcome": "reject",
            "judge": "human_node_reviewer" if sentence_id else "sentence_traceability_gate",
            "llm_used": bool(item.get("validation", {}).get("llm_used")),
            "reason": reason.strip() or "A human rejected this candidate under the approved node-validity rubric.",
            "reviewed_at": reviewed_at,
        }
        decided_item = {**item, "status": "rejected", "validation": validation}
        rejections = jsonl_read(run_dir / "node_rejections.jsonl")
        rejections = [row for row in rejections if row.get("candidate_id") != candidate_id]
        rejections.append(decided_item)
        jsonl_write(run_dir / "node_rejections.jsonl", rejections)

    remaining = [row for row in queue if row.get("candidate_id") != candidate_id]
    jsonl_write(run_dir / "review_queue.jsonl", remaining)
    jsonl_write(run_dir / "node_review_candidates.jsonl", remaining)
    jsonl_write(run_dir / "mentions.jsonl", mentions)

    judgments = jsonl_read(run_dir / "candidate_judgments.jsonl")
    judgments = [row for row in judgments if row.get("candidate_id") != candidate_id]
    judgments.append(judgment_artifact(
        decided_item,
        decision=effective_decision,
        validation=decided_item["validation"],
        rubric=rubric,
    ))
    jsonl_write(run_dir / "candidate_judgments.jsonl", judgments)

    entities = canonical_entities(mentions)
    jsonl_write(run_dir / "canonical_entities.jsonl", entities)
    old_pairs = {row["resolution_id"]: row for row in jsonl_read(run_dir / "similar_nodes_review.jsonl")}
    threshold = float(json_read(CONFIG_PATH)["similarity"]["candidate_threshold"])
    pairs = similarity_candidates(entities, threshold)
    for pair in pairs:
        previous = old_pairs.get(pair["resolution_id"])
        if previous and previous.get("status") in {"same", "distinct"}:
            pair.update({
                "status": previous["status"],
                "chosen_canonical_name": previous.get("chosen_canonical_name", ""),
                "reviewed_at": previous.get("reviewed_at", ""),
            })
    jsonl_write(run_dir / "similar_nodes_review.jsonl", pairs)
    merged = materialize_merges(run_dir)
    json_write(run_dir / "lexicon_snapshot.json", Lexicon().data)

    manifest = json_read(run_dir / "manifest.json")
    manifest["counts"].update({
        "accepted_mentions": len(mentions),
        "canonical_entities": len(entities),
        "node_review_candidates": len(remaining),
        "rejected_node_candidates": len(jsonl_read(run_dir / "node_rejections.jsonl")),
        "similar_node_candidates": len(pairs),
        "merged_entities": len(merged),
        "lexicon_entries": len(Lexicon().entries),
    })
    json_write(run_dir / "manifest.json", manifest)
    return {
        "item": decided_item,
        "effective_decision": effective_decision,
        "mentions": mentions,
        "entities": merged,
        "similar_nodes": pairs,
        "manifest": manifest,
    }


def apply_reference_resolution(run_dir: Path, resolution_id: str, decision: str, target_mention_id: str = "") -> dict:
    if decision not in {"resolve", "ignore"}:
        raise ValueError("decision must be 'resolve' or 'ignore'")
    queue = jsonl_read(run_dir / "reference_resolution_review.jsonl")
    item = next((row for row in queue if row["resolution_id"] == resolution_id), None)
    if item is None:
        raise KeyError("Unknown reference-resolution candidate")
    if item.get("status") != "pending":
        raise ValueError("This reference has already been reviewed.")

    mentions = jsonl_read(run_dir / "mentions.jsonl")
    if decision == "resolve":
        candidate = next(
            (row for row in item["candidate_targets"] if row["target_mention_id"] == target_mention_id),
            None,
        )
        if candidate is None:
            raise ValueError("The selected target is not one of the grounded antecedent candidates.")
        target = next((row for row in mentions if row["mention_id"] == target_mention_id), None)
        if target is None:
            raise ValueError("The selected antecedent mention is no longer available.")
        reviewed_at = now_iso()
        mention = {
            "candidate_id": resolution_id,
            "mention_id": stable_id("mention", resolution_id, target_mention_id, "human"),
            "document_id": item["document_id"],
            "label": target["label"],
            "surface_text": item["surface_text"],
            "canonical_name": target.get("canonical_name") or target["surface_text"],
            "canonical_id": target.get("canonical_id", ""),
            "confidence": 1.0,
            "extraction_method": "human_reference_resolution",
            "status": "accepted",
            "source": item["source"],
            "reference_resolution": {
                "status": "human_resolved",
                "resolution_id": resolution_id,
                "confidence": 1.0,
                "target_mention_id": target_mention_id,
                "target_canonical_name": target.get("canonical_name") or target["surface_text"],
                "antecedent_sentence_id": candidate["antecedent_sentence_id"],
                "antecedent_evidence_quote": candidate["antecedent_evidence_quote"],
                "reviewed_at": reviewed_at,
                "reason": "A human selected this target from the finite, evidence-grounded antecedent list.",
            },
            "validation": {
                "outcome": "accept", "judge": "human_reference_reviewer", "llm_used": False,
                "reason": "The vague surface mention was retained after explicit human antecedent selection.",
            },
        }
        mentions.append(mention)
        item.update({"status": "resolved", "selected_target_mention_id": target_mention_id, "reviewed_at": reviewed_at})
    else:
        item.update({"status": "ignored", "selected_target_mention_id": "", "reviewed_at": now_iso()})

    jsonl_write(run_dir / "mentions.jsonl", mentions)
    jsonl_write(run_dir / "reference_resolution_review.jsonl", queue)
    entities = canonical_entities(mentions)
    jsonl_write(run_dir / "canonical_entities.jsonl", entities)

    old_pairs = {row["resolution_id"]: row for row in jsonl_read(run_dir / "similar_nodes_review.jsonl")}
    threshold = float(json_read(CONFIG_PATH)["similarity"]["candidate_threshold"])
    pairs = similarity_candidates(entities, threshold)
    for pair in pairs:
        previous = old_pairs.get(pair["resolution_id"])
        if previous and previous.get("status") in {"same", "distinct"}:
            pair.update({
                "status": previous["status"],
                "chosen_canonical_name": previous.get("chosen_canonical_name", ""),
                "reviewed_at": previous.get("reviewed_at", ""),
            })
    jsonl_write(run_dir / "similar_nodes_review.jsonl", pairs)
    merged = materialize_merges(run_dir)

    manifest = json_read(run_dir / "manifest.json")
    manifest["counts"].update({
        "accepted_mentions": len(mentions),
        "canonical_entities": len(entities),
        "reference_review_pending": sum(row.get("status") == "pending" for row in queue),
        "reference_mentions_resolved": sum(bool(row.get("reference_resolution")) for row in mentions),
        "similar_node_candidates": len(pairs),
        "merged_entities": len(merged),
    })
    json_write(run_dir / "manifest.json", manifest)
    validation_path = run_dir / "validation_report.json"
    if validation_path.is_file():
        validation = json_read(validation_path)
        validation.setdefault("reference_resolution", {}).update({
            "resolved_mentions": sum(bool(row.get("reference_resolution")) for row in mentions),
            "pending_human_review": sum(row.get("status") == "pending" for row in queue),
            "reviewed_ignored": sum(row.get("status") == "ignored" for row in queue),
        })
        json_write(validation_path, validation)
    return {"item": item, "mentions": mentions, "entities": merged, "similar_nodes": pairs, "manifest": manifest}


def apply_resolution(run_dir: Path, resolution_id: str, decision: str, chosen_name: str = "") -> dict:
    if decision not in {"same", "distinct"}:
        raise ValueError("decision must be 'same' or 'distinct'")
    pairs = jsonl_read(run_dir / "similar_nodes_review.jsonl")
    pair = next((row for row in pairs if row["resolution_id"] == resolution_id), None)
    if pair is None:
        raise KeyError("Unknown similarity candidate")
    allowed_names = {pair["left"]["canonical_name"], pair["right"]["canonical_name"], pair["recommended_canonical_name"]}
    canonical_name = chosen_name.strip() or pair["recommended_canonical_name"]
    if canonical_name not in allowed_names:
        raise ValueError("The canonical name must be one of the grounded candidate names.")
    pair.update({"status": decision, "chosen_canonical_name": canonical_name if decision == "same" else "", "reviewed_at": now_iso()})
    jsonl_write(run_dir / "similar_nodes_review.jsonl", pairs)
    decision_row = {
        "decision_id": stable_id("decision", run_dir.name, resolution_id, pair["reviewed_at"]),
        "run_id": run_dir.name, "resolution_id": resolution_id, "decision": decision,
        "chosen_canonical_name": canonical_name if decision == "same" else "",
        "label": pair["label"], "left_entity_id": pair["left"]["entity_id"],
        "right_entity_id": pair["right"]["entity_id"], "left_name": pair["left"]["canonical_name"],
        "right_name": pair["right"]["canonical_name"], "reviewed_at": pair["reviewed_at"],
    }
    append_jsonl(DECISIONS_PATH, decision_row)
    if decision == "same":
        Lexicon().add_reviewed_aliases(
            pair["label"], canonical_name,
            sorted({*pair["left"]["aliases"], *pair["right"]["aliases"], pair["left"]["canonical_name"], pair["right"]["canonical_name"]}),
            {"decision_id": decision_row["decision_id"], "run_id": run_dir.name, "reviewed_at": pair["reviewed_at"]},
        )
    merged = materialize_merges(run_dir)
    from Pipeline.Edge_Extraction import materialize_canonical_relationships
    canonical_relationships = materialize_canonical_relationships(
        run_dir, merged, jsonl_read(run_dir / "assertions.jsonl")
    )
    from Pipeline.Question_Evaluation.evaluator import evaluate_question_answerability
    question_evaluation = evaluate_question_answerability(
        run_dir, merged, canonical_relationships, jsonl_read(run_dir / "assertions.jsonl")
    )
    return {
        "pair": pair,
        "merged_entities": merged,
        "canonical_relationships": canonical_relationships,
        "question_answerability": question_evaluation,
        "lexicon_version": json_read(LEXICON_PATH)["version"],
    }
