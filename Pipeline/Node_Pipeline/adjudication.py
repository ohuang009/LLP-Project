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

from .common import CONFIG_PATH, GENERIC_TERMS, ONTOLOGY_PATH, ROOT, RUBRIC_PATH, WORKSPACE, json_read, normalized, stable_id
from .contracts import judgment_artifact, rubric_metadata

def _prior_node_comparison(candidate: dict, ontology_label: str, canonical_name: str) -> dict:
    options = [
        row for row in candidate.get("candidate_metadata", {}).get("prior_nodes_same_type", [])
        if row["label"] == ontology_label
    ]
    matches = [
        row for row in options
        if normalized(row["canonical_name"]) == normalized(canonical_name)
    ]
    matched = matches[0] if len(matches) == 1 else None
    return {
        "sentence_id": candidate["source"]["sentence_id"],
        "status": "same" if matched else ("no_exact_match" if options else "not_compared"),
        "selected_label": ontology_label,
        "llm_cleaned_name": canonical_name,
        "compared_node_ids": [row["node_id"] for row in options],
        "matched_node_id": matched["node_id"] if matched else "",
        "matched_canonical_name": matched["canonical_name"] if matched else "",
        "matched_aliases": matched.get("aliases", []) if matched else [],
        "matched_source_sentence_ids": matched.get("source_sentence_ids", []) if matched else [],
        "policy": "Only an exact DeepSeek-returned canonical name among supplied same-type prior nodes maps automatically.",
    }


def _canonical_name_is_grounded(
    candidate: dict, canonical_name: str, comparison: dict,
) -> bool:
    """Allow source-span cleanup or an exact same-type prior mapping, never a new identity."""
    if comparison.get("status") == "same" and comparison.get("matched_node_id"):
        surface = normalized(candidate.get("surface_text", ""))
        prior_names = {
            normalized(comparison.get("matched_canonical_name", "")),
            *(normalized(value) for value in comparison.get("matched_aliases", [])),
        }
        return any(
            value and surface and (
                value == surface
                or (min(len(value), len(surface)) >= 4 and (value in surface or surface in value))
            )
            for value in prior_names
        )
    surface = normalized(candidate.get("surface_text", ""))
    canonical = normalized(canonical_name)
    return bool(canonical and surface and canonical in surface)


def _ontology_label_is_scibert_supported(candidate: dict, ontology_label: str) -> bool:
    alternatives = candidate.get("candidate_metadata", {}).get("typing_alternatives", [])
    supported = {
        candidate.get("label", ""),
        *(row.get("label", "") for row in alternatives),
    } - {"", "NONE"}
    return not supported or ontology_label in supported


def optional_llm_review(
    review: list[dict], progress: Callable[[str, str, int], None],
    *, node_judge=None,
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], dict]:
    config = json_read(CONFIG_PATH)
    spec = config["llm_node_judge"]
    rubric = json_read(RUBRIC_PATH)
    status = {
        "enabled": bool(spec.get("enabled")), "used": False, "provider": spec.get("provider", "ollama"),
        "model": spec.get("model", ""), "reason": "", **rubric_metadata(rubric),
    }
    untraceable: list[dict] = []
    traceable: list[dict] = []
    for row in review:
        sentence_id = str(row.get("source", {}).get("sentence_id", "")).strip()
        if sentence_id:
            traceable.append(row)
            continue
        validation = {
            **row.get("validation", {}),
            "outcome": "reject",
            "judge": "sentence_traceability_gate",
            "llm_used": False,
            "reason": "Rejected before LLM adjudication because no specific sentence ID points to the source evidence.",
        }
        untraceable.append({**row, "status": "rejected", "validation": validation})
    review = traceable
    if not review or not spec.get("enabled"):
        status["reason"] = "No unreviewed candidates." if not review else "Disabled in configuration."
        judgments = [
            judgment_artifact(row, decision="review", validation=row["validation"], rubric=rubric)
            for row in review
        ]
        judgments.extend(
            judgment_artifact(row, decision="reject", validation=row["validation"], rubric=rubric)
            for row in untraceable
        )
        return [], review, untraceable, judgments, [], status
    try:
        judge = node_judge
        if judge is None:
            from Pipeline.Knowledge_Graph_Core.llm_node_judge import MandatoryLLMNodeJudge
            from Pipeline.Knowledge_Graph_Core.ontology import Ontology
            llm_config = {"llm_node_judge": {**spec, "required": True}}
            judge = MandatoryLLMNodeJudge(Ontology(ONTOLOGY_PATH), llm_config, ROOT / "cache" / "llm_node_judge")
        judge.verify_runtime()
    except Exception as exc:
        status["reason"] = str(exc).splitlines()[0]
        unavailable = []
        for row in review:
            validation = {
                **row["validation"], "judge": "human_review_after_llm_unavailable",
                "reason": status["reason"],
            }
            row["validation"] = validation
            unavailable.append(judgment_artifact(
                row, decision="review", validation=validation, rubric=rubric,
            ))
        unavailable.extend(
            judgment_artifact(row, decision="reject", validation=row["validation"], rubric=rubric)
            for row in untraceable
        )
        return [], review, untraceable, unavailable, [], status

    progress("adjudicating", "Local DeepSeek is validating candidates against their 2-3 sentence context.", 56)
    grouped: dict[str, dict] = {}
    index: dict[str, dict] = {}
    for candidate in review:
        sentence_id = candidate["source"]["sentence_id"]
        index[candidate["candidate_id"]] = candidate
        grouped.setdefault(sentence_id, {
            "sentence_id": sentence_id,
            "section_title": candidate["source"]["section_title"],
            "text": candidate["source"]["evidence_quote"],
            "context_sentences": candidate["source"]["context_sentences"],
            "context_instruction": "Use all context sentences to resolve references, but any returned node span must occur exactly in text, the target sentence.",
            "validity_rubric": rubric,
            "allow_discovery": bool(spec.get("allow_llm_discovery", False)),
            "allow_new_class_candidates": bool(spec.get("allow_new_class_candidates", False)),
            "candidates": [],
        })["candidates"].append({
            "candidate_id": candidate["candidate_id"], "surface_text": candidate["surface_text"],
            "start_char": candidate["source"]["start_char"], "end_char": candidate["source"]["end_char"],
            "suggested_label": candidate["label"], "candidate_method": candidate["extraction_method"],
            "deterministically_cleaned_name": candidate.get("canonical_name") or candidate["surface_text"],
            "scibert_alternatives": candidate.get("candidate_metadata", {}).get("typing_alternatives", []),
            "prior_nodes_same_type": candidate.get("candidate_metadata", {}).get("prior_nodes_same_type", []),
        })
    try:
        results = judge.judge(list(grouped.values()))
    except Exception as exc:
        status["reason"] = f"Local DeepSeek became unavailable: {str(exc).splitlines()[0]}"
        unavailable = []
        for row in review:
            validation = {
                **row["validation"], "judge": "human_review_after_llm_unavailable",
                "reason": status["reason"],
            }
            row["validation"] = validation
            unavailable.append(judgment_artifact(
                row, decision="review", validation=validation, rubric=rubric,
            ))
        unavailable.extend(
            judgment_artifact(row, decision="reject", validation=row["validation"], rubric=rubric)
            for row in untraceable
        )
        return [], review, untraceable, unavailable, judge.audit_records, status

    accepted: list[dict] = []
    remaining: list[dict] = []
    rejected: list[dict] = list(untraceable)
    judgments: list[dict] = []
    for result in results.values():
        for judgment in result["candidate_judgments"]:
            audit = judgment.get("_llm_audit", result.get("_llm_audit", {}))
            candidate = index[judgment["candidate_id"]]
            comparison = _prior_node_comparison(
                candidate, judgment["ontology_label"], judgment["canonical_name"],
            )
            accepted_name = comparison["matched_canonical_name"] or judgment["canonical_name"]
            identity_is_clear = normalized(accepted_name) not in GENERIC_TERMS
            identity_is_grounded = _canonical_name_is_grounded(candidate, accepted_name, comparison)
            ontology_label_is_supported = _ontology_label_is_scibert_supported(
                candidate, judgment["ontology_label"],
            )
            normalization = {
                "sentence_id": candidate["source"]["sentence_id"],
                "original_surface_text": candidate["surface_text"],
                "deterministic": candidate.get("candidate_metadata", {}).get("deterministic_normalization", {}),
                "deepseek_cleaned_name": judgment["canonical_name"],
                "model": audit.get("model", spec.get("model")),
            }
            if (
                judgment["decision"] == "accept" and identity_is_clear
                and identity_is_grounded and ontology_label_is_supported
            ):
                accepted.append({
                    **candidate,
                    "mention_id": stable_id("mention", candidate["candidate_id"], "llm"),
                    "label": judgment["ontology_label"], "canonical_name": accepted_name,
                    "canonical_id": comparison["matched_node_id"] or candidate.get("canonical_id", ""),
                    "status": "accepted", "extraction_method": "llm_judged_candidate",
                    "validation": {
                        "outcome": "accept", "judge": "structured_llm_node_judge", "llm_used": True,
                        "model": audit.get("model", spec.get("model")), "request_hash": audit.get("request_hash", ""),
                        "prompt_version": audit.get("prompt_version", ""), "reason": judgment["reason"],
                        "confidence": judgment["confidence"], "definition": judgment["definition"],
                        "normalization": normalization, "prior_node_comparison": comparison,
                    },
                })
            elif (
                judgment["decision"] in {"accept", "review"}
                and identity_is_clear and identity_is_grounded
            ):
                review_reason = judgment["reason"]
                if judgment["decision"] == "accept" and not ontology_label_is_supported:
                    review_reason = (
                        "Human review is required because DeepSeek's ontology label was outside the "
                        "pretrained SciBERT candidate types. " + review_reason
                    )
                remaining.append({
                    **candidate,
                    "label": judgment["ontology_label"] if judgment["ontology_label"] != "NONE" else candidate["label"],
                    "canonical_name": judgment["canonical_name"],
                    "validation": {
                        "outcome": "review", "judge": "structured_llm_node_judge", "llm_used": True,
                        "model": audit.get("model", spec.get("model")), "request_hash": audit.get("request_hash", ""),
                        "prompt_version": audit.get("prompt_version", ""), "reason": review_reason,
                        "confidence": judgment["confidence"], "definition": judgment["definition"],
                        "normalization": normalization, "prior_node_comparison": comparison,
                    },
                })
            else:
                reason = judgment["reason"]
                if not identity_is_clear:
                    prefix = (
                        "Rejected because the node remained vague or generic after normalization and was not "
                        "mapped to a clearer same-type prior node. "
                    )
                    if not reason.startswith(prefix):
                        reason = prefix + reason
                elif not identity_is_grounded:
                    prefix = (
                        "Rejected because the proposed canonical name cannot be traced to the exact candidate "
                        "span and is not an exact same-type prior-node match. "
                    )
                    if not reason.startswith(prefix):
                        reason = prefix + reason
                rejected.append({
                    **candidate, "status": "rejected",
                    "label": judgment["ontology_label"] if judgment["ontology_label"] != "NONE" else candidate["label"],
                    "canonical_name": judgment["canonical_name"],
                    "validation": {
                        "outcome": "reject", "judge": "structured_llm_node_judge", "llm_used": True,
                        "model": audit.get("model", spec.get("model")), "request_hash": audit.get("request_hash", ""),
                        "prompt_version": audit.get("prompt_version", ""), "reason": reason,
                        "confidence": judgment["confidence"], "definition": judgment["definition"],
                        "normalization": normalization, "prior_node_comparison": comparison,
                    },
                })
        # LLM discoveries remain reviewable: exact span validation makes them grounded, but human confirmation protects precision.
        target = grouped[result["sentence_id"]]
        for discovery in result["discoveries"]:
            start, end = int(discovery["start_char"]), int(discovery["end_char"])
            synthetic = {
                "candidate_id": stable_id("candidate", result["sentence_id"], start, end, discovery["ontology_label"], "llm_discovery"),
                "document_id": next(iter(index.values()))["document_id"], "label": discovery["ontology_label"],
                "surface_text": discovery["surface_text"], "canonical_name": discovery["canonical_name"], "canonical_id": "",
                "confidence": discovery["confidence"], "extraction_method": "llm_discovery", "status": "review",
                "source": {
                    "kind": "sentence_span", "sentence_id": result["sentence_id"], "section_title": target["section_title"],
                    "pages": next((c["pages"] for c in target["context_sentences"] if c["role"] == "target"), []),
                    "start_char": start, "end_char": end, "quote": discovery["surface_text"],
                    "evidence_quote": target["text"], "context_sentences": target["context_sentences"],
                    "context_sentence_ids": [c["sentence_id"] for c in target["context_sentences"]],
                    "context_policy": "target sentence plus previous and/or next narrative sentence within the same section; 2–3 sentences when available",
                },
                "validation": {"outcome": "review", "judge": "structured_llm_node_judge", "llm_used": True, "reason": discovery["reason"]},
            }
            remaining.append(synthetic)
    judged_ids = {row["candidate_id"] for row in accepted + remaining + rejected}
    for row in review:
        if row["candidate_id"] not in judged_ids:
            remaining.append(row)
    for row in accepted:
        judgments.append(judgment_artifact(
            row, decision="accept", validation=row["validation"], rubric=rubric,
        ))
    for row in remaining:
        judgments.append(judgment_artifact(
            row, decision="review", validation=row["validation"], rubric=rubric,
        ))
    for row in rejected:
        judgments.append(judgment_artifact(
            row, decision="reject", validation=row["validation"], rubric=rubric,
        ))
    status.update({"used": True, "reason": "The configured model completed structured candidate adjudication."})
    return accepted, remaining, rejected, judgments, judge.audit_records, status
