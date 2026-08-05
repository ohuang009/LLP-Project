"""Stable artifact contracts for the node-extraction boundary.

Candidate generation is recall-oriented. Adjudication is precision-oriented.
These helpers keep those two records explicit and auditable.
"""
from __future__ import annotations

import hashlib
import json


def rubric_metadata(rubric: dict) -> dict:
    encoded = json.dumps(rubric, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return {
        "rubric_version": str(rubric.get("rubric_version", "")),
        "rubric_hash": hashlib.sha256(encoded).hexdigest(),
    }


def candidate_artifact(candidate: dict, *, trusted: bool) -> dict:
    """Return the public, generator-facing candidate representation."""
    source = candidate["source"]
    return {
        "schema_version": "1.0",
        "candidate_id": candidate.get("candidate_id") or candidate["mention_id"],
        "document_id": candidate["document_id"],
        "sentence_id": source.get("sentence_id", ""),
        "surface_text": candidate["surface_text"],
        "suggested_label": candidate["label"],
        "suggested_canonical_name": candidate.get("canonical_name", candidate["surface_text"]),
        "candidate_generators": candidate.get("candidate_generators") or [{
            "name": candidate.get("extraction_method", "unknown"),
            "confidence": float(candidate.get("confidence", 0.0)),
        }],
        "trusted": trusted,
        "candidate_metadata": candidate.get("candidate_metadata", {}),
        "source": source,
    }


def judgment_artifact(candidate: dict, *, decision: str, validation: dict, rubric: dict) -> dict:
    """Return one durable decision for one supplied candidate."""
    source = candidate["source"]
    return {
        "schema_version": "1.0",
        "candidate_id": candidate.get("candidate_id") or candidate["mention_id"],
        "document_id": candidate["document_id"],
        "sentence_id": source.get("sentence_id", ""),
        "surface_text": candidate["surface_text"],
        "decision": decision,
        "ontology_label": candidate.get("label", ""),
        "canonical_name": candidate.get("canonical_name", candidate["surface_text"]),
        "definition": validation.get("definition", ""),
        "judge": validation.get("judge", "unknown"),
        "llm_used": bool(validation.get("llm_used")),
        "model": validation.get("model", ""),
        **rubric_metadata(rubric),
        "reason": validation.get("reason", "No reason recorded."),
        "confidence": float(validation.get("confidence", candidate.get("confidence", 0.0))),
        "normalization": validation.get("normalization", candidate.get("candidate_metadata", {}).get("deterministic_normalization", {})),
        "prior_node_comparison": validation.get("prior_node_comparison", {}),
        "source": source,
    }
