"""Normalize relationship proposals and turn validated proposals into artifacts."""
from __future__ import annotations

from Pipeline.core import stable_id
from Pipeline.ontology import Ontology

from .context import endpoint_payload, endpoint_type
from .passes import RELATIONSHIP_RUBRIC
from .validation import deterministic_gates, inferred_candidate


def prepare_proposals(paragraph: dict, generated: list[dict]) -> list[dict]:
    """Normalize generated rows and assign stable proposal identifiers."""
    return [{
        "proposal_id": stable_id("edge-proposal", paragraph["paragraph_id"], index),
        "subject": endpoint_payload(raw.get("subject", {})),
        "predicate": str(raw.get("predicate") or ""),
        "object": endpoint_payload(raw.get("object", {})),
        "evidence_sentence_ids": list(raw.get("evidence_sentence_ids") or []),
        "generator_reason": str(raw.get("reason") or ""),
    } for index, raw in enumerate(generated, 1)]


def proposal_artifacts(
    paragraph: dict, proposal: dict, generated_count: int, minimum: int,
    decision: dict, mentions: dict[str, dict], ontology: Ontology,
) -> tuple[dict, dict | None, list[dict]]:
    """Apply gates and return a candidate, optional assertion, and inferred nodes."""
    gates = deterministic_gates(paragraph, proposal, mentions, ontology)
    unknown_rows = [
        row for role in ("subject", "object")
        if (row := inferred_candidate(paragraph, proposal, role))
    ]
    requested = str(decision.get("decision") or "needs_review")
    if not all(gates.values()):
        status = "rejected"
    elif unknown_rows:
        status = "needs_review"
    elif requested == "accepted":
        status = "accepted"
    else:
        status = requested if requested in {"needs_review", "rejected"} else "needs_review"

    subject_id = str(proposal["subject"].get("mention_id") or "")
    object_id = str(proposal["object"].get("mention_id") or "")
    candidate = {
        "candidate_id": proposal["proposal_id"], "paragraph_id": paragraph["paragraph_id"],
        "section_id": paragraph["section_id"], "document_id": paragraph["document_id"],
        "subject_mention_id": subject_id,
        "subject_text": mentions.get(subject_id, {}).get(
            "canonical_name", proposal["subject"].get("inferred_name", ""),
        ),
        "subject_label": endpoint_type(proposal["subject"], mentions),
        "predicate": proposal["predicate"],
        "object_mention_id": object_id,
        "object_text": mentions.get(object_id, {}).get(
            "canonical_name", proposal["object"].get("inferred_name", ""),
        ),
        "object_label": endpoint_type(proposal["object"], mentions),
        "evidence_sentence_ids": proposal["evidence_sentence_ids"],
        "context_quotes": [row["text"] for row in paragraph["sentences"]],
        "inferred_node_candidate_ids": [row["candidate_id"] for row in unknown_rows],
        "status": status, "rubric_version": RELATIONSHIP_RUBRIC["version"],
        "gates": gates, "llm_validation": decision,
        "refinement": proposal.get("refinement", ""),
        "generation_count": generated_count,
        "generation_minimum_met": generated_count >= minimum,
    }
    if status != "accepted":
        return candidate, None, unknown_rows

    confidence = max(0.0, min(1.0, float(decision.get("confidence", 1.0))))
    assertion = {
        "assertion_id": stable_id("assertion", proposal["proposal_id"], subject_id, object_id),
        **{key: candidate[key] for key in (
            "candidate_id", "document_id", "paragraph_id", "subject_mention_id", "subject_text",
            "subject_label", "predicate", "object_mention_id", "object_text", "object_label",
            "evidence_sentence_ids", "context_quotes",
        )},
        "confidence": confidence, "status": "accepted", "scope": "paragraph",
        "evidence_quote": " ".join(
            row["text"] for row in paragraph["sentences"]
            if row["sentence_id"] in proposal["evidence_sentence_ids"]
        ),
        "extraction_method": "qwen_ollama_generate_refine_validate",
        "relationship_decision": decision, "gates": gates,
    }
    return candidate, assertion, unknown_rows
