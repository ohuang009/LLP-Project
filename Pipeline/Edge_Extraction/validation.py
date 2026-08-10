"""Deterministic relationship gates and inferred-node review candidates."""
from __future__ import annotations

from Pipeline.core import stable_id
from Pipeline.ontology import Ontology

from .context import endpoint_type


def deterministic_gates(
    paragraph: dict, proposal: dict, mentions: dict[str, dict], ontology: Ontology,
) -> dict:
    """Evaluate structural, ontology, and provenance requirements for a proposal."""
    sentence_ids = {row["sentence_id"] for row in paragraph["sentences"]}
    subject_id = str(proposal["subject"].get("mention_id") or "")
    object_id = str(proposal["object"].get("mention_id") or "")
    subject_type = endpoint_type(proposal["subject"], mentions)
    object_type = endpoint_type(proposal["object"], mentions)
    evidence = proposal.get("evidence_sentence_ids") or []
    sentence_text = {row["sentence_id"]: row["text"] for row in paragraph["sentences"]}

    def inferred_endpoint_is_grounded(endpoint: dict) -> bool:
        if endpoint.get("mention_id"):
            return True
        sentence_id = endpoint.get("evidence_sentence_id")
        quote = str(endpoint.get("evidence_quote") or "")
        return bool(
            endpoint.get("inferred_name")
            and endpoint.get("inferred_type") in ontology.nodes
            and sentence_id in sentence_ids
            and quote
            and quote in sentence_text[sentence_id]
        )

    return {
        "at_least_one_existing_node": subject_id in mentions or object_id in mentions,
        "known_ids_exist": (
            (not subject_id or subject_id in mentions)
            and (not object_id or object_id in mentions)
        ),
        "evidence_in_paragraph": bool(evidence) and all(value in sentence_ids for value in evidence),
        "predicate_exists": proposal.get("predicate") in ontology.relationships,
        "domain_range_valid": ontology.valid_relationship(
            proposal.get("predicate", ""), subject_type, object_type,
        ),
        "not_self_relationship": not (subject_id and subject_id == object_id),
        "inferred_endpoints_grounded": all(
            inferred_endpoint_is_grounded(endpoint)
            for endpoint in (proposal["subject"], proposal["object"])
        ),
    }


def inferred_candidate(paragraph: dict, proposal: dict, role: str) -> dict | None:
    """Create a review-queue node for a proposal's inferred endpoint."""
    endpoint = proposal[role]
    if endpoint.get("mention_id"):
        return None
    candidate_id = stable_id(
        "inferred-node", paragraph["document_id"], paragraph["paragraph_id"],
        endpoint.get("inferred_name"), endpoint.get("inferred_type"),
        endpoint.get("evidence_sentence_id"),
    )
    sentence = next((
        row for row in paragraph["sentences"]
        if row["sentence_id"] == endpoint.get("evidence_sentence_id")
    ), {})
    return {
        "candidate_id": candidate_id, "origin": "relationship_inference",
        "status": "needs_review",
        "document_id": paragraph["document_id"], "paragraph_id": paragraph["paragraph_id"],
        "sentence_id": endpoint.get("evidence_sentence_id", ""),
        "sentence_text": sentence.get("text", ""),
        "surface_text": endpoint.get("evidence_quote", ""),
        "canonical_name": endpoint.get("inferred_name", ""),
        "label": endpoint.get("inferred_type", ""),
        "specific_name": endpoint.get("inferred_name", ""),
        "llm_type": endpoint.get("inferred_type", ""),
        "start": None, "end": None,
        "relationship_proposal_id": proposal["proposal_id"], "endpoint_role": role,
        "source": {
            "kind": "sentence_span", "sentence_id": endpoint.get("evidence_sentence_id", ""),
            "paragraph_id": paragraph["paragraph_id"], "section_id": paragraph["section_id"],
            "section_title": paragraph["section_title"],
            "evidence_quote": endpoint.get("evidence_quote", ""),
            "context_sentence_ids": [row["sentence_id"] for row in paragraph["sentences"]],
            "context_sentences": paragraph["sentences"],
            "context_policy": "all sentences in paragraph",
        },
    }
