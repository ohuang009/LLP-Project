"""Deterministic provenance validation for extracted nodes and relationships."""
from __future__ import annotations


def validate_provenance(parsed: dict, mentions: list[dict], relationships: list[dict]) -> dict:
    """Verify paragraph, sentence, span, and endpoint provenance."""
    sentence_to_paragraph = {
        sentence["id"]: paragraph["id"]
        for section in parsed.get("sections", [])
        for paragraph in section.get("paragraphs", [])
        for sentence in paragraph.get("sentences", [])
    }
    failures = []
    for mention in mentions:
        source = mention.get("source", {})
        if sentence_to_paragraph.get(source.get("sentence_id")) != source.get("paragraph_id"):
            failures.append({"id": mention.get("mention_id"), "failure": "mention_sentence_paragraph_mismatch"})
        if source.get("evidence_quote", "") != mention.get("surface_text", ""):
            failures.append({"id": mention.get("mention_id"), "failure": "mention_quote_changed"})
    mention_ids = {row["mention_id"] for row in mentions}
    for edge in relationships:
        paragraph_id = edge.get("paragraph_id")
        evidence_ids = edge.get("evidence_sentence_ids", [])
        if not paragraph_id or any(sentence_to_paragraph.get(sid) != paragraph_id for sid in evidence_ids):
            failures.append({"id": edge.get("candidate_id"), "failure": "edge_evidence_outside_paragraph"})
        known = [edge.get("subject_mention_id"), edge.get("object_mention_id")]
        if not any(value in mention_ids for value in known):
            failures.append({"id": edge.get("candidate_id"), "failure": "edge_has_no_existing_node"})
    return {
        "status": "PASS" if not failures else "FAIL", "failures": failures,
        "mentions_checked": len(mentions), "relationships_checked": len(relationships),
    }
