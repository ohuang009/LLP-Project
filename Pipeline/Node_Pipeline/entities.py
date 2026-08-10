"""Convert accepted node candidates into mentions and canonical entities."""
from __future__ import annotations

from Pipeline.core import normalized, stable_id

from .passes import NODE_RUBRIC


def _context_for(batch: dict, sentence_id: str) -> tuple[list[str], list[dict]]:
    sentence = next(row for row in batch["sentences"] if row["sentence_id"] == sentence_id)
    context = []
    for role, text in (
        ("previous", sentence["previous_sentence"]),
        ("target", sentence["text"]),
        ("next", sentence["next_sentence"]),
    ):
        if text:
            context.append({
                "role": role,
                "text": text,
                "sentence_id": sentence_id if role == "target" else "",
            })
    return [row["sentence_id"] for row in context if row["sentence_id"]], context


def mention_from_candidate(
    row: dict, batch: dict, *, method: str = "qwen_ollama_three_pass",
) -> dict:
    """Convert an accepted candidate into a provenance-rich mention artifact."""
    context_ids, context = _context_for(batch, row["sentence_id"])
    return {
        "candidate_id": row["candidate_id"],
        "mention_id": stable_id("mention", row["candidate_id"], row["llm_type"], row["specific_name"]),
        "document_id": row["document_id"],
        "label": row["llm_type"],
        "surface_text": row["surface_text"],
        "canonical_name": row["specific_name"],
        "confidence": 1.0,
        "extraction_method": method,
        "status": "accepted",
        "source": {
            "kind": "sentence_span", "sentence_id": row["sentence_id"],
            "paragraph_id": row["paragraph_id"], "section_id": row["section_id"],
            "section_title": row["section_title"], "start_char": row["start"],
            "end_char": row["end"], "evidence_quote": row["surface_text"],
            "context_sentence_ids": context_ids, "context_sentences": context,
            "context_policy": "previous, target, and next sentence",
        },
        "validation": {
            "outcome": "accept", "judge": "qwen_ollama_node_judgment",
            "reason": row.get("judgment_reason", "Accepted by a human reviewer."),
            "rubric": NODE_RUBRIC,
            "scibert_type": row.get("scibert_type", "NONE"),
            "scibert_score": row.get("scibert_score", 0),
        },
    }


def canonical_entities(mentions: list[dict]) -> list[dict]:
    """Group mentions by ontology label and normalized canonical name."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for mention in mentions:
        groups.setdefault((mention["label"], normalized(mention["canonical_name"])), []).append(mention)
    entities = []
    for (label, _), rows in groups.items():
        name = rows[0]["canonical_name"]
        entities.append({
            "entity_id": stable_id("entity", label, normalized(name)),
            "label": label, "canonical_name": name,
            "aliases": sorted({row["surface_text"] for row in rows}, key=str.casefold),
            "mention_ids": [row["mention_id"] for row in rows],
            "mention_count": len(rows), "identity_scope": "canonical",
        })
    return sorted(entities, key=lambda row: (row["label"], normalized(row["canonical_name"])))
