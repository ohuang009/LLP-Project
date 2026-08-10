"""Apply human decisions to queued node and inferred-endpoint candidates."""
from __future__ import annotations

from pathlib import Path

from Pipeline.core import ONTOLOGY_PATH, json_read, json_write, jsonl_read, jsonl_write, stable_id
from Pipeline.lexicon import update_lexicon

from .entities import canonical_entities, mention_from_candidate


def apply_node_review(
    run_dir: Path, candidate_id: str, decision: str, *, ontology_label: str = "",
    canonical_name: str = "", reason: str = "",
) -> dict:
    """Add a queued node only after explicit human acceptance."""
    if decision not in {"accept", "reject"}:
        raise ValueError("decision must be accept or reject")
    queue = jsonl_read(run_dir / "review_queue.jsonl")
    item = next((row for row in queue if row.get("candidate_id") == candidate_id), None)
    if item is None:
        raise KeyError("Unknown or already reviewed candidate")
    remaining = [row for row in queue if row.get("candidate_id") != candidate_id]
    mentions = jsonl_read(run_dir / "mentions.jsonl")
    if decision == "accept":
        label = ontology_label or item.get("label") or item.get("llm_type")
        allowed = set(json_read(ONTOLOGY_PATH).get("nodes", {}))
        if label not in allowed:
            raise ValueError("Choose an existing ontology type")
        item = {
            **item, "llm_type": label,
            "specific_name": canonical_name or item.get("canonical_name") or item["surface_text"],
            "judgment_reason": reason or "Accepted by a human reviewer.",
        }
        batch = {"sentences": [{
            "sentence_id": item["sentence_id"], "previous_sentence": "",
            "text": item["sentence_text"], "next_sentence": "",
        }]}
        accepted_mention = mention_from_candidate(item, batch, method="human_review")
        mentions.append(accepted_mention)
    else:
        rejected = jsonl_read(run_dir / "node_rejections.jsonl")
        rejected.append({
            **item, "status": "disregarded",
            "judgment_reason": reason or "Rejected by a human reviewer.",
        })
        jsonl_write(run_dir / "node_rejections.jsonl", rejected)

    entities = canonical_entities(mentions)
    if decision == "accept":
        update_lexicon(entities)
    jsonl_write(run_dir / "review_queue.jsonl", remaining)
    jsonl_write(run_dir / "node_review_candidates.jsonl", remaining)
    jsonl_write(run_dir / "mentions.jsonl", mentions)
    jsonl_write(run_dir / "canonical_entities.jsonl", entities)
    jsonl_write(run_dir / "canonical_entities_merged.jsonl", entities)

    if decision == "accept" and item.get("origin") == "relationship_inference":
        _reconnect_reviewed_endpoint(run_dir, item, candidate_id, accepted_mention, entities)

    manifest = json_read(run_dir / "manifest.json", {})
    manifest.setdefault("counts", {}).update({
        "accepted_mentions": len(mentions), "canonical_entities": len(entities),
        "node_review_candidates": len(remaining),
    })
    json_write(run_dir / "manifest.json", manifest)
    return {
        "item": item, "mentions": mentions, "entities": entities,
        "manifest": manifest, "effective_decision": decision,
    }


def _reconnect_reviewed_endpoint(
    run_dir: Path, item: dict, candidate_id: str, accepted_mention: dict,
    entities: list[dict],
) -> None:
    """Reconnect a human-approved inferred endpoint to its relationship proposal."""
    candidates = jsonl_read(run_dir / "relationship_candidates.jsonl")
    assertions = jsonl_read(run_dir / "assertions.jsonl")
    for relationship in candidates:
        if relationship.get("candidate_id") != item.get("relationship_proposal_id"):
            continue
        relationship[f"{item.get('endpoint_role')}_mention_id"] = accepted_mention["mention_id"]
        relationship["inferred_node_candidate_ids"] = [
            value for value in relationship.get("inferred_node_candidate_ids", [])
            if value != candidate_id
        ]
        validator_accepted = relationship.get("llm_validation", {}).get("decision") == "accepted"
        if (
            not relationship["inferred_node_candidate_ids"]
            and validator_accepted
            and all(relationship.get("gates", {}).values())
        ):
            relationship["status"] = "accepted"
            if not any(row.get("candidate_id") == relationship["candidate_id"] for row in assertions):
                assertions.append({
                    "assertion_id": stable_id(
                        "assertion", relationship["candidate_id"],
                        relationship["subject_mention_id"], relationship["object_mention_id"],
                    ),
                    **{key: relationship[key] for key in (
                        "candidate_id", "document_id", "paragraph_id", "subject_mention_id", "subject_text",
                        "subject_label", "predicate", "object_mention_id", "object_text", "object_label",
                        "evidence_sentence_ids", "context_quotes",
                    )},
                    "confidence": 1.0, "status": "accepted", "scope": "paragraph",
                    "evidence_quote": " ".join(relationship.get("context_quotes", [])),
                    "extraction_method": "qwen_ollama_generate_refine_validate_human_endpoint",
                    "relationship_decision": relationship.get("llm_validation", {}),
                    "gates": relationship.get("gates", {}),
                })
    jsonl_write(run_dir / "relationship_candidates.jsonl", candidates)
    jsonl_write(run_dir / "assertions.jsonl", assertions)
    from Pipeline.Edge_Extraction import materialize_canonical_relationships
    materialize_canonical_relationships(run_dir, entities, assertions)
