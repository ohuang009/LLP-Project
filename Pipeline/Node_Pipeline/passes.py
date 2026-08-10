"""Payloads and LLM passes used to clean, type, and judge node candidates."""
from __future__ import annotations

from Pipeline.core import parse_tsv_rows
from Pipeline.llm import OllamaClient

from .batching import candidate_rows


NODE_RUBRIC = {
    "version": "gold_evidence_v1",
    "accepted": "Exact raw span; one resolved referent; one evidence-supported type; reusable graph identity.",
    "needs_review": "Grounded and useful, but exactly one of referent, boundary, name, or type remains ambiguous.",
    "disregard": "Generic, malformed, unsupported, unresolved, typed NONE, or not a standalone graph identity.",
    "boundary": "Do not accept a finite clause except as Observation, Claim, or Challenge.",
}


def cleanup_payload(batch: dict) -> dict:
    return {
        "paragraph_id": batch["paragraph_id"],
        "sentences": [{
            "sentence_id": sentence["sentence_id"],
            "previous_sentence": sentence["previous_sentence"],
            "target_sentence": sentence["text"],
            "next_sentence": sentence["next_sentence"],
            "nodes": [{
                "candidate_id": row["candidate_id"],
                "raw_text": row["surface_text"], "roles": row["roles"],
            } for row in [*sentence["subjects"], *sentence["objects"]]],
        } for sentence in batch["sentences"]],
    }


def cleanup_candidates(client: OllamaClient, batches: list[dict]) -> list[dict]:
    candidates = [row for batch in batches for row in candidate_rows(batch)]
    if not candidates:
        return []
    response = client.complete_text(
        "node_cleanup",
        """Resolve each raw grammatical span to one concise entity name. Use only its previous, target,
and next sentence. Never split, merge, or invent an entity; raw_text is immutable. Resolve pronouns and
abbreviations only when one referent is explicit, otherwise mark vague. Paragraphs are independent; do
not carry facts between them. Return one tab-separated line per candidate:
candidate_id<TAB>specific_name<TAB>S|V
S means specific and V means vague. Use exactly three columns and an actual tab character.
Example: candidate_123\tWaterRAG\tS
No header, JSON, Markdown, explanation, or tabs inside names.""",
        {"paragraphs": [cleanup_payload(batch) for batch in batches]},
    )
    decisions = {
        candidate_id: {"specific_name": name, "specificity": "specific" if code == "S" else "vague"}
        for candidate_id, name, code in parse_tsv_rows(response, 3)
        if code in {"S", "V"}
    }
    output = []
    for candidate in candidates:
        decision = decisions.get(candidate["candidate_id"], {})
        specific_name = str(decision.get("specific_name") or candidate["surface_text"]).strip()
        output.append({
            **candidate,
            "specific_name": specific_name,
            "cleanup_specificity": decision.get("specificity", "vague"),
            "cleanup_reason": "specific" if decision.get("specificity") == "specific"
                              else "vague_or_missing_response",
        })
    return output


def typing_payload(unit: tuple[dict, list[dict]]) -> dict:
    batch, candidates = unit
    return {
        "paragraph_id": batch["paragraph_id"],
        "nodes": [{
            "candidate_id": row["candidate_id"], "raw_text": row["surface_text"],
            "specific_name": row["specific_name"], "sentence": row["sentence_text"],
            "scibert_type": row["scibert_type"], "scibert_score": row["scibert_score"],
            "scibert_alternatives": row["scibert_alternatives"],
            "previous_types": row.get("previous_types", []),
        } for row in candidates],
    }


def type_candidates(
    client: OllamaClient, units: list[tuple[dict, list[dict]]], ontology: dict,
) -> list[dict]:
    candidates = [row for _, rows in units for row in rows]
    if not candidates:
        return []
    allowed = [label for label in ontology["nodes"] if label not in {"Mention", "EvidenceFragment"}]
    response = client.complete_text(
        "node_ontology_typing",
        """Type the resolved referent, not nearby words. Choose one supplied type or NONE; SciBERT and
previous_types are advisory evidence. Prefer a previous exact-name type when the current sentence agrees,
but never copy it over contradictory evidence. Key distinctions: Agent=role-bearing system component; AISystem=whole AI platform;
Model=language/ML/process model; Tool=external interface or executable; SoftwareArtifact=code/package;
Dataset=curated records/corpus; DataSource=origin; Method=procedure; Task=performed objective;
Metric=measure name; Observation=reported value/result; Challenge=limitation. Return one tab-separated
line per candidate: candidate_id<TAB>TYPE|NONE. Paragraphs are independent. No header, JSON, Markdown,
or explanation. Use exactly two columns and an actual tab character.
Example: candidate_123\tAISystem""",
        {"ontology_types": allowed, "paragraphs": [typing_payload(unit) for unit in units]},
    )
    decisions = {candidate_id: label for candidate_id, label in parse_tsv_rows(response, 2)}
    allowed_set = set(allowed)
    output = []
    for candidate in candidates:
        label = str(decisions.get(candidate["candidate_id"]) or "NONE")
        if label not in allowed_set:
            label = "NONE"
        output.append({
            **candidate, "llm_type": label,
            "typing_reason": "qwen_tsv_type" if candidate["candidate_id"] in decisions
                             else "missing_or_invalid_response",
        })
    return output


def judgment_payload(unit: tuple[dict, list[dict]]) -> dict:
    batch, candidates = unit
    return {
        "paragraph_id": batch["paragraph_id"],
        "nodes": [{
            "candidate_id": row["candidate_id"], "raw_text": row["surface_text"],
            "specific_name": row["specific_name"], "ontology_type": row["llm_type"],
            "sentence": row["sentence_text"], "cleanup_specificity": row["cleanup_specificity"],
        } for row in candidates],
    }


def judge_candidates(client: OllamaClient, units: list[tuple[dict, list[dict]]]) -> list[dict]:
    candidates = [row for _, rows in units for row in rows]
    if not candidates:
        return []
    response = client.complete_text(
        "node_judgment",
        """Apply the decision rubric exactly. Use needs_review only for one genuine human-resolvable
ambiguity, not as a fallback. A contextual pronoun may pass when specific_name resolves one explicit
referent. Return one tab-separated line per candidate: candidate_id<TAB>A|R|D.
A=accepted, R=needs review, D=disregard. Paragraphs are independent. No header, JSON, Markdown,
or explanation. Use exactly two columns and an actual tab character.
Example: candidate_123\tA""",
        {"rubric": NODE_RUBRIC, "paragraphs": [judgment_payload(unit) for unit in units]},
    )
    decision_names = {"A": "accepted", "R": "needs_review", "D": "disregard"}
    decisions = {
        candidate_id: decision_names[code]
        for candidate_id, code in parse_tsv_rows(response, 2)
        if code in decision_names
    }
    output = []
    for candidate in candidates:
        status = decisions.get(candidate["candidate_id"], "needs_review")
        if candidate["llm_type"] == "NONE":
            status = "disregard"
        if status not in {"accepted", "needs_review", "disregard"}:
            status = "needs_review"
        output.append({
            **candidate, "decision": status,
            "judgment_reason": "qwen_tsv_decision" if candidate["candidate_id"] in decisions
                               else "missing_or_invalid_response",
        })
    return output
