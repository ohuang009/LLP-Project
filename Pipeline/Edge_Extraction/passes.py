"""LLM generation, refinement, and validation passes for relationships."""
from __future__ import annotations

from Pipeline.core import parse_tsv_rows
from Pipeline.llm import OllamaClient
from Pipeline.ontology import Ontology

from .context import predicate_payload


RELATIONSHIP_RUBRIC = {
    "version": "gold_evidence_v1",
    "accepted": "Direct positive entailment; unambiguous endpoints; correct direction; one atomic, query-useful fact.",
    "needs_review": "The fact is entailed, but one endpoint identity or predicate choice is genuinely ambiguous.",
    "rejected": "Unsupported, proximity-only, purpose-only, negated, hypothetical, duplicate, vague, or wrong direction.",
}


def generation_payload(unit: tuple[dict, list[dict]], ontology: Ontology) -> dict:
    paragraph, nodes = unit
    return {
        "paragraph_id": paragraph["paragraph_id"],
        "paragraph": paragraph,
        "accepted_nodes": [{
            "mention_id": row["mention_id"], "name": row["canonical_name"],
            "type": row["label"], "sentence_id": row["source"]["sentence_id"],
        } for row in nodes],
        "predicates": predicate_payload(ontology, {row["label"] for row in nodes}),
    }


def _parsed_endpoint(
    kind: str, value: str, label: str, sentence_id: str, quote: str,
) -> dict:
    if kind == "K":
        return {"mention_id": value}
    if kind == "I":
        return {
            "inferred_name": value, "inferred_type": label,
            "evidence_sentence_id": sentence_id, "evidence_quote": quote,
        }
    return {}


def generate_relationships(
    client: OllamaClient, units: list[tuple[dict, list[dict]]],
    ontology: Ontology, minimum: int, maximum: int,
) -> dict[str, list[dict]]:
    response = client.complete_text(
        "relationship_generation",
        f"""For each independent paragraph, extract {minimum}-{maximum} distinct atomic relationships when
the text supports them; never pad with unsupported facts. Scan first for PROPOSES, HAS_AGENT, USES_MODEL, PERFORMS, EVALUATED_ON,
EVALUATED_BY, REPORTS, and MEASURES, then other supplied predicates. Priority is not evidence. Preserve
negation and comparison direction. Every edge needs at least one accepted mention_id. One inferred
endpoint is allowed only with an exact quote and one supplied type. Never mix evidence between paragraphs.
Return one 13-column tab-separated line per relationship:
paragraph_id<TAB>S_KIND<TAB>S_ID_OR_NAME<TAB>S_TYPE<TAB>S_SENTENCE<TAB>S_QUOTE<TAB>PREDICATE<TAB>
O_KIND<TAB>O_ID_OR_NAME<TAB>O_TYPE<TAB>O_SENTENCE<TAB>O_QUOTE<TAB>EVIDENCE_SENTENCE_IDS
K means a known mention ID; I means an inferred name. Leave unused known-endpoint columns blank.
Separate multiple evidence IDs with commas. Use exactly 13 columns and actual tab characters.
No header, JSON, Markdown, or explanation.""",
        {
            "paragraphs": [generation_payload(unit, ontology) for unit in units],
            "node_types": list(ontology.nodes),
        },
    )
    paragraph_ids = {paragraph["paragraph_id"] for paragraph, _ in units}
    only_id = next(iter(paragraph_ids)) if len(paragraph_ids) == 1 else ""
    output = {paragraph_id: [] for paragraph_id in paragraph_ids}
    for values in parse_tsv_rows(response, 13):
        (
            paragraph_id, subject_kind, subject_value, subject_type, subject_sentence,
            subject_quote, predicate, object_kind, object_value, object_type,
            object_sentence, object_quote, evidence,
        ) = values
        paragraph_id = paragraph_id or only_id
        if paragraph_id in output and len(output[paragraph_id]) < maximum:
            output[paragraph_id].append({
                "subject": _parsed_endpoint(
                    subject_kind, subject_value, subject_type, subject_sentence, subject_quote,
                ),
                "predicate": predicate,
                "object": _parsed_endpoint(
                    object_kind, object_value, object_type, object_sentence, object_quote,
                ),
                "evidence_sentence_ids": [
                    value.strip() for value in evidence.split(",") if value.strip()
                ],
            })
    return output


def refinement_payload(
    unit: tuple[dict, list[dict], list[dict]], ontology: Ontology,
) -> dict:
    paragraph, nodes, proposals = unit
    return {
        "paragraph_id": paragraph["paragraph_id"],
        "paragraph": paragraph,
        "relationships": proposals,
        "predicates": predicate_payload(ontology, {row["label"] for row in nodes}),
    }


def refine_relationships(
    client: OllamaClient, units: list[tuple[dict, list[dict], list[dict]]],
    ontology: Ontology,
) -> dict[str, list[dict]]:
    proposals = [row for _, _, rows in units for row in rows]
    if not proposals:
        return {paragraph["paragraph_id"]: [] for paragraph, _, _ in units}
    response = client.complete_text(
        "relationship_refinement",
        """Correct predicate, direction, endpoint specificity, and trigger-sentence evidence. Preserve
negation. Do not add facts or replace an endpoint entity. Paragraphs are independent. Return one
6-column tab-separated patch line per proposal:
proposal_id<TAB>K|D<TAB>PREDICATE|=<TAB>EVIDENCE_IDS|=<TAB>SUBJECT_INFERRED_NAME|=<TAB>OBJECT_INFERRED_NAME|=
K keeps the proposal; D drops a duplicate or unsupported proposal. = leaves a field unchanged.
Separate evidence IDs with commas. Use exactly six columns and actual tab characters.
Example: edge_123\tK\t=\t=\t=\t=
No header, JSON, Markdown, or explanation.""",
        {"paragraphs": [refinement_payload(unit, ontology) for unit in units]},
    )
    by_id = {
        proposal_id: {
            "action": action, "predicate": predicate, "evidence": evidence,
            "subject_name": subject_name, "object_name": object_name,
        }
        for proposal_id, action, predicate, evidence, subject_name, object_name
        in parse_tsv_rows(response, 6)
        if action in {"K", "D"}
    }
    output: dict[str, list[dict]] = {paragraph["paragraph_id"]: [] for paragraph, _, _ in units}
    for paragraph, _, paragraph_proposals in units:
        for proposal in paragraph_proposals:
            patch = by_id.get(proposal["proposal_id"])
            if patch and patch["action"] == "D":
                continue
            subject = dict(proposal["subject"])
            obj = dict(proposal["object"])
            if patch and not subject.get("mention_id") and patch["subject_name"] != "=":
                subject["inferred_name"] = patch["subject_name"]
            if patch and not obj.get("mention_id") and patch["object_name"] != "=":
                obj["inferred_name"] = patch["object_name"]
            predicate = proposal["predicate"]
            evidence_ids = proposal["evidence_sentence_ids"]
            if patch and patch["predicate"] != "=":
                predicate = patch["predicate"]
            if patch and patch["evidence"] != "=":
                evidence_ids = [
                    value.strip() for value in patch["evidence"].split(",") if value.strip()
                ]
            output[paragraph["paragraph_id"]].append({
                **proposal, "subject": subject, "object": obj,
                "predicate": predicate, "evidence_sentence_ids": evidence_ids,
                "refinement": "qwen_tsv_patch" if patch else "missing_patch_kept_unchanged",
            })
    return output


def validation_payload(unit: tuple[dict, list[dict]]) -> dict:
    paragraph, proposals = unit
    return {
        "paragraph_id": paragraph["paragraph_id"],
        "paragraph": paragraph,
        "relationships": proposals,
    }


def validate_relationships(
    client: OllamaClient, units: list[tuple[dict, list[dict]]],
) -> dict[str, dict]:
    proposals = [row for _, rows in units for row in rows]
    if not proposals:
        return {}
    response = client.complete_text(
        "relationship_validation",
        """Independently apply the rubric to every proposal. Judge semantic entailment and endpoint
identity; deterministic code separately checks IDs, ontology signatures, and provenance. needs_review is
only for an entailed fact with one genuine ambiguity. Paragraphs are independent. Return one
4-column tab-separated line per proposal:
proposal_id<TAB>A|R|X<TAB>confidence_0_to_1<TAB>code
A=accepted, R=needs review, X=rejected. Use exactly four columns and actual tab characters.
Example: edge_123\tA\t0.95\tentailed
No header, JSON, Markdown, or explanation.""",
        {"rubric": RELATIONSHIP_RUBRIC, "paragraphs": [validation_payload(unit) for unit in units]},
    )
    decision_names = {"A": "accepted", "R": "needs_review", "X": "rejected"}
    output = {}
    for proposal_id, decision, confidence, code in parse_tsv_rows(response, 4):
        if decision not in decision_names:
            continue
        try:
            score = float(confidence)
        except ValueError:
            score = 0.0
        output[proposal_id] = {
            "proposal_id": proposal_id, "decision": decision_names[decision],
            "confidence": max(0.0, min(1.0, score)), "code": code,
            "reason": code,
        }
    return output
