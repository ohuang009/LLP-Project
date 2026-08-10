"""Prepare paragraph, node, endpoint, and ontology context for edge extraction."""
from __future__ import annotations

from Pipeline.ontology import Ontology


# These predicates answer the fixed competency questions and were the dominant
# false negatives in the human EWS-R07 gold set. Priority affects scan order,
# never acceptance: the paragraph must still directly entail every edge.
PRIORITY_PREDICATES = (
    "PROPOSES", "HAS_AGENT", "USES_MODEL", "PERFORMS",
    "EVALUATED_ON", "EVALUATED_BY", "REPORTS", "MEASURES",
)


def predicate_payload(
    ontology: Ontology, anchor_types: set[str] | None = None,
) -> list[dict]:
    """Return compact legal signatures, optionally filtered to known anchors."""
    priority = {name: index for index, name in enumerate(PRIORITY_PREDICATES)}
    rows = []
    for name, spec in ontology.relationships.items():
        domain = list(spec.get("domain") or [])
        range_ = list(spec.get("range") or [])
        if anchor_types and not anchor_types.intersection([*domain, *range_]):
            continue
        rows.append({
            "p": name, "d": domain, "r": range_,
            "m": str(spec.get("definition") or ""),
        })
    return sorted(rows, key=lambda row: (priority.get(row["p"], len(priority)), row["p"]))


def paragraphs(parsed: dict) -> list[dict]:
    """Flatten parsed sections into paragraph records with sentence provenance."""
    return [{
        "document_id": parsed["document"]["id"],
        "section_id": section["id"], "section_title": section.get("title", ""),
        "paragraph_id": paragraph["id"],
        "paragraph_text": paragraph.get("text", ""),
        "sentences": [{
            "sentence_id": sentence["id"], "text": sentence["text"],
            "pages": sentence.get("pages", []),
        } for sentence in paragraph.get("sentences", [])],
    } for section in parsed.get("sections", []) for paragraph in section.get("paragraphs", [])]


def nodes_for_paragraph(paragraph: dict, mentions: list[dict]) -> list[dict]:
    """Return accepted mentions grounded in this paragraph."""
    sentence_ids = {row["sentence_id"] for row in paragraph["sentences"]}
    return [row for row in mentions if row.get("source", {}).get("sentence_id") in sentence_ids]


def endpoint_payload(endpoint: dict) -> dict:
    """Normalize a known or inferred relationship endpoint."""
    if endpoint.get("mention_id"):
        return {"mention_id": str(endpoint["mention_id"])}
    return {
        "inferred_name": str(endpoint.get("inferred_name") or "").strip(),
        "inferred_type": str(endpoint.get("inferred_type") or "NONE").strip(),
        "evidence_sentence_id": str(endpoint.get("evidence_sentence_id") or "").strip(),
        "evidence_quote": str(endpoint.get("evidence_quote") or "").strip(),
    }


def endpoint_type(endpoint: dict, mentions: dict[str, dict]) -> str:
    """Resolve an endpoint's ontology type from a mention or inferred data."""
    known = mentions.get(str(endpoint.get("mention_id") or ""))
    return known["label"] if known else str(endpoint.get("inferred_type") or "NONE")
