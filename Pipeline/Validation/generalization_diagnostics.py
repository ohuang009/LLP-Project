from __future__ import annotations

from collections import Counter
import re


CORE_AI = {"AISystem", "Agent", "Model", "Method"}
WATER_CONTEXT = {
    "WaterSystem", "WaterSource", "WaterMatrix", "TreatmentPlant",
    "TreatmentProcess", "TreatmentTrain", "TreatmentUnit",
}
DATA = {"Dataset", "DataSource", "Sample", "SamplingEvent"}
EVALUATION = {"Metric", "Observation", "Experiment"}
GENERIC_NAMES = {
    "agent", "ai", "algorithm", "approach", "data", "dataset", "framework",
    "llm", "method", "model", "pipeline", "process", "result", "study", "system",
    "task", "tool", "water",
}
FALSE_AGENT_ROLES = {
    "and", "are", "based", "driven", "employed", "feedback", "functions", "into",
    "management", "operations", "optimization", "parameters", "per", "results", "state",
    "trajectory", "values",
}


def _sentences(parsed: dict) -> list[dict]:
    """Handle sentences for this stage. It measures whether the graph is traceable, complete, and generalizable."""
    return [
        {**sentence, "section_title": section.get("title", "")}
        for section in parsed.get("sections", [])
        for paragraph in section.get("paragraphs", [])
        for sentence in paragraph.get("sentences", [])
    ]


def _normalized(value: str) -> str:
    """Handle normalized for this stage. It measures whether the graph is traceable, complete, and generalizable."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def build_document_profile(parsed: dict) -> dict:
    """Infer a read-only extraction profile; it never creates graph facts."""
    sentences = _sentences(parsed)
    title = str(parsed.get("document", {}).get("title", ""))
    lead = [
        row for row in sentences
        if re.search(r"\b(?:abstract|introduction|highlights?)\b", row["section_title"], re.I)
    ][:80]
    text = " ".join([title, *(row["text"] for row in (lead or sentences[:60]))])
    archetype_patterns = {
        "distribution_network_modeling": r"\b(?:water distribution|WDS|WDN|EPANET|hydraulic)\b",
        "wastewater_treatment": r"\b(?:wastewater|WWTP|activated sludge|aeration)\b",
        "drinking_water_treatment": r"\b(?:drinking water treatment|DWTP|potable water)\b",
        "desalination": r"\b(?:desalination|deionization|salinity)\b",
        "groundwater": r"\b(?:groundwater|aquifer|inverse model)\b",
        "irrigation": r"\b(?:irrigation|soil moisture|crop water)\b",
        "water_quality_prediction": r"\b(?:water quality|bacteriological|contaminant|forecast)\b",
    }
    archetypes = [name for name, pattern in archetype_patterns.items() if re.search(pattern, text, re.I)]
    proposal_signals = [
        {"sentence_id": row["id"], "quote": row["text"], "pages": row.get("pages", [])}
        for row in sentences
        if re.search(r"\b(?:we|this (?:paper|study|work))\b[^.;]{0,60}\b(?:propos|introduc|develop|present)", row["text"], re.I)
    ][:10]
    return {
        "schema_version": "1.0",
        "title": title,
        "archetypes": archetypes or ["unclassified_engineered_water"],
        "proposal_signals": proposal_signals,
        "profile_scope": "title plus abstract/introduction lead text",
        "writes_graph_facts": False,
    }


def build_generalization_diagnostics(
    parsed: dict,
    mentions: list[dict],
    entities: list[dict],
    relationships: list[dict],
    assertions: list[dict],
    question_evaluation: dict,
) -> dict:
    """Build generalization diagnostics. It measures whether the graph is traceable, complete, and generalizable."""
    profile = build_document_profile(parsed)
    label_counts = Counter(row.get("label", "") for row in entities)
    extraction_channels = Counter(row.get("extraction_method", "unknown") for row in mentions)
    connected_ids = {
        value
        for row in relationships
        for value in (row.get("subject_entity_id"), row.get("object_entity_id"))
        if value
    }
    orphan_entities = [row for row in entities if row.get("entity_id") not in connected_ids]
    generic_entities = [
        row for row in entities if _normalized(str(row.get("canonical_name", ""))) in GENERIC_NAMES
    ]
    suspicious_agents = []
    for row in entities:
        if row.get("label") != "Agent":
            continue
        name = str(row.get("canonical_name", ""))
        role = name.rsplit("_", 1)[-1].casefold() if "_" in name else ""
        if role in FALSE_AGENT_ROLES:
            suspicious_agents.append(row)

    expected_groups = [
        ("ai_technology", CORE_AI),
        ("water_context", WATER_CONTEXT),
        ("task_or_objective", {"Task", "Intervention"}),
        ("data_or_sample", DATA),
        ("evaluation_or_result", EVALUATION),
    ]
    role_coverage = []
    for role, labels in expected_groups:
        observed = sum(label_counts.get(label, 0) for label in labels)
        role_coverage.append({
            "role": role,
            "labels": sorted(labels),
            "observed_entities": observed,
            "covered": observed > 0,
        })
    covered_roles = sum(row["covered"] for row in role_coverage)
    cq_answerable = int(question_evaluation.get("answerable_questions", 0))
    cq_total = int(question_evaluation.get("total_questions", 0))
    warnings = []
    if suspicious_agents:
        warnings.append("suspicious_agent_identifier_pattern")
    if generic_entities:
        warnings.append("generic_identity_leakage")
    if entities and len(orphan_entities) / len(entities) > 0.75:
        warnings.append("high_orphan_rate")
    if relationships and len(relationships) / max(1, len(entities)) < 0.2:
        warnings.append("low_relationship_density")
    if cq_total and cq_answerable / cq_total < 0.3:
        warnings.append("low_competency_question_coverage")
    if covered_roles < 3:
        warnings.append("low_archetype_role_coverage")
    return {
        "schema_version": "1.0",
        "status": "NEEDS_REVIEW" if warnings else "PASS_WITHOUT_GOLD_STANDARD",
        "document_profile": profile,
        "ner": {
            "architecture": "predicate-first SVO arguments + pretrained SciBERT ontology typing + local Qwen adjudication",
            "accepted_mentions_by_channel": dict(sorted(extraction_channels.items())),
            "optional_external_adapter": "not installed",
            "span_policy": "every accepted mention must match exact source characters",
        },
        "graph": {
            "entities": len(entities),
            "relationships": len(relationships),
            "relationship_density": round(len(relationships) / max(1, len(entities)), 4),
            "orphan_entities": len(orphan_entities),
            "orphan_rate": round(len(orphan_entities) / max(1, len(entities)), 4),
            "label_counts": dict(sorted(label_counts.items())),
            "role_coverage": role_coverage,
            "covered_roles": covered_roles,
            "total_roles": len(role_coverage),
        },
        "quality_warnings": warnings,
        "suspicious_agents": [row.get("canonical_name", "") for row in suspicious_agents],
        "generic_entities": [row.get("canonical_name", "") for row in generic_entities],
        "competency_questions": {"answerable": cq_answerable, "total": cq_total},
        "note": "These are recall and precision diagnostics, not substitutes for human gold annotations.",
    }
