from __future__ import annotations

import re


CORE_AI_LABELS = {"AISystem", "Agent", "Model"}


def _sentences(parsed: dict) -> list[dict]:
    """Handle sentences for this stage. It measures whether the graph is traceable, complete, and generalizable."""
    return [
        sentence
        for section in parsed.get("sections", [])
        for paragraph in section.get("paragraphs", [])
        for sentence in paragraph.get("sentences", [])
    ]


def _signal(sentences: list[dict], pattern: str) -> list[dict]:
    """Handle signal for this stage. It measures whether the graph is traceable, complete, and generalizable."""
    regex = re.compile(pattern, re.I)
    return [
        {"sentence_id": row["id"], "quote": row["text"], "pages": row.get("pages", [])}
        for row in sentences
        if regex.search(row["text"])
    ]


def evaluate_semantic_completeness(
    parsed: dict,
    entities: list[dict],
    relationships: list[dict],
    assertions: list[dict],
    question_evaluation: dict,
) -> dict:
    """Report likely recall gaps separately from structural/traceability validation."""
    sentences = _sentences(parsed)
    label_counts: dict[str, int] = {}
    for entity in entities:
        label_counts[entity["label"]] = label_counts.get(entity["label"], 0) + 1

    system_signals = _signal(
        sentences,
        r"\b(?:introduc(?:e|es|ed|ing)|propos(?:e|es|ed|ing)|develop(?:s|ed|ing)?)\b"
        r"[^.;]{0,140}\b(?:system|framework|platform|assistant|agent|pipeline)\b",
    )
    agent_signals = _signal(
        sentences,
        r"\b(?:two|three|four|five|\d+)\s+(?:expert\s+)?agents?\b",
    )
    model_identifier = (
        r"(?:[A-Z][A-Za-z0-9]*[A-Z0-9][A-Za-z0-9]*|[A-Z][A-Za-z0-9]*(?:[-:./][A-Za-z0-9.]+)+)"
    )
    model_cue = r"\b(?:model|models|language model|LLMs?|backbone|regressor)\b"
    model_signals = _signal(
        sentences,
        rf"(?:{model_cue}[^.;]{{0,100}}{model_identifier}|{model_identifier}[^.;]{{0,35}}{model_cue})",
    )

    connected_ids = {
        entity_id
        for relationship in relationships
        for entity_id in (relationship["subject_entity_id"], relationship["object_entity_id"])
    }
    core_entities = [row for row in entities if row["label"] in CORE_AI_LABELS]
    connected_core = [row for row in core_entities if row["entity_id"] in connected_ids]
    traceable_assertions = [
        row for row in assertions
        if row.get("evidence_quote") and (row.get("evidence_sentence_ids") or row.get("scope") == "document_metadata")
    ]

    checks = [
        {
            "id": "introduced_ai_system",
            "label": "Introduced AI system captured",
            "applicable": bool(system_signals),
            "passed": not system_signals or label_counts.get("AISystem", 0) > 0,
            "expected_signals": len(system_signals),
            "detected_entities": label_counts.get("AISystem", 0),
            "evidence": system_signals[:3],
        },
        {
            "id": "enumerated_agents",
            "label": "Enumerated agents captured",
            "applicable": bool(agent_signals),
            "passed": not agent_signals or label_counts.get("Agent", 0) >= 2,
            "expected_signals": len(agent_signals),
            "detected_entities": label_counts.get("Agent", 0),
            "evidence": agent_signals[:3],
        },
        {
            "id": "named_models",
            "label": "Named models captured",
            "applicable": bool(model_signals),
            "passed": not model_signals or label_counts.get("Model", 0) > 0,
            "expected_signals": len(model_signals),
            "detected_entities": label_counts.get("Model", 0),
            "evidence": model_signals[:3],
        },
        {
            "id": "core_graph_connectivity",
            "label": "Core AI nodes participate in facts",
            "applicable": len(core_entities) >= 2,
            "passed": len(core_entities) < 2 or len(connected_core) >= 2,
            "expected_signals": len(core_entities),
            "detected_entities": len(connected_core),
            "evidence": [],
        },
        {
            "id": "relationship_traceability",
            "label": "Relationships retain exact evidence",
            "applicable": bool(assertions),
            "passed": not assertions or len(traceable_assertions) == len(assertions),
            "expected_signals": len(assertions),
            "detected_entities": len(traceable_assertions),
            "evidence": [],
        },
    ]
    applicable = [row for row in checks if row["applicable"]]
    passed = [row for row in applicable if row["passed"]]
    failures = [row for row in applicable if not row["passed"]]
    return {
        "schema_version": "1.0",
        "status": "COMPLETE" if not failures else "INCOMPLETE",
        "coverage_score": round(len(passed) / len(applicable), 3) if applicable else 1.0,
        "checks_passed": len(passed),
        "checks_applicable": len(applicable),
        "checks": checks,
        "question_coverage": {
            "answerable": question_evaluation.get("answerable_questions", 0),
            "total": question_evaluation.get("total_questions", 0),
        },
        "note": "This is a recall-oriented warning layer; ontology integrity and traceability remain separate validation gates.",
    }
