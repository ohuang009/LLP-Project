from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from Pipeline.Node_Pipeline.common import json_write


QUESTION_SPECS = (
    ("system", "What AI system does the paper propose?", {"PROPOSES"}),
    ("agents", "Which agents make up the proposed AI system?", {"HAS_AGENT"}),
    ("models", "Which AI models are used, and by which system or agent?", {"USES_MODEL"}),
    ("tasks", "What problems or tasks does the AI system tackle?", {"PERFORMS"}),
    ("tools", "Which software tools or simulators does the system use?", {"USES_TOOL"}),
    ("methods", "Which workflow or oversight methods does the system use?", {"USES_METHOD"}),
    ("benchmarks", "On which datasets or benchmark networks was it evaluated?", {"EVALUATED_ON"}),
    ("metrics", "Which metrics are used to evaluate performance?", {"EVALUATED_BY", "MEASURES"}),
    ("results", "What quantitative results does the paper report?", {"REPORTS"}),
    ("limitations", "What limitations of the system does the paper identify?", {"HAS_LIMITATION"}),
)


def _answer_rows(key: str, relationships: list[dict], entities: dict[str, dict]) -> list[str]:
    rows: list[str] = []
    for relationship in relationships:
        source = entities.get(relationship["subject_entity_id"], {})
        target = entities.get(relationship["object_entity_id"], {})
        source_name = source.get("canonical_name", relationship["subject_entity_id"])
        target_name = target.get("canonical_name", relationship["object_entity_id"])
        predicate = relationship["predicate"]
        if key in {"models", "tasks"}:
            value = f"{source_name} -> {target_name}"
        elif key == "metrics" and predicate == "MEASURES":
            value = target_name
        elif key == "results":
            value = target_name
        else:
            value = target_name
        if value not in rows:
            rows.append(value)
    return rows


def evaluate_question_answerability(
    run_dir: Path,
    entities: list[dict],
    relationships: list[dict],
    assertions: list[dict],
) -> dict:
    """Evaluate only graph-materialized facts; parsed prose is evidence, never an answer shortcut."""
    entity_index = {row["entity_id"]: row for row in entities}
    assertion_index = {row["assertion_id"]: row for row in assertions}
    by_predicate: dict[str, list[dict]] = defaultdict(list)
    for relationship in relationships:
        by_predicate[relationship["predicate"]].append(relationship)

    questions: list[dict] = []
    for key, question, predicates in QUESTION_SPECS:
        matched = [row for predicate in predicates for row in by_predicate.get(predicate, [])]
        answers = _answer_rows(key, matched, entity_index)
        evidence: list[dict] = []
        seen_quotes: set[str] = set()
        for relationship in matched:
            for assertion_id in relationship.get("assertion_ids", []):
                assertion = assertion_index.get(assertion_id)
                if not assertion:
                    continue
                quote = assertion.get("evidence_quote", "")
                if not quote or quote in seen_quotes:
                    continue
                seen_quotes.add(quote)
                evidence.append({
                    "quote": quote,
                    "pages": assertion.get("pages", []),
                    "sentence_ids": assertion.get("evidence_sentence_ids", []),
                    "assertion_id": assertion_id,
                })
        status = "answerable" if answers and evidence else "not_answerable"
        questions.append({
            "id": key,
            "question": question,
            "status": status,
            "answers": answers,
            "relationship_types": sorted(predicates),
            "evidence": evidence,
            "criterion": "At least one graph fact and one exact supporting paper sentence are required.",
        })

    answerable = sum(row["status"] == "answerable" for row in questions)
    result = {
        "schema_version": "1.0",
        "evaluation_scope": "canonical graph facts only",
        "total_questions": len(questions),
        "answerable_questions": answerable,
        "coverage_percent": round(100 * answerable / max(1, len(questions)), 1),
        "questions": questions,
    }
    json_write(run_dir / "question_answerability.json", result)
    return result
