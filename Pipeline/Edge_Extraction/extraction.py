"""Orchestrate relationship generation, refinement, validation, and artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from Pipeline.core import (
    CONFIG_PATH, ONTOLOGY_PATH, dynamic_token_batches, estimated_json_tokens,
    json_read, jsonl_read, jsonl_write,
)
from Pipeline.llm import OllamaClient
from Pipeline.ontology import Ontology

from .context import nodes_for_paragraph, paragraphs
from .passes import (
    RELATIONSHIP_RUBRIC, generate_relationships, generation_payload,
    refine_relationships, refinement_payload, validate_relationships,
    validation_payload,
)
from .proposals import prepare_proposals, proposal_artifacts


def run_relationship_extraction(
    parsed: dict, accepted_mentions: list[dict], run_dir: Path,
    progress: Callable[..., None] | None = None, *, llm_model: str | None = None,
    client: OllamaClient | None = None,
) -> dict:
    """Run three relationship LLM passes for every paragraph containing an accepted node."""
    config = json_read(CONFIG_PATH)
    client = client or OllamaClient.from_config(config, model=llm_model)
    audit_start = len(client.audit)
    ontology = Ontology(ONTOLOGY_PATH)
    spec = config.get("relationships", {})
    minimum = int(spec.get("minimum_per_paragraph", 0))
    maximum = int(spec.get("maximum_per_paragraph", 10))
    mentions = {row["mention_id"]: row for row in accepted_mentions}

    eligible = [
        (paragraph, nodes_for_paragraph(paragraph, accepted_mentions))
        for paragraph in paragraphs(parsed)
    ]
    eligible = [(paragraph, nodes) for paragraph, nodes in eligible if nodes]
    generation_groups = dynamic_token_batches(
        eligible, config=config,
        input_value=lambda unit: generation_payload(unit, ontology),
        expected_output_tokens=lambda _: max(1, maximum) * 45,
        fixed_input_tokens=650 + estimated_json_tokens(list(ontology.nodes)),
    )
    generated_by_paragraph: dict[str, list[dict]] = {}
    for group in generation_groups:
        generated_by_paragraph.update(
            generate_relationships(client, group, ontology, minimum, maximum),
        )

    prepared_by_paragraph = {
        paragraph["paragraph_id"]: prepare_proposals(
            paragraph, generated_by_paragraph.get(paragraph["paragraph_id"], []),
        )
        for paragraph, _ in eligible
    }
    refinement_units = [
        (paragraph, nodes, prepared_by_paragraph[paragraph["paragraph_id"]])
        for paragraph, nodes in eligible if prepared_by_paragraph[paragraph["paragraph_id"]]
    ]
    refinement_groups = dynamic_token_batches(
        refinement_units, config=config,
        input_value=lambda unit: refinement_payload(unit, ontology),
        expected_output_tokens=lambda unit: 12 * len(unit[2]), fixed_input_tokens=500,
    )
    refined_by_paragraph: dict[str, list[dict]] = {
        paragraph["paragraph_id"]: [] for paragraph, _ in eligible
    }
    for group in refinement_groups:
        refined_by_paragraph.update(refine_relationships(client, group, ontology))

    validation_units = [
        (paragraph, refined_by_paragraph[paragraph["paragraph_id"]])
        for paragraph, _ in eligible if refined_by_paragraph[paragraph["paragraph_id"]]
    ]
    validation_groups = dynamic_token_batches(
        validation_units, config=config, input_value=validation_payload,
        expected_output_tokens=lambda unit: 8 * len(unit[1]),
        fixed_input_tokens=400 + estimated_json_tokens(RELATIONSHIP_RUBRIC),
    )
    llm_validations: dict[str, dict] = {}
    for group in validation_groups:
        llm_validations.update(validate_relationships(client, group))

    candidates: list[dict] = []
    inferred: dict[str, dict] = {}
    assertions: list[dict] = []
    for paragraph, _ in eligible:
        paragraph_id = paragraph["paragraph_id"]
        generated_count = len(generated_by_paragraph.get(paragraph_id, []))
        for proposal in refined_by_paragraph.get(paragraph_id, []):
            candidate, assertion, unknown_rows = proposal_artifacts(
                paragraph, proposal, generated_count, minimum,
                llm_validations.get(proposal["proposal_id"], {}), mentions, ontology,
            )
            candidates.append(candidate)
            if assertion:
                assertions.append(assertion)
            for row in unknown_rows:
                inferred[row["candidate_id"]] = row

    inferred_rows = list(inferred.values())
    existing_review = [
        row for row in jsonl_read(run_dir / "review_queue.jsonl")
        if row.get("origin") != "relationship_inference"
    ]
    jsonl_write(run_dir / "review_queue.jsonl", [*existing_review, *inferred_rows])
    jsonl_write(run_dir / "node_review_candidates.jsonl", [*existing_review, *inferred_rows])
    jsonl_write(run_dir / "relationship_candidates.jsonl", candidates)
    jsonl_write(run_dir / "relationship_unknown_nodes.jsonl", inferred_rows)
    jsonl_write(run_dir / "assertions.jsonl", assertions)
    jsonl_write(
        run_dir / "relationship_rejections.jsonl",
        [row for row in candidates if row["status"] == "rejected"],
    )
    jsonl_write(run_dir / "ollama_relationship_calls.jsonl", client.audit[audit_start:])
    if progress:
        progress(
            "relationships", "Paragraph relationships generated, refined, and validated.",
            80, "complete", {
                "eligible_paragraphs": len(eligible), "candidates": len(candidates),
                "accepted": len(assertions),
                "inferred_nodes_needing_review": len(inferred_rows),
            },
        )
    return {
        "mentions": accepted_mentions, "candidates": candidates, "assertions": assertions,
        "unknown_nodes": inferred_rows, "llm_calls": client.audit[audit_start:],
        "counts": {
            "eligible_paragraphs": len(eligible), "candidates": len(candidates),
            "accepted": len(assertions),
            "needs_review": sum(row["status"] == "needs_review" for row in candidates),
            "rejected": sum(row["status"] == "rejected" for row in candidates),
        },
    }
