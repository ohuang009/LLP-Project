"""Top-level PDF runners for node-only and complete extraction workflows."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from Pipeline.core import CONFIG_PATH, copy_source, json_read, json_write, jsonl_write, now_iso
from Pipeline.lexicon import update_lexicon
from Pipeline.llm import OllamaClient
from Pipeline.Paper_Parsing import parse_pdf

from .entities import canonical_entities
from .extraction import Progress, _emit, extract_nodes
from .provenance import validate_provenance


def _parse_source(
    pdf_path: Path, run_dir: Path, include_section_titles: set[str] | None = None,
) -> dict:
    """Parse and persist the source PDF, optionally retaining selected sections."""
    parsed = parse_pdf(pdf_path.read_bytes(), pdf_path.name)
    if include_section_titles:
        wanted = {value.casefold() for value in include_section_titles}
        parsed = deepcopy(parsed)
        parsed["sections"] = [
            row for row in parsed.get("sections", [])
            if row.get("title", "").casefold() in wanted
        ]
    json_write(run_dir / "parsed.json", parsed)
    copy_source(pdf_path, run_dir / "source.pdf")
    return parsed


def run_extraction(
    pdf_path: Path, run_dir: Path, progress: Progress | None = None, *,
    llm_model: str | None = None, client: OllamaClient | None = None,
    include_section_titles: set[str] | None = None,
) -> dict:
    """Run the complete PDF -> nodes -> relationships pipeline without publishing."""
    client = client or OllamaClient.from_config(json_read(CONFIG_PATH), model=llm_model)
    run_dir.mkdir(parents=True, exist_ok=True)
    parsed = _parse_source(pdf_path, run_dir, include_section_titles)
    _emit(progress, "parsing", "PDF parsed with sentence and paragraph provenance.", 15, parsed.get("summary", {}))
    node_result = extract_nodes(parsed, run_dir, client=client, model=llm_model, progress=progress)
    lexicon_update = update_lexicon(node_result["entities"])

    from Pipeline.Edge_Extraction import (
        materialize_canonical_relationships, run_relationship_extraction,
    )
    relationship_result = run_relationship_extraction(
        parsed, node_result["mentions"], run_dir, progress,
        llm_model=llm_model, client=client,
    )
    entities = canonical_entities(relationship_result["mentions"])
    jsonl_write(run_dir / "mentions.jsonl", relationship_result["mentions"])
    jsonl_write(run_dir / "canonical_entities.jsonl", entities)
    jsonl_write(run_dir / "canonical_entities_merged.jsonl", entities)
    relationships = materialize_canonical_relationships(
        run_dir, entities, relationship_result["assertions"],
    )
    validation = validate_provenance(
        parsed, relationship_result["mentions"], relationship_result["candidates"],
    )

    from Pipeline.Question_Evaluation.evaluator import evaluate_question_answerability
    from Pipeline.Validation.generalization_diagnostics import build_generalization_diagnostics
    from Pipeline.Validation.semantic_completeness import evaluate_semantic_completeness
    questions = evaluate_question_answerability(
        run_dir, entities, relationships, relationship_result["assertions"],
    )
    completeness = evaluate_semantic_completeness(
        parsed, entities, relationships, relationship_result["assertions"], questions,
    )
    diagnostics = build_generalization_diagnostics(
        parsed, relationship_result["mentions"], entities, relationships,
        relationship_result["assertions"], questions,
    )
    json_write(run_dir / "semantic_completeness.json", completeness)
    json_write(run_dir / "generalization_diagnostics.json", diagnostics)
    validation["semantic_completeness"] = completeness
    validation["generalization_diagnostics"] = diagnostics
    json_write(run_dir / "validation_report.json", validation)

    manifest = {
        "schema_version": "3.0", "run_id": run_dir.name, "created_at": now_iso(),
        "document": parsed["document"], "parser_summary": parsed.get("summary", {}),
        "pipeline": "raw_svo_scibert_qwen_ollama_nodes_and_paragraph_edges",
        "llm": {"provider": "ollama", "model": client.model},
        "graph_publication": {"state": "not_added", "add_count": 0},
        "counts": {
            "accepted_mentions": len(relationship_result["mentions"]),
            "canonical_entities": len(entities),
            "node_review_candidates": len(node_result["review"]),
            "rejected_node_candidates": len(node_result["rejected"]),
            "relationships": len(relationships),
            "relationship_review_candidates": sum(
                row["status"] == "needs_review" for row in relationship_result["candidates"]
            ),
            "answerable_questions": questions["answerable_questions"],
            "total_questions": questions["total_questions"],
            "lexicon_entries": lexicon_update["total"],
        },
        "semantic_completeness": completeness,
        "outputs": [
            "source.pdf", "parsed.json", "raw_svo.jsonl", "grammatical_triples.jsonl",
            "paragraph_batches.jsonl", "scibert_typings.jsonl", "node_candidates.jsonl",
            "candidate_judgments.jsonl", "mentions.jsonl", "review_queue.jsonl",
            "node_rejections.jsonl", "canonical_entities_merged.jsonl",
            "ollama_node_calls.jsonl", "relationship_candidates.jsonl",
            "relationship_unknown_nodes.jsonl", "assertions.jsonl",
            "relationship_rejections.jsonl", "ollama_relationship_calls.jsonl",
            "canonical_relationships.jsonl", "question_answerability.json",
            "semantic_completeness.json", "generalization_diagnostics.json",
            "validation_report.json", "manifest.json",
        ],
        "validation": validation,
    }
    json_write(run_dir / "manifest.json", manifest)
    _emit(
        progress, "complete",
        "Extraction complete; Neo4j publication awaits explicit approval.",
        100, manifest["counts"],
    )
    return {"run_dir": str(run_dir), **manifest}


def run_node_only_extraction(
    pdf_path: Path, run_dir: Path, progress: Progress | None = None, *,
    include_section_titles: set[str] | None = None, llm_model: str | None = None,
    client: OllamaClient | None = None,
) -> dict:
    """Run parsing and node extraction without relationship extraction."""
    client = client or OllamaClient.from_config(json_read(CONFIG_PATH), model=llm_model)
    run_dir.mkdir(parents=True, exist_ok=True)
    # Keep the node-only runner's historical behavior: this option is accepted
    # for API compatibility, but section filtering belongs to the full runner.
    parsed = _parse_source(pdf_path, run_dir)
    result = extract_nodes(parsed, run_dir, client=client, model=llm_model, progress=progress)
    lexicon_update = update_lexicon(result["entities"])
    manifest = {
        "run_id": run_dir.name,
        "document": parsed["document"],
        "parser_summary": parsed.get("summary", {}),
        "counts": {
            "accepted_mentions": len(result["mentions"]),
            "canonical_entities": len(result["entities"]),
            "lexicon_entries": lexicon_update["total"],
            "node_review_candidates": len(result["review"]),
            "rejected_node_candidates": len(result["rejected"]),
        },
        "graph_publication": {"state": "not_added", "add_count": 0},
    }
    json_write(run_dir / "manifest.json", manifest)
    return manifest
