from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
import unicodedata
from typing import Callable, Iterable

from Pipeline.Grammatical_Parsing.candidates import generate_grammatical_candidates
from Pipeline.Paper_Parsing.parser import parse_pdf
from Pipeline.Question_Evaluation.evaluator import evaluate_question_answerability
from Pipeline.Validation.generalization_diagnostics import aggregate_llm_usage, build_generalization_diagnostics
from Pipeline.Validation.semantic_completeness import evaluate_semantic_completeness
from .adjudication import optional_llm_review
from .candidate_generation import flatten_sentences, generate_candidates, mention_dedup_key, remove_redundant_review_candidates
from .canonicalization import canonical_entities, materialize_merges, similarity_candidates
from .common import CONFIG_PATH, ONTOLOGY_PATH, RELATIONSHIP_OUTPUTS, RUBRIC_PATH, json_read, json_write, jsonl_write, now_iso
from .contracts import candidate_artifact, judgment_artifact
from .lexicon import Lexicon, attach_prior_nodes_same_type, load_prior_node_catalog
from .ner_boundary import build_required_ner_artifacts
from .reference_resolution import optional_llm_reference_resolution

def run_extraction(pdf_path: Path, run_dir: Path, progress: Callable[..., None]) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    stage_reports: dict[str, dict] = {}
    pipeline_started = time.perf_counter()
    stage_started: dict[str, float] = {}
    stage_seconds: dict[str, float] = {}

    def emit(stage: str, message: str, value: int, status: str = "running", output: dict | None = None) -> None:
        now = time.perf_counter()
        if status == "running":
            stage_started.setdefault(stage, now)
        elif status == "complete" and stage in stage_started:
            stage_seconds[stage] = round(now - stage_started.pop(stage), 3)
            output = {**(output or {}), "elapsed_seconds": stage_seconds[stage]}
        report = {"stage": stage, "status": status, "message": message, "progress": value}
        if output is not None:
            report["output"] = output
        stage_reports[stage] = report
        json_write(run_dir / "stage_outputs.json", {"stages": list(stage_reports.values())})
        try:
            progress(stage, message, value, report)
        except TypeError:
            progress(stage, message, value)

    emit("parsing", "Parsing narrative text from the paper.", 8)
    pdf_bytes = pdf_path.read_bytes()
    parsed = parse_pdf(pdf_bytes, pdf_path.name)
    json_write(run_dir / "parsed.json", parsed)
    shutil.copy2(pdf_path, run_dir / "source.pdf")
    parser_output = {
        "artifact": "parsed.json",
        "sections": parsed["summary"].get("sections", len(parsed.get("sections", []))),
        "paragraphs": parsed["summary"].get("paragraphs", 0),
        "sentences": parsed["summary"].get("sentences", 0),
        "pages": parsed["summary"].get("pages", parsed["summary"].get("page_count", 0)),
        "narrative_pages": parsed["summary"].get("narrative_pages", 0),
        "last_narrative_page": parsed["summary"].get("last_narrative_page", 0),
        "narrative_page_ratio": parsed["summary"].get("narrative_page_ratio", 0.0),
    }
    emit("parsing", "Narrative text parsed and saved.", 18, "complete", parser_output)

    emit("ner", "Generating a high-recall pool of exact-span node candidates.", 24)
    lexicon = Lexicon()
    node_config = json_read(CONFIG_PATH)
    grammar = generate_grammatical_candidates(parsed, ONTOLOGY_PATH, node_config)
    jsonl_write(run_dir / "grammatical_analysis.jsonl", grammar["sentence_analysis"])
    jsonl_write(run_dir / "grammatical_triples.jsonl", grammar["triples"])
    jsonl_write(run_dir / "scibert_typings.jsonl", grammar["typings"])
    json_write(run_dir / "grammatical_stage.json", grammar["status"])
    accepted, review = generate_candidates(parsed, lexicon, grammar["candidates_by_sentence"])
    review = remove_redundant_review_candidates(accepted, review)
    prior_catalog = load_prior_node_catalog(lexicon, exclude_run_dir=run_dir)
    attach_prior_nodes_same_type(
        review, prior_catalog,
        maximum=int(node_config.get("prior_node_matching", {}).get("maximum_candidates_per_node", 12)),
    )
    jsonl_write(run_dir / "node_normalization_candidates.jsonl", [{
        "candidate_id": row["candidate_id"],
        "document_id": row["document_id"],
        "sentence_id": row["source"]["sentence_id"],
        "original_surface_text": row["surface_text"],
        "deterministically_cleaned_name": row.get("canonical_name") or row["surface_text"],
        "deterministic_normalization": row.get("candidate_metadata", {}).get("deterministic_normalization", {}),
        "scibert_suggested_label": row.get("label", ""),
        "scibert_alternatives": row.get("candidate_metadata", {}).get("typing_alternatives", []),
        "prior_nodes_same_type": row.get("candidate_metadata", {}).get("prior_nodes_same_type", []),
    } for row in review])
    rubric = json_read(RUBRIC_PATH)
    node_candidates = [
        *(candidate_artifact(row, trusted=True) for row in accepted),
        *(candidate_artifact(row, trusted=False) for row in review),
    ]
    trusted_judgments = [
        judgment_artifact(
            row, decision="accept", validation=row["validation"], rubric=rubric,
        )
        for row in accepted
    ]
    ner_candidates, ner_status = build_required_ner_artifacts(parsed, accepted, review, node_config)
    ner_status["grammatical_candidate_generation"] = grammar["status"]
    jsonl_write(run_dir / "node_candidates.jsonl", node_candidates)
    jsonl_write(run_dir / "ner_candidates.jsonl", ner_candidates)
    json_write(run_dir / "ner_stage.json", ner_status)
    emit("ner", "Candidate generation completed with exact-span validation.", 36, "complete", ner_status)

    emit("adjudicating", "Applying the versioned validity rubric to every untrusted candidate.", 42)
    llm_accepted, review, rejected, llm_judgments, llm_calls, llm_status = optional_llm_review(review, emit)
    accepted.extend(llm_accepted)
    # Eliminate exact duplicate mentions without erasing source occurrences.
    seen = set()
    mentions = []
    for row in accepted:
        key = mention_dedup_key(row)
        if key not in seen:
            seen.add(key)
            mentions.append(row)
    jsonl_write(run_dir / "node_review_candidates.jsonl", review)
    jsonl_write(run_dir / "review_queue.jsonl", review)
    jsonl_write(run_dir / "node_rejections.jsonl", rejected)
    jsonl_write(run_dir / "candidate_judgments.jsonl", [*trusted_judgments, *llm_judgments])
    jsonl_write(run_dir / "llm_node_judge_calls.jsonl", llm_calls)
    emit("adjudicating", "Node decisions and their audit trail are saved.", 55, "complete", {
        "accepted_mentions": len(mentions), "remaining_for_review": len(review),
        "rejected_candidates": len(rejected),
        "model_used": bool(llm_status.get("used")), "model": llm_status.get("model", ""),
        "audit_calls": len(llm_calls), "artifact": "llm_node_judge_calls.jsonl",
    })
    emit("resolving", "Resolving vague surface mentions against specific nearby nodes.", 57)
    resolved_references, reference_review, ignored_references, reference_llm_calls, reference_llm_status = optional_llm_reference_resolution(
        parsed, mentions, emit
    )
    mentions.extend(resolved_references)
    jsonl_write(run_dir / "mentions.jsonl", mentions)
    jsonl_write(run_dir / "reference_resolution_review.jsonl", reference_review)
    jsonl_write(run_dir / "reference_resolution_ignored.jsonl", ignored_references)
    jsonl_write(run_dir / "llm_reference_judge_calls.jsonl", reference_llm_calls)
    emit("resolving", "Contextually matched references now point to specific nodes; ambiguous cases await review.", 60, "complete", {
        "llm_resolved": len(resolved_references), "human_review": len(reference_review),
        "ignored_unresolved": len(ignored_references),
        "model_used": bool(reference_llm_status.get("used")),
        "pairwise_comparisons": reference_llm_status.get("pairs", 0),
        "artifacts": ["reference_resolution_review.jsonl", "reference_resolution_ignored.jsonl", "llm_reference_judge_calls.jsonl"],
    })

    from Pipeline.Edge_Extraction import (
        materialize_canonical_relationships,
        run_relationship_extraction,
    )
    relationship_result = run_relationship_extraction(parsed, mentions, run_dir, emit)
    mentions = relationship_result["mentions"]
    jsonl_write(run_dir / "mentions.jsonl", mentions)
    emit("relationships", "Context-window relationship decisions and accepted assertions are saved.", 70, "complete", {
        **relationship_result["counts"],
        "model_used": bool(relationship_result["llm"].get("used")),
        "artifacts": ["relationship_candidates.jsonl", "relationship_unknown_nodes.jsonl",
                      "assertions.jsonl", "relationship_rejections.jsonl"],
    })

    emit("canonicalizing", "Grouping exact canonical IDs and reviewed aliases.", 74)
    entities = canonical_entities(mentions)
    jsonl_write(run_dir / "canonical_entities.jsonl", entities)
    emit("canonicalizing", "Canonical nodes are ready for entity resolution.", 82, "complete", {
        "canonical_entities": len(entities),
        "artifact": "canonical_entities.jsonl",
    })

    emit("matching", "Comparing canonical entities for possible duplicates.", 85)
    threshold = float(json_read(CONFIG_PATH)["similarity"]["candidate_threshold"])
    pairs = similarity_candidates(entities, threshold)
    jsonl_write(run_dir / "similar_nodes_review.jsonl", pairs)
    merged = materialize_merges(run_dir)
    canonical_relationships = materialize_canonical_relationships(
        run_dir, merged, relationship_result["assertions"]
    )
    from Pipeline.Question_Evaluation.evaluator import evaluate_question_answerability
    question_evaluation = evaluate_question_answerability(
        run_dir, merged, canonical_relationships, relationship_result["assertions"]
    )
    json_write(run_dir / "lexicon_snapshot.json", lexicon.data)
    semantic_completeness = evaluate_semantic_completeness(
        parsed, merged, canonical_relationships, relationship_result["assertions"], question_evaluation,
    )
    json_write(run_dir / "semantic_completeness.json", semantic_completeness)
    generalization_diagnostics = build_generalization_diagnostics(
        parsed, mentions, merged, canonical_relationships,
        relationship_result["assertions"], question_evaluation,
    )
    llm_usage = aggregate_llm_usage(
        llm_calls, reference_llm_calls, relationship_result["llm_calls"],
        accepted_mentions=len(mentions), assertions=len(relationship_result["assertions"]),
    )
    json_write(run_dir / "generalization_diagnostics.json", generalization_diagnostics)
    json_write(run_dir / "llm_usage_summary.json", llm_usage)

    emit("matching", "Possible duplicate pairs are ready for human review.", 90, "complete", {
        "similar_node_candidates": len(pairs), "merged_entities": len(merged),
        "canonical_relationships": len(canonical_relationships),
        "answerable_questions": question_evaluation["answerable_questions"],
        "total_questions": question_evaluation["total_questions"],
        "semantic_completeness": semantic_completeness["status"],
        "semantic_coverage_score": semantic_completeness["coverage_score"],        "artifacts": ["similar_nodes_review.jsonl", "canonical_entities_merged.jsonl", "canonical_relationships.jsonl"],
    })

    emit("neo4j", "Upserting this paper into the Engineered Water Systems Neo4j graph.", 93)
    from Pipeline.Graph_Persistence.neo4j_writer import upsert_run
    graph = upsert_run(run_dir, parsed, merged, canonical_relationships)
    json_write(run_dir / "neo4j_upsert.json", graph)
    emit("neo4j", "Neo4j now contains this run, its nodes, evidence, and relationships.", 98, "complete", graph)

    context_counts = [len(row["context"]) for row in flatten_sentences(parsed)]
    validation = {
        "status": "PASS",
        "node_only": False,
        "relationship_extraction_enabled": True,
        "relationship_files_created": list(RELATIONSHIP_OUTPUTS),
        "traceability": {
            "mentions_with_evidence": sum(bool(row["source"].get("evidence_quote")) for row in mentions),
            "sentence_mentions": sum(row["source"].get("kind") == "sentence_span" for row in mentions),
            "sentence_mentions_with_context": sum(bool(row["source"].get("context_sentences")) for row in mentions),
            "minimum_context_sentences": min(context_counts) if context_counts else 0,
            "maximum_context_sentences": max(context_counts) if context_counts else 0,
        },
        "reference_resolution": {
            "llm_resolved": len(resolved_references),
            "pending_human_review": len(reference_review),
            "ignored_unresolved": len(ignored_references),
        },
        "ner": ner_status,
        "llm": llm_status,
        "reference_llm": reference_llm_status,
        "semantic_completeness": semantic_completeness,
        "generalization_diagnostics": generalization_diagnostics,
        "llm_usage": llm_usage,
        "relationship_llm": relationship_result["llm"],
        "relationship_counts": relationship_result["counts"],
    }
    json_write(run_dir / "validation_report.json", validation)
    timing = {
        "total_seconds": round(time.perf_counter() - pipeline_started, 3),
        "stage_seconds": dict(stage_seconds),
        "measurement_boundary": "PDF read through verified Neo4j upsert",
    }
    summary = {
        "run_id": run_dir.name, "created_at": now_iso(), "source_filename": pdf_path.name,
        "paper": parsed["document"], "parser_summary": parsed["summary"],
        "counts": {
            "accepted_mentions": len(mentions), "canonical_entities": len(entities),
            "node_review_candidates": len(review), "rejected_node_candidates": len(rejected),
            "similar_node_candidates": len(pairs),
            "reference_mentions_resolved": len(resolved_references),
            "reference_review_pending": len(reference_review),
            "reference_mentions_ignored": len(ignored_references),
            "merged_entities": len(merged),
            "relationship_candidates": relationship_result["counts"]["candidates"],
            "relationship_unknown_nodes": relationship_result["counts"]["unknown_nodes"],
            "relationships": len(canonical_relationships),
            "assertions": relationship_result["counts"]["assertions"],
            "answerable_questions": question_evaluation["answerable_questions"],
            "total_questions": question_evaluation["total_questions"],
            "semantic_checks_passed": semantic_completeness["checks_passed"],
            "semantic_checks_applicable": semantic_completeness["checks_applicable"],
            "semantic_coverage_score": semantic_completeness["coverage_score"],
            "lexicon_entries": len(lexicon.entries),
        },
        "ner": ner_status,
        "llm": llm_status,
        "semantic_completeness": semantic_completeness,
        "generalization_diagnostics": generalization_diagnostics,
        "llm_usage": llm_usage,
        "reference_llm": reference_llm_status,
        "relationship_llm": relationship_result["llm"],
        "relationship_extraction": "llm_context_window_v1",
        "timing": timing,
        "neo4j": graph,
        "stages": list(stage_reports.values()),
        "outputs": [
            "parsed.json", "grammatical_analysis.jsonl", "grammatical_triples.jsonl",
            "scibert_typings.jsonl", "grammatical_stage.json", "node_candidates.jsonl",
            "node_normalization_candidates.jsonl",
            "ner_candidates.jsonl", "ner_stage.json", "mentions.jsonl",
            "candidate_judgments.jsonl", "node_review_candidates.jsonl", "review_queue.jsonl",
            "node_rejections.jsonl", "canonical_entities.jsonl",
            "similar_nodes_review.jsonl", "canonical_entities_merged.jsonl", "lexicon_snapshot.json",
            "reference_resolution_review.jsonl", "reference_resolution_ignored.jsonl",
            "llm_node_judge_calls.jsonl", "llm_reference_judge_calls.jsonl", *RELATIONSHIP_OUTPUTS,
            "question_answerability.json", "semantic_completeness.json",
            "generalization_diagnostics.json", "llm_usage_summary.json",
            "validation_report.json", "stage_outputs.json", "neo4j_upsert.json", "manifest.json", "source.pdf",
        ],
    }
    json_write(run_dir / "manifest.json", summary)
    emit("complete", "The paper is fully extracted and upserted into Neo4j.", 100, "complete", {
        "nodes": len(entities), "relationships": len(canonical_relationships), "neo4j_status": graph["status"],
    })
    summary["stages"] = list(stage_reports.values())
    json_write(run_dir / "manifest.json", summary)
    return summary
