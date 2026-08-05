"""Stable node-only orchestration boundary.

This module intentionally never imports relationship_pipeline. It protects the
approved node-only workbench from experimental relationship work elsewhere in
the project.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
import sys
from typing import Callable

WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from Pipeline.Paper_Parsing.parser import parse_pdf

from Pipeline.Node_Pipeline import (
    CONFIG_PATH, LEXICON_PATH, ONTOLOGY_PATH, RUBRIC_PATH, Lexicon, canonical_entities, flatten_sentences,
    attach_prior_nodes_same_type, generate_candidates, json_read, json_write, jsonl_write,
    load_prior_node_catalog, materialize_merges, mention_dedup_key, now_iso,
    optional_llm_reference_resolution, optional_llm_review, remove_redundant_review_candidates,
    similarity_candidates,
)
from Pipeline.Node_Pipeline.contracts import candidate_artifact, judgment_artifact
from Pipeline.Grammatical_Parsing.candidates import generate_grammatical_candidates


def run_node_only_extraction(
    pdf_path: Path,
    run_dir: Path,
    progress: Callable[[str, str, int], None],
    *,
    include_section_titles: set[str] | None = None,
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    progress("parsing", "Parsing embedded narrative text; tables, figures, captions, and references remain excluded.", 15)
    parsed = parse_pdf(pdf_path.read_bytes(), pdf_path.name)
    test_scope = "full_paper"
    if include_section_titles:
        wanted = {title.casefold().strip() for title in include_section_titles}
        parsed = deepcopy(parsed)
        parsed["sections"] = [
            section for section in parsed.get("sections", [])
            if str(section.get("title", "")).casefold().strip() in wanted
        ]
        paragraph_count = sum(len(section.get("paragraphs", [])) for section in parsed["sections"])
        sentence_count = sum(
            len(paragraph.get("sentences", []))
            for section in parsed["sections"] for paragraph in section.get("paragraphs", [])
        )
        parsed["summary"].update({
            "sections": len(parsed["sections"]),
            "paragraphs": paragraph_count,
            "sentences": sentence_count,
        })
        test_scope = "sections:" + ",".join(sorted(include_section_titles))
    json_write(run_dir / "parsed.json", parsed)
    shutil.copy2(pdf_path, run_dir / "source.pdf")

    progress("extracting", "Accepting trusted metadata and lexicon matches, then generating high-recall candidates.", 34)
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
        "candidate_id": row["candidate_id"], "document_id": row["document_id"],
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
        judgment_artifact(row, decision="accept", validation=row["validation"], rubric=rubric)
        for row in accepted
    ]
    llm_accepted, review, rejected, llm_judgments, llm_calls, llm_status = optional_llm_review(review, progress)
    accepted.extend(llm_accepted)

    seen = set()
    mentions = []
    for row in accepted:
        key = mention_dedup_key(row)
        if key not in seen:
            seen.add(key)
            mentions.append(row)
    progress("resolving", "Comparing vague phrases with possible nearby nodes in context; ambiguous cases go to human review.", 62)
    resolved_references, reference_review, ignored_references, reference_llm_calls, reference_llm_status = optional_llm_reference_resolution(
        parsed, mentions, progress
    )
    mentions.extend(resolved_references)
    jsonl_write(run_dir / "mentions.jsonl", mentions)
    jsonl_write(run_dir / "node_candidates.jsonl", node_candidates)
    jsonl_write(run_dir / "node_review_candidates.jsonl", review)
    jsonl_write(run_dir / "review_queue.jsonl", review)
    jsonl_write(run_dir / "node_rejections.jsonl", rejected)
    jsonl_write(run_dir / "candidate_judgments.jsonl", [*trusted_judgments, *llm_judgments])
    jsonl_write(run_dir / "llm_node_judge_calls.jsonl", llm_calls)
    jsonl_write(run_dir / "llm_reference_judge_calls.jsonl", reference_llm_calls)
    jsonl_write(run_dir / "reference_resolution_review.jsonl", reference_review)
    jsonl_write(run_dir / "reference_resolution_ignored.jsonl", ignored_references)

    progress("canonicalizing", "Grouping exact canonical IDs and reviewed aliases; no fuzzy pair is auto-merged.", 72)
    entities = canonical_entities(mentions)
    jsonl_write(run_dir / "canonical_entities.jsonl", entities)

    progress("matching", "Comparing entities after extraction and preparing the human similarity queue.", 84)
    threshold = float(json_read(CONFIG_PATH)["similarity"]["candidate_threshold"])
    pairs = similarity_candidates(entities, threshold)
    jsonl_write(run_dir / "similar_nodes_review.jsonl", pairs)
    merged = materialize_merges(run_dir)
    json_write(run_dir / "lexicon_snapshot.json", lexicon.data)

    progress("neo4j", "Writing one node per canonical ontology entity with sentence mentions embedded as properties; relationship extraction remains disabled.", 92)
    from Pipeline.Graph_Persistence.neo4j_writer import upsert_run
    graph = upsert_run(run_dir, parsed, merged, [])
    json_write(run_dir / "neo4j_upsert.json", graph)

    context_counts = [len(row["context"]) for row in flatten_sentences(parsed)]
    validation = {
        "status": "PASS", "node_only": True,
        "relationship_extraction_enabled": False, "relationship_files_created": [],
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
        "llm": llm_status,
        "test_scope": test_scope,
        "grammatical_candidate_generation": grammar["status"],
        "reference_llm": reference_llm_status,
        "neo4j": graph,
    }
    json_write(run_dir / "validation_report.json", validation)
    summary = {
        "run_id": run_dir.name, "created_at": now_iso(), "source_filename": pdf_path.name,
        "paper": parsed["document"], "parser_summary": parsed["summary"], "test_scope": test_scope,
        "counts": {
            "accepted_mentions": len(mentions), "canonical_entities": len(entities),
            "node_review_candidates": len(review), "rejected_node_candidates": len(rejected),
            "similar_node_candidates": len(pairs),
            "reference_mentions_resolved": len(resolved_references),
            "reference_review_pending": len(reference_review),
            "reference_mentions_ignored": len(ignored_references),
            "merged_entities": len(merged), "relationships": 0, "lexicon_entries": len(lexicon.entries),
        },
        "llm": llm_status, "grammatical_candidate_generation": grammar["status"],
        "reference_llm": reference_llm_status, "neo4j": graph,
        "relationship_extraction": "disabled_by_node_only_boundary",
        "outputs": [
            "parsed.json", "grammatical_analysis.jsonl", "grammatical_triples.jsonl",
            "scibert_typings.jsonl", "grammatical_stage.json", "node_candidates.jsonl",
            "node_normalization_candidates.jsonl",
            "mentions.jsonl", "candidate_judgments.jsonl",
            "node_review_candidates.jsonl", "review_queue.jsonl", "node_rejections.jsonl", "canonical_entities.jsonl",
            "similar_nodes_review.jsonl", "canonical_entities_merged.jsonl", "lexicon_snapshot.json",
            "reference_resolution_review.jsonl", "reference_resolution_ignored.jsonl",
            "llm_node_judge_calls.jsonl", "llm_reference_judge_calls.jsonl",
            "validation_report.json", "neo4j_upsert.json", "manifest.json", "source.pdf",
        ],
    }
    json_write(run_dir / "manifest.json", summary)
    progress("complete", "Node extraction is ready for evidence inspection and human similarity review.", 100)
    return summary
