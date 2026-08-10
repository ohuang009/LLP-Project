"""Orchestrate raw SVO extraction and the three node decision passes."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from Pipeline.core import (
    CONFIG_PATH, ONTOLOGY_PATH, dynamic_token_batches, estimated_json_tokens,
    json_read, jsonl_write,
)
from Pipeline.Grammatical_Parsing.candidates import extract_raw_svo, type_with_scibert
from Pipeline.lexicon import load_lexicon, previous_types
from Pipeline.llm import OllamaClient

from .batching import build_paragraph_batches, candidate_rows
from .entities import canonical_entities, mention_from_candidate
from .passes import (
    NODE_RUBRIC, cleanup_candidates, cleanup_payload, judge_candidates,
    judgment_payload, type_candidates, typing_payload,
)


Progress = Callable[..., None]


def _emit(progress: Progress | None, stage: str, message: str, percent: int, output: dict) -> None:
    if progress:
        progress(stage, message, percent, "complete", output)


def extract_nodes(
    parsed: dict, run_dir: Path, *, client: OllamaClient | None = None,
    model: str | None = None, progress: Progress | None = None, parser=None, typer=None,
    prior_lexicon: list[dict[str, str]] | None = None,
) -> dict:
    """Run raw SVO, Qwen cleanup, SciBERT, Qwen typing, and judgment."""
    config = json_read(CONFIG_PATH)
    ontology = json_read(ONTOLOGY_PATH)
    client = client or OllamaClient.from_config(config, model=model)
    raw = extract_raw_svo(parsed, parser=parser)
    batches = build_paragraph_batches(parsed, raw)
    jsonl_write(run_dir / "raw_svo.jsonl", raw["sentences"])
    jsonl_write(run_dir / "grammatical_triples.jsonl", raw["triples"])
    jsonl_write(run_dir / "paragraph_batches.jsonl", batches)
    _emit(progress, "ner", "Raw main-clause SVO and context fragments extracted.", 25, {
        "sentences": len(raw["sentences"]), "triples": len(raw["triples"]),
    })

    active_batches = [batch for batch in batches if candidate_rows(batch)]
    cleaned_by_paragraph: dict[str, list[dict]] = {batch["paragraph_id"]: [] for batch in batches}
    all_cleaned: list[dict] = []
    cleanup_groups = dynamic_token_batches(
        active_batches, config=config, input_value=cleanup_payload,
        expected_output_tokens=lambda batch: 14 * len(candidate_rows(batch)),
        fixed_input_tokens=350,
    )
    for group in cleanup_groups:
        rows = cleanup_candidates(client, group)
        all_cleaned.extend(rows)
        for row in rows:
            cleaned_by_paragraph[row["paragraph_id"]].append(row)

    scibert_input = [{**row, "node_name": row["specific_name"]} for row in all_cleaned]
    typed = type_with_scibert(scibert_input, ONTOLOGY_PATH, config, typer=typer)
    prior_lexicon = load_lexicon() if prior_lexicon is None else prior_lexicon
    typed = [{
        **row,
        "previous_types": previous_types(
            prior_lexicon, row.get("specific_name"), row.get("surface_text"),
        ),
    } for row in typed]
    typed_index = {row["candidate_id"]: row for row in typed}
    jsonl_write(run_dir / "scibert_typings.jsonl", typed)
    _emit(progress, "adjudicating", "SciBERT supplied advisory ontology types, including NONE.", 40, {
        "candidates": len(typed),
        "untyped": sum(row["scibert_type"] == "NONE" for row in typed),
    })

    typing_units = [
        (batch, [typed_index[row["candidate_id"]] for row in cleaned_by_paragraph[batch["paragraph_id"]]])
        for batch in active_batches
    ]
    typing_groups = dynamic_token_batches(
        typing_units, config=config, input_value=typing_payload,
        expected_output_tokens=lambda unit: 8 * len(unit[1]),
        fixed_input_tokens=500 + estimated_json_tokens(ontology.get("nodes", [])),
    )
    ontology_typed: list[dict] = []
    for group in typing_groups:
        ontology_typed.extend(type_candidates(client, group, ontology))
    typed_by_paragraph: dict[str, list[dict]] = {batch["paragraph_id"]: [] for batch in batches}
    for row in ontology_typed:
        typed_by_paragraph[row["paragraph_id"]].append(row)

    judgment_units = [
        (batch, typed_by_paragraph[batch["paragraph_id"]])
        for batch in active_batches
    ]
    judgment_groups = dynamic_token_batches(
        judgment_units, config=config, input_value=judgment_payload,
        expected_output_tokens=lambda unit: 6 * len(unit[1]),
        fixed_input_tokens=350 + estimated_json_tokens(NODE_RUBRIC),
    )
    judgments: list[dict] = []
    for group in judgment_groups:
        judgments.extend(judge_candidates(client, group))

    batch_index = {batch["paragraph_id"]: batch for batch in batches}
    mentions = [
        mention_from_candidate(row, batch_index[row["paragraph_id"]])
        for row in judgments if row["decision"] == "accepted"
    ]
    review = [{
        **row, "status": "needs_review", "origin": "svo_node",
        "canonical_name": row["specific_name"],
        "label": "" if row["llm_type"] == "NONE" else row["llm_type"],
        "source": mention_from_candidate(
            {**row, "llm_type": row["llm_type"] if row["llm_type"] != "NONE" else "NONE"},
            batch_index[row["paragraph_id"]],
        )["source"],
    } for row in judgments if row["decision"] == "needs_review"]
    rejected = [{**row, "status": "disregarded"} for row in judgments if row["decision"] == "disregard"]
    entities = canonical_entities(mentions)
    jsonl_write(run_dir / "node_candidates.jsonl", judgments)
    jsonl_write(run_dir / "candidate_judgments.jsonl", judgments)
    jsonl_write(run_dir / "mentions.jsonl", mentions)
    jsonl_write(run_dir / "review_queue.jsonl", review)
    jsonl_write(run_dir / "node_review_candidates.jsonl", review)
    jsonl_write(run_dir / "node_rejections.jsonl", rejected)
    jsonl_write(run_dir / "canonical_entities.jsonl", entities)
    jsonl_write(run_dir / "canonical_entities_merged.jsonl", entities)
    jsonl_write(run_dir / "ollama_node_calls.jsonl", client.audit)
    _emit(progress, "resolving", "Three Qwen/Ollama node passes complete.", 55, {
        "accepted_mentions": len(mentions), "needs_review": len(review),
        "disregarded": len(rejected),
    })
    return {
        "mentions": mentions, "entities": entities, "review": review,
        "rejected": rejected, "judgments": judgments, "llm_calls": client.audit,
    }
