from __future__ import annotations

from datetime import datetime, timezone
from collections.abc import Callable
from collections import Counter
import json
from pathlib import Path
import re
import sys
import traceback

from . import __version__
from .edge_extractor import EdgeExtractor
from .entity_resolution import EntityResolver
from .llm_node_judge import MandatoryLLMNodeJudge, PROMPT_VERSION
from .llm_relationship_judge import (
    MandatoryLLMRelationshipJudge,
    PROMPT_VERSION as RELATIONSHIP_PROMPT_VERSION,
)
from .models import sha256_bytes, sha256_file, stable_id, write_json, write_jsonl
from .node_extractor import NodeExtractor
from .ontology import Ontology
from .review_queues import build_review_queues
from .validation import validate_run


def _slug(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return value[:90] or "document"


def run_pipeline(
    papers_dir: Path,
    output_dir: Path,
    ontology_path: Path,
    lexicon_path: Path,
    config_path: Path,
    parser_project: Path,
    *,
    limit: int | None = None,
    reuse_parsed: bool = False,
    progress_callback: Callable[[str, dict], None] | None = None,
) -> dict:
    def notify(event: str, **payload: object) -> None:
        if progress_callback is not None:
            progress_callback(event, payload)

    started = datetime.now(timezone.utc)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    ontology = Ontology(ontology_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    llm_judge = MandatoryLLMNodeJudge(ontology, config, output_dir / "llm_cache")
    llm_judge.verify_runtime()
    relationship_judge = MandatoryLLMRelationshipJudge(
        ontology, config, output_dir / "relationship_llm_cache"
    )
    relationship_judge.verify_runtime()
    node_extractor = NodeExtractor(ontology, lexicon_path, config, llm_judge)
    edge_extractor = EdgeExtractor(ontology, relationship_judge, config=config)
    resolver = EntityResolver(config)
    notify(
        "run_started",
        output_dir=str(output_dir.resolve()),
        papers_dir=str(papers_dir.resolve()),
    )

    parser_project = parser_project.resolve()  # Retained in the public signature for compatibility.
    from Pipeline.Paper_Parsing.pipeline import parse_pdf

    pdfs = sorted(papers_dir.glob("*.pdf"))
    if limit is not None:
        pdfs = pdfs[:limit]
    documents: list[dict] = []
    all_mentions: list[dict] = []
    all_node_review_candidates: list[dict] = []
    all_llm_node_judge_calls: list[dict] = []
    all_relationship_candidates: list[dict] = []
    all_relationship_unknown_nodes: list[dict] = []
    all_llm_relationship_judge_calls: list[dict] = []
    all_assertions: list[dict] = []
    all_rejections: list[dict] = []
    failures: list[dict] = []
    document_manifest: list[dict] = []

    for index, pdf_path in enumerate(pdfs, 1):
        print(f"[{index}/{len(pdfs)}] {pdf_path.name}", flush=True)
        notify("paper_started", filename=pdf_path.name, index=index, total=len(pdfs))
        pdf_bytes = pdf_path.read_bytes()
        content_sha256 = sha256_bytes(pdf_bytes)
        pipeline_document_id = f"doc_{content_sha256[:16]}"
        document_dir = output_dir / "documents" / f"{index:02d}-{_slug(pdf_path.stem)}"
        parsed_path = document_dir / "parsed.json"
        try:
            if reuse_parsed and parsed_path.is_file():
                parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
            else:
                parsed = parse_pdf(pdf_bytes, pdf_path.name)
                parsed["document"]["pipeline_document_id"] = pipeline_document_id
                parsed["document"]["content_sha256"] = content_sha256
                parsed["document"]["parser_version"] = "Pipeline-Paper-Parsing-1.1"
                parsed["document"]["source_path"] = str(pdf_path.resolve())
                write_json(parsed_path, parsed)
            notify(
                "parsed_complete",
                filename=pdf_path.name,
                parsed_path=str(parsed_path.resolve()),
                summary=parsed.get("summary", {}),
            )

            mentions, node_review_candidates, node_rejections, llm_calls = node_extractor.extract(
                parsed, pipeline_document_id
            )
            notify(
                "nodes_complete",
                filename=pdf_path.name,
                mention_count=len(mentions),
                review_count=len(node_review_candidates),
                rejection_count=len(node_rejections),
                llm_call_count=len(llm_calls),
                llm_model=llm_judge.model,
                llm_discovery_count=sum(m.get("extraction_method") == "llm_discovery" for m in mentions),
                new_class_candidate_count=sum(
                    row.get("candidate_validation", {}).get("decision_type") == "new_ontology_class_candidate"
                    for row in node_review_candidates
                ),
                artifact_path=str((output_dir / "mentions.jsonl").resolve()),
            )
            assertions, edge_rejections = edge_extractor.extract(parsed, mentions)
            relationship_candidates = list(edge_extractor.candidates)
            relationship_unknown_nodes = list(edge_extractor.unknown_nodes)
            relationship_llm_calls = list(edge_extractor.llm_calls)
            notify(
                "edges_complete",
                filename=pdf_path.name,
                assertion_count=len(assertions),
                candidate_count=len(relationship_candidates),
                accepted_count=sum(row.get("status") == "accepted" for row in relationship_candidates),
                review_count=sum(row.get("status") == "review" for row in relationship_candidates),
                rejected_count=sum(row.get("status") == "rejected" for row in relationship_candidates),
                rejection_count=len(edge_rejections),
                artifact_path=str((output_dir / "assertions.jsonl").resolve()),
                candidate_artifact_path=str((output_dir / "relationship_candidates.jsonl").resolve()),
                relationships=relationship_candidates,
            )
            all_mentions.extend(mentions)
            all_node_review_candidates.extend(node_review_candidates)
            all_llm_node_judge_calls.extend(llm_calls)
            all_relationship_candidates.extend(relationship_candidates)
            all_relationship_unknown_nodes.extend(relationship_unknown_nodes)
            all_llm_relationship_judge_calls.extend(relationship_llm_calls)
            all_assertions.extend(assertions)
            all_rejections.extend(node_rejections)
            all_rejections.extend(edge_rejections)
            documents.append({
                "pipeline_document_id": pipeline_document_id,
                "source_path": str(pdf_path.resolve()),
                "parsed_path": str(parsed_path.resolve()),
                "parsed": parsed,
            })
            document_manifest.append({
                "pipeline_document_id": pipeline_document_id,
                "filename": pdf_path.name,
                "content_sha256": content_sha256,
                "parsed_json": str(parsed_path.relative_to(output_dir).as_posix()),
                "title": parsed["document"].get("title"),
                "sentences": parsed.get("summary", {}).get("sentences", 0),
                "mentions": len(mentions),
                "assertions": len(assertions),
                "node_review_candidates": len(node_review_candidates),
                "llm_node_judge_calls": len(llm_calls),
                "relationship_candidates": len(relationship_candidates),
                "relationship_unknown_nodes": len(relationship_unknown_nodes),
                "llm_relationship_judge_calls": len(relationship_llm_calls),
                "status": "processed",
            })
            notify(
                "paper_complete",
                filename=pdf_path.name,
                mention_count=len(mentions),
                assertion_count=len(assertions),
            )
        except Exception as exc:
            failure = {
                "filename": pdf_path.name,
                "content_sha256": content_sha256,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            all_rejections.append({"stage": "paper_pipeline", "reason": "processing_failure", **failure})
            document_manifest.append({
                "pipeline_document_id": pipeline_document_id,
                "filename": pdf_path.name,
                "content_sha256": content_sha256,
                "status": "failed",
                "error": str(exc),
            })
            notify(
                "paper_failed",
                filename=pdf_path.name,
                error_type=type(exc).__name__,
                error=str(exc),
            )

    resolutions, calibration = resolver.resolve(all_mentions)
    review_queues = build_review_queues(resolutions)
    notify(
        "resolution_complete",
        decision_count=len(resolutions),
        high_review_count=len(review_queues["high"]),
        uncertain_review_count=len(review_queues["uncertain_middle"]),
        keep_distinct_count=len(review_queues["low"]),
        artifact_path=str((output_dir / "resolution_decisions.jsonl").resolve()),
    )
    run_id = stable_id("run", started.isoformat(), sha256_file(ontology_path), len(pdfs))
    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "pipeline_version": __version__,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "ontology_path": str(ontology_path.resolve()),
        "ontology_sha256": sha256_file(ontology_path),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "lexicon_path": str(lexicon_path.resolve()),
        "lexicon_sha256": sha256_file(lexicon_path),
        "llm_node_judge": {
            "required": True,
            "provider": llm_judge.provider,
            "model": llm_judge.model,
            "reasoning_effort": llm_judge.reasoning_effort,
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": llm_judge.prompt_hash,
            "calls": len(all_llm_node_judge_calls),
        },
        "llm_relationship_judge": {
            "required": True,
            "provider": relationship_judge.provider,
            "model": relationship_judge.model,
            "reasoning_effort": relationship_judge.reasoning_effort,
            "prompt_version": RELATIONSHIP_PROMPT_VERSION,
            "prompt_sha256": relationship_judge.prompt_hash,
            "calls": len(all_llm_relationship_judge_calls),
            "strategy": "llm_context_window_v1",
        },
        "entity_resolution": {
            "method": "embedding_cosine",
            "model": resolver.embedding_backend.model_name,
            "high_confidence_threshold": config["entity_resolution"]["high_review_threshold"],
            "medium_confidence_threshold": config["entity_resolution"]["uncertain_review_threshold"],
            "auto_merge": False,
        },
        "parser_project": str(parser_project),
        "papers_directory": str(papers_dir.resolve()),
        "documents_requested": len(pdfs),
        "documents_processed": len(documents),
        "documents_failed": len(failures),
        "documents": document_manifest,
        "artifacts": {
            "mentions": "mentions.jsonl",
            "node_review_candidates": "node_review_candidates.jsonl",
            "llm_node_judge_calls": "llm_node_judge_calls.jsonl",
            "llm_discovered_nodes": "llm_discovered_nodes.jsonl",
            "new_ontology_class_candidates": "new_ontology_class_candidates.jsonl",
            "assertions": "assertions.jsonl",
            "relationship_candidates": "relationship_candidates.jsonl",
            "relationship_unknown_nodes": "relationship_unknown_nodes.jsonl",
            "llm_relationship_judge_calls": "llm_relationship_judge_calls.jsonl",
            "resolution_decisions": "resolution_decisions.jsonl",
            "rejections": "rejections.jsonl",
            "review_high_confidence": "review_high_confidence.jsonl",
            "review_uncertain_middle": "review_uncertain_middle.jsonl",
            "keep_distinct_low": "keep_distinct_low.jsonl",
            "validation_report": "validation_report.json",
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    write_jsonl(output_dir / "mentions.jsonl", all_mentions)
    write_jsonl(output_dir / "node_review_candidates.jsonl", all_node_review_candidates)
    write_jsonl(output_dir / "llm_node_judge_calls.jsonl", all_llm_node_judge_calls)
    write_jsonl(
        output_dir / "llm_discovered_nodes.jsonl",
        [m for m in all_mentions if m.get("extraction_method") == "llm_discovery"],
    )
    write_jsonl(
        output_dir / "new_ontology_class_candidates.jsonl",
        [row for row in all_node_review_candidates
         if row.get("candidate_validation", {}).get("decision_type") == "new_ontology_class_candidate"],
    )
    write_jsonl(output_dir / "assertions.jsonl", all_assertions)
    write_jsonl(output_dir / "relationship_candidates.jsonl", all_relationship_candidates)
    write_jsonl(output_dir / "relationship_unknown_nodes.jsonl", all_relationship_unknown_nodes)
    write_jsonl(output_dir / "llm_relationship_judge_calls.jsonl", all_llm_relationship_judge_calls)
    write_jsonl(output_dir / "resolution_decisions.jsonl", resolutions)
    write_jsonl(output_dir / "rejections.jsonl", all_rejections)
    write_jsonl(output_dir / "review_high_confidence.jsonl", review_queues["high"])
    write_jsonl(output_dir / "review_uncertain_middle.jsonl", review_queues["uncertain_middle"])
    write_jsonl(output_dir / "keep_distinct_low.jsonl", review_queues["low"])
    notify(
        "artifacts_written",
        manifest_path=str((output_dir / "manifest.json").resolve()),
        mentions_path=str((output_dir / "mentions.jsonl").resolve()),
        node_review_candidates_path=str((output_dir / "node_review_candidates.jsonl").resolve()),
        llm_calls_path=str((output_dir / "llm_node_judge_calls.jsonl").resolve()),
        llm_discoveries_path=str((output_dir / "llm_discovered_nodes.jsonl").resolve()),
        new_class_candidates_path=str((output_dir / "new_ontology_class_candidates.jsonl").resolve()),
        assertions_path=str((output_dir / "assertions.jsonl").resolve()),
        relationship_candidates_path=str((output_dir / "relationship_candidates.jsonl").resolve()),
        relationship_unknown_nodes_path=str((output_dir / "relationship_unknown_nodes.jsonl").resolve()),
        relationship_llm_calls_path=str((output_dir / "llm_relationship_judge_calls.jsonl").resolve()),
        resolutions_path=str((output_dir / "resolution_decisions.jsonl").resolve()),
        rejections_path=str((output_dir / "rejections.jsonl").resolve()),
    )
    validation = validate_run(
        output_dir,
        ontology,
        config,
        documents,
        all_mentions,
        all_assertions,
        resolutions,
    )
    candidate_rejections = [row for row in all_rejections if row.get("stage") == "llm_candidate_judgment"]
    sentence_mentions = [m for m in all_mentions if m.get("source", {}).get("kind") == "sentence_span"]
    llm_valid = bool(all_llm_node_judge_calls) and all(row.get("llm_used") for row in all_llm_node_judge_calls)
    llm_valid = llm_valid and all(
        m.get("attributes", {}).get("candidate_validation", {}).get("llm_used") is True
        for m in sentence_mentions
    )
    if not llm_valid:
        validation["errors"].append({"code": "mandatory_llm_node_judgment_missing"})
        validation["error_count"] += 1
        validation["status"] = "fail"
    validation["candidate_validation"] = {
        "validator": PROMPT_VERSION,
        "llm_used": llm_valid,
        "provider": llm_judge.provider,
        "model": llm_judge.model,
        "llm_calls": len(all_llm_node_judge_calls),
        "accepted_mentions": len(all_mentions),
        "llm_discoveries": sum(m.get("extraction_method") == "llm_discovery" for m in all_mentions),
        "review_candidates": len(all_node_review_candidates),
        "rejected_candidates": len(candidate_rejections),
        "rejection_reasons": dict(sorted(Counter(
            row.get("reason", "unknown") for row in candidate_rejections
        ).items())),
    }
    semantic_relationship_candidates = [
        row for row in all_relationship_candidates if row.get("requires_llm")
    ]
    relationship_llm_valid = all(
        row.get("llm_decision", {}).get("llm_used") is True
        for row in semantic_relationship_candidates
    )
    if semantic_relationship_candidates and not relationship_llm_valid:
        validation["errors"].append({"code": "mandatory_llm_relationship_judgment_missing"})
        validation["error_count"] += 1
        validation["status"] = "fail"
    validation["relationship_extraction"] = {
        "strategy": "llm_context_window_v1",
        "prompt_version": RELATIONSHIP_PROMPT_VERSION,
        "llm_used": relationship_llm_valid,
        "provider": relationship_judge.provider,
        "model": relationship_judge.model,
        "llm_calls": len(all_llm_relationship_judge_calls),
        "candidates": len(all_relationship_candidates),
        "accepted": sum(row.get("status") == "accepted" for row in all_relationship_candidates),
        "review": sum(row.get("status") == "review" for row in all_relationship_candidates),
        "unresolved": sum(row.get("status") == "unresolved" for row in all_relationship_candidates),
        "rejected": sum(row.get("status") == "rejected" for row in all_relationship_candidates),
        "unknown_nodes": len(all_relationship_unknown_nodes),
    }
    validation["metrics"]["relationship_candidates"] = len(all_relationship_candidates)
    validation["metrics"]["relationship_review"] = sum(
        row.get("status") == "review" for row in all_relationship_candidates
    )
    validation["entity_resolution_calibration"] = calibration
    validation["processing_failures"] = failures
    if failures:
        validation["status"] = "fail"
        validation["error_count"] += len(failures)
        validation["errors"].append({"code": "paper_processing_failures", "count": len(failures)})
    write_json(output_dir / "validation_report.json", validation)
    manifest["validation_status"] = validation["status"]
    manifest["counts"] = validation["metrics"]
    manifest["artifact_sha256"] = {
        name: sha256_file(output_dir / relative)
        for name, relative in manifest["artifacts"].items()
        if (output_dir / relative).is_file()
    }
    write_json(output_dir / "manifest.json", manifest)
    notify(
        "validation_complete",
        status=validation["status"],
        error_count=validation["error_count"],
        warning_count=validation["warning_count"],
        metrics=validation["metrics"],
        validation_path=str((output_dir / "validation_report.json").resolve()),
        manifest_path=str((output_dir / "manifest.json").resolve()),
    )
    print(
        f"Validation {validation['status'].upper()}: {validation['error_count']} errors, "
        f"{validation['warning_count']} warnings, {len(all_mentions)} mentions, "
        f"{len(all_assertions)} assertions",
        flush=True,
    )
    return validation
