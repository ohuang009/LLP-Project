"""Single command-line entry point for the maintained pipeline stages."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from Pipeline.core import RUNS_DIR, copy_source, json_read, json_write, jsonl_read, jsonl_write


STAGES = ("server", "parse", "nodes", "relationships", "full", "publish",
          "graph-summary", "remove-paper", "reset-graph")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="python -m Pipeline")
    value.add_argument("--stage", choices=STAGES, default="full")
    value.add_argument("--input", type=Path)
    value.add_argument("--output", "--run-dir", dest="output", type=Path)
    value.add_argument("--model", help="Installed Ollama model; defaults to pipeline.json")
    value.add_argument("--document-id")
    value.add_argument("--run-id", default="")
    value.add_argument("--confirm")
    value.add_argument("--json", action="store_true")
    value.add_argument("--quiet", action="store_true")
    return value


def _output(stage: str, requested: Path | None) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (requested or RUNS_DIR / f"cli_{stage}_{stamp}").resolve()


def _input(args: argparse.Namespace) -> Path:
    if not args.input or not args.input.is_file() or args.input.suffix.casefold() != ".pdf":
        raise SystemExit(f"--input must name an existing PDF for stage {args.stage}")
    return args.input.resolve()


def _run_dir(args: argparse.Namespace) -> Path:
    if not args.output or not args.output.is_dir():
        raise SystemExit(f"--output must name an existing run directory for stage {args.stage}")
    return args.output.resolve()


def _progress(quiet: bool):
    def emit(stage: str, message: str, percent: int, *_: object) -> None:
        if not quiet:
            print(f"[{percent:>3}%] {stage}: {message}", flush=True)
    return emit


def run(args: argparse.Namespace) -> dict:
    if args.stage == "server":
        from Pipeline.GENERAL.run import main
        main()
        return {"stage": "server", "status": "stopped"}
    if args.stage in {"parse", "nodes", "full"}:
        pdf = _input(args)
        run_dir = _output(args.stage, args.output)
        if args.stage == "parse":
            from Pipeline.Paper_Parsing import parse_pdf
            run_dir.mkdir(parents=True, exist_ok=True)
            parsed = parse_pdf(pdf.read_bytes(), pdf.name)
            json_write(run_dir / "parsed.json", parsed)
            copy_source(pdf, run_dir / "source.pdf")
            return {"stage": "parse", "run_dir": str(run_dir), "summary": parsed.get("summary", {})}
        from Pipeline.Node_Pipeline import run_extraction, run_node_only_extraction
        function = run_extraction if args.stage == "full" else run_node_only_extraction
        return function(pdf, run_dir, _progress(args.quiet), llm_model=args.model)
    if args.stage == "relationships":
        from Pipeline.Edge_Extraction import materialize_canonical_relationships, run_relationship_extraction
        from Pipeline.Node_Pipeline import canonical_entities
        run_dir = _run_dir(args)
        parsed = json_read(run_dir / "parsed.json")
        mentions = jsonl_read(run_dir / "mentions.jsonl")
        result = run_relationship_extraction(parsed, mentions, run_dir, _progress(args.quiet), llm_model=args.model)
        entities = canonical_entities(result["mentions"])
        jsonl_write(run_dir / "canonical_entities_merged.jsonl", entities)
        relationships = materialize_canonical_relationships(run_dir, entities, result["assertions"])
        return {"stage": "relationships", "run_dir": str(run_dir),
                "candidates": len(result["candidates"]), "relationships": len(relationships)}

    from Pipeline.Graph_Persistence.neo4j_writer import graph_summary, remove_document, reset_graph, upsert_run
    if args.stage == "graph-summary":
        return graph_summary()
    if args.stage == "remove-paper":
        if not args.document_id:
            raise SystemExit("--document-id is required")
        return remove_document(args.document_id, args.run_id)
    if args.stage == "reset-graph":
        if args.confirm != "RESET":
            raise SystemExit("Refusing to reset Neo4j without --confirm RESET")
        return reset_graph()
    run_dir = _run_dir(args)
    return upsert_run(run_dir, json_read(run_dir / "parsed.json"),
                      jsonl_read(run_dir / "canonical_entities_merged.jsonl"),
                      jsonl_read(run_dir / "canonical_relationships.jsonl"))


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = run(args)
    if args.stage != "server" and not args.quiet:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0
