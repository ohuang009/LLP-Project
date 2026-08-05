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

from Pipeline.Node_Pipeline.common import ROOT, WORKSPACE, json_read, jsonl_write
from .common import CONFIG_PATH, ONTOLOGY_PATH
from .observation_mentions import assemble_observation_mentions

def run_relationship_extraction(
    parsed: dict,
    mentions: list[dict],
    run_dir: Path,
    progress: Callable[[str, str, int], None],
) -> dict:
    from Pipeline.Knowledge_Graph_Core.edge_extractor import EdgeExtractor
    from Pipeline.Knowledge_Graph_Core.llm_relationship_judge import MandatoryLLMRelationshipJudge
    from Pipeline.Knowledge_Graph_Core.ontology import Ontology

    config = json_read(CONFIG_PATH)
    ontology = Ontology(ONTOLOGY_PATH)
    augmented_mentions = assemble_observation_mentions(parsed, mentions)
    progress(
        "relationships",
        "Extracting and judging relationships from target sentences with previous/next context.",
        66,
    )

    judge = None
    llm_status = {
        "enabled": bool(config["llm_relationship_judge"].get("enabled", True)),
        "used": False,
        "provider": config["llm_relationship_judge"].get("provider", "ollama"),
        "model": config["llm_relationship_judge"].get("model", ""),
        "reason": "",
    }
    if llm_status["enabled"]:
        try:
            judge = MandatoryLLMRelationshipJudge(
                ontology, config, ROOT / "cache" / "llm_relationship_judge"
            )
            judge.verify_runtime()
            llm_status["used"] = True
        except Exception as exc:
            llm_status["reason"] = str(exc).splitlines()[0]
            judge = None
    else:
        llm_status["reason"] = "Disabled in configuration."

    extractor = EdgeExtractor(ontology, judge, config=config)
    assertions, rejections = extractor.extract(parsed, augmented_mentions)
    candidates = extractor.candidates
    unknown_nodes = extractor.unknown_nodes
    calls = extractor.llm_calls

    if judge is None:
        llm_status["reason"] = (
            llm_status["reason"]
            or "Relationship model unavailable; no semantic fallback edges were created."
        )

    jsonl_write(run_dir / "relationship_candidates.jsonl", candidates)
    jsonl_write(run_dir / "relationship_unknown_nodes.jsonl", unknown_nodes)
    jsonl_write(run_dir / "assertions.jsonl", assertions)
    jsonl_write(run_dir / "relationship_rejections.jsonl", rejections)
    jsonl_write(run_dir / "llm_relationship_judge_calls.jsonl", calls)
    return {
        "mentions": augmented_mentions,
        "candidates": candidates,
        "unknown_nodes": unknown_nodes,
        "assertions": assertions,
        "rejections": rejections,
        "llm_calls": calls,
        "llm": llm_status,
        "counts": {
            "candidates": len(candidates),
            "accepted": sum(row.get("status") == "accepted" for row in candidates),
            "review": sum(row.get("status") == "review" for row in candidates),
            "unresolved": sum(row.get("status") == "unresolved" for row in candidates),
            "rejected": sum(row.get("status") == "rejected" for row in candidates),
            "unknown_nodes": len(unknown_nodes),
            "assertions": len(assertions),
            "observations": sum(row["label"] == "Observation" for row in augmented_mentions),
        },
    }
