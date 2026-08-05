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

# Cross-stage runtime assets live in Pipeline/GENERAL; generated audit runs
# live in the repository-level PipelineAudits folder.
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = PIPELINE_ROOT.parent
ROOT = PIPELINE_ROOT / "GENERAL"
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
RUNS_DIR = WORKSPACE / "PipelineAudits" / "runs"
LEXICON_PATH = ROOT / "lexicon" / "lexicon.json"
DECISIONS_PATH = ROOT / "review" / "entity_resolution_decisions.jsonl"
ONTOLOGY_PATH = ROOT / "ontology" / "ontology.json"
CONFIG_PATH = ROOT / "config" / "node_extraction.json"
RUBRIC_PATH = ROOT / "config" / "node_validity_rubric.json"

RELATIONSHIP_OUTPUTS: tuple[str, ...] = (
    "relationship_candidates.jsonl", "assertions.jsonl", "relationship_rejections.jsonl",
    "relationship_unknown_nodes.jsonl", "llm_relationship_judge_calls.jsonl",
    "canonical_relationships.jsonl",
)
PROVENANCE_ONLY_LABELS = {"EvidenceFragment", "Mention"}
SCOPED_LABELS = {
    "Experiment", "Observation", "Claim", "Sample", "SamplingEvent",
    "Condition", "Parameter",
}
GENERIC_TERMS = {
    "model", "method", "system", "agent", "data", "result", "study", "approach",
    "tool", "task", "framework", "process", "technology", "technique", "analysis",
    "this model", "the model", "our model", "this method", "the method", "this system",
    "the system", "this platform", "the platform", "new platform", "current system",
    "this approach", "the approach", "it", "they", "these results",
    "llm based", "llmbased", "ai based", "aibased",
}

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_id(prefix: str, *values: object) -> str:
    raw = "\u241f".join(str(v) for v in values)
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:20]}"


def normalized(text: str) -> str:
    value = text.casefold().replace("–", "-").replace("—", "-")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def json_read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def jsonl_read(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def jsonl_write(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def copy_sample(sample_path: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sample_path, target)
    return target
