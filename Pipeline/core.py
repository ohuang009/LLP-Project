"""Small shared primitives for the paper-to-graph pipeline.

This module deliberately contains only paths, stable identifiers, and JSON I/O.
Keeping these mechanics here prevents extraction stages from importing one
another just to read an artifact.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Callable, Iterable


PIPELINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_ROOT.parent
GENERAL_DIR = PIPELINE_ROOT / "GENERAL"
# ROOT is retained as the website's GENERAL asset root.
ROOT = GENERAL_DIR
CONFIG_PATH = GENERAL_DIR / "config" / "pipeline.json"
ONTOLOGY_PATH = GENERAL_DIR / "ontology" / "ontology.json"
LEXICON_PATH = GENERAL_DIR / "lexicon" / "nodes.json"
RUNS_DIR = PROJECT_ROOT / "PipelineAudits" / "runs"
SCOPED_LABELS = {
    "Experiment", "Observation", "Claim", "Sample", "SamplingEvent",
    "Condition", "Parameter",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_id(prefix: str, *values: object) -> str:
    raw = "\u241f".join(str(value) for value in values)
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:20]}"


def normalized(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def estimated_json_tokens(value: object, chars_per_token: float = 3.2) -> int:
    """Conservatively estimate model tokens without loading another tokenizer."""
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return max(1, int(len(encoded) / chars_per_token) + 1)


def dynamic_token_batches(
    items: Iterable[Any], *, config: dict,
    input_value: Callable[[Any], object],
    expected_output_tokens: Callable[[Any], int],
    fixed_input_tokens: int = 0,
) -> list[list[Any]]:
    """Greedily pack ordered items within both input and output token budgets.

    Paragraph order is preserved. Output budgeting matters because structured
    JSON can overflow well before the model's input context is full.
    """
    ollama = config.get("ollama", {})
    batching = config.get("batching", {})
    chars_per_token = float(batching.get("chars_per_token", 3.2))
    context_window = int(ollama.get("context_window", 32768))
    max_tokens = int(ollama.get("max_tokens", 4096))
    safety = int(batching.get("context_safety_tokens", 2048))
    input_budget = int(batching.get(
        "input_token_budget", max(1024, context_window - max_tokens - safety)
    ))
    output_budget = int(batching.get("output_token_budget", max(512, int(max_tokens * 0.75))))

    groups: list[list[Any]] = []
    current: list[Any] = []
    current_input = fixed_input_tokens
    current_output = 0
    for item in items:
        item_input = estimated_json_tokens(input_value(item), chars_per_token)
        item_output = max(1, int(expected_output_tokens(item)))
        would_overflow = current and (
            current_input + item_input > input_budget
            or current_output + item_output > output_budget
        )
        if would_overflow:
            groups.append(current)
            current = []
            current_input = fixed_input_tokens
            current_output = 0
        current.append(item)
        current_input += item_input
        current_output += item_output
    if current:
        groups.append(current)
    return groups


def parse_tsv_rows(text: str, columns: int) -> list[list[str]]:
    """Parse an exact-width TSV response, ignoring prose, headers, and fences."""
    rows: list[list[str]] = []
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("```"):
            continue
        if "\t" not in line and "\\t" in line:
            line = line.replace("\\t", "\t")
        values = [value.strip() for value in line.split("\t")]
        if len(values) == columns:
            rows.append(values)
    return rows


def json_read(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def jsonl_read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def jsonl_write(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def copy_source(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
