"""Persistent, version-controlled memory of previously accepted nodes and types."""
from __future__ import annotations

import json
from pathlib import Path
import threading

from Pipeline.core import LEXICON_PATH, ONTOLOGY_PATH, json_read, normalized


_LOCK = threading.Lock()
_FIELDS = {"name", "type"}


def load_lexicon(path: Path | None = None) -> list[dict[str, str]]:
    """Load and validate the minimal ``[{name, type}]`` registry."""
    target = path or LEXICON_PATH
    if not target.exists():
        return []
    value = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Lexicon must be a JSON array: {target}")
    allowed_types = set(json_read(ONTOLOGY_PATH).get("nodes", {}))
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != _FIELDS:
            raise ValueError(f"Lexicon row {index} must contain only name and type")
        name = str(item["name"]).strip()
        node_type = str(item["type"]).strip()
        if not name or node_type not in allowed_types:
            raise ValueError(f"Lexicon row {index} has an empty name or unknown ontology type")
        key = (normalized(name), node_type)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"name": name, "type": node_type})
    return sorted(rows, key=lambda row: (normalized(row["name"]), row["type"]))


def previous_types(lexicon: list[dict[str, str]], *names: object) -> list[str]:
    """Return types previously stored for an exact normalized node name."""
    wanted = {normalized(name) for name in names if normalized(name)}
    return sorted({row["type"] for row in lexicon if normalized(row["name"]) in wanted})


def update_lexicon(entities: list[dict], path: Path | None = None) -> dict[str, int]:
    """Merge accepted canonical node/type pairs and atomically persist them."""
    target = path or LEXICON_PATH
    with _LOCK:
        existing = load_lexicon(target)
        by_key = {(normalized(row["name"]), row["type"]): row for row in existing}
        before = len(by_key)
        allowed_types = set(json_read(ONTOLOGY_PATH).get("nodes", {}))
        for entity in entities:
            name = str(entity.get("canonical_name", "")).strip()
            node_type = str(entity.get("label", "")).strip()
            if not name or node_type not in allowed_types:
                raise ValueError("Accepted lexicon entities require a canonical_name and ontology label")
            key = (normalized(name), node_type)
            by_key.setdefault(key, {"name": name, "type": node_type})

        rows = sorted(by_key.values(), key=lambda row: (normalized(row["name"]), row["type"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        try:
            temporary.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return {"added": len(rows) - before, "total": len(rows)}
