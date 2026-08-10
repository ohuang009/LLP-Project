"""Deterministic ontology loading and relationship compatibility checks."""
from __future__ import annotations

import json
from pathlib import Path


class Ontology:
    def __init__(self, path: Path) -> None:
        self.data = json.loads(path.read_text(encoding="utf-8"))
        self.nodes: dict[str, dict] = self.data["nodes"]
        self.relationships: dict[str, dict] = self.data["relationships"]
        self.parents: dict[str, list[str]] = self.data.get("parents", {})

    def ancestors(self, label: str) -> set[str]:
        result = {label}
        pending = list(self.parents.get(label, []))
        while pending:
            parent = pending.pop()
            if parent not in result:
                result.add(parent)
                pending.extend(self.parents.get(parent, []))
        return result

    def compatible_class(self, actual: str, allowed: list[str]) -> bool:
        return bool(self.ancestors(actual) & set(allowed))

    def valid_relationship(self, predicate: str, source_label: str, target_label: str) -> bool:
        spec = self.relationships.get(predicate)
        return bool(spec and self.compatible_class(source_label, spec["domain"])
                    and self.compatible_class(target_label, spec["range"]))
