from __future__ import annotations

import json
from pathlib import Path

from .similarity import cosine_vectors, idf_for, vector


class Ontology:
    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text(encoding="utf-8"))
        self.nodes: dict[str, dict] = self.data["nodes"]
        self.relationships: dict[str, dict] = self.data["relationships"]
        self.parents: dict[str, list[str]] = self.data.get("parents", {})
        self._class_labels = list(self.nodes)
        self._class_references = [
            " ".join([
                label,
                self.nodes[label]["definition"],
                " ".join(self.nodes[label].get("properties", [])),
                self.nodes[label].get("category", ""),
            ])
            for label in self._class_labels
        ]
        self._class_idf = idf_for(self._class_references)
        self._class_vectors = [vector(text, self._class_idf) for text in self._class_references]

    def ancestors(self, label: str) -> set[str]:
        result = {label}
        pending = list(self.parents.get(label, []))
        while pending:
            parent = pending.pop()
            if parent in result:
                continue
            result.add(parent)
            pending.extend(self.parents.get(parent, []))
        return result

    def compatible_class(self, actual: str, allowed: list[str]) -> bool:
        return bool(self.ancestors(actual) & set(allowed))

    def valid_relationship(self, predicate: str, source_label: str, target_label: str) -> bool:
        relationship = self.relationships.get(predicate)
        if not relationship:
            return False
        return (
            self.compatible_class(source_label, relationship["domain"])
            and self.compatible_class(target_label, relationship["range"])
        )

    def class_scores(self, text: str) -> list[dict]:
        candidate = vector(text, self._class_idf)
        scored = [
            {"label": label, "cosine": round(cosine_vectors(candidate, reference), 6)}
            for label, reference in zip(self._class_labels, self._class_vectors, strict=True)
        ]
        return sorted(scored, key=lambda item: (-item["cosine"], item["label"]))
