from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from Pipeline.lexicon import load_lexicon, previous_types, update_lexicon


class PersistentLexiconTests(unittest.TestCase):
    def test_registry_contains_only_unique_node_name_and_type_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nodes.json"
            first = update_lexicon([
                {"canonical_name": "Qwen 3", "label": "Model", "aliases": ["Qwen"]},
                {"canonical_name": "EPANET", "label": "Tool"},
            ], path)
            second = update_lexicon([
                {"canonical_name": " qwen 3 ", "label": "Model"},
                {"canonical_name": "EPANET", "label": "SoftwareArtifact"},
            ], path)

            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual({"name", "type"}, set(raw[0]))
            self.assertTrue(all(set(row) == {"name", "type"} for row in raw))
            self.assertEqual({"added": 2, "total": 2}, first)
            self.assertEqual({"added": 1, "total": 3}, second)
            self.assertEqual(raw, load_lexicon(path))
            self.assertEqual(["Model"], previous_types(raw, "QWEN 3"))
            self.assertEqual(["SoftwareArtifact", "Tool"], previous_types(raw, "epanet"))

    def test_registry_rejects_extra_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nodes.json"
            path.write_text('[{"name":"EPANET","type":"Tool","alias":"EPANET 2"}]', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "only name and type"):
                load_lexicon(path)


if __name__ == "__main__":
    unittest.main()
