from __future__ import annotations

import unittest

from Pipeline.GENERAL.run import (
    _ollama_catalog, advance_stage_reports, graph_publication,
    mark_manifest_removed_by_reset,
)


class GraphPublicationTests(unittest.TestCase):
    def test_model_catalog_enables_installed_qwen(self) -> None:
        catalog = _ollama_catalog({"models": [{
            "name": "qwen3:4b-instruct", "size": 123,
            "details": {"parameter_size": "4.0B", "quantization_level": "Q4_K_M"},
        }]}, "qwen3:4b-instruct")
        self.assertTrue(catalog["available"])
        self.assertEqual("qwen3:4b-instruct", catalog["models"][0]["name"])

    def test_model_catalog_blocks_when_configured_qwen_is_missing(self) -> None:
        catalog = _ollama_catalog({"models": [{"name": "llama3.2:latest"}]}, "qwen3:4b-instruct")
        self.assertFalse(catalog["available"])
        self.assertIn("not installed", catalog["error"])

    def test_progress_reports_wrap_stage_output_with_required_metadata(self) -> None:
        rows = advance_stage_reports([], "parsing", "Read paper", 15, {"sentences": 12})
        rows = advance_stage_reports(rows, "ner", "Raw SVO", 25, {"triples": 4})
        self.assertEqual(["parsing", "ner"], [row["stage"] for row in rows])
        self.assertEqual({"sentences": 12}, rows[0]["output"])
        self.assertEqual("complete", rows[1]["status"])

    def test_progress_reports_ignore_malformed_legacy_rows(self) -> None:
        rows = advance_stage_reports([{"sentences": 12}], "ner", "Raw SVO", 25, {"triples": 4})
        self.assertEqual("ner", rows[0]["stage"])

    def test_new_evaluated_run_has_not_been_added(self) -> None:
        state = graph_publication({"neo4j": {"status": "not_added"}})
        self.assertEqual(state["state"], "not_added")
        self.assertEqual(state["add_count"], 0)

    def test_legacy_upsert_receipt_counts_as_the_one_allowed_add(self) -> None:
        state = graph_publication({
            "created_at": "2026-08-05T12:00:00Z",
            "neo4j": {"status": "upserted"},
        })
        self.assertEqual(state["state"], "added")
        self.assertEqual(state["add_count"], 1)

    def test_removed_state_keeps_the_add_count_terminal(self) -> None:
        state = graph_publication({
            "graph_publication": {
                "state": "removed", "add_count": 1,
                "added_at": "2026-08-05T12:00:00Z", "removed_at": "2026-08-05T13:00:00Z",
            }
        })
        self.assertEqual(state["state"], "removed")
        self.assertEqual(state["add_count"], 1)

    def test_graph_reset_marks_added_run_removed_and_terminal(self) -> None:
        manifest = {"graph_publication": {"state": "added", "add_count": 1}}
        changed = mark_manifest_removed_by_reset(manifest, "2026-08-05T14:00:00Z")
        self.assertTrue(changed)
        self.assertEqual(manifest["graph_publication"]["state"], "removed")
        self.assertEqual(manifest["graph_publication"]["add_count"], 1)
        self.assertEqual(manifest["neo4j"]["reason"], "graph_reset")

    def test_graph_reset_leaves_unpublished_run_available(self) -> None:
        manifest = {"graph_publication": {"state": "not_added", "add_count": 0}}
        changed = mark_manifest_removed_by_reset(manifest, "2026-08-05T14:00:00Z")
        self.assertFalse(changed)
        self.assertEqual(manifest["graph_publication"]["state"], "not_added")


if __name__ == "__main__":
    unittest.main()
