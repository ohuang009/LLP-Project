from __future__ import annotations

import unittest

from Pipeline.Sector_Discovery.discovery import (
    _agent_candidates,
    _discover_agent_aliases,
)


from Pipeline.Knowledge_Graph_Core.llm_relationship_judge import MandatoryLLMRelationshipJudge


class AgentEnumerationRepairTests(unittest.TestCase):
    def test_acronym_plus_ordinary_words_do_not_become_agents(self) -> None:
        rows = [
            {"sentence_id": "s1", "text": "The agents use LLM driven reasoning and LLM into API calls."},
            {"sentence_id": "s2", "text": "Agents compare WDN state and WDN optimization results."},
        ]
        self.assertEqual(_discover_agent_aliases(rows), {})
        self.assertEqual(_agent_candidates(rows, "doc"), {})

    def test_explicit_local_identifier_family_is_retained(self) -> None:
        rows = [{
            "sentence_id": "s1",
            "text": "Three expert agents - EA know, EA data, and EA hybrid - were designed.",
        }]
        candidates = _agent_candidates(rows, "doc")
        self.assertEqual(
            {row.canonical_name for row in candidates["s1"]},
            {"EA_know", "EA_data", "EA_hybrid"},
        )


class RelationshipFaultIsolationTests(unittest.TestCase):
    @staticmethod
    def _batch() -> list[dict]:
        return [
            {"target_sentence_id": "s1", "context_sentences": []},
            {"target_sentence_id": "s2", "context_sentences": []},
        ]

    def test_unknown_non_object_and_missing_windows_degrade_to_empty(self) -> None:
        judge = object.__new__(MandatoryLLMRelationshipJudge)
        parsed = {"windows": {
            "s1": "invalid",
            "unknown": {"proposals": []},
        }}
        normalized, issues = judge._normalize_result(parsed, self._batch())
        self.assertEqual(list(normalized["windows"]), ["s1", "s2"])
        self.assertTrue(all(row["proposals"] == [] for row in normalized["windows"].values()))
        self.assertEqual(
            {row["kind"] for row in issues},
            {"non_object_window", "unknown_target_sentence_id", "missing_target_sentence_id"},
        )

    def test_schema_allows_empty_endpoints_and_empty_relationship_lists(self) -> None:
        judge = object.__new__(MandatoryLLMRelationshipJudge)
        judge.ontology = type("OntologyStub", (), {
            "nodes": {"AISystem": {}, "Model": {}},
            "relationships": {"USES_MODEL": {}},
        })()
        nodes = [
            {"node_id": "system", "ontology_class": "AISystem"},
            {"node_id": "model", "ontology_class": "Model"},
        ]
        schema = judge._schema(self._batch(), nodes, ["USES_MODEL"])
        windows = schema["properties"]["windows"]
        self.assertEqual(windows["required"], ["s1", "s2"])
        proposal = windows["properties"]["s1"]["properties"]["proposals"]["items"]
        self.assertEqual(
            proposal["properties"]["head_node_id"]["enum"], ["", "system", "model"]
        )
        self.assertEqual(
            proposal["properties"]["tail_node_id"]["enum"], ["", "system", "model"]
        )
        self.assertEqual(proposal["properties"]["predicate"]["enum"], ["USES_MODEL"])
        self.assertNotIn("minItems", windows["properties"]["s1"]["properties"]["proposals"])


if __name__ == "__main__":
    unittest.main()
