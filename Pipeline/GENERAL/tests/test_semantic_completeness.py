import unittest

from Pipeline.Validation.semantic_completeness import evaluate_semantic_completeness


PARSED = {
    "sections": [{"id": "sec1", "paragraphs": [{"id": "p1", "sentences": [
        {"id": "s1", "text": "We introduce a portable artificial intelligence platform using Mistral-7B model.", "pages": [1]},
        {"id": "s2", "text": "Three expert agents - EA know, EA data, and EA hybrid - were designed.", "pages": [1]},
    ]}]}],
}


class SemanticCompletenessTests(unittest.TestCase):
    def test_missing_expected_core_nodes_is_reported_as_incomplete(self):
        result = evaluate_semantic_completeness(PARSED, [], [], [], {
            "answerable_questions": 0, "total_questions": 10,
        })
        self.assertEqual(result["status"], "INCOMPLETE")
        failed = {row["id"] for row in result["checks"] if row["applicable"] and not row["passed"]}
        self.assertIn("introduced_ai_system", failed)
        self.assertIn("enumerated_agents", failed)
        self.assertIn("named_models", failed)

    def test_grounded_connected_core_with_traceability_is_complete(self):
        entities = [
            {"entity_id": "sys", "label": "AISystem"},
            {"entity_id": "agent", "label": "Agent"},
            {"entity_id": "agent2", "label": "Agent"},
            {"entity_id": "agent3", "label": "Agent"},            {"entity_id": "model", "label": "Model"},
        ]
        relationships = [
            {"subject_entity_id": "sys", "predicate": "HAS_AGENT", "object_entity_id": "agent"},
            {"subject_entity_id": "agent", "predicate": "USES_MODEL", "object_entity_id": "model"},
        ]
        assertions = [{
            "evidence_quote": PARSED["sections"][0]["paragraphs"][0]["sentences"][0]["text"],
            "evidence_sentence_ids": ["s1"],
        }]
        result = evaluate_semantic_completeness(PARSED, entities, relationships, assertions, {
            "answerable_questions": 8, "total_questions": 10,
        })
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["coverage_score"], 1.0)
        self.assertEqual(result["question_coverage"], {"answerable": 8, "total": 10})


if __name__ == "__main__":
    unittest.main()
