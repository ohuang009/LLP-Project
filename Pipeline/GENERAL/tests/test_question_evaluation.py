from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from Pipeline.Question_Evaluation.evaluator import evaluate_question_answerability


class QuestionEvaluationTests(unittest.TestCase):
    def test_question_requires_graph_fact_and_exact_evidence(self):
        entities = [
            {"entity_id": "pub", "canonical_name": "Paper"},
            {"entity_id": "sys", "canonical_name": "EPANET-Agentic"},
        ]
        relationships = [{
            "subject_entity_id": "pub", "predicate": "PROPOSES",
            "object_entity_id": "sys", "assertion_ids": ["a1"],
        }]
        assertions = [{
            "assertion_id": "a1", "evidence_quote": "We propose EPANET-Agentic.",
            "pages": [1], "evidence_sentence_ids": ["s1"],
        }]
        with tempfile.TemporaryDirectory() as tmp:
            result = evaluate_question_answerability(Path(tmp), entities, relationships, assertions)
        system = next(row for row in result["questions"] if row["id"] == "system")
        self.assertEqual(system["status"], "answerable")
        self.assertEqual(system["answers"], ["EPANET-Agentic"])
        self.assertEqual(system["evidence"][0]["quote"], "We propose EPANET-Agentic.")
        agents = next(row for row in result["questions"] if row["id"] == "agents")
        self.assertEqual(agents["status"], "not_answerable")


if __name__ == "__main__":
    unittest.main()
