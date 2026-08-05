from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from Pipeline.Edge_Extraction import (
    assemble_observation_mentions,
    materialize_canonical_relationships,
)


class RelationshipPipelineTests(unittest.TestCase):
    @staticmethod
    def parsed(text: str) -> dict:
        return {
            "document": {"id": "doc1"},
            "sections": [{"id": "sec1", "title": "Results", "paragraphs": [{
                "id": "para1",
                "sentences": [{"id": "sent1", "text": text, "pages": [4]}],
            }]}],
        }

    def test_quantitative_result_creates_condition_aware_observation(self):
        text = "GAC removed 88% of PFOS at pH 7.2 and temperature 25 Â°C."
        rows = assemble_observation_mentions(self.parsed(text), [])
        self.assertEqual(len(rows), 1)
        observation = rows[0]
        self.assertEqual(observation["label"], "Observation")
        self.assertEqual(observation["source"]["quote"], text)
        self.assertTrue(observation["attributes"]["measurements"])
        self.assertEqual(
            {row["name"].casefold() for row in observation["attributes"]["conditions"]},
            {"ph", "temperature"},
        )

    def test_figure_number_is_not_misread_as_hour_measurement(self):
        text = "Fig. 8 highlights two representative hallucination cases."
        self.assertEqual(assemble_observation_mentions(self.parsed(text), []), [])

    def test_existing_observation_is_not_duplicated(self):
        text = "The model achieved 92% accuracy."
        mention = {
            "mention_id": "observation", "document_id": "doc1", "label": "Observation",
            "surface_text": text, "status": "accepted",
            "source": {"sentence_id": "sent1"},
        }
        rows = assemble_observation_mentions(self.parsed(text), [mention])
        self.assertEqual(rows, [mention])

    def test_canonical_relationship_aggregates_distinct_assertion_support(self):
        entities = [
            {"entity_id": "process_gac", "mention_ids": ["m1"]},
            {"entity_id": "contaminant_pfos", "mention_ids": ["m2"]},
        ]
        assertions = [
            {"assertion_id": "a1", "subject_mention_id": "m1", "predicate": "REMOVES",
             "object_mention_id": "m2", "evidence_sentence_ids": ["s1"]},
            {"assertion_id": "a2", "subject_mention_id": "m1", "predicate": "REMOVES",
             "object_mention_id": "m2", "evidence_sentence_ids": ["s2"]},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            rows = materialize_canonical_relationships(Path(tmp), entities, assertions)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["support_count"], 2)
        self.assertEqual(rows[0]["assertion_ids"], ["a1", "a2"])
        self.assertEqual(rows[0]["evidence_sentence_ids"], ["s1", "s2"])


if __name__ == "__main__":
    unittest.main()
