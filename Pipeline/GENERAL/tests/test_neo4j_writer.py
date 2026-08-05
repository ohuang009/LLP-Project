from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from Pipeline.Graph_Persistence import neo4j_writer


class Neo4jWriterTests(unittest.TestCase):
    def test_reusable_identity_aligns_to_another_paper_and_rewires_relationships(self) -> None:
        response = {"results": [{"data": [
            {"row": ["Task", "decision support", "current_copy", ["doc_current"]]},
            {"row": ["Task", "decision support", "shared_copy", ["doc_other"]]},
        ]}]}
        entities = [
            {
                "entity_id": "local_task", "label": "Task",
                "canonical_name": "Decision Support", "identity_scope": "canonical",
            },
            {
                "entity_id": "local_observation", "label": "Observation",
                "canonical_name": "Accuracy was 90%", "identity_scope": "occurrence",
            },
        ]
        relationships = [{
            "subject_entity_id": "local_observation", "predicate": "ABOUT",
            "object_entity_id": "local_task",
        }]
        with patch.object(neo4j_writer, "_post", return_value=response):
            aligned, rewired, remap = neo4j_writer.align_reusable_entity_ids(
                entities, relationships, "doc_current",
            )

        self.assertEqual(remap, {"local_task": "shared_copy"})
        self.assertEqual(aligned[0]["entity_id"], "shared_copy")
        self.assertEqual(aligned[1]["entity_id"], "local_observation")
        self.assertEqual(rewired[0]["subject_entity_id"], "local_observation")
        self.assertEqual(rewired[0]["object_entity_id"], "shared_copy")


    def test_mentions_are_embedded_on_one_canonical_ontology_node(self) -> None:
        entities = [{
            "entity_id": "model_gpt41", "label": "Model", "canonical_name": "GPT-4.1",
            "aliases": ["GPT-4.1", "GPT 4.1"], "mention_ids": ["m1", "m2"],
        }]
        mentions = [
            {
                "mention_id": "m1", "document_id": "doc1", "label": "Model",
                "surface_text": "GPT-4.1", "extraction_method": "lexicon",
                "source": {"kind": "sentence_span", "sentence_id": "s1", "evidence_quote": "GPT-4.1 answered the question.", "pages": [1]},
            },
            {
                "mention_id": "m2", "document_id": "doc1", "label": "Model",
                "surface_text": "it", "extraction_method": "reference_resolution",
                "source": {"kind": "sentence_span", "sentence_id": "s2", "evidence_quote": "It produced a concise answer.", "pages": [1]},
                "reference_resolution": {
                    "status": "auto_resolved", "antecedent_sentence_id": "s1",
                    "antecedent_evidence_quote": "GPT-4.1 answered the question.",
                },
            },
        ]
        rows = neo4j_writer.build_entity_rows("run1", {"id": "doc1"}, entities, mentions)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["mention_ids"], ["m1", "m2"])
        self.assertEqual(rows[0]["mention_source_locators"], ["s1", "s2"])
        self.assertEqual(rows[0]["mention_sentence_ids"], ["s1", "s2"])
        self.assertEqual(json.loads(rows[0]["mentions"][1])["sentence"], "It produced a concise answer.")
        self.assertEqual(rows[0]["antecedent_sentence_ids"], ["", "s1"])

    def test_document_metadata_mentions_use_exact_field_provenance(self) -> None:
        entities = [{
            "entity_id": "person_1", "label": "Person", "canonical_name": "Ada Example",
            "aliases": ["Ada Example"], "mention_ids": ["m1"],
        }]
        mentions = [{
            "mention_id": "m1", "document_id": "doc1", "label": "Person",
            "surface_text": "Ada Example", "extraction_method": "structural_metadata",
            "source": {
                "kind": "document_metadata", "field": "document.authors_text",
                "evidence_quote": "Ada Example, Ben Example",
            },
        }]
        rows = neo4j_writer.build_entity_rows("run1", {"id": "doc1"}, entities, mentions)
        record = json.loads(rows[0]["mentions"][0])
        self.assertEqual(record["source_kind"], "document_metadata")
        self.assertEqual(record["source_locator"], "document.authors_text")
        self.assertTrue(record["sentence_id"].startswith("metadata-sentence_"))
        self.assertEqual(record["evidence_quote"], "Ada Example, Ben Example")

    def test_duplicate_canonical_nodes_are_rejected(self) -> None:
        rows = [
            {"entity_id": "a", "label": "Model", "canonical_name": "GPT-4.1"},
            {"entity_id": "b", "label": "Model", "canonical_name": "gpt 4.1"},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate canonical node"):
            neo4j_writer.validate_entities_against_ontology(rows)

    def test_scoped_observations_may_have_the_same_display_value(self) -> None:
        rows = [
            {"entity_id": "o1", "label": "Observation", "canonical_name": "Accuracy was 80%."},
            {"entity_id": "o2", "label": "Observation", "canonical_name": "Accuracy was 80%."},
        ]
        result = neo4j_writer.validate_entities_against_ontology(rows)
        self.assertEqual(result["validated"], 2)

    def test_provenance_classes_cannot_become_instance_nodes(self) -> None:
        with self.assertRaisesRegex(ValueError, "embedded provenance"):
            neo4j_writer.validate_entities_against_ontology([{
                "entity_id": "mention_1", "label": "Mention", "canonical_name": "this model",
            }])

    def test_concrete_ontology_labels_are_applied_to_canonical_entities(self) -> None:
        calls: list[list[dict]] = []
        rows = [
            {"id": "entity_model", "label": "Model"},
            {"id": "entity_process", "label": "TreatmentProcess"},
        ]
        with patch.object(neo4j_writer, "_post", side_effect=lambda statements: calls.append(statements) or {}):
            result = neo4j_writer.apply_ontology_labels(rows)

        statements = [item["statement"] for call in calls for item in call]
        self.assertTrue(any("SET e:Model" in statement for statement in statements))
        self.assertTrue(any("SET e:TreatmentProcess" in statement for statement in statements))
        self.assertEqual(result["typed_entities"], 2)
        self.assertEqual(result["ontology_labels"], ["Model", "TreatmentProcess"])

    def test_unknown_entity_class_is_rejected_before_neo4j_write(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown or unsafe ontology class"):
            neo4j_writer.apply_ontology_labels([{"id": "x", "label": "MadeUpClass"}])

    def test_relationship_domain_and_range_are_enforced(self) -> None:
        entities = [
            {"entity_id": "paper", "label": "Publication"},
            {"entity_id": "author", "label": "Person"},
        ]
        result = neo4j_writer.validate_relationships_against_ontology(entities, [{
            "predicate": "AUTHORED_BY",
            "subject_entity_id": "paper",
            "object_entity_id": "author",
        }])
        self.assertEqual(result, {"validated": 1, "violations": 0})

        with self.assertRaisesRegex(ValueError, "subject class Person is outside domain"):
            neo4j_writer.validate_relationships_against_ontology(entities, [{
                "predicate": "AUTHORED_BY",
                "subject_entity_id": "author",
                "object_entity_id": "paper",
            }])

    def test_undefined_relationship_is_rejected(self) -> None:
        entities = [
            {"entity_id": "a", "label": "Model"},
            {"entity_id": "b", "label": "Metric"},
        ]
        with self.assertRaisesRegex(ValueError, "not defined by ontology"):
            neo4j_writer.validate_relationships_against_ontology(entities, [{
                "predicate": "VAGUELY_RELATED_TO",
                "subject_entity_id": "a",
                "object_entity_id": "b",
            }])


if __name__ == "__main__":
    unittest.main()
