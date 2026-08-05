from __future__ import annotations

import unittest

from Pipeline.Node_Pipeline import rule_spans
from Pipeline.Sector_Discovery.discovery import _model_candidates


PROFILE = {
    "construction_vocabulary": {
        "model_cues": ["model", "models", "LLM", "LLMs", "regressor"],
    }
}


class OntologyGuidedNERGeneralizationTests(unittest.TestCase):
    def test_author_name_ending_in_rag_is_not_an_ai_system(self) -> None:
        spans = rule_spans("Ferrag et al. proposed a workflow.")
        self.assertFalse(any(row.label == "AISystem" for row in spans))

    def test_true_uppercase_rag_product_is_retained(self) -> None:
        spans = rule_spans("WaterRAG supports wastewater decisions.")
        self.assertTrue(any(row.label == "AISystem" for row in spans))

    def test_common_water_and_chemical_names_are_not_ai_models(self) -> None:
        rows = [{
            "sentence_id": "s1",
            "text": "The LLM agent evaluates WDNs such as Anytown using the Extended Debye-H model and Ca-X chemistry.",
        }]
        values = {row.canonical_name for row in _model_candidates(rows, "doc", PROFILE)["s1"]}
        self.assertNotIn("WDNs", values)
        self.assertNotIn("Anytown", values)
        self.assertNotIn("Ca-X", values)

    def test_named_model_near_model_cue_is_retained(self) -> None:
        rows = [{"sentence_id": "s1", "text": "The language model DeepSeek-V3 generated code."}]
        values = {row.canonical_name for row in _model_candidates(rows, "doc", PROFILE)["s1"]}
        self.assertIn("DeepSeek-V3", values)

    def test_uppercase_scientific_models_near_model_cues_are_retained(self) -> None:
        rows = [
            {"sentence_id": "s1", "text": "We propose the SPI-GNN model for hydraulic estimation."},
            {"sentence_id": "s2", "text": "The MADDPG model controls dissolved oxygen."},
        ]
        result = _model_candidates(rows, "doc", PROFILE)
        values = {row.canonical_name for candidates in result.values() for row in candidates}
        self.assertTrue({"SPI-GNN", "MADDPG"} <= values)

    def test_water_assets_and_named_role_compounds_are_typed(self) -> None:
        spans = rule_spans(
            "HydroCoder controls a drinking water treatment plant supplied by groundwater."
        )
        observed = {(row.label, row.canonical_name or "") for row in spans}
        self.assertTrue(any(row.label == "Agent" for row in spans))
        self.assertTrue(any(row.label == "TreatmentPlant" for row in spans))
        self.assertTrue(any(row.label == "WaterSource" for row in spans))


if __name__ == "__main__":
    unittest.main()
