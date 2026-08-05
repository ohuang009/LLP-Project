from collections import Counter
import unittest

from Pipeline.Sector_Discovery.discovery import discover_document_candidates


PROFILE = {"construction_vocabulary": {
    "system_heads": ["system", "framework", "platform", "assistant", "agent", "pipeline"],
    "agent_heads": ["agent", "expert agent"],
    "model_cues": ["model", "models", "LLM", "LLMs"],
    "task_patterns": ["process\\s+optimization", "closed-loop\\s+control"],
    "dataset_patterns": [
        "\\d+\\s+(?:published\\s+|scientific\\s+)?(?:papers|studies)",
        "\\d+\\s+(?:structured\\s+)?data\\s+points",
        "(?:standardized\\s+)?[A-Za-z0-9-]+\\s+question\\s+dataset",
    ],
    "model_patterns": ["gradient\\s+boosting\\s+(?:regression|regressor)"],
    "tool_patterns": ["Raspberry\\s+Pi(?:\\s+\\d+)?", "Ollama"],
    "process_patterns": ["membrane\\s+capacitive\\s+deionization", "MCDI"],
    "metric_patterns": ["(?:coefficient\\s+of\\s+determination|R2)"],
    "identity_stopwords": [
        "a", "an", "the", "proposed", "new", "portable", "artificial", "intelligence",
        "ai", "expert", "agent", "system", "framework", "platform", "pipeline",
    ],
}}


def source_rows(*sentences):
    return [{"sentence_id": f"s{i}", "text": text, "section_title": "Methods"}
            for i, text in enumerate(sentences)]


class SectorDiscoveryTests(unittest.TestCase):
    def flatten(self, *sentences):
        result = discover_document_candidates(source_rows(*sentences), "doc1", PROFILE)
        return [candidate for candidates in result.values() for candidate in candidates]

    def test_descriptive_systems_cluster_and_spaced_agents_propagate(self):
        candidates = self.flatten(
            "Here, we introduce an intelligent on-device platform for desalination.",
            "This study develops and validates a portable, on-device artificial intelligence platform.",
            "Three expert agents - EA know, EA data, and EA hybrid - were designed.",
            "EA hybrid uses a compact model for process optimization.",
        )
        systems = [row for row in candidates if row.label == "AISystem"]
        self.assertEqual(len(systems), 2)
        self.assertEqual(len({row.canonical_id for row in systems}), 1)
        self.assertEqual({row.canonical_name for row in candidates if row.label == "Agent"},
                         {"EA_know", "EA_data", "EA_hybrid"})

    def test_false_models_are_rejected_but_enumerated_models_are_kept(self):
        candidates = self.flatten(
            "The LLM models include Mistral, TinyLlama, Gemma3:1B, and SmolLM on Raspberry Pi.",
            "The model was saved in GGUF format after the October-2025 release (Table S1).",        )
        models = {row.canonical_name for row in candidates if row.label == "Model"}
        self.assertTrue({"Mistral", "TinyLlama", "Gemma3:1B", "SmolLM"} <= models)
        self.assertTrue({"GGUF", "October-2025", "Raspberry", "Pi", "S1"}.isdisjoint(models))
    def test_profile_types_and_counted_datasets_are_canonicalized(self):
        candidates = self.flatten(
            "The corpus includes 320 published studies and 320 scientific papers.",
            "The model was fine-tuned on 6000 data points and 6000 structured data points.",
            "MCDI used Raspberry Pi 5 and achieved R2 during process optimization.",
        )
        by_label = {
            label: Counter(row.canonical_name for row in candidates if row.label == label)
            for label in {row.label for row in candidates}
        }
        self.assertEqual(set(by_label["Dataset"]), {"320-paper corpus", "6000 structured data points"})
        self.assertIn("membrane capacitive deionization", by_label["TreatmentProcess"])
        self.assertIn("Raspberry Pi 5", by_label["Tool"])
        self.assertIn("coefficient of determination (R2)", by_label["Metric"])
        self.assertIn("process optimization", by_label["Task"])


if __name__ == "__main__":
    unittest.main()
