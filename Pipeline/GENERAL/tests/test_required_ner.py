from __future__ import annotations

import unittest

from Pipeline.Node_Pipeline import build_required_ner_artifacts


class RequiredNERTests(unittest.TestCase):
    @staticmethod
    def parsed(*, pages: int = 1, last_page: int = 1) -> dict:
        text = "SPI-GNN estimates hydraulic state in water distribution systems."
        return {
            "document": {"id": "doc"},
            "summary": {
                "pages": pages,
                "last_narrative_page": last_page,
                "narrative_page_ratio": last_page / pages,
            },
            "sections": [{
                "id": "sec", "title": "Abstract", "paragraphs": [{
                    "id": "para", "sentences": [{
                        "id": "sent", "text": text, "pages": [1], "ordinal": 0,
                    }],
                }],
            }],
        }

    @staticmethod
    def config() -> dict:
        return {"ner": {
            "enabled": True,
            "required": True,
            "minimum_narrative_page_ratio": 0.5,
            "trusted_channels": ["persistent_lexicon"],
            "channels": ["persistent_lexicon", "scibert_svo_argument"],
        }}

    def test_required_ner_materializes_exact_span_audit(self) -> None:
        text = self.parsed()["sections"][0]["paragraphs"][0]["sentences"][0]["text"]
        start = text.index("SPI-GNN")
        accepted = [{
            "mention_id": "m1", "label": "Model", "surface_text": "SPI-GNN",
            "canonical_name": "SPI-GNN", "confidence": 0.96,
            "extraction_method": "persistent_lexicon",
            "source": {
                "kind": "sentence_span", "sentence_id": "sent", "start_char": start,
                "end_char": start + len("SPI-GNN"), "pages": [1],
            },
        }]
        rows, status = build_required_ner_artifacts(self.parsed(), accepted, [], self.config())
        self.assertEqual(len(rows), 1)
        self.assertEqual(status["status"], "complete")
        self.assertTrue(status["required"])

    def test_untrusted_generator_cannot_bypass_adjudication(self) -> None:
        text = self.parsed()["sections"][0]["paragraphs"][0]["sentences"][0]["text"]
        start = text.index("SPI-GNN")
        accepted = [{
            "mention_id": "m1", "label": "Model", "surface_text": "SPI-GNN",
            "canonical_name": "SPI-GNN", "confidence": 0.96,
            "extraction_method": "scibert_svo_argument",
            "source": {
                "kind": "sentence_span", "sentence_id": "sent", "start_char": start,
                "end_char": start + len("SPI-GNN"), "pages": [1],
            },
        }]
        with self.assertRaisesRegex(RuntimeError, "Required NER validation failed"):
            build_required_ner_artifacts(self.parsed(), accepted, [], self.config())

    def test_required_ner_cannot_be_disabled(self) -> None:
        config = self.config()
        config["ner"]["enabled"] = False
        with self.assertRaisesRegex(RuntimeError, "required but disabled"):
            build_required_ner_artifacts(self.parsed(), [], [], config)

    def test_required_ner_blocks_suspicious_early_parse_cutoff(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "suspiciously early"):
            build_required_ner_artifacts(self.parsed(pages=20, last_page=4), [], [], self.config())


if __name__ == "__main__":
    unittest.main()
