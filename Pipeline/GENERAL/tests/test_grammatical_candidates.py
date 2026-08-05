from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from Pipeline.Grammatical_Parsing.candidates import (
    SciBERTOntologyTyper,
    SpacyPredicateTripleExtractor,
    generate_grammatical_candidates,
)


class FakeTyper:
    def type_arguments(self, arguments: list[dict]) -> list[dict]:
        labels = {
            "The membrane bioreactor": "TreatmentUnit",
            "dissolved contaminants": "Contaminant",
            "wastewater": "WaterMatrix",
            "WaterRAG": "AISystem",
            "evidence": "EvidenceFragment",
            "reports": "Publication",
        }
        return [{
            **row,
            "selected_label": labels.get(row["surface_text"], "Method"),
            "selected_score": .8,
            "applicable": True,
            "alternatives": [{"label": labels.get(row["surface_text"], "Method"), "score": .8}],
            "typing_method": "fake_scibert",
            "model": "fake-scibert",
            "threshold": .25,
        } for row in arguments]


class FakeEncoder:
    def encode(self, texts, **_kwargs):
        vectors = []
        for text in texts:
            folded = text.casefold()
            if "ontology type: model" in folded:
                vectors.append([1.0, 0.0])
            elif "ontology type: dataset" in folded:
                vectors.append([0.0, 1.0])
            elif "dataset" in folded:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([1.0, 0.0])
        return np.asarray(vectors, dtype=float)


class AmbiguousFakeTyper:
    def type_arguments(self, arguments: list[dict]) -> list[dict]:
        return [{
            **row,
            "selected_label": "",
            "selected_score": .63,
            "score_margin": .001,
            "applicable": False,
            "alternatives": [
                {"label": "TreatmentUnit", "score": .63},
                {"label": "WaterMatrix", "score": .629},
            ],
            "typing_method": "fake_scibert",
            "model": "fake-scibert",
            "threshold": .25,
            "min_margin": .015,
        } for row in arguments]


class GrammaticalCandidateTests(unittest.TestCase):
    @staticmethod
    def parsed(text: str) -> dict:
        return {
            "document": {"id": "doc", "title": "Test paper"},
            "sections": [{"id": "sec", "title": "Methods", "paragraphs": [{
                "id": "p", "sentences": [{
                    "id": "s1", "ordinal": 0, "text": text, "pages": [1],
                }],
            }]}],
        }

    @staticmethod
    def parser():
        try:
            return SpacyPredicateTripleExtractor.load("en_core_web_sm")
        except Exception as exc:
            raise unittest.SkipTest(f"spaCy English parser is not installed: {exc}")

    def test_complete_pos_parse_and_predicate_first_triples(self) -> None:
        text = "The membrane bioreactor removes dissolved contaminants from wastewater."
        result = generate_grammatical_candidates(
            self.parsed(text),
            Path("Pipeline/GENERAL/ontology/ontology.json"),
            {"grammatical_candidate_generation": {
                "enabled": True, "required": True, "parser_model": "en_core_web_sm",
                "scibert_model": "fake-scibert", "typing_threshold": .25, "typing_top_k": 3,
            }},
            parser=self.parser(), typer=FakeTyper(),
        )
        analysis = result["sentence_analysis"][0]
        token_by_text = {row["text"]: row for row in analysis["tokens"]}
        self.assertEqual(token_by_text["The"]["coarse_pos"], "DET")
        self.assertEqual(token_by_text["removes"]["coarse_pos"], "VERB")
        self.assertEqual(token_by_text["bioreactor"]["dependency"], "nsubj")

        triples = result["triples"]
        values = {
            (row["subject"]["text"], row["predicate"]["text"], row["object"]["text"])
            for row in triples
        }
        self.assertIn(("The membrane bioreactor", "removes", "dissolved contaminants"), values)
        self.assertIn(("The membrane bioreactor", "removes from", "wastewater"), values)

        candidates = result["candidates_by_sentence"]["s1"]
        surfaces = {row["canonical_name"] for row in candidates}
        self.assertEqual(
            surfaces,
            {"membrane bioreactor", "dissolved contaminants", "wastewater"},
        )
        subject = next(row for row in candidates if row["canonical_name"] == "membrane bioreactor")
        self.assertEqual(subject["metadata"]["deterministic_normalization"]["original_text"], "The membrane bioreactor")
        self.assertEqual(
            subject["metadata"]["deterministic_normalization"]["transformations"][0]["operation"],
            "remove_leading_article",
        )
        self.assertTrue(all(row["method"] == "scibert_svo_argument" for row in candidates))

    def test_conjoined_predicates_inherit_the_subject(self) -> None:
        result = generate_grammatical_candidates(
            self.parsed("WaterRAG retrieves evidence and generates reports."),
            Path("Pipeline/GENERAL/ontology/ontology.json"),
            {"grammatical_candidate_generation": {
                "enabled": True, "required": True, "parser_model": "en_core_web_sm",
                "scibert_model": "fake-scibert", "typing_threshold": .25, "typing_top_k": 3,
            }},
            parser=self.parser(), typer=FakeTyper(),
        )
        values = {
            (row["subject"]["text"], row["predicate"]["text"], row["object"]["text"])
            for row in result["triples"]
        }
        self.assertIn(("WaterRAG", "retrieves", "evidence"), values)
        self.assertIn(("WaterRAG", "generates", "reports"), values)

    def test_pretrained_scibert_similarity_assigns_provisional_type(self) -> None:
        ontology = {"nodes": {
            "Model": {"category": "Technology", "definition": "A computational model."},
            "Dataset": {"category": "Evidence", "definition": "A scientific dataset."},
        }}
        typer = SciBERTOntologyTyper(
            ontology, "fake-scibert", threshold=.5, top_k=2, min_margin=.1,
            cache_dir=Path("tmp/fake-cache"), encoder=FakeEncoder(),
        )
        result = typer.type_arguments([{
            "surface_text": "benchmark dataset", "sentence_text": "We evaluated a benchmark dataset."
        }])[0]
        self.assertTrue(result["applicable"])
        self.assertEqual(result["selected_label"], "Dataset")
        self.assertEqual(result["typing_method"], "pretrained_scibert_candidate_ontology_similarity")
        self.assertGreaterEqual(result["score_margin"], .1)

    def test_ambiguous_scibert_type_does_not_remove_grammar_candidate(self) -> None:
        result = generate_grammatical_candidates(
            self.parsed("The membrane bioreactor removes contaminants."),
            Path("Pipeline/GENERAL/ontology/ontology.json"),
            {"grammatical_candidate_generation": {
                "enabled": True, "required": True, "parser_model": "en_core_web_sm",
                "scibert_model": "fake-scibert", "typing_threshold": .25,
                "typing_min_margin": .015, "typing_top_k": 3,
            }},
            parser=self.parser(), typer=AmbiguousFakeTyper(),
        )
        candidates = result["candidates_by_sentence"]["s1"]
        self.assertEqual(len(candidates), 2)
        self.assertTrue(all(candidate["label"] == "" for candidate in candidates))
        self.assertTrue(all(not candidate["metadata"]["typing_applicable"] for candidate in candidates))
        self.assertEqual(result["status"]["generated_candidates"], 2)
        self.assertEqual(result["status"]["typed_candidates"], 0)


if __name__ == "__main__":
    unittest.main()
