"""Deterministic contracts for the compact LLM-focused pipeline."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from Pipeline.Edge_Extraction.pipeline import _deterministic_gates, run_relationship_extraction
from Pipeline.Grammatical_Parsing.candidates import RawSVOExtractor
from Pipeline.ontology import Ontology
from Pipeline.Node_Pipeline.pipeline import build_paragraph_batches, extract_nodes, validate_provenance
from Pipeline.core import ONTOLOGY_PATH, dynamic_token_batches, parse_tsv_rows
from Pipeline.llm import OllamaClient


def parsed_document() -> dict:
    return {
        "document": {"id": "doc-1", "title": "Test", "filename": "test.pdf"},
        "summary": {"sentences": 1, "paragraphs": 1},
        "sections": [{
            "id": "section-1", "title": "Results", "paragraphs": [{
                "id": "paragraph-1", "text": "The membrane removes contaminants while energy use increases.",
                "sentences": [{
                    "id": "sentence-1", "text": "The membrane removes contaminants while energy use increases.",
                    "pages": [1],
                }],
            }],
        }],
    }


def two_paragraph_document() -> dict:
    parsed = deepcopy(parsed_document())
    second = deepcopy(parsed["sections"][0]["paragraphs"][0])
    second["id"] = "paragraph-2"
    second["sentences"][0]["id"] = "sentence-2"
    parsed["sections"][0]["paragraphs"].append(second)
    parsed["summary"] = {"sentences": 2, "paragraphs": 2}
    return parsed


class FakeTyper:
    def type_arguments(self, arguments: list[dict]) -> list[dict]:
        output = []
        for row in arguments:
            is_membrane = "membrane" in row["node_name"].casefold()
            output.append({
                **row,
                "scibert_type": "TreatmentUnit" if is_membrane else "NONE",
                "scibert_score": .7 if is_membrane else .1,
                "scibert_margin": .2,
                "scibert_alternatives": [{"label": "TreatmentUnit" if is_membrane else "Contaminant", "score": .7}],
                "scibert_applicable": is_membrane,
                "scibert_model": "fake-scibert",
            })
        return output


class NodeTransport:
    def __init__(self) -> None:
        self.typing_candidate_ids: list[str] = []
        self.typing_payload: dict = {}

    def __call__(self, body: dict) -> dict:
        system = body["messages"][0]["content"]
        payload = json.loads(body["messages"][1]["content"])
        if "Resolve each raw grammatical span" in system:
            nodes = [
                node for paragraph in payload["paragraphs"]
                for sentence in paragraph["sentences"] for node in sentence["nodes"]
            ]
            result = "\n".join(
                f"{row['candidate_id']}\t{row['raw_text'].removeprefix('The ').title()}\tS"
                for row in nodes
            )
        elif "Type the resolved referent" in system:
            self.typing_payload = payload
            nodes = [node for paragraph in payload["paragraphs"] for node in paragraph["nodes"]]
            self.typing_candidate_ids = [row["candidate_id"] for row in nodes]
            result = "\n".join(
                f"{row['candidate_id']}\t"
                f"{'TreatmentUnit' if 'membrane' in row['specific_name'].casefold() else 'Contaminant'}"
                for row in nodes
            )
        else:
            nodes = [node for paragraph in payload["paragraphs"] for node in paragraph["nodes"]]
            result = "\n".join(f"{row['candidate_id']}\tA" for row in nodes)
        return {"choices": [{"message": {"content": result}}], "usage": {}}


class RelationshipTransport:
    def __init__(self) -> None:
        self.generation_payload: dict = {}

    def __call__(self, body: dict) -> dict:
        system = body["messages"][0]["content"]
        payload = json.loads(body["messages"][1]["content"])
        if "For each independent paragraph" in system:
            self.generation_payload = payload
            relationships: list[str] = []
            for paragraph in payload["paragraphs"]:
                nodes = paragraph["accepted_nodes"]
                subject = next(row for row in nodes if row["type"] == "TreatmentUnit")
                obj = next(row for row in nodes if row["type"] == "Contaminant")
                relationships.extend("\t".join([
                    paragraph["paragraph_id"], "K", subject["mention_id"], "", "", "",
                    "REMOVES", "K", obj["mention_id"], "", "", "", subject["sentence_id"],
                ]) for _ in range(5))
            result = "\n".join(relationships)
        elif "Correct predicate" in system:
            relationships = [
                row for paragraph in payload["paragraphs"]
                for row in paragraph["relationships"]
            ]
            result = "\n".join(
                f"{row['proposal_id']}\tK\t=\t=\t=\t=" for row in relationships
            )
        else:
            relationships = [
                row for paragraph in payload["paragraphs"]
                for row in paragraph["relationships"]
            ]
            result = "\n".join(
                f"{row['proposal_id']}\tA\t1.0\tentailed" for row in relationships
            )
        return {"choices": [{"message": {"content": result}}], "usage": {}}


class PipelineContractsTest(unittest.TestCase):
    def test_dynamic_batching_uses_input_and_output_budgets(self):
        config = {
            "ollama": {"context_window": 1000, "max_tokens": 400},
            "batching": {
                "chars_per_token": 1, "context_safety_tokens": 100,
                "input_token_budget": 500, "output_token_budget": 100,
            },
        }
        items = [{"text": "a" * 50}, {"text": "b" * 50}, {"text": "c" * 50}]
        groups = dynamic_token_batches(
            items, config=config, input_value=lambda row: row,
            expected_output_tokens=lambda _: 45, fixed_input_tokens=20,
        )
        self.assertEqual([2, 1], [len(group) for group in groups])

    def test_tsv_parser_ignores_prose_fences_and_wrong_width_rows(self):
        text = "result follows\n```text\ncandidate-1\tModel\nbad\trow\textra\ncandidate-2\\tDataset\n```"
        self.assertEqual(
            [["candidate-1", "Model"], ["candidate-2", "Dataset"]],
            parse_tsv_rows(text, 2),
        )

    def test_ollama_returns_plain_text_without_requesting_json_mode(self):
        captured = {}
        def transport(body):
            captured.update(body)
            return {"message": {"content": "candidate-1\tModel"}}
        client = OllamaClient(transport=lambda _: {
            "message": {"content": "candidate-1\tModel"}
        })
        client.transport = transport
        result = client.complete_text("node_ontology_typing", "Return TSV.", {})
        self.assertEqual([["candidate-1", "Model"]], parse_tsv_rows(result, 2))
        self.assertNotIn("format", captured)

    def test_raw_svo_preserves_articles_and_separates_context(self):
        extractor = RawSVOExtractor.load("en_core_web_sm")
        record = {"document_id": "d", "section_id": "s", "paragraph_id": "p",
                  "sentence_id": "x", "text": "The membrane removes contaminants while energy use increases."}
        analyses, triples = extractor.analyze_records([record])
        self.assertEqual("The membrane", triples[0]["subject"]["text"])
        self.assertEqual("removes", triples[0]["verb"]["text"])
        self.assertEqual("contaminants", triples[0]["object"]["text"])
        self.assertIn("energy use increases", analyses[0]["context"][0]["text"])

    def test_paragraph_batch_keeps_subject_object_and_context_separate(self):
        extractor = RawSVOExtractor.load("en_core_web_sm")
        records = [{"document_id": "doc-1", "section_id": "section-1", "section_title": "Results",
                    "paragraph_id": "paragraph-1", "sentence_id": "sentence-1",
                    "text": "The membrane removes contaminants while energy use increases."}]
        analyses, triples = extractor.analyze_records(records)
        batch = build_paragraph_batches(parsed_document(), {"sentences": analyses, "triples": triples})[0]
        self.assertEqual("The membrane", batch["sentences"][0]["subjects"][0]["surface_text"])
        self.assertEqual("contaminants", batch["sentences"][0]["objects"][0]["surface_text"])
        self.assertTrue(batch["sentences"][0]["context"])

    def test_all_candidates_reach_qwen_typing_even_when_scibert_says_none(self):
        transport = NodeTransport()
        client = OllamaClient(transport=transport)
        extractor = RawSVOExtractor.load("en_core_web_sm")
        with tempfile.TemporaryDirectory() as directory:
            result = extract_nodes(
                parsed_document(), Path(directory), client=client, parser=extractor,
                typer=FakeTyper(), prior_lexicon=[{"name": "Membrane", "type": "TreatmentUnit"}],
            )
            self.assertEqual(2, len(transport.typing_candidate_ids))
            self.assertEqual(2, len(result["mentions"]))
            self.assertIn("ontology_types", transport.typing_payload)
            self.assertNotIn("ontology", transport.typing_payload)
            typing_nodes = [
                row for paragraph in transport.typing_payload["paragraphs"]
                for row in paragraph["nodes"]
            ]
            membrane = next(row for row in typing_nodes
                            if "membrane" in row["specific_name"].casefold())
            self.assertEqual(["TreatmentUnit"], membrane["previous_types"])
            self.assertEqual(["node_cleanup", "node_ontology_typing", "node_judgment"],
                             [row["stage"] for row in client.audit])

    def test_relationships_have_three_passes_and_exact_provenance(self):
        node_client = OllamaClient(transport=NodeTransport())
        extractor = RawSVOExtractor.load("en_core_web_sm")
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            nodes = extract_nodes(parsed_document(), run_dir, client=node_client,
                                  parser=extractor, typer=FakeTyper())
            edge_transport = RelationshipTransport()
            edge_client = OllamaClient(transport=edge_transport)
            result = run_relationship_extraction(parsed_document(), nodes["mentions"], run_dir, client=edge_client)
            self.assertEqual(5, len(result["candidates"]))
            self.assertTrue(all(row["generation_minimum_met"] for row in result["candidates"]))
            self.assertTrue(all(row["status"] == "accepted" for row in result["candidates"]))
            self.assertEqual(["relationship_generation", "relationship_refinement", "relationship_validation"],
                             [row["stage"] for row in edge_client.audit])
            predicate_names = {
                row["p"] for row in edge_transport.generation_payload["paragraphs"][0]["predicates"]
            }
            self.assertIn("REMOVES", predicate_names)
            self.assertNotIn("PROPOSES", predicate_names)
            self.assertTrue(all(set(row) == {"p", "d", "r", "m"}
                                for row in edge_transport.generation_payload["paragraphs"][0]["predicates"]))
            validation = validate_provenance(parsed_document(), nodes["mentions"], result["candidates"])
            self.assertEqual("PASS", validation["status"])

    def test_multiple_paragraphs_share_one_call_per_pass(self):
        extractor = RawSVOExtractor.load("en_core_web_sm")
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            node_client = OllamaClient(transport=NodeTransport())
            nodes = extract_nodes(
                two_paragraph_document(), run_dir, client=node_client,
                parser=extractor, typer=FakeTyper(), prior_lexicon=[],
            )
            self.assertEqual(3, len(node_client.audit))

            edge_transport = RelationshipTransport()
            edge_client = OllamaClient(transport=edge_transport)
            edges = run_relationship_extraction(
                two_paragraph_document(), nodes["mentions"], run_dir, client=edge_client,
            )
            self.assertEqual(3, len(edge_client.audit))
            self.assertEqual(2, len(edge_transport.generation_payload["paragraphs"]))
            self.assertEqual(10, len(edges["candidates"]))

    def test_inferred_endpoint_quote_must_be_exact(self):
        paragraph = {
            "paragraph_id": "paragraph-1",
            "sentences": [{"sentence_id": "sentence-1", "text": "AquaSense uses RiverBench."}],
        }
        mentions = {"known": {"mention_id": "known", "label": "AISystem"}}
        proposal = {
            "subject": {"mention_id": "known"},
            "predicate": "USES_DATASET",
            "object": {
                "inferred_name": "RiverBench", "inferred_type": "Dataset",
                "evidence_sentence_id": "sentence-1", "evidence_quote": "invented dataset",
            },
            "evidence_sentence_ids": ["sentence-1"],
        }
        gates = _deterministic_gates(paragraph, proposal, mentions, Ontology(ONTOLOGY_PATH))
        self.assertFalse(gates["inferred_endpoints_grounded"])


if __name__ == "__main__":
    unittest.main()
