from __future__ import annotations

import json
from copy import deepcopy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]

from Pipeline.Knowledge_Graph_Core.entity_resolution import EntityResolver
from Pipeline.Knowledge_Graph_Core.edge_extractor import EdgeExtractor
from Pipeline.Knowledge_Graph_Core.llm_node_judge import LLMNodeJudgeError, MandatoryLLMNodeJudge
from Pipeline.Knowledge_Graph_Core.llm_reference_judge import MandatoryLLMReferenceJudge
from Pipeline.Knowledge_Graph_Core.llm_relationship_judge import MandatoryLLMRelationshipJudge
from Pipeline.Knowledge_Graph_Core.models import normalize_name, stable_id
from Pipeline.Knowledge_Graph_Core.node_extractor import NodeExtractor
from Pipeline.Knowledge_Graph_Core.ontology import Ontology
from Pipeline.Knowledge_Graph_Core.review_queues import build_review_queues


class PipelineUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ontology = Ontology(ROOT / "config" / "ontology.json")
        cls.config = json.loads((ROOT / "config" / "extraction.json").read_text(encoding="utf-8"))

    def test_llm_judge_is_mandatory_without_credential(self):
        config = deepcopy(self.config)
        config["llm_node_judge"].update({"provider": "openai", "model": "test-openai-model"})
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(LLMNodeJudgeError):
                MandatoryLLMNodeJudge(self.ontology, config, Path(tmp))

    def test_ollama_provider_requires_no_api_key(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            judge = MandatoryLLMNodeJudge(
                self.ontology, self.config, Path(tmp), transport=lambda _request: {}
            )
        self.assertEqual(judge.provider, "ollama")
        self.assertEqual(judge.model, "llama3.1:8b-instruct-q3_K_S")

    def test_ollama_node_judge_expands_context_only_for_oversized_payloads(self):
        config = deepcopy(self.config)
        config["llm_node_judge"].update({
            "context_window": 32768,
            "maximum_context_window": 65536,
            "max_output_tokens": 4096,
        })
        schema = {"type": "object", "properties": {}}
        with tempfile.TemporaryDirectory() as tmp:
            judge = MandatoryLLMNodeJudge(
                self.ontology, config, Path(tmp), transport=lambda _request: {}
            )
            normal = judge._request_payload({"sentences": [{"text": "short"}]}, schema)
            oversized = judge._request_payload(
                {"sentences": [{"text": "x" * 120_000}]}, schema
            )
        self.assertEqual(normal["options"]["num_ctx"], 32768)
        self.assertEqual(oversized["options"]["num_ctx"], 65536)

    def test_ollama_node_judge_splits_candidate_heavy_sentence_and_merges_audits(self):
        config = deepcopy(self.config)
        config["llm_node_judge"].update({
            "batch_sentences": 1,
            "context_window": 32768,
            "maximum_context_window": 65536,
            "max_output_tokens": 4096,
        })
        requests = []

        def transport(payload):
            requests.append(payload)
            user = json.loads(payload["messages"][1]["content"])
            sentence = user["sentences"][0]
            judgments = [{
                "candidate_id": candidate["candidate_id"],
                "decision": "reject", "ontology_label": "NONE",
                "canonical_name": candidate["surface_text"],
                "definition": "A test candidate.",
                "reason": "Rejected by the test transport.", "confidence": 0.9,
            } for candidate in sentence["candidates"]]
            return {
                "message": {"role": "assistant", "content": json.dumps({"sentences": [{
                    "sentence_id": sentence["sentence_id"],
                    "candidate_judgments": judgments,
                    "discoveries": [], "new_class_candidates": [],
                }]})},
                "done": True,
            }

        candidates = [{
            "candidate_id": f"candidate{index}", "surface_text": "WaterRAG",
            "start_char": 0, "end_char": 8, "suggested_label": "AISystem",
            "candidate_method": "scibert_svo_argument",
            "deterministically_cleaned_name": "WaterRAG", "scibert_alternatives": [],
            "prior_nodes_same_type": [{
                "label": "AISystem", "canonical_name": "x" * 30_000,
            }],
        } for index in range(4)]
        sentence = [{
            "sentence_id": "sent1", "section_title": "Abstract",
            "text": "WaterRAG supports operators.", "candidates": candidates,
        }]
        with tempfile.TemporaryDirectory() as tmp:
            judge = MandatoryLLMNodeJudge(
                self.ontology, config, Path(tmp), transport=transport,
            )
            result = judge.judge(sentence)["sent1"]
        self.assertGreater(len(requests), 1)
        self.assertEqual(len(result["candidate_judgments"]), 4)
        self.assertEqual(len(result["_llm_audits"]), len(requests))
        self.assertTrue(all(row.get("_llm_audit") for row in result["candidate_judgments"]))

    def test_reference_judge_sends_contextual_pair_and_returns_same(self):
        config = deepcopy(self.config)
        config["llm_reference_judge"] = deepcopy(config["llm_node_judge"])
        config["llm_reference_judge"]["batch_pairs"] = 12

        def transport(request):
            user = json.loads(request["messages"][1]["content"])
            self.assertEqual(user["pairs"][0]["reference"]["surface_text"], "this model")
            self.assertEqual(user["pairs"][0]["candidate"]["canonical_name"], "WaterRAG")
            return {
                "model": request["model"], "created_at": "2026-07-20T00:00:00Z", "done": True,
                "message": {"role": "assistant", "content": json.dumps({"judgments": [{
                    "pair_id": "pair1", "decision": "SAME",
                    "reason": "The second sentence continues describing WaterRAG.",
                    "confidence": .97,
                }]})},
            }

        pairs = [{
            "pair_id": "pair1",
            "reference": {
                "surface_text": "this model",
                "evidence_quote": "This model improved retrieval accuracy.",
                "context_sentences": [
                    {"text": "WaterRAG was evaluated."},
                    {"text": "This model improved retrieval accuracy."},
                ],
            },
            "candidate": {
                "surface_text": "WaterRAG", "canonical_name": "WaterRAG",
                "ontology_label": "AISystem", "evidence_quote": "WaterRAG was evaluated.",
            },
        }]
        with tempfile.TemporaryDirectory() as tmp:
            judge = MandatoryLLMReferenceJudge(
                self.ontology, config, Path(tmp), transport=transport
            )
            result = judge.judge(pairs)
        self.assertEqual(result["pair1"]["decision"], "SAME")
        self.assertTrue(result["pair1"]["_llm_audit"]["llm_used"])

    def test_lexicon_contains_only_specific_anchors(self):
        lexicons = json.loads((ROOT / "config" / "lexicons.json").read_text(encoding="utf-8"))
        forbidden = {
            "agent", "model", "method", "system", "framework", "tool", "api", "solver",
            "simulator", "dataset", "data", "optimization", "prompting", "forecasting",
            "retrieval", "classification", "accuracy", "precision", "recall", "limitation",
            "uncertainty", "environmental system", "real network", "organism", "species",
            "process", "policy document", "regulation", "standard", "pump", "pipe",
            "reservoir", "plant", "facility", "sensor", "university", "institute",
            "agency", "utility", "company", "research group",
        }
        values = {value.casefold() for phrases in lexicons.values() for value in phrases}
        self.assertFalse(values & forbidden, f"Generic lexicon entries found: {sorted(values & forbidden)}")

    def test_llm_judges_candidates_and_discovers_out_of_lexicon_node(self):
        def transport(request):
            self.assertEqual(request["model"], "llama3.1:8b-instruct-q3_K_S")
            self.assertIsInstance(request["format"], dict)
            self.assertFalse(request["stream"])
            user = json.loads(request["messages"][1]["content"])
            sentence_results = []
            for sentence in user["sentences"]:
                judgments = [{
                    "candidate_id": candidate["candidate_id"],
                    "decision": "accept",
                    "ontology_label": candidate["suggested_label"],
                    "canonical_name": candidate["surface_text"],
                    "definition": "A specific paper entity accepted by the mandatory LLM judge.",
                    "reason": "Specific named concept.",
                    "confidence": 0.96,
                } for candidate in sentence["candidates"]]
                discoveries = []
                value = "HydroSim-X"
                if value in sentence["text"]:
                    start = sentence["text"].index(value)
                    discoveries.append({
                        "start_char": start, "end_char": start + len(value),
                        "surface_text": value, "ontology_label": "Tool",
                        "canonical_name": value,
                        "definition": "A named simulation tool discovered outside the specific lexicon.",
                        "reason": "Named tool absent from candidate anchors.", "confidence": 0.94,
                    })
                sentence_results.append({
                    "sentence_id": sentence["sentence_id"],
                    "candidate_judgments": judgments,
                    "discoveries": discoveries,
                    "new_class_candidates": [],
                })
            return {"model": "llama3.1:8b", "created_at": "2026-07-14T00:00:00Z",
                    "message": {"role": "assistant", "content": json.dumps({"sentences": sentence_results})},
                    "prompt_eval_count": 10, "eval_count": 10, "done": True}

        parsed = {
            "document": {"title": "Test paper", "filename": "test.pdf", "authors_text": "A. Author"},
            "sections": [{
                "id": "sec1", "title": "Methods", "ordinal": 1,
                "paragraphs": [{
                    "id": "para1", "ordinal": 1,
                    "sentences": [{
                        "id": "sent1", "ordinal": 1, "pages": [1],
                        "text": "The Orchestrating Agent used HydroSim-X.",
                    }],
                }],
            }],
        }
        with tempfile.TemporaryDirectory() as tmp:
            judge = MandatoryLLMNodeJudge(
                self.ontology, self.config, Path(tmp) / "cache", transport=transport
            )
            extractor = NodeExtractor(
                self.ontology, ROOT / "config" / "lexicons.json", self.config, judge
            )
            mentions, reviews, rejections, calls = extractor.extract(parsed, "doc_test")
        sentence_mentions = [m for m in mentions if m["source"]["kind"] == "sentence_span"]
        self.assertTrue(any(m["label"] == "Agent" for m in sentence_mentions))
        self.assertTrue(any(m["label"] == "Tool" and m["extraction_method"] == "llm_discovery"
                            for m in sentence_mentions))
        self.assertTrue(all(m["attributes"]["candidate_validation"]["llm_used"]
                            for m in sentence_mentions))
        self.assertFalse(reviews)
        self.assertTrue(calls and calls[0]["llm_used"])
        self.assertEqual(calls[0]["provider"], "ollama")
        self.assertTrue(calls[0]["response_id"].startswith("ollama_"))

    def test_missing_llm_candidate_judgment_follows_reject_policy(self):
        def incomplete_transport(_request):
            return {"id": "resp_bad", "output_text": json.dumps({"sentences": [{
                "sentence_id": "sent1", "candidate_judgments": [],
                "discoveries": [], "new_class_candidates": [],
            }]})}
        sentence = [{
            "sentence_id": "sent1", "section_title": "Methods", "text": "EPANET was used.",
            "candidates": [{"candidate_id": "c1", "surface_text": "EPANET", "start_char": 0,
                            "end_char": 6, "suggested_label": "Tool", "candidate_method": "specific_lexicon"}],
        }]
        with tempfile.TemporaryDirectory() as tmp:
            judge = MandatoryLLMNodeJudge(
                self.ontology, self.config, Path(tmp), transport=incomplete_transport
            )
            result = judge.judge(sentence)
        self.assertEqual(result["sent1"]["candidate_judgments"][0]["decision"], "reject")
        self.assertTrue(judge.audit_records[0]["invalid_outputs"])

    def test_llm_discovery_must_match_exact_source_span(self):
        def invented_span_transport(_request):
            return {"id": "resp_invented", "output_text": json.dumps({"sentences": [{
                "sentence_id": "sent1", "candidate_judgments": [],
                "discoveries": [{
                    "start_char": 0, "end_char": 6, "surface_text": "EPANEX",
                    "ontology_label": "Tool", "canonical_name": "EPANEX",
                    "definition": "Invented tool text.", "reason": "Invalid test output.",
                    "confidence": 0.9,
                }],
                "new_class_candidates": [],
            }]})}
        sentence = [{
            "sentence_id": "sent1", "section_title": "Methods", "text": "EPANET was used.",
            "candidates": [],
        }]
        with tempfile.TemporaryDirectory() as tmp:
            judge = MandatoryLLMNodeJudge(
                self.ontology, self.config, Path(tmp), transport=invented_span_transport
            )
            result = judge.judge(sentence)
        self.assertEqual(result["sent1"]["discoveries"], [])
        self.assertEqual(len(judge.audit_records[0]["invalid_outputs"]), 1)
        self.assertEqual(
            judge.audit_records[0]["invalid_outputs"][0]["raw_output"]["surface_text"],
            "EPANEX",
        )

    def test_llm_invalid_span_can_be_configured_to_fail_closed(self):
        def invented_span_transport(_request):
            return {"id": "resp_invented", "output_text": json.dumps({"sentences": [{
                "sentence_id": "sent1", "candidate_judgments": [],
                "discoveries": [{
                    "start_char": 0, "end_char": 6, "surface_text": "EPANEX",
                    "ontology_label": "Tool", "canonical_name": "EPANEX",
                    "definition": "Invented tool text.", "reason": "Invalid test output.",
                    "confidence": 0.9,
                }],
                "new_class_candidates": [],
            }]})}
        sentence = [{
            "sentence_id": "sent1", "section_title": "Methods", "text": "EPANET was used.",
            "candidates": [],
        }]
        config = deepcopy(self.config)
        config["llm_node_judge"]["invalid_output_policy"] = "fail"
        with tempfile.TemporaryDirectory() as tmp:
            judge = MandatoryLLMNodeJudge(
                self.ontology, config, Path(tmp), transport=invented_span_transport
            )
            with self.assertRaises(LLMNodeJudgeError):
                judge.judge(sentence)

    def test_llm_real_text_with_wrong_offset_is_aligned_and_audited(self):
        def wrong_offset_transport(_request):
            return {"model": "llama3.1:8b", "message": {"role": "assistant", "content": json.dumps({
                "sentences": [{
                    "sentence_id": "sent1", "candidate_judgments": [],
                    "discoveries": [{
                        "start_char": 0, "end_char": 6, "surface_text": "EPANET",
                        "ontology_label": "Tool", "canonical_name": "EPANET",
                        "definition": "A named hydraulic simulation tool.",
                        "reason": "Specific named tool.", "confidence": 0.95,
                    }],
                    "new_class_candidates": [],
                }]
            })}, "done": True}
        sentence = [{
            "sentence_id": "sent1", "section_title": "Methods",
            "text": "The model used EPANET.", "candidates": [],
        }]
        with tempfile.TemporaryDirectory() as tmp:
            judge = MandatoryLLMNodeJudge(
                self.ontology, self.config, Path(tmp), transport=wrong_offset_transport
            )
            result = judge.judge(sentence)
        discovery = result["sent1"]["discoveries"][0]
        self.assertEqual(discovery["start_char"], 15)
        self.assertEqual(discovery["end_char"], 21)
        self.assertEqual(len(judge.audit_records[0]["span_corrections"]), 1)

    def test_stable_id_is_deterministic(self):
        self.assertEqual(stable_id("x", "a", 1), stable_id("x", "a", 1))

    def test_missing_llm_candidate_judgment_is_quarantined_for_review(self):
        config = deepcopy(self.config)
        config["llm_node_judge"]["invalid_output_policy"] = "review_and_audit"

        def transport(_payload: dict) -> dict:
            return {
                "message": {"content": json.dumps({
                    "sentences": [{
                        "sentence_id": "1", "candidate_judgments": [],
                        "discoveries": [], "new_class_candidates": [],
                    }]
                })},
                "done": True,
            }

        sentence = [{
            "sentence_id": "sent1", "section_title": "Abstract", "text": "WaterRAG supports operators.",
            "candidates": [{
                "candidate_id": "candidate1", "surface_text": "WaterRAG", "start_char": 0,
                "end_char": 8, "suggested_label": "AISystem", "candidate_method": "scibert_svo_argument",
                "deterministically_cleaned_name": "WaterRAG", "scibert_alternatives": [],
                "prior_nodes_same_type": [],
            }],
        }]
        with tempfile.TemporaryDirectory() as tmp:
            judge = MandatoryLLMNodeJudge(
                self.ontology, config, Path(tmp), transport=transport,
            )
            result = judge.judge(sentence)
        judgment = result["sent1"]["candidate_judgments"][0]
        self.assertEqual(judgment["decision"], "review")
        self.assertEqual(judgment["candidate_id"], "candidate1")
        self.assertTrue(judge.audit_records[0]["invalid_outputs"])
        self.assertTrue(any(
            row["correction_type"] == "single_sentence_id_restored"
            for row in judge.audit_records[0]["format_corrections"]
        ))

    def test_malformed_structured_json_is_retried(self):
        config = deepcopy(self.config)
        config["llm_node_judge"]["structured_output_retries"] = 1
        attempts = 0

        def transport(_payload: dict) -> dict:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return {"output_text": '{"sentences":[', "done": True}
            return {"output_text": json.dumps({"sentences": [{
                "sentence_id": "sent1",
                "candidate_judgments": [{
                    "candidate_id": "candidate1", "decision": "accept", "ontology_label": "AISystem",
                    "canonical_name": "WaterRAG", "definition": "A water-industry AI system.",
                    "reason": "The named system is explicit.", "confidence": 0.9,
                }],
                "discoveries": [], "new_class_candidates": [],
            }]}), "done": True}

        sentence = [{
            "sentence_id": "sent1", "section_title": "Abstract", "text": "WaterRAG supports operators.",
            "candidates": [{
                "candidate_id": "candidate1", "surface_text": "WaterRAG", "start_char": 0,
                "end_char": 8, "suggested_label": "AISystem", "candidate_method": "scibert_svo_argument",
                "deterministically_cleaned_name": "WaterRAG", "scibert_alternatives": [],
                "prior_nodes_same_type": [],
            }],
        }]
        with tempfile.TemporaryDirectory() as tmp:
            judge = MandatoryLLMNodeJudge(self.ontology, config, Path(tmp), transport=transport)
            result = judge.judge(sentence)
        self.assertEqual(attempts, 2)
        self.assertEqual(result["sent1"]["candidate_judgments"][0]["decision"], "accept")
        self.assertEqual(judge.audit_records[0]["status"], "complete_with_quarantine_or_retry")

    def test_name_normalization(self):
        self.assertEqual(normalize_name("Water–MAS"), "water mas")

    def test_domain_range_accepts_subclass(self):
        self.assertTrue(self.ontology.valid_relationship("TREATS", "TreatmentProcess", "WaterMatrix"))
        self.assertTrue(self.ontology.valid_relationship("ABOUT", "Observation", "Contaminant"))

    def test_domain_range_rejects_invalid_direction(self):
        self.assertFalse(self.ontology.valid_relationship("AUTHORED_BY", "Person", "Publication"))

    @staticmethod
    def _relationship_fixture():
        sentences = [
            {"id": "previous", "text": "WaterRAG is the proposed system.", "pages": [1]},
            {"id": "target", "text": "It uses GPT-4 for generation.", "pages": [1]},
            {"id": "next", "text": "The evaluation follows.", "pages": [1]},
        ]
        parsed = {
            "document": {"id": "doc1", "title": "Context test"},
            "sections": [{"id": "section", "title": "Methods", "paragraphs": [
                {"id": "paragraph", "sentences": sentences},
            ]}],
        }
        mentions = [
            {"mention_id": "system", "document_id": "doc1", "label": "AISystem",
             "surface_text": "WaterRAG", "canonical_name": "WaterRAG", "status": "accepted",
             "source": {"sentence_id": "previous", "start_char": 0, "end_char": 8}},
            {"mention_id": "model", "document_id": "doc1", "label": "Model",
             "surface_text": "GPT-4", "canonical_name": "GPT-4", "status": "accepted",
             "source": {"sentence_id": "target", "start_char": 8, "end_char": 13}},
        ]
        return parsed, mentions

    def test_edge_extractor_uses_llm_selected_existing_nodes_and_all_context(self):
        class ContextJudge:
            audit_records = []

            def extract(self, windows, accepted_nodes):
                self.audit_records.append({"llm_used": True})
                return {
                    row["target_sentence_id"]: {
                        "target_sentence_id": row["target_sentence_id"],
                        "proposals": ([{
                            "head_node_id": "system", "head_text": "WaterRAG",
                            "head_proposed_label": "AISystem",
                            "tail_node_id": "model", "tail_text": "GPT-4",
                            "tail_proposed_label": "Model", "predicate": "USES_MODEL",
                            "predicate_trigger_text": "uses",
                            "predicate_trigger_sentence_id": "target", "decision": "accept",
                            "attribution": "source_statement", "negated": False,
                            "modal": False, "hypothetical": False, "comparative": False,
                            "causal": False, "reason": "Explicit model use.", "confidence": 0.97,
                        }] if row["target_sentence_id"] == "target" else []),
                    }
                    for row in windows
                }

        parsed, mentions = self._relationship_fixture()
        assertions, _ = EdgeExtractor(self.ontology, ContextJudge(), self.config).extract(
            parsed, mentions
        )
        self.assertEqual(len(assertions), 1)
        edge = assertions[0]
        self.assertEqual(edge["subject_mention_id"], "system")
        self.assertEqual(edge["object_mention_id"], "model")
        self.assertEqual(edge["evidence_sentence_ids"], ["previous", "target", "next"])
        self.assertEqual(edge["predicate_trigger_text"], "uses")
        self.assertEqual(edge["predicate_trigger_sentence_id"], "target")
        self.assertEqual(edge["extraction_method"], "llm_context_window_v1")

    def test_edge_extractor_audits_unseen_endpoint_without_creating_edge(self):
        class UnknownNodeJudge:
            audit_records = []

            def extract(self, windows, accepted_nodes):
                return {row["target_sentence_id"]: {
                    "target_sentence_id": row["target_sentence_id"],
                    "proposals": ([{
                        "head_node_id": "system", "head_text": "WaterRAG",
                        "head_proposed_label": "AISystem", "tail_node_id": "",
                        "tail_text": "NovaLM", "tail_proposed_label": "Model",
                        "predicate": "USES_MODEL", "predicate_trigger_text": "uses",
                        "predicate_trigger_sentence_id": "target", "decision": "accept",
                        "attribution": "source_statement", "negated": False,
                        "modal": False, "hypothetical": False, "comparative": False,
                        "causal": False, "reason": "Model is absent from the node catalog.",
                        "confidence": 0.95,
                    }] if row["target_sentence_id"] == "target" else []),
                } for row in windows}

        parsed, mentions = self._relationship_fixture()
        parsed["sections"][0]["paragraphs"][0]["sentences"][1]["text"] = (
            "It uses NovaLM for generation."
        )
        extractor = EdgeExtractor(self.ontology, UnknownNodeJudge(), self.config)
        assertions, _ = extractor.extract(parsed, mentions)
        self.assertEqual(assertions, [])
        self.assertEqual(len(extractor.unknown_nodes), 1)
        self.assertEqual(extractor.unknown_nodes[0]["surface_text"], "NovaLM")
        self.assertEqual(extractor.unknown_nodes[0]["role"], "tail")


    def test_relationship_llm_selects_only_candidate_permitted_predicate(self):
        def transport(request):
            window = json.loads(request["messages"][1]["content"])["sentence_windows"][0]
            return {
                "model": "llama3.1:8b",
                "message": {"role": "assistant", "content": json.dumps({"windows": {
                    window["target_sentence_id"]: {"proposals": []},
                }})},
                "done": True,
            }

        window = {"target_sentence_id": "sent1", "section_title": "Methods",
                  "context_sentences": [{"sentence_id": "sent1", "role": "target",
                                         "text": "WaterRAG uses GPT-4."}]}
        with tempfile.TemporaryDirectory() as tmp:
            judge = MandatoryLLMRelationshipJudge(
                self.ontology, self.config, Path(tmp), transport=transport
            )
            result = judge.extract([window], [])
        self.assertEqual(result["sent1"]["proposals"], [])
        self.assertTrue(judge.audit_records[0]["llm_used"])

    def test_resolution_never_auto_merges(self):
        class FakeEmbeddingBackend:
            model_name = "fake-semantic-model"

            @staticmethod
            def encode(texts):
                return [[1.0, 0.0] if "Shanghai" in text else [0.95, 0.05] for text in texts]

        resolver = EntityResolver(self.config, FakeEmbeddingBackend())
        base = {
            "label": "EnvironmentalSystem",
            "document_id": "d",
            "confidence": 1,
            "status": "accepted",
            "extraction_method": "test",
            "ontology_matches": [],
            "attributes": {},
            "source": {"kind": "document_metadata"},
        }
        mentions = [
            {**base, "mention_id": "m1", "surface_text": "water distribution network", "normalized_name": "water distribution network", "definition": "A water distribution network in Shanghai."},
            {**base, "mention_id": "m2", "surface_text": "WDN", "normalized_name": "wdn", "definition": "The WDN is a water distribution network in Shanghai."},
        ]
        decisions, _ = resolver.resolve(mentions)
        self.assertTrue(decisions)
        self.assertTrue(all(not decision["auto_merged"] for decision in decisions))
        self.assertTrue(all("embedding_cosine" in decision for decision in decisions))
        self.assertTrue(all("cosine_distance" in decision for decision in decisions))
        self.assertTrue(all(decision["confidence_level"] in {"high", "medium", "low"}
                            for decision in decisions))

    def test_embedding_resolution_finds_semantic_match_without_shared_name_tokens(self):
        class FakeEmbeddingBackend:
            model_name = "fake-semantic-model"

            @staticmethod
            def encode(texts):
                vectors = []
                for text in texts:
                    if "EPANET" in text or "water distribution simulation package" in text:
                        vectors.append([1.0, 0.02, 0.0])
                    else:
                        vectors.append([0.0, 0.0, 1.0])
                return vectors

        resolver = EntityResolver(self.config, FakeEmbeddingBackend())
        base = {
            "label": "Tool", "document_id": "d", "confidence": 1,
            "status": "accepted", "extraction_method": "test", "ontology_matches": [],
            "attributes": {}, "source": {"kind": "sentence_span"},
        }
        mentions = [
            {**base, "mention_id": "m1", "surface_text": "EPANET", "normalized_name": "epanet",
             "definition": "Software for modelling hydraulic behavior in water distribution networks."},
            {**base, "mention_id": "m2", "surface_text": "water distribution simulation package",
             "normalized_name": "water distribution simulation package",
             "definition": "A hydraulic network analysis program."},
        ]
        decisions, _ = resolver.resolve(mentions)
        self.assertEqual(len(decisions), 1)
        self.assertGreater(decisions[0]["embedding_cosine"], 0.99)
        self.assertEqual(decisions[0]["confidence_level"], "high")
        self.assertEqual(decisions[0]["score_basis"], "embedding_cosine")

    def test_review_queue_collapses_repeated_name_pairs(self):
        base = {
            "band": "high", "action": "review_high_confidence_match", "label": "Model",
            "left_surface_text": "GPT-4", "right_surface_text": "GPT 4",
            "combined_cosine": 0.9, "auto_merged": False,
        }
        queues = build_review_queues([
            {**base, "resolution_id": "resolution_1"},
            {**base, "resolution_id": "resolution_2", "combined_cosine": 0.92},
        ])
        self.assertEqual(len(queues["high"]), 1)
        self.assertEqual(queues["high"][0]["decision_count"], 2)


if __name__ == "__main__":
    unittest.main()
