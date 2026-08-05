from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from Pipeline.Node_Pipeline import (
    apply_node_review,
    apply_reference_resolution,
    canonical_entities,
    flatten_sentences,
    json_write,
    jsonl_read,
    jsonl_write,
    Lexicon,
    lexicon_spans,
    generate_candidates,
    introduced_system_spans,
    optional_llm_review,
    optional_llm_reference_resolution,
    materialize_merges,
    Span,
    mention_dedup_key,
    rule_spans,
    resolve_references,
    remove_redundant_review_candidates,
    select_spans,
    similarity_candidates,
    similarity_features,
)

class NodePipelineTests(unittest.TestCase):
    def test_nested_candidates_are_retained_for_high_recall_adjudication(self) -> None:
        text = "specific energy consumption"
        spans = select_spans([
            Span(9, len(text), "Metric", "persistent_lexicon", 0.99,
                 canonical_name="energy consumption"),
            Span(0, len(text), "Metric", "scibert_svo_argument", 0.90,
                 canonical_name="specific energy consumption"),
        ], text)
        self.assertEqual(len(spans), 2)
        self.assertEqual(
            {text[row.start:row.end] for row in spans},
            {"energy consumption", "specific energy consumption"},
        )


    def test_explicit_introduction_discovers_unseen_system_names(self) -> None:
        for text, expected in [
            ("This study introduces LLM-EPANET, an agent-based framework for water modeling.", "LLM-EPANET"),
            ("We propose WaterAdmin, a bi-level AI-agent-based framework for operations.", "WaterAdmin"),
        ]:
            spans = introduced_system_spans(text)
            self.assertEqual([text[row.start:row.end] for row in spans], [expected])

    def test_scibert_svo_arguments_are_reviewable_and_lexicon_stays_trusted(self) -> None:
        intro = "This study introduces LLM-EPANET, an agent-based framework for water modeling."
        later = "LLM-EPANET enables interaction with EPANET for a water distribution system (WDS)."
        parsed = {
            "document": {"id": "doc", "filename": "paper.pdf", "title": "Paper", "authors_text": ""},
            "sections": [{"id": "sec", "title": "Abstract", "paragraphs": [{
                "id": "p", "sentences": [
                    {"id": "s1", "ordinal": 0, "text": intro, "pages": [1]},
                    {"id": "s2", "ordinal": 1, "text": later, "pages": [1]},
                ],
            }]}],
        }
        start = intro.index("LLM-EPANET")
        grammatical = {"s1": [{
            "start": start, "end": start + len("LLM-EPANET"), "label": "AISystem",
            "canonical_name": "LLM-EPANET", "confidence": .71,
            "method": "scibert_svo_argument",
            "metadata": {"grammatical_roles": ["object"], "triple_ids": ["t1"]},
        }]}
        accepted, review = generate_candidates(parsed, Lexicon(), grammatical)
        systems = [row for row in review if row["label"] == "AISystem"]
        self.assertEqual(len(systems), 1)
        self.assertTrue(all(row["canonical_name"] == "LLM-EPANET" for row in systems))
        self.assertEqual(systems[0]["extraction_method"], "scibert_svo_argument")
        self.assertEqual(systems[0]["candidate_metadata"]["grammatical_roles"], ["object"])
        tools = [row for row in accepted if row["label"] == "Tool"]
        self.assertEqual([row["surface_text"] for row in tools], ["EPANET"])

    def test_only_metadata_and_unambiguous_lexicon_matches_are_trusted(self) -> None:
        text = "EPANET supports membrane capacitive deionization (MCDI)."
        parsed = {
            "document": {"id": "doc", "filename": "paper.pdf", "title": "Paper", "authors_text": ""},
            "sections": [{"id": "sec", "title": "Methods", "paragraphs": [{
                "id": "p", "sentences": [{"id": "s1", "ordinal": 0, "text": text, "pages": [1]}],
            }]}],
        }
        accepted, review = generate_candidates(parsed, Lexicon())
        self.assertTrue(any(row["surface_text"] == "EPANET" for row in accepted))
        self.assertFalse(any(row["surface_text"] == "MCDI" for row in review))
        self.assertTrue(all(
            row["extraction_method"] in {"structural_metadata", "persistent_lexicon"}
            for row in accepted
        ))

    def test_ambiguous_persistent_alias_requires_adjudication(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "lexicon.json"
            json_write(path, {"version": 1, "entries": [
                {
                    "canonical_id": "tool_shared", "label": "Tool", "canonical_name": "Shared Tool",
                    "aliases": ["SHARED"], "status": "human_reviewed",
                },
                {
                    "canonical_id": "model_shared", "label": "Model", "canonical_name": "Shared Model",
                    "aliases": ["SHARED"], "status": "human_reviewed",
                },
            ]})
            spans = lexicon_spans("SHARED was evaluated.", Lexicon(path))
        self.assertEqual(len(spans), 2)
        self.assertTrue(all(row.method == "persistent_lexicon_ambiguous" for row in spans))

    def test_llm_accept_review_and_reject_are_all_audited(self) -> None:
        sentence = "AquaNet used uncertain phrase and generic result."

        def candidate(candidate_id: str, surface: str, label: str) -> dict:
            start = sentence.index(surface)
            return {
                "candidate_id": candidate_id, "document_id": "doc", "label": label,
                "surface_text": surface, "canonical_name": surface, "canonical_id": "",
                "confidence": .5, "extraction_method": "scibert_svo_argument", "status": "review",
                "source": {
                    "kind": "sentence_span", "sentence_id": "s1", "section_title": "Results",
                    "pages": [1], "start_char": start, "end_char": start + len(surface),
                    "quote": surface, "evidence_quote": sentence,
                    "context_sentences": [{"sentence_id": "s1", "role": "target", "text": sentence, "pages": [1]}],
                },
                "validation": {
                    "outcome": "review", "judge": "high_recall_candidate_generator",
                    "llm_used": False, "reason": "Generated for adjudication.",
                },
            }

        review = [
            candidate("accept_me", "AquaNet", "AISystem"),
            candidate("review_me", "uncertain phrase", "Method"),
            candidate("reject_me", "generic result", "Observation"),
        ]
        review[0]["candidate_metadata"] = {"prior_nodes_same_type": [{
            "node_id": "ai_aquanet", "label": "AISystem", "canonical_name": "AquaNet",
            "aliases": ["AquaNet"], "source_sentence_ids": ["old-sentence"],
        }]}

        class Judge:
            audit_records = [{
                "provider": "test", "model": "deepseek-test", "request_hash": "hash",
                "prompt_version": "test_v1", "llm_used": True,
            }]

            def verify_runtime(self) -> None:
                return None

            def judge(self, records: list[dict]) -> dict[str, dict]:
                decisions = {
                    "accept_me": ("accept", "AISystem", "AquaNet"),
                    "review_me": ("review", "Method", "uncertain phrase"),
                    "reject_me": ("reject", "NONE", "generic result"),
                }
                return {records[0]["sentence_id"]: {
                    "sentence_id": records[0]["sentence_id"],
                    "candidate_judgments": [{
                        "candidate_id": row["candidate_id"], "decision": decisions[row["candidate_id"]][0],
                        "ontology_label": decisions[row["candidate_id"]][1],
                        "canonical_name": decisions[row["candidate_id"]][2],
                        "definition": "Test definition.", "reason": "Test rubric decision.", "confidence": .9,
                    } for row in records[0]["candidates"]],
                    "discoveries": [], "new_class_candidates": [],
                    "_llm_audit": self.audit_records[0],
                }}

        accepted, remaining, rejected, judgments, calls, status = optional_llm_review(
            review, lambda *_: None, node_judge=Judge(),
        )
        self.assertEqual((len(accepted), len(remaining), len(rejected)), (1, 1, 1))
        self.assertEqual({row["decision"] for row in judgments}, {"accept", "review", "reject"})
        self.assertEqual(accepted[0]["canonical_id"], "ai_aquanet")
        self.assertEqual(accepted[0]["validation"]["prior_node_comparison"]["sentence_id"], "s1")
        self.assertEqual(accepted[0]["validation"]["prior_node_comparison"]["matched_source_sentence_ids"], ["old-sentence"])
        self.assertTrue(all(row["rubric_version"] for row in judgments))
        self.assertEqual(calls, Judge.audit_records)
        self.assertTrue(status["used"])

    def test_missing_sentence_id_is_rejected_before_llm(self) -> None:
        candidate = {
            "candidate_id": "untraceable", "document_id": "doc", "label": "ScientificConcept",
            "surface_text": "water quality", "canonical_name": "water quality", "canonical_id": "",
            "confidence": .5, "extraction_method": "scibert_svo_argument", "status": "review",
            "source": {"kind": "sentence_span", "sentence_id": "", "evidence_quote": "water quality"},
            "validation": {"outcome": "review", "judge": "generator", "llm_used": False},
        }

        class Judge:
            audit_records = []

            def verify_runtime(self) -> None:
                raise AssertionError("The LLM must not receive an untraceable candidate.")

        accepted, remaining, rejected, judgments, _, _ = optional_llm_review(
            [candidate], lambda *_: None, node_judge=Judge(),
        )
        self.assertEqual((accepted, remaining), ([], []))
        self.assertEqual(rejected[0]["validation"]["judge"], "sentence_traceability_gate")
        self.assertEqual(judgments[0]["decision"], "reject")
        self.assertEqual(judgments[0]["sentence_id"], "")

    def test_hallucinated_canonical_name_is_rejected(self) -> None:
        sentence = "A citation-supported review was generated."
        candidate = {
            "candidate_id": "candidate1", "document_id": "doc", "label": "ScientificConcept",
            "surface_text": "citation-supported review", "canonical_name": "citation-supported review",
            "canonical_id": "", "confidence": .5, "extraction_method": "scibert_svo_argument",
            "status": "review", "candidate_metadata": {},
            "source": {
                "kind": "sentence_span", "sentence_id": "s1", "section_title": "Abstract", "pages": [1],
                "start_char": 2, "end_char": 27, "quote": "citation-supported review",
                "evidence_quote": sentence, "context_sentences": [],
            },
            "validation": {"outcome": "review", "judge": "generator", "llm_used": False},
        }

        class Judge:
            audit_records = [{"model": "deepseek-test", "request_hash": "h", "prompt_version": "v"}]

            def verify_runtime(self) -> None:
                return None

            def judge(self, records: list[dict]) -> dict[str, dict]:
                return {"s1": {
                    "sentence_id": "s1", "discoveries": [], "new_class_candidates": [],
                    "candidate_judgments": [{
                        "candidate_id": "candidate1", "decision": "accept", "ontology_label": "BiologicalEntity",
                        "canonical_name": "Peroxisome", "definition": "A cellular organelle.",
                        "reason": "Model hallucination.", "confidence": .9,
                    }], "_llm_audit": self.audit_records[0],
                }}

        accepted, remaining, rejected, _, _, _ = optional_llm_review(
            [candidate], lambda *_: None, node_judge=Judge(),
        )
        self.assertEqual((accepted, remaining), ([], []))
        self.assertIn("cannot be traced", rejected[0]["validation"]["reason"])

    def test_trusted_exact_span_removes_redundant_review_candidate(self) -> None:
        source = {"sentence_id": "s1", "start_char": 0, "end_char": 8}
        accepted = [{"document_id": "doc", "surface_text": "WaterRAG", "source": source}]
        review = [
            {"candidate_id": "same", "document_id": "doc", "surface_text": "WaterRAG", "source": source},
            {"candidate_id": "other", "document_id": "doc", "surface_text": "water treatment",
             "source": {"sentence_id": "s1", "start_char": 20, "end_char": 35}},
        ]
        self.assertEqual(
            [row["candidate_id"] for row in remove_redundant_review_candidates(accepted, review)],
            ["other"],
        )

    def test_human_accept_adds_only_sentence_grounded_review_node(self) -> None:
        candidate = {
            "candidate_id": "candidate_condition", "document_id": "doc", "label": "Condition",
            "surface_text": "anaerobic conditions", "canonical_name": "anaerobic conditions",
            "canonical_id": "", "confidence": .6, "extraction_method": "scibert_svo_argument",
            "status": "review",
            "source": {
                "kind": "sentence_span", "sentence_id": "sentence_42", "section_title": "Methods",
                "pages": [2], "start_char": 18, "end_char": 38, "quote": "anaerobic conditions",
                "evidence_quote": "The samples used anaerobic conditions.", "context_sentences": [],
            },
            "validation": {"outcome": "review", "judge": "structured_llm_node_judge", "llm_used": True},
        }
        with TemporaryDirectory() as directory:
            run_dir = Path(directory)
            jsonl_write(run_dir / "review_queue.jsonl", [candidate])
            jsonl_write(run_dir / "node_review_candidates.jsonl", [candidate])
            jsonl_write(run_dir / "mentions.jsonl", [])
            jsonl_write(run_dir / "node_rejections.jsonl", [])
            jsonl_write(run_dir / "candidate_judgments.jsonl", [])
            jsonl_write(run_dir / "similar_nodes_review.jsonl", [])
            json_write(run_dir / "manifest.json", {"counts": {}, "outputs": []})
            result = apply_node_review(
                run_dir, "candidate_condition", "accept",
                ontology_label="Condition", canonical_name="anaerobic conditions",
            )
            mentions = jsonl_read(run_dir / "mentions.jsonl")
            self.assertEqual(result["effective_decision"], "accept")
            self.assertEqual(mentions[0]["source"]["sentence_id"], "sentence_42")
            self.assertEqual(mentions[0]["validation"]["judge"], "human_node_reviewer")
            self.assertEqual(jsonl_read(run_dir / "review_queue.jsonl"), [])

    def test_mas_suffix_does_not_turn_thomas_into_ai_system(self) -> None:
        text = "Thomas, A.W., 2021."
        self.assertFalse(any(row.label == "AISystem" for row in rule_spans(text)))

    def test_generic_the_benchmark_is_not_a_dataset(self) -> None:
        text = "The benchmark was used for comparison."
        self.assertFalse(any(row.label == "Dataset" for row in rule_spans(text)))

    def test_rag_abbreviation_normalizes_to_method(self) -> None:
        spans = rule_spans("The system uses an RAG technique.")
        rag = next(row for row in spans if row.label == "Method")
        self.assertEqual(rag.canonical_name, "retrieval-augmented generation")

    def test_provider_model_repairs_spacing_around_hyphens(self) -> None:
        text = "We evaluated openai/gpt- 4.1-mini and anthropic/claude-3.7-sonnet."
        spans = [row for row in rule_spans(text) if row.label == "Model"]
        names = {row.canonical_name for row in spans}
        self.assertIn("openai/gpt-4.1-mini", names)
        self.assertIn("anthropic/claude-3.7-sonnet", names)
        self.assertNotIn("openai/gpt", names)

    def test_llm_based_adjective_is_not_an_ai_system(self) -> None:
        text = "An LLM-based framework supports engineers."
        spans = select_spans(rule_spans(text), text)
        self.assertFalse(any(row.label == "AISystem" for row in spans))

    def test_definite_framework_after_explicit_introduction_auto_resolves(self) -> None:
        intro = "This study introduces AquaPilot, an AI framework for water operations."
        followup = "The framework combines retrieval-augmented generation with simulation."
        parsed = {
            "document": {"id": "doc_reference"},
            "sections": [{"id": "sec", "title": "Abstract", "paragraphs": [{
                "id": "p1", "sentences": [
                    {"id": "s1", "ordinal": 0, "text": intro, "pages": [1]},
                    {"id": "s2", "ordinal": 1, "text": followup, "pages": [1]},
                ],
            }]}],
        }
        start = intro.index("AquaPilot")
        mentions = [{
            "mention_id": "m1", "document_id": "doc_reference", "label": "AISystem",
            "surface_text": "AquaPilot", "canonical_name": "AquaPilot", "status": "accepted",
            "source": {
                "kind": "sentence_span", "sentence_id": "s1", "start_char": start,
                "end_char": start + len("AquaPilot"), "evidence_quote": intro,
            },
        }]
        resolved, review, ignored = resolve_references(parsed, mentions)
        self.assertEqual((len(resolved), len(review), len(ignored)), (1, 0, 0))
        self.assertEqual(resolved[0]["surface_text"], "The framework")
        self.assertEqual(resolved[0]["canonical_name"], "AquaPilot")

    def test_that_complementizer_is_not_resolved_as_that_model(self) -> None:
        parsed, mentions = self.reference_fixture(
            "The results suggest that model capability affects failures.",
            [("GPT-4.1", "GPT-4.1", "Model")],
        )
        resolved, review, ignored = resolve_references(parsed, mentions)
        all_resolutions = [*resolved, *review, *ignored]
        self.assertFalse(any(row["surface_text"].casefold() == "that model" for row in all_resolutions))

    def test_model_variant_suffix_is_not_swallowed_by_base_lexicon_entry(self) -> None:
        text = "WaterRAG was compared with GPT-4.1 Search."
        lexicon_matches = lexicon_spans(text, Lexicon())
        self.assertFalse(any(row.canonical_name == "GPT-4.1" for row in lexicon_matches))
        rule_matches = rule_spans(text)
        self.assertTrue(any(text[row.start:row.end] == "GPT-4.1 Search" for row in rule_matches))

    def test_equal_looking_statistics_remain_occurrence_scoped(self) -> None:
        mentions = []
        for index in range(2):
            mentions.append({
                "mention_id": f"m{index}", "document_id": "doc1", "label": "Observation",
                "surface_text": "The accuracy was 80%.",
                "canonical_name": "The accuracy was 80%.",
                "canonical_id": f"observation_sentence_{index}",
                "source": {"evidence_quote": "The accuracy was 80%.", "sentence_id": f"s{index}"},
            })
        entities = canonical_entities(mentions)
        self.assertEqual(len(entities), 2)
        self.assertTrue(all(row["identity_scope"] == "occurrence" for row in entities))

    def test_same_named_entity_from_lexicon_and_discovery_is_one_node(self) -> None:
        mentions = [
            {
                "mention_id": "m1", "document_id": "doc", "label": "AISystem",
                "surface_text": "EPANET-Agentic", "canonical_name": "EPANET-Agentic",
                "canonical_id": "ai_epanet_agentic", "extraction_method": "persistent_lexicon",
                "source": {"evidence_quote": "We introduce EPANET-Agentic.", "sentence_id": "s1"},
            },
            {
                "mention_id": "m2", "document_id": "doc", "label": "AISystem",
                "surface_text": "EPANET- Agentic", "canonical_name": "EPANET-Agentic",
                "canonical_id": "entity_discovered", "extraction_method": "document_alias",
                "source": {"evidence_quote": "EPANET- Agentic performed well.", "sentence_id": "s2"},
            },
        ]
        entities = canonical_entities(mentions)
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]["entity_id"], "ai_epanet_agentic")
        self.assertEqual(entities[0]["mention_count"], 2)

    def test_unmerged_entity_keeps_its_persistent_identifier(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir = Path(directory)
            jsonl_write(run_dir / "canonical_entities.jsonl", [{
                "entity_id": "model_gpt41", "label": "Model", "canonical_name": "GPT-4.1",
                "aliases": ["GPT-4.1"], "mention_ids": ["m1"], "mention_count": 1, "evidence": [],
            }])
            jsonl_write(run_dir / "similar_nodes_review.jsonl", [])
            merged = materialize_merges(run_dir)
            self.assertEqual(merged[0]["entity_id"], "model_gpt41")

    @staticmethod
    def reference_fixture(target_text: str, antecedents: list[tuple[str, str, str]]) -> tuple[dict, list[dict]]:
        first_text = "WaterRAG was evaluated with GPT-4.1."
        parsed = {
            "document": {"id": "doc_reference"},
            "sections": [{"id": "sec", "title": "Results", "paragraphs": [{
                "id": "p1", "sentences": [
                    {"id": "s1", "ordinal": 0, "text": first_text, "pages": [1]},
                    {"id": "s2", "ordinal": 1, "text": target_text, "pages": [1]},
                ],
            }]}],
        }
        mentions = []
        for index, (surface, canonical, label) in enumerate(antecedents):
            start = first_text.index(surface)
            mentions.append({
                "mention_id": f"m{index}", "document_id": "doc_reference", "label": label,
                "surface_text": surface, "canonical_name": canonical, "canonical_id": f"c{index}",
                "source": {
                    "kind": "sentence_span", "sentence_id": "s1", "start_char": start,
                    "end_char": start + len(surface), "evidence_quote": first_text, "pages": [1],
                },
            })
        return parsed, mentions

    def test_context_crosses_paragraphs_but_not_sections(self) -> None:
        parsed = {"sections": [{"id": "sec", "title": "Methods", "paragraphs": [
            {"id": "p1", "sentences": [{"id": "s1", "ordinal": 0, "text": "First sentence.", "pages": [1]}]},
            {"id": "p2", "sentences": [
                {"id": "s2", "ordinal": 0, "text": "Second sentence.", "pages": [1]},
                {"id": "s3", "ordinal": 1, "text": "Third sentence.", "pages": [1]},
            ]},
        ]}]}
        rows = flatten_sentences(parsed)
        self.assertEqual([len(row["context"]) for row in rows], [2, 3, 2])
        self.assertEqual([row["role"] for row in rows[1]["context"]], ["previous", "target", "next"])

    def test_distinct_authors_in_one_metadata_field_are_not_deduplicated(self) -> None:
        def author(name: str) -> dict:
            return {
                "document_id": "doc", "label": "Person", "canonical_name": name,
                "surface_text": name,
                "source": {"kind": "document_metadata", "field": "document.authors_text"},
            }

        self.assertNotEqual(mention_dedup_key(author("Mudi Zhai")), mention_dedup_key(author("Qingyun Zeng")))

    def test_vague_model_is_not_a_node_candidate(self) -> None:
        text = "This model improved removal efficiency, while GPT-4 served as the comparator."
        spans = select_spans(rule_spans(text), text)
        surfaces = {text[row.start:row.end] for row in spans}
        self.assertNotIn("This model", surfaces)
        self.assertIn("removal efficiency", surfaces)
        self.assertIn("GPT-4", surfaces)

    def test_unique_compatible_vague_reference_is_resolved_to_existing_node(self) -> None:
        parsed, mentions = self.reference_fixture(
            "This model improved removal efficiency.", [("WaterRAG", "WaterRAG", "AISystem")]
        )
        resolved, review, ignored = resolve_references(parsed, mentions)
        self.assertEqual((len(resolved), len(review), len(ignored)), (1, 0, 0))
        self.assertEqual(resolved[0]["surface_text"], "This model")
        self.assertEqual(resolved[0]["canonical_name"], "WaterRAG")
        self.assertEqual(resolved[0]["reference_resolution"]["antecedent_sentence_id"], "s1")
        entity = canonical_entities([*mentions, *resolved])[0]
        self.assertNotIn("This model", entity["aliases"])

    def test_contextual_llm_same_maps_vague_mention_to_specific_node(self) -> None:
        parsed, mentions = self.reference_fixture(
            "The model improved removal efficiency.", [("WaterRAG", "WaterRAG", "AISystem")]
        )

        class SameJudge:
            def __init__(self) -> None:
                self.audit_records = [{
                    "provider": "test", "model": "context-model", "request_hash": "hash",
                    "prompt_version": "test_v1", "llm_used": True,
                }]
                self.pairs = []

            def verify_runtime(self) -> None:
                return None

            def judge(self, pairs: list[dict]) -> dict[str, dict]:
                self.pairs = pairs
                return {
                    pair["pair_id"]: {
                        "pair_id": pair["pair_id"], "decision": "SAME", "confidence": .96,
                        "reason": "The second sentence continues describing WaterRAG.",
                        "_llm_audit": self.audit_records[0],
                    }
                    for pair in pairs
                }

        judge = SameJudge()
        resolved, review, ignored, calls, status = optional_llm_reference_resolution(
            parsed, mentions, lambda *_: None, reference_judge=judge
        )
        self.assertEqual((len(resolved), len(review), len(ignored)), (1, 0, 0))
        self.assertEqual(resolved[0]["surface_text"], "The model")
        self.assertEqual(resolved[0]["canonical_name"], "WaterRAG")
        self.assertEqual(resolved[0]["reference_resolution"]["status"], "llm_resolved")
        self.assertTrue(resolved[0]["validation"]["llm_used"])
        self.assertEqual(judge.pairs[0]["reference"]["context_sentences"][-1]["text"], "The model improved removal efficiency.")
        self.assertEqual(judge.pairs[0]["candidate"]["evidence_quote"], "WaterRAG was evaluated with GPT-4.1.")
        self.assertEqual(calls, judge.audit_records)
        self.assertTrue(status["used"])

    def test_contextual_llm_uncertain_keeps_pair_for_human_review(self) -> None:
        parsed, mentions = self.reference_fixture(
            "This model improved removal efficiency.",
            [("WaterRAG", "WaterRAG", "AISystem"), ("GPT-4.1", "GPT-4.1", "Model")],
        )

        class UncertainJudge:
            audit_records: list[dict] = []

            def verify_runtime(self) -> None:
                return None

            def judge(self, pairs: list[dict]) -> dict[str, dict]:
                return {
                    pair["pair_id"]: {
                        "pair_id": pair["pair_id"], "decision": "UNCERTAIN", "confidence": .5,
                        "reason": "Both systems occur in the preceding context.", "_llm_audit": {},
                    }
                    for pair in pairs
                }

        resolved, review, ignored, _, status = optional_llm_reference_resolution(
            parsed, mentions, lambda *_: None, reference_judge=UncertainJudge()
        )
        self.assertEqual((len(resolved), len(review), len(ignored)), (0, 1, 0))
        self.assertEqual(len(review[0]["llm_pair_judgments"]), 2)
        self.assertTrue(status["used"])

    def test_ambiguous_vague_reference_requires_human_review(self) -> None:
        parsed, mentions = self.reference_fixture(
            "This model improved removal efficiency.",
            [("WaterRAG", "WaterRAG", "AISystem"), ("GPT-4.1", "GPT-4.1", "Model")],
        )
        resolved, review, ignored = resolve_references(parsed, mentions)
        self.assertEqual((len(resolved), len(review), len(ignored)), (0, 1, 0))
        self.assertEqual({row["canonical_name"] for row in review[0]["candidate_targets"]}, {"WaterRAG", "GPT-4.1"})

    def test_weak_definite_reference_is_not_automatically_resolved(self) -> None:
        parsed, mentions = self.reference_fixture(
            "The model improved removal efficiency.", [("WaterRAG", "WaterRAG", "AISystem")]
        )
        resolved, review, ignored = resolve_references(parsed, mentions)
        self.assertEqual((len(resolved), len(review), len(ignored)), (0, 1, 0))

    def test_agentic_system_reference_prefers_ai_system_over_nearby_tool(self) -> None:
        sentences = [
            ("s1", "We introduce EPANET-Agentic."),
            ("s2", "EPANET provides hydraulic simulation."),
            ("s3", "This system features an Orchestrator-centred multi-agent architecture."),
        ]
        parsed = {
            "document": {"id": "doc_reference"},
            "sections": [{"id": "sec", "title": "Methods", "paragraphs": [{
                "id": "p1", "sentences": [
                    {"id": sentence_id, "ordinal": index, "text": text, "pages": [1]}
                    for index, (sentence_id, text) in enumerate(sentences)
                ],
            }]}],
        }
        mentions = []
        for mention_id, sentence_id, label, surface in (
            ("ai", "s1", "AISystem", "EPANET-Agentic"),
            ("tool", "s2", "Tool", "EPANET"),
        ):
            text = dict(sentences)[sentence_id]
            start = text.index(surface)
            mentions.append({
                "mention_id": mention_id, "document_id": "doc_reference", "label": label,
                "surface_text": surface, "canonical_name": surface, "status": "accepted",
                "source": {
                    "kind": "sentence_span", "sentence_id": sentence_id,
                    "start_char": start, "end_char": start + len(surface), "evidence_quote": text,
                },
            })
        resolved, review, ignored = resolve_references(parsed, mentions)
        target = next(row for row in resolved if row["surface_text"].casefold() == "this system")
        self.assertEqual(target["label"], "AISystem")
        self.assertEqual(target["canonical_name"], "EPANET-Agentic")
        self.assertEqual((len(review), len(ignored)), (0, 0))

    def test_unresolvable_vague_reference_is_ignored(self) -> None:
        parsed, _ = self.reference_fixture("This model improved removal efficiency.", [])
        resolved, review, ignored = resolve_references(parsed, [])
        self.assertEqual((len(resolved), len(review), len(ignored)), (0, 0, 1))

    def test_human_reference_choice_adds_traceable_mention_without_alias_pollution(self) -> None:
        parsed, mentions = self.reference_fixture(
            "The model improved removal efficiency.", [("WaterRAG", "WaterRAG", "AISystem")]
        )
        _, review, _ = resolve_references(parsed, mentions)
        with TemporaryDirectory() as directory:
            run_dir = Path(directory)
            jsonl_write(run_dir / "mentions.jsonl", mentions)
            jsonl_write(run_dir / "reference_resolution_review.jsonl", review)
            jsonl_write(run_dir / "similar_nodes_review.jsonl", [])
            json_write(run_dir / "manifest.json", {"counts": {}, "outputs": []})
            result = apply_reference_resolution(
                run_dir, review[0]["resolution_id"], "resolve", review[0]["candidate_targets"][0]["target_mention_id"]
            )
            saved_mentions = jsonl_read(run_dir / "mentions.jsonl")
            self.assertEqual(len(saved_mentions), 2)
            self.assertEqual(saved_mentions[-1]["reference_resolution"]["status"], "human_resolved")
            self.assertNotIn("The model", result["entities"][0]["aliases"])

    def test_different_explicit_versions_are_not_duplicate_candidates(self) -> None:
        left = {"canonical_name":"GPT-4", "aliases":["GPT-4"]}
        right = {"canonical_name":"GPT-4.1", "aliases":["GPT-4.1"]}
        score, reasons, features = similarity_features(left, right)
        self.assertEqual(score, 0.0)
        self.assertTrue(features["version_conflict"])
        self.assertIn("different explicit model or software versions", reasons)

    def test_lead_verb_is_not_a_contaminant(self) -> None:
        text = "These choices lead to lower energy consumption."
        spans = rule_spans(text)
        self.assertFalse(any(row.label == "Contaminant" for row in spans))

    def test_epanet_is_a_tool_not_a_standard(self) -> None:
        text = "EPANET is an open-source hydraulic simulation tool."
        spans = rule_spans(text)
        self.assertTrue(any(row.label == "Tool" and text[row.start:row.end] == "EPANET" for row in spans))
        self.assertFalse(any(row.label == "Standard" for row in spans))

    def test_possessive_system_names_do_not_create_fragment_nodes(self) -> None:
        text = "Microsoft's Magentic-One differs from Microsoft's GraphRAG."
        values = {text[row.start:row.end] for row in rule_spans(text) if row.label == "AISystem"}
        self.assertNotIn("s Magentic", values)
        self.assertNotIn("s GraphRAG", values)
        self.assertFalse(any(value.startswith("s ") for value in values))

    def test_named_agents_and_models_are_specific_candidates(self) -> None:
        text = "TaskExecutor is powered by DeepSeek-V3, while DataAnalyzer uses Qwen-VL-Max."
        spans = rule_spans(text)
        values = {(row.label, row.canonical_name or text[row.start:row.end]) for row in spans}
        self.assertIn(("Agent", "TaskExecutor"), values)
        self.assertIn(("Agent", "DataAnalyzer"), values)
        self.assertIn(("Model", "DeepSeek-V3"), values)
        self.assertIn(("Model", "Qwen-VL-Max"), values)

    def test_challenge_inflections_share_one_canonical_name(self) -> None:
        text = "Hallucinations and hallucination risk can cause reasoning errors."
        spans = [row for row in rule_spans(text) if row.label == "Challenge"]
        names = [row.canonical_name for row in spans]
        self.assertEqual(names.count("hallucination"), 2)
        self.assertIn("reasoning error", names)


if __name__ == "__main__":
    unittest.main()
