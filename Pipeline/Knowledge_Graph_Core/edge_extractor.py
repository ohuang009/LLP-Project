from __future__ import annotations

from .models import stable_id
from .ontology import Ontology


ACCEPTED_ATTRIBUTIONS = {"current_study", "source_statement", "author_interpretation"}


class EdgeExtractor:
    """LLM-only semantic relationship extraction over three-sentence windows.

    The model may select only nodes that already occur in ``mentions``. A proposal
    with an unseen endpoint is never materialized as an edge; the endpoint is
    retained in ``unknown_nodes`` for a later user-defined workflow.
    """

    def __init__(
        self,
        ontology: Ontology,
        relationship_judge=None,
        config: dict | None = None,
    ) -> None:
        self.ontology = ontology
        self.relationship_judge = relationship_judge
        relationship_config = (config or {}).get("relationship_extraction", {})
        self.minimum_confidence = float(
            relationship_config.get("minimum_accept_confidence", 0.75)
        )
        self.candidates: list[dict] = []
        self.unknown_nodes: list[dict] = []
        self.llm_calls: list[dict] = []

    def extract(self, parsed: dict, mentions: list[dict]) -> tuple[list[dict], list[dict]]:
        self.candidates = []
        self.unknown_nodes = []
        self.llm_calls = []
        assertions: list[dict] = []
        rejections: list[dict] = []

        accepted_mentions = [
            row for row in mentions
            if row.get("status", "accepted") == "accepted" and row.get("mention_id")
        ]
        mention_index = {row["mention_id"]: row for row in accepted_mentions}
        publication = next(
            (row for row in accepted_mentions if row.get("label") == "Publication"), None
        )

        # Authorship is parsed document metadata rather than a semantic inference.
        if publication is not None:
            for person in (row for row in accepted_mentions if row.get("label") == "Person"):
                if self.ontology.valid_relationship("AUTHORED_BY", "Publication", "Person"):
                    assertions.append(
                        self._metadata_assertion(
                            "AUTHORED_BY",
                            publication,
                            person,
                            parsed.get("document", {}).get("authors_text", ""),
                        )
                    )

        sentence_index, windows = self._context_windows(parsed)
        if not windows:
            return assertions, rejections

        if self.relationship_judge is None:
            rejections.append({
                "rejection_id": stable_id(
                    "reject", parsed.get("document", {}).get("id", ""),
                    "relationship_llm_unavailable",
                ),
                "stage": "relationship_extraction",
                "reason": "relationship_llm_unavailable_no_semantic_fallback",
                "sentence_count": len(windows),
            })
            return sorted(assertions, key=lambda row: row["assertion_id"]), rejections

        node_catalog = self._node_catalog(accepted_mentions)
        audit_start = len(self.relationship_judge.audit_records)
        results = self.relationship_judge.extract(windows, node_catalog)
        self.llm_calls = list(self.relationship_judge.audit_records[audit_start:])

        unknown_index: dict[tuple, dict] = {}
        for window in windows:
            target_sentence_id = window["target_sentence_id"]
            result = results.get(target_sentence_id, {"proposals": []})
            audit = result.get("_llm_audit", {})
            for proposal_number, proposal in enumerate(result.get("proposals", []), 1):
                candidate = self._candidate_from_proposal(
                    parsed,
                    window,
                    proposal,
                    proposal_number,
                    mention_index,
                    sentence_index,
                    audit,
                )
                self.candidates.append(candidate)

                missing_roles = []
                if not candidate["subject_mention_id"]:
                    missing_roles.append("head")
                if not candidate["object_mention_id"]:
                    missing_roles.append("tail")
                if candidate["status"] == "unresolved":
                    for role in missing_roles:
                        unknown = self._unknown_node(candidate, proposal, role)
                        key = (
                            unknown["target_sentence_id"], unknown["role"],
                            unknown["surface_text"].casefold(), unknown["predicate"],
                        )
                        unknown_index[key] = unknown

                if candidate["status"] == "accepted":
                    assertions.append(self._assertion_from_candidate(candidate))
                else:
                    rejections.append({
                        "rejection_id": stable_id(
                            "reject", candidate["candidate_id"], candidate["status"]
                        ),
                        "stage": "relationship_acceptance",
                        "reason": candidate["rejection_reason"],
                        "candidate_id": candidate["candidate_id"],
                        "predicate": candidate["allowed_predicates"][0],
                        "target_sentence_id": target_sentence_id,
                        "evidence_sentence_ids": candidate["evidence_sentence_ids"],
                        "gates": candidate["gates"],
                    })

        self.candidates.sort(key=lambda row: row["candidate_id"])
        self.unknown_nodes = sorted(
            unknown_index.values(), key=lambda row: row["unknown_node_id"]
        )

        unique: dict[tuple, dict] = {}
        for assertion in assertions:
            key = (
                assertion["predicate"], assertion["subject_mention_id"],
                assertion["object_mention_id"],
                tuple(assertion.get("evidence_sentence_ids", [])),
            )
            prior = unique.get(key)
            if prior is None or assertion.get("confidence", 0) > prior.get("confidence", 0):
                unique[key] = assertion
        return sorted(unique.values(), key=lambda row: row["assertion_id"]), rejections

    @staticmethod
    def _context_windows(parsed: dict) -> tuple[dict[str, dict], list[dict]]:
        sentence_index: dict[str, dict] = {}
        windows: list[dict] = []
        for section in parsed.get("sections", []):
            section_rows: list[tuple[dict, dict]] = []
            for paragraph in section.get("paragraphs", []):
                for sentence in paragraph.get("sentences", []):
                    sentence_index[sentence["id"]] = sentence
                    section_rows.append((sentence, paragraph))

            for index, (sentence, paragraph) in enumerate(section_rows):
                start = max(0, index - 1)
                stop = min(len(section_rows), index + 2)
                context = []
                for context_index in range(start, stop):
                    context_sentence, context_paragraph = section_rows[context_index]
                    context.append({
                        "sentence_id": context_sentence["id"],
                        "role": (
                            "target" if context_index == index
                            else "previous" if context_index < index
                            else "next"
                        ),
                        "text": context_sentence["text"],
                        "paragraph_id": context_paragraph.get("id", ""),
                        "pages": list(context_sentence.get("pages", [])),
                    })
                windows.append({
                    "target_sentence_id": sentence["id"],
                    "section_id": section.get("id", ""),
                    "section_title": section.get("title", ""),
                    "paragraph_id": paragraph.get("id", ""),
                    "context_sentences": context,
                })
        return sentence_index, windows

    @staticmethod
    def _node_catalog(mentions: list[dict]) -> list[dict]:
        return [{
            "node_id": row["mention_id"],
            "ontology_class": row.get("label", ""),
            "canonical_name": row.get("canonical_name") or row.get("surface_text", ""),
            "surface_text": row.get("surface_text", ""),
        } for row in mentions]

    def _candidate_from_proposal(
        self,
        parsed: dict,
        window: dict,
        proposal: dict,
        proposal_number: int,
        mention_index: dict[str, dict],
        sentence_index: dict[str, dict],
        audit: dict,
    ) -> dict:
        predicate = str(proposal.get("predicate", ""))
        head_id = str(proposal.get("head_node_id") or "")
        tail_id = str(proposal.get("tail_node_id") or "")
        head = mention_index.get(head_id)
        tail = mention_index.get(tail_id)
        endpoint_repairs: dict[str, str] = {}
        if head is None:
            head = self._unique_exact_node_match(
                proposal.get("head_text", ""),
                proposal.get("head_proposed_label", ""),
                mention_index,
            )
            if head is not None:
                head_id = head["mention_id"]
                endpoint_repairs["head"] = "unique_exact_catalog_name_or_surface"
        if tail is None:
            tail = self._unique_exact_node_match(
                proposal.get("tail_text", ""),
                proposal.get("tail_proposed_label", ""),
                mention_index,
            )
            if tail is not None:
                tail_id = tail["mention_id"]
                endpoint_repairs["tail"] = "unique_exact_catalog_name_or_surface"
        trigger_text = str(proposal.get("predicate_trigger_text") or "")
        trigger_sentence_id = str(proposal.get("predicate_trigger_sentence_id") or "")
        trigger_sentence = sentence_index.get(trigger_sentence_id)
        window_sentence_ids = [row["sentence_id"] for row in window["context_sentences"]]
        trigger_exact = bool(
            trigger_text
            and trigger_sentence_id in window_sentence_ids
            and trigger_sentence
            and trigger_text in trigger_sentence.get("text", "")
        )
        trigger_start = (
            trigger_sentence["text"].find(trigger_text) if trigger_exact else None
        )
        trigger_end = trigger_start + len(trigger_text) if trigger_start is not None else None
        try:
            confidence = float(proposal.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0.0
        decision = str(proposal.get("decision", "reject"))
        attribution = str(proposal.get("attribution", "unknown"))
        context_texts = [row["text"] for row in window["context_sentences"]]
        head_unknown_exact = bool(
            head or (
                proposal.get("head_text")
                and any(str(proposal["head_text"]) in text for text in context_texts)
            )
        )
        tail_unknown_exact = bool(
            tail or (
                proposal.get("tail_text")
                and any(str(proposal["tail_text"]) in text for text in context_texts)
            )
        )

        gates = {
            "llm_confirmed": decision == "accept",
            "endpoints_resolved": bool(head and tail),
            "nodes_previously_accepted": bool(head and tail),
            "unseen_endpoint_text_exact": head_unknown_exact and tail_unknown_exact,
            "not_self_relationship": bool(head and tail and head_id != tail_id),
            "predicate_permitted": predicate in self.ontology.relationships,
            "domain_range": bool(
                head and tail and self.ontology.valid_relationship(
                    predicate, head.get("label", ""), tail.get("label", "")
                )
            ),
            "predicate_trigger_exact": trigger_exact,
            "not_negated": not bool(proposal.get("negated", False)),
            "not_modal": not bool(proposal.get("modal", False)),
            "not_hypothetical": not bool(proposal.get("hypothetical", False)),
            "attribution_accepted": attribution in ACCEPTED_ATTRIBUTIONS,
        }

        if decision != "accept":
            status, rejection_reason = "rejected", "llm_judged_relationship_absent"
        elif (not head or not tail) and head_unknown_exact and tail_unknown_exact:
            status, rejection_reason = "unresolved", "llm_selected_unseen_endpoint"
        elif confidence < self.minimum_confidence:
            status, rejection_reason = "review", "llm_confidence_below_acceptance_threshold"
        elif not all(gates.values()):
            status, rejection_reason = "rejected", "relationship_acceptance_gate_failed"
        else:
            status, rejection_reason = "accepted", ""

        evidence_sentence_ids = list(window_sentence_ids)
        context_quotes = context_texts
        candidate_id = stable_id(
            "relationship-window", parsed.get("document", {}).get("id", ""),
            window["target_sentence_id"], proposal_number, predicate,
            head_id or proposal.get("head_text", ""),
            tail_id or proposal.get("tail_text", ""),
        )
        source_span = (head or {}).get("source", {})
        target_span = (tail or {}).get("source", {})
        return {
            "candidate_id": candidate_id,
            "document_id": parsed.get("document", {}).get("id", ""),
            "target_sentence_id": window["target_sentence_id"],
            "sentence_id": trigger_sentence_id,
            "paragraph_id": window["paragraph_id"],
            "section_id": window["section_id"],
            "section_title": window["section_title"],
            "pages": sorted({
                page for row in window["context_sentences"] for page in row.get("pages", [])
            }),
            "subject_mention_id": head_id if head else "",
            "subject_label": head.get("label", "") if head else proposal.get("head_proposed_label", ""),
            "subject_text": head.get("surface_text", "") if head else proposal.get("head_text", ""),
            "subject_start_char": source_span.get("start_char"),
            "subject_end_char": source_span.get("end_char"),
            "object_mention_id": tail_id if tail else "",
            "object_label": tail.get("label", "") if tail else proposal.get("tail_proposed_label", ""),
            "object_text": tail.get("surface_text", "") if tail else proposal.get("tail_text", ""),
            "object_start_char": target_span.get("start_char"),
            "object_end_char": target_span.get("end_char"),
            "allowed_predicates": [predicate],
            "predicate_trigger_text": trigger_text,
            "predicate_trigger_sentence_id": trigger_sentence_id,
            "predicate_trigger_start_char": trigger_start,
            "predicate_trigger_end_char": trigger_end,
            "trigger_text": trigger_text,
            "evidence_sentence_ids": evidence_sentence_ids,
            "evidence_quote": " ".join(context_quotes),
            "context_quotes": context_quotes,
            "context_policy": "target sentence plus previous and next sentence within the same section when available",
            "attribution": attribution,
            "flags": {
                "negated": bool(proposal.get("negated", False)),
                "modal": bool(proposal.get("modal", False)),
                "hypothetical": bool(proposal.get("hypothetical", False)),
                "comparative": bool(proposal.get("comparative", False)),
                "causal": bool(proposal.get("causal", False)),
            },
            "channel": "llm_context_window_v1",
            "requires_llm": True,
            "status": status,
            "rejection_reason": rejection_reason,
            "decision_reason": str(proposal.get("reason") or "No model reason supplied."),
            "confidence": confidence,
            "endpoint_repairs": endpoint_repairs,
            "gates": gates,
            "llm_decision": {
                "decision": decision,
                "predicate": predicate,
                "reason": str(proposal.get("reason") or "No model reason supplied."),
                "confidence": confidence,
                "llm_used": True,
                "provider": audit.get("provider", ""),
                "model": audit.get("model", ""),
                "response_id": audit.get("response_id", ""),
                "request_hash": audit.get("request_hash", ""),
                "prompt_version": audit.get("prompt_version", ""),
                "prompt_sha256": audit.get("prompt_sha256", ""),
            },
        }

    @staticmethod
    def _unique_exact_node_match(
        text: object,
        proposed_label: object,
        mention_index: dict[str, dict],
    ) -> dict | None:
        normalized = str(text or "").strip().casefold()
        if not normalized:
            return None
        label = str(proposed_label or "")
        matches = [
            row for row in mention_index.values()
            if (not label or label == "UNKNOWN" or row.get("label") == label)
            and normalized in {
                str(row.get("surface_text") or "").strip().casefold(),
                str(row.get("canonical_name") or "").strip().casefold(),
            }
        ]
        unique = {row["mention_id"]: row for row in matches}
        return next(iter(unique.values())) if len(unique) == 1 else None

    @staticmethod
    def _unknown_node(candidate: dict, proposal: dict, role: str) -> dict:
        prefix = "head" if role == "head" else "tail"
        surface = str(proposal.get(f"{prefix}_text") or "")
        proposed_label = str(proposal.get(f"{prefix}_proposed_label") or "UNKNOWN")
        return {
            "unknown_node_id": stable_id(
                "relationship-unknown-node", candidate["document_id"],
                candidate["target_sentence_id"], role, surface, candidate["allowed_predicates"][0],
            ),
            "document_id": candidate["document_id"],
            "target_sentence_id": candidate["target_sentence_id"],
            "role": role,
            "surface_text": surface,
            "proposed_ontology_label": proposed_label,
            "predicate": candidate["allowed_predicates"][0],
            "known_counterpart_node_id": (
                candidate["object_mention_id"] if role == "head"
                else candidate["subject_mention_id"]
            ),
            "predicate_trigger_text": candidate["predicate_trigger_text"],
            "predicate_trigger_sentence_id": candidate["predicate_trigger_sentence_id"],
            "evidence_sentence_ids": candidate["evidence_sentence_ids"],
            "evidence_quote": candidate["evidence_quote"],
            "reason": candidate["decision_reason"],
            "confidence": candidate["confidence"],
            "status": "pending_user_defined_node_workflow",
        }

    @staticmethod
    def _assertion_from_candidate(candidate: dict) -> dict:
        predicate = candidate["allowed_predicates"][0]
        assertion_id = stable_id(
            "assertion", predicate, candidate["subject_mention_id"],
            candidate["object_mention_id"], ",".join(candidate["evidence_sentence_ids"]),
        )
        return {
            "assertion_id": assertion_id,
            "candidate_id": candidate["candidate_id"],
            "document_id": candidate["document_id"],
            "predicate": predicate,
            "subject_mention_id": candidate["subject_mention_id"],
            "subject_label": candidate["subject_label"],
            "subject_text": candidate["subject_text"],
            "object_mention_id": candidate["object_mention_id"],
            "object_label": candidate["object_label"],
            "object_text": candidate["object_text"],
            "confidence": round(min(candidate["confidence"], 0.99), 6),
            "confidence_band": "high" if candidate["confidence"] >= 0.9 else "medium",
            "status": "accepted",
            "scope": "context_window",
            "target_sentence_id": candidate["target_sentence_id"],
            "predicate_text": candidate["predicate_trigger_text"],
            "predicate_trigger_text": candidate["predicate_trigger_text"],
            "predicate_trigger_sentence_id": candidate["predicate_trigger_sentence_id"],
            "predicate_trigger_start_char": candidate["predicate_trigger_start_char"],
            "predicate_trigger_end_char": candidate["predicate_trigger_end_char"],
            **candidate["flags"],
            "attribution": candidate["attribution"],
            "ontology_valid": True,
            "evidence_sentence_ids": candidate["evidence_sentence_ids"],
            "evidence_quote": candidate["evidence_quote"],
            "paragraph_id": candidate["paragraph_id"],
            "section_id": candidate["section_id"],
            "pages": candidate["pages"],
            "subject_start_char": candidate["subject_start_char"],
            "subject_end_char": candidate["subject_end_char"],
            "object_start_char": candidate["object_start_char"],
            "object_end_char": candidate["object_end_char"],
            "context_quotes": candidate["context_quotes"],
            "context_policy": candidate["context_policy"],
            "extraction_method": "llm_context_window_v1",
            "extraction_channel": "llm_context_window_v1",
            "relationship_decision": candidate["llm_decision"],
            "gates": candidate["gates"],
        }

    @staticmethod
    def _metadata_assertion(predicate: str, source: dict, target: dict, quote: str) -> dict:
        return {
            "assertion_id": stable_id(
                "assertion", predicate, source["mention_id"], target["mention_id"], "metadata"
            ),
            "document_id": source["document_id"],
            "predicate": predicate,
            "subject_mention_id": source["mention_id"],
            "subject_label": source["label"],
            "subject_text": source["surface_text"],
            "object_mention_id": target["mention_id"],
            "object_label": target["label"],
            "object_text": target["surface_text"],
            "confidence": 0.99,
            "confidence_band": "high",
            "status": "accepted",
            "scope": "document_metadata",
            "target_sentence_id": None,
            "predicate_text": None,
            "predicate_trigger_text": None,
            "predicate_trigger_sentence_id": None,
            "predicate_trigger_start_char": None,
            "predicate_trigger_end_char": None,
            "negated": False,
            "modal": False,
            "hypothetical": False,
            "comparative": False,
            "causal": False,
            "attribution": "current_study",
            "ontology_valid": True,
            "evidence_sentence_ids": [],
            "evidence_quote": quote,
            "paragraph_id": None,
            "section_id": None,
            "pages": [],
            "subject_start_char": None,
            "subject_end_char": None,
            "object_start_char": None,
            "object_end_char": None,
            "context_quotes": [],
            "context_policy": "document metadata",
            "extraction_method": "structural_metadata",
            "extraction_channel": "structural_metadata",
            "relationship_decision": {
                "decision": "accept", "predicate": predicate,
                "llm_used": False, "reason": "Parsed document authorship metadata.",
            },
            "gates": {
                "llm_confirmed": True,
                "endpoints_resolved": True,
                "nodes_previously_accepted": True,
                "unseen_endpoint_text_exact": True,
                "not_self_relationship": True,
                "predicate_permitted": True,
                "domain_range": True,
                "predicate_trigger_exact": True,
                "not_negated": True,
                "not_modal": True,
                "not_hypothetical": True,
                "attribution_accepted": True,
            },
        }
