from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .llm_node_judge import LLMNodeJudgeError, MandatoryLLMNodeJudge
from .ontology import Ontology


PROMPT_VERSION = "llm_context_relationship_extractor_v1"

SYSTEM_PROMPT = """You extract and judge scientific knowledge-graph relationships.

You receive:
- the complete catalog of nodes already accepted for the current paper;
- the permitted ontology relationships and their domain/range definitions; and
- windows containing a target sentence, its previous sentence, and its next sentence when available.

For each window, identify possible directed relationships and judge whether the supplied text actually asserts them.

Hard rules:
1. A materializable relationship must use head_node_id and tail_node_id values from accepted_nodes. Never invent a node ID.
2. If an endpoint matches an accepted node by name or surface text, you MUST copy that node's node_id. Use an empty node ID only when no accepted node matches the endpoint. For an unseen endpoint, copy its shortest exact surface text into head_text or tail_text; the proposal will be audited, not materialized.
3. Endpoints are optional. Return an empty proposals list when there is no suitable relationship. Never force a head, tail, or relationship.
4. Select only a supplied ontology predicate whose domain and range match the endpoint classes.
5. Act as the final semantic judge. Use decision=accept only when the relationship is clearly asserted by the window. Use decision=reject for mere co-mention, ambiguity, negation, speculation, hypotheticals, or unsupported directionality.
6. The target sentence is focal. Previous and next sentences are context for coreference and multi-sentence facts; do not extract unrelated facts that occur only in a neighboring sentence.
7. predicate_trigger_text must be the shortest exact, non-empty phrase that expresses the predicate, such as "uses", "trained on", or "outperformed". It must be a substring of one supplied context sentence, and predicate_trigger_sentence_id must identify that sentence.
8. Mark negated, modal, hypothetical, comparative, causal, and attribution fields accurately. Prior-work attribution is cited_study, not current_study.
9. Do not alter source text and do not create facts not present in the supplied sentences.
10. Return exactly one windows object property for every supplied target_sentence_id and no others.
"""


class MandatoryLLMRelationshipJudge(MandatoryLLMNodeJudge):
    """Generate endpoints and predicates, then judge them in one structured call."""

    def __init__(
        self,
        ontology: Ontology,
        config: dict,
        cache_dir: Path,
        *,
        api_key: str | None = None,
        transport=None,
    ) -> None:
        effective = deepcopy(config)
        relationship_config = deepcopy(
            config.get("llm_relationship_judge") or config["llm_node_judge"]
        )
        relationship_config["required"] = True
        effective["llm_node_judge"] = relationship_config
        super().__init__(ontology, effective, cache_dir, api_key=api_key, transport=transport)
        self.batch_size = int(relationship_config.get("batch_windows", 1))
        self.prompt_hash = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()

    def extract(self, windows: list[dict], accepted_nodes: list[dict]) -> dict[str, dict]:
        results: dict[str, dict] = {}
        for offset in range(0, len(windows), self.batch_size):
            batch_number = offset // self.batch_size + 1
            batch = windows[offset:offset + self.batch_size]
            try:
                parsed, audit = self._extract_batch(batch, accepted_nodes, batch_number)
            except Exception as exc:
                parsed = {"windows": {
                    row["target_sentence_id"]: {"proposals": []} for row in batch
                }}
                audit = self._fallback_audit(batch, accepted_nodes, batch_number, exc)
            self.audit_records.append(audit)
            for target_sentence_id, row in parsed["windows"].items():
                row["target_sentence_id"] = target_sentence_id
                row["_llm_audit"] = audit
                results[target_sentence_id] = row
        return results

    def _extract_batch(
        self,
        batch: list[dict],
        accepted_nodes: list[dict],
        batch_number: int,
    ) -> tuple[dict, dict]:
        eligible_predicates = self._eligible_predicates(accepted_nodes)
        relationship_reference = [{
            "predicate": name,
            "definition": spec.get("definition", ""),
            "domain": spec.get("domain", []),
            "range": spec.get("range", []),
        } for name, spec in self.ontology.relationships.items() if name in eligible_predicates]
        payload = {
            "accepted_nodes": accepted_nodes,
            "ontology_relationships": relationship_reference,
            "sentence_windows": batch,
        }
        schema = self._schema(batch, accepted_nodes, eligible_predicates)
        request_payload = self._request_payload(payload, schema)
        request_hash = hashlib.sha256(
            json.dumps(request_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        cache_path = self.cache_dir / f"{request_hash}.json"
        cached = bool(self.config.get("cache_responses")) and cache_path.is_file()
        if cached:
            response_payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if not self._response_complete(response_payload):
                cached = False
        if not cached:
            response_payload = self._send(request_payload)
            if self.config.get("cache_responses"):
                cache_path.write_text(
                    json.dumps(response_payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )

        try:
            raw = json.loads(self._output_text(response_payload))
        except json.JSONDecodeError as exc:
            raise LLMNodeJudgeError(f"Relationship LLM returned invalid JSON: {exc}") from exc
        parsed, issues = self._normalize_result(raw, batch)
        audit = {
            "batch_number": batch_number,
            "provider": self.provider,
            "model": self.model,
            "response_id": self._response_id(response_payload),
            "request_hash": request_hash,
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": self.prompt_hash,
            "target_sentence_ids": [row["target_sentence_id"] for row in batch],
            "window_count": len(batch),
            "accepted_node_count": len(accepted_nodes),
            "proposal_count": sum(len(row["proposals"]) for row in parsed["windows"].values()),
            "cached": cached,
            "usage": self._usage(response_payload),
            "runtime_created_at": response_payload.get("created_at", ""),
            "runtime_done_reason": response_payload.get("done_reason", ""),
            "llm_used": True,
            "status": "complete" if not issues else "complete_with_safe_repairs",
            "normalization_issues": issues,
            "safe_repair_count": len(issues),
        }
        return parsed, audit

    def _request_payload(self, user_payload: dict, schema: dict) -> dict:
        if self.provider == "ollama":
            return {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps({
                        **user_payload, "required_output_json_schema": schema,
                    }, ensure_ascii=False)},
                ],
                "stream": False,
                "format": schema,
                "options": {
                    "temperature": float(self.config.get("temperature", 0)),
                    "num_ctx": int(self.config.get("context_window", 32768)),
                    "seed": int(self.config.get("seed", 42)),
                    "num_predict": int(self.config.get("max_output_tokens", 8192)),
                },
                "keep_alive": str(self.config.get("keep_alive", "30m")),
            }
        return {
            "model": self.model,
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "reasoning": {"effort": self.reasoning_effort},
            "text": {"format": {
                "type": "json_schema",
                "name": "scientific_context_relationships",
                "strict": True,
                "schema": schema,
            }},
            "store": False,
        }

    @staticmethod
    def _normalize_result(parsed: dict, batch: list[dict]) -> tuple[dict, list[dict]]:
        expected = {row["target_sentence_id"] for row in batch}
        raw_rows = parsed.get("windows", {}) if isinstance(parsed, dict) else {}
        if not isinstance(raw_rows, dict):
            raw_rows = {}
        accepted: dict[str, dict] = {}
        issues: list[dict] = []
        for target_id, row in raw_rows.items():
            if not isinstance(row, dict):
                issues.append({"kind": "non_object_window", "value": target_id})
                continue
            if target_id not in expected:
                issues.append({"kind": "unknown_target_sentence_id", "value": target_id})
                continue
            proposals = row.get("proposals", [])
            if not isinstance(proposals, list):
                issues.append({"kind": "non_array_proposals", "value": target_id})
                proposals = []
            accepted[target_id] = {
                "proposals": [proposal for proposal in proposals if isinstance(proposal, dict)],
            }
        for row in batch:
            target_id = row["target_sentence_id"]
            if target_id not in accepted:
                issues.append({"kind": "missing_target_sentence_id", "value": target_id})
                accepted[target_id] = {"proposals": []}
        return {"windows": {
            row["target_sentence_id"]: accepted[row["target_sentence_id"]] for row in batch
        }}, issues

    def _fallback_audit(
        self,
        batch: list[dict],
        accepted_nodes: list[dict],
        batch_number: int,
        exc: Exception,
    ) -> dict:
        return {
            "batch_number": batch_number,
            "provider": self.provider,
            "model": self.model,
            "response_id": "",
            "request_hash": "",
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": self.prompt_hash,
            "target_sentence_ids": [row["target_sentence_id"] for row in batch],
            "window_count": len(batch),
            "accepted_node_count": len(accepted_nodes),
            "proposal_count": 0,
            "cached": False,
            "usage": {},
            "llm_used": True,
            "status": "safe_fallback_no_relationships",
            "error_type": type(exc).__name__,
            "error": str(exc).splitlines()[0][:500],
            "normalization_issues": [{"kind": "batch_failure"}],
            "safe_repair_count": len(batch),
        }

    def _eligible_predicates(self, accepted_nodes: list[dict]) -> list[str]:
        labels = {row.get("ontology_class", "") for row in accepted_nodes}
        return [
            name for name, spec in self.ontology.relationships.items()
            if any(self.ontology.compatible_class(label, spec.get("domain", [])) for label in labels)
            and any(self.ontology.compatible_class(label, spec.get("range", [])) for label in labels)
            and name not in {"CONTAINS_FRAGMENT", "CONTAINS_MENTION", "REFERS_TO"}
        ]

    def _schema(
        self,
        windows: list[dict] | int,
        accepted_nodes: list[dict] | None = None,
        eligible_predicates: list[str] | None = None,
    ) -> dict:
        # The integer form remains useful to lightweight schema unit tests.
        window_count = windows if isinstance(windows, int) else len(windows)
        target_ids = (
            [] if isinstance(windows, int)
            else [row["target_sentence_id"] for row in windows]
        )
        accepted_nodes = accepted_nodes or []
        labels = [*self.ontology.nodes, "UNKNOWN"]
        predicates = eligible_predicates or [*self.ontology.relationships]
        node_ids = ["", *[row["node_id"] for row in accepted_nodes]]
        context_sentence_ids = sorted({
            sentence["sentence_id"]
            for window in ([] if isinstance(windows, int) else windows)
            for sentence in window.get("context_sentences", [])
        })
        proposal = {
            "type": "object",
            "properties": {
                "head_node_id": {"type": "string", **({"enum": node_ids} if accepted_nodes else {})},
                "head_text": {"type": "string"},
                "head_proposed_label": {"type": "string", "enum": labels},
                "tail_node_id": {"type": "string", **({"enum": node_ids} if accepted_nodes else {})},
                "tail_text": {"type": "string"},
                "tail_proposed_label": {"type": "string", "enum": labels},
                "predicate": {"type": "string", "enum": predicates},
                "predicate_trigger_text": {"type": "string", "minLength": 1},
                "predicate_trigger_sentence_id": {
                    "type": "string", "minLength": 1,
                    **({"enum": context_sentence_ids} if context_sentence_ids else {}),
                },
                "decision": {"type": "string", "enum": ["accept", "reject"]},
                "attribution": {"type": "string", "enum": [
                    "current_study", "source_statement", "author_interpretation",
                    "cited_study", "background", "unknown",
                ]},
                "negated": {"type": "boolean"},
                "modal": {"type": "boolean"},
                "hypothetical": {"type": "boolean"},
                "comparative": {"type": "boolean"},
                "causal": {"type": "boolean"},
                "reason": {"type": "string", "minLength": 1},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": [
                "head_node_id", "head_text", "head_proposed_label",
                "tail_node_id", "tail_text", "tail_proposed_label", "predicate",
                "predicate_trigger_text", "predicate_trigger_sentence_id", "decision",
                "attribution", "negated", "modal", "hypothetical", "comparative",
                "causal", "reason", "confidence",
            ],
            "additionalProperties": False,
        }
        window_result = {
            "type": "object",
            "properties": {
                "proposals": {"type": "array", "items": proposal, "maxItems": 12},
            },
            "required": ["proposals"],
            "additionalProperties": False,
        }
        return {
            "type": "object",
            "properties": {"windows": {
                "type": "object",
                "properties": {
                    target_id: window_result for target_id in target_ids
                } if target_ids else {},
                "required": target_ids,
                "additionalProperties": False,
            }},
            "required": ["windows"],
            "additionalProperties": False,
        }


# Backwards-compatible import name used by existing integrations.
OpenAIResponsesRelationshipJudge = MandatoryLLMRelationshipJudge
