from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .llm_node_judge import LLMNodeJudgeError, MandatoryLLMNodeJudge
from .ontology import Ontology


PROMPT_VERSION = "llm_reference_judge_v1"
SYSTEM_PROMPT = """You resolve contextual references for a scientific knowledge graph.

Each input pair contains one vague reference mention (for example, "this model") and one
specific candidate node (for example, "WaterRAG"), together with the exact sentences in
which both are used. Decide whether the two mentions refer to the same real entity.

Return exactly one decision for every pair_id:
- SAME: the supplied context clearly indicates that the vague reference denotes the candidate node.
- DIFFERENT: the context clearly indicates that it denotes another entity.
- UNCERTAIN: the context is insufficient or more than one antecedent remains plausible.

Hard rules:
1. Compare only the supplied pair. Never invent another node or rewrite source evidence.
2. Grammatical proximity alone is not enough; use discourse continuity, semantic role, and context.
3. Prefer UNCERTAIN whenever SAME is not well supported.
4. A generic reference is never a new canonical node.
5. Return every pair_id exactly once and return no additional pairs.
"""


class MandatoryLLMReferenceJudge(MandatoryLLMNodeJudge):
    """Contextual SAME/DIFFERENT/UNCERTAIN adjudication for mention pairs."""

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
        reference_config = deepcopy(config["llm_reference_judge"])
        reference_config["required"] = True
        effective["llm_node_judge"] = reference_config
        super().__init__(ontology, effective, cache_dir, api_key=api_key, transport=transport)
        self.batch_size = int(reference_config.get("batch_pairs", 12))
        self.prompt_hash = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()

    def judge(self, pairs: list[dict]) -> dict[str, dict]:
        results: dict[str, dict] = {}
        for offset in range(0, len(pairs), self.batch_size):
            batch = pairs[offset:offset + self.batch_size]
            parsed, audit = self._judge_batch(batch, offset // self.batch_size + 1)
            self.audit_records.append(audit)
            for judgment in parsed["judgments"]:
                judgment["_llm_audit"] = audit
                results[judgment["pair_id"]] = judgment
        return results

    def _judge_batch(self, batch: list[dict], batch_number: int) -> tuple[dict, dict]:
        schema = self._schema(len(batch))
        request_payload = self._request_payload({"pairs": batch}, schema)
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
            parsed = json.loads(self._output_text(response_payload))
        except json.JSONDecodeError as exc:
            raise LLMNodeJudgeError(f"Reference LLM returned invalid JSON: {exc}") from exc
        self._validate_result(parsed, batch)
        audit = {
            "batch_number": batch_number,
            "provider": self.provider,
            "model": self.model,
            "response_id": self._response_id(response_payload),
            "request_hash": request_hash,
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": self.prompt_hash,
            "pair_ids": [row["pair_id"] for row in batch],
            "pair_count": len(batch),
            "cached": cached,
            "usage": self._usage(response_payload),
            "runtime_created_at": response_payload.get("created_at", ""),
            "runtime_done_reason": response_payload.get("done_reason", ""),
            "llm_used": True,
            "status": "complete",
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
                    "num_ctx": int(self.config.get("context_window", 8192)),
                    "seed": int(self.config.get("seed", 42)),
                    "num_predict": int(self.config.get("max_output_tokens", 4096)),
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
                "name": "contextual_reference_judgments",
                "strict": True,
                "schema": schema,
            }},
            "store": False,
        }

    @staticmethod
    def _validate_result(parsed: dict, batch: list[dict]) -> None:
        if not isinstance(parsed, dict) or not isinstance(parsed.get("judgments"), list):
            raise LLMNodeJudgeError("Reference LLM result is missing the judgments array.")
        expected_ids = {row["pair_id"] for row in batch}
        actual_ids = [row.get("pair_id") for row in parsed["judgments"]]
        if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected_ids:
            raise LLMNodeJudgeError(
                "Reference LLM must return exactly one judgment for every pair ID."
            )

    @staticmethod
    def _schema(pair_count: int) -> dict:
        judgment = {
            "type": "object",
            "properties": {
                "pair_id": {"type": "string"},
                "decision": {"type": "string", "enum": ["SAME", "DIFFERENT", "UNCERTAIN"]},
                "reason": {"type": "string", "minLength": 1},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["pair_id", "decision", "reason", "confidence"],
            "additionalProperties": False,
        }
        return {
            "type": "object",
            "properties": {"judgments": {
                "type": "array",
                "items": judgment,
                "minItems": pair_count,
                "maxItems": pair_count,
            }},
            "required": ["judgments"],
            "additionalProperties": False,
        }
