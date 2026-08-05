from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .ontology import Ontology


PROMPT_VERSION = "llm_node_judge_v7_cross_paper_traceable"
SYSTEM_PROMPT = """You are the mandatory node-adjudication and open-vocabulary discovery stage for a scientific knowledge graph.

Your job is not to maximize extraction volume. Your job is to produce a small, specific, useful, evidence-grounded set of nodes.

For every supplied candidate, perform these operations in order:
1. Clean its deterministically_cleaned_name further, removing grammatical debris while preserving every scientifically meaningful qualifier. Put the result in canonical_name. Never alter the supplied source span.
2. Select one ontology type when supported. Compare the cleaned candidate only with prior_nodes_same_type whose label equals that selected ontology type. Never compare or merge nodes of different types.
3. If it is the same entity as one supplied prior node, return that prior node's canonical_name exactly. Otherwise return a concise new canonical name.
4. Apply the validity rubric and return exactly one judgment:
- accept: it is a clear standalone entity or scientifically useful concept, including a broad cross-domain concept.
- review: it is grounded and potentially useful, but its boundary, ontology type, or identity needs a human decision.
- reject: it lacks sentence traceability, remains vague after attempted resolution, is malformed, or is not a standalone graph node.

Apply the supplied validity_rubric to every candidate. Candidate generation is deliberately high recall; your adjudication should protect precision.

Only inspect a sentence for missed nodes when allow_discovery is true. When it is false, return an empty discoveries array. Only propose a new ontology class when allow_new_class_candidates is true; otherwise return an empty new_class_candidates array.

Hard rules:
1. Prefer named systems, models, agent roles, tools, datasets, methods, metrics, environmental systems, chemicals, organisms, locations, policies, organizations, and domain-specific tasks.
2. Broad concepts are eligible nodes. Reject only bare generic or vague references such as model, method, system, it, or this approach when they cannot be resolved to one supplied, same-type prior node.
3. A phrase naming multiple agent roles is not one node; send it to review or discover the exact individual role spans.
4. Do not invent text. Every discovery span and surface text must exactly match the supplied sentence using zero-based start_char and end_char offsets.
5. Do not create relationships. Do not infer facts not present in the sentence.
6. Map accepted nodes to one existing ontology class. Use new_class_candidates only when the input explicitly allows it and no existing class can represent the concept.
7. Return concise canonical names and definitions grounded in the sentence.
8. Paper centrality is not a validity requirement. Retain clear, grounded concepts from background and related work so the graph can express connections across papers.
9. Respect the ontology definitions when assigning a class. A software package, simulator, or executable such as EPANET is a Tool; an AI system built around a tool, such as EPANET-Agentic, is an AISystem; and a named functional role such as Coding Agent is an Agent. Do not promote a Tool to AISystem merely because an AI system uses it.
10. Treat a suggested ontology label as a strong evidence-based anchor. Change it only when the sentence and ontology definitions clearly support a different existing class, and explain that change in the reason.
11. Return each supplied sentence_id exactly once. Do not rediscover a span that already appears in that sentence's candidate list.
12. canonical_name, definition, and reason must never be empty. The definition must describe the named entity, agree with the selected ontology class, and use the sentence context; never copy an unrelated class definition.
13. prior_nodes_same_type is a closed comparison list. If none is the same entity, do not force a match. Explain any chosen prior canonical name in the judgment reason.
14. canonical_name must remain a readable scientific noun phrase. Preserve ordinary spaces, established hyphens, abbreviations, and capitalization; never invent CamelCase such as MembraneBioreactor when the source says membrane bioreactor.
15. A candidate without a specific non-empty sentence_id must be rejected. Never accept or review an untraceable candidate.
16. If a vague candidate can be resolved to a supplied same-type prior node, return that prior node's canonical_name exactly; otherwise reject it rather than leaving it vague.
"""


class LLMNodeJudgeError(RuntimeError):
    pass


Transport = Callable[[dict], dict]


class MandatoryLLMNodeJudge:
    def __init__(
        self,
        ontology: Ontology,
        config: dict,
        cache_dir: Path,
        *,
        api_key: str | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.ontology = ontology
        self.config = config["llm_node_judge"]
        if not self.config.get("required"):
            raise LLMNodeJudgeError("The LLM node judge must be configured as required.")
        self.provider = str(self.config.get("provider", "")).casefold()
        if self.provider not in {"openai", "ollama"}:
            raise LLMNodeJudgeError(
                f"Unsupported mandatory LLM provider: {self.provider or 'missing'}."
            )
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.transport = transport
        if self.provider == "openai" and self.transport is None and not self.api_key:
            raise LLMNodeJudgeError(
                "Mandatory LLM node judgment cannot run: OPENAI_API_KEY is not configured."
            )
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = str(self.config["model"])
        self.reasoning_effort = str(self.config.get("reasoning_effort", "medium"))
        self.base_url = str(
            self.config.get(
                "base_url",
                "http://127.0.0.1:11434" if self.provider == "ollama"
                else "https://api.openai.com",
            )
        ).rstrip("/")
        self.batch_size = int(self.config.get("batch_sentences", 10))
        self.audit_records: list[dict] = []
        self.prompt_hash = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()

    def verify_runtime(self) -> None:
        """Fail before parsing when the configured real model runtime is unavailable."""
        if self.transport is not None or self.provider == "openai":
            return
        request = Request(f"{self.base_url}/api/tags", method="GET")
        try:
            with urlopen(request, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMNodeJudgeError(
                "Mandatory local LLM judgment cannot run: Ollama is not reachable at "
                f"{self.base_url}. Start Ollama, then run `ollama pull {self.model}`."
            ) from exc
        installed = {
            value
            for row in payload.get("models", [])
            for value in (row.get("name", ""), row.get("model", ""))
        }
        if self.model not in installed:
            raise LLMNodeJudgeError(
                f"Mandatory local LLM model {self.model!r} is not installed. "
                f"Run `ollama pull {self.model}`."
            )

    def judge(self, sentence_records: list[dict]) -> dict[str, dict]:
        expanded_records: list[dict] = []
        split_occurred = False
        for sentence in sentence_records:
            fragments = self._split_oversized_sentence(sentence)
            split_occurred = split_occurred or len(fragments) > 1
            expanded_records.extend(fragments)

        results: dict[str, dict] = {}
        # Keep fragments of one oversized sentence in separate requests. Normal
        # runs retain the configured batch size and therefore retain identical
        # request hashes/cache behavior.
        batch_size = 1 if split_occurred else self.batch_size
        for offset in range(0, len(expanded_records), batch_size):
            batch = expanded_records[offset:offset + batch_size]
            parsed, audit = self._judge_batch(batch, offset // batch_size + 1)
            self.audit_records.append(audit)
            expected_ids = {row["sentence_id"] for row in batch}
            actual_ids = {row["sentence_id"] for row in parsed["sentences"]}
            if actual_ids != expected_ids:
                raise LLMNodeJudgeError(
                    f"LLM batch sentence IDs differ from input: expected {sorted(expected_ids)}, got {sorted(actual_ids)}"
                )
            for row in parsed["sentences"]:
                for judgment in row.get("candidate_judgments", []):
                    judgment["_llm_audit"] = audit
                existing = results.get(row["sentence_id"])
                if existing is None:
                    row["_llm_audit"] = audit
                    row["_llm_audits"] = [audit]
                    results[row["sentence_id"]] = row
                    continue
                for field in ("candidate_judgments", "discoveries", "new_class_candidates"):
                    existing.setdefault(field, []).extend(row.get(field, []))
                existing.setdefault("_llm_audits", []).append(audit)
        return results

    def _split_oversized_sentence(self, sentence: dict) -> list[dict]:
        """Split only candidate-heavy sentences that exceed normal num_ctx.

        Each fragment keeps the exact sentence/context and sentence ID; only
        the candidate list is partitioned. Later aggregation restores one
        sentence result and retains a separate audit hash for every fragment.
        """
        if self.provider != "ollama" or len(sentence.get("candidates", [])) <= 1:
            return [sentence]
        ontology_reference = [
            {
                "label": label,
                "definition": spec["definition"],
                "category": spec.get("category", ""),
            }
            for label, spec in self.ontology.nodes.items()
        ]
        schema = self._schema([sentence])
        payload = self._request_payload(
            {"ontology_classes": ontology_reference, "sentences": [sentence]}, schema
        )
        configured_context = int(self.config.get("context_window", 32768))
        if int(payload.get("options", {}).get("num_ctx", configured_context)) <= configured_context:
            return [sentence]

        candidates = list(sentence["candidates"])
        midpoint = max(1, len(candidates) // 2)
        left = {**sentence, "candidates": candidates[:midpoint]}
        right = {**sentence, "candidates": candidates[midpoint:]}
        return [
            *self._split_oversized_sentence(left),
            *self._split_oversized_sentence(right),
        ]

    def _judge_batch(self, batch: list[dict], batch_number: int) -> tuple[dict, dict]:
        ontology_reference = [
            {
                "label": label,
                "definition": spec["definition"],
                "category": spec.get("category", ""),
            }
            for label, spec in self.ontology.nodes.items()
        ]
        user_payload = {
            "ontology_classes": ontology_reference,
            "sentences": batch,
        }
        schema = self._schema(batch)
        request_payload = self._request_payload(user_payload, schema)
        request_hash = hashlib.sha256(
            json.dumps(request_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        cache_path = self.cache_dir / f"{request_hash}.json"
        cached = False
        response_payload: dict = {}
        parsed: dict | None = None
        validation_result: tuple[list[dict], list[dict], list[dict]] | None = None
        structured_errors: list[str] = []
        responses: list[tuple[dict, bool]] = []
        if bool(self.config.get("cache_responses")) and cache_path.is_file():
            try:
                cached_response = json.loads(cache_path.read_text(encoding="utf-8"))
                if self._response_complete(cached_response):
                    responses.append((cached_response, True))
            except (json.JSONDecodeError, OSError) as exc:
                structured_errors.append(f"Unreadable cached response: {exc}")

        structured_retries = int(self.config.get("structured_output_retries", 1))
        for attempt in range(structured_retries + 1):
            if responses:
                candidate_response, from_cache = responses.pop(0)
            else:
                attempt_payload = json.loads(json.dumps(request_payload))
                if self.provider == "ollama" and attempt:
                    attempt_payload.setdefault("options", {})["seed"] = int(
                        self.config.get("seed", 42)
                    ) + attempt
                candidate_response = self._send(attempt_payload)
                from_cache = False
            try:
                candidate_parsed = json.loads(self._output_text(candidate_response))
                candidate_validation = self._validate_batch_result(candidate_parsed, batch)
            except (json.JSONDecodeError, LLMNodeJudgeError) as exc:
                structured_errors.append(str(exc))
                if from_cache:
                    cache_path.unlink(missing_ok=True)
                continue
            response_payload = candidate_response
            parsed = candidate_parsed
            validation_result = candidate_validation
            cached = from_cache
            if not from_cache and self.config.get("cache_responses"):
                cache_path.write_text(
                    json.dumps(response_payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
            break

        if parsed is None or validation_result is None:
            if str(self.config.get("invalid_output_policy", "")).casefold() == "fail":
                detail = structured_errors[-1] if structured_errors else "unknown structured-output failure"
                raise LLMNodeJudgeError(f"LLM structured output failed closed: {detail}")
            fallback_decision = (
                "review" if str(self.config.get("invalid_output_policy", "")).casefold().startswith("review")
                else "reject"
            )
            parsed = {"sentences": []}
            for sentence in batch:
                parsed["sentences"].append({
                    "sentence_id": sentence["sentence_id"],
                    "candidate_judgments": [{
                        "candidate_id": candidate["candidate_id"],
                        "decision": fallback_decision,
                        "ontology_label": candidate.get("suggested_label", "NONE")
                        if candidate.get("suggested_label", "") in self.ontology.nodes else "NONE",
                        "canonical_name": candidate.get("deterministically_cleaned_name") or candidate.get("surface_text") or "unresolved candidate",
                        "definition": "The local model did not produce a valid structured judgment for this grounded candidate.",
                        "reason": "The sentence batch was quarantined because DeepSeek did not return valid structured JSON after retry.",
                        "confidence": 0.0,
                    } for candidate in sentence.get("candidates", [])],
                    "discoveries": [],
                    "new_class_candidates": [],
                })
            validation_result = self._validate_batch_result(parsed, batch)
            response_payload = response_payload or {
                "id": f"quarantine_{request_hash[:24]}",
                "done_reason": "structured_output_quarantined",
            }
        span_corrections, invalid_outputs, format_corrections = validation_result
        if structured_errors:
            invalid_outputs.extend({
                "sentence_id": row["sentence_id"],
                "output_type": "sentence_batch",
                "reason": "Structured-output retry: " + error,
            } for row in batch for error in structured_errors)
        audit = {
            "batch_number": batch_number,
            "provider": self.provider,
            "model": self.model,
            "response_id": self._response_id(response_payload),
            "request_hash": request_hash,
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": self.prompt_hash,
            "sentence_ids": [row["sentence_id"] for row in batch],
            "candidate_count": sum(len(row.get("candidates", [])) for row in batch),
            "cached": cached,
            "usage": self._usage(response_payload),
            "runtime_created_at": response_payload.get("created_at", ""),
            "runtime_done_reason": response_payload.get("done_reason", ""),
            "span_corrections": span_corrections,
            "invalid_outputs": invalid_outputs,
            "format_corrections": format_corrections,
            "llm_used": True,
            "status": "complete" if not structured_errors else "complete_with_quarantine_or_retry",
        }
        return parsed, audit

    def _request_payload(self, user_payload: dict, schema: dict) -> dict:
        if self.provider == "ollama":
            grounded_payload = {
                **user_payload,
                "required_output_json_schema": schema,
            }
            user_content = json.dumps(grounded_payload, ensure_ascii=False)
            configured_context = int(self.config.get("context_window", 32768))
            maximum_context = max(
                configured_context,
                int(self.config.get("maximum_context_window", configured_context)),
            )
            max_output_tokens = int(self.config.get("max_output_tokens", 4096))
            # Ollama counts the system prompt, user payload, and structured
            # output schema against num_ctx. Candidate-heavy scientific
            # sentences can exceed the normal context even when batching is
            # already one sentence at a time. Use a conservative character to
            # token estimate and grow only those requests to the configured
            # ceiling; ordinary requests retain the original payload/hash and
            # continue to reuse their cached judgments.
            estimated_prompt_tokens = (
                len(SYSTEM_PROMPT) + len(user_content)
                + len(json.dumps(schema, ensure_ascii=False)) + 2
            ) // 3
            required_context = estimated_prompt_tokens + max_output_tokens + 1024
            request_context = configured_context
            while request_context < required_context and request_context < maximum_context:
                request_context = min(request_context * 2, maximum_context)
            return {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "stream": False,
                "format": schema,
                "options": {
                    "temperature": float(self.config.get("temperature", 0)),
                    "num_ctx": request_context,
                    "seed": int(self.config.get("seed", 42)),
                    "num_predict": max_output_tokens,
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
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "scientific_node_judgments",
                    "strict": True,
                    "schema": schema,
                }
            },
            "store": False,
        }

    def _send(self, payload: dict) -> dict:
        if self.transport is not None:
            return self.transport(payload)
        encoded = json.dumps(payload).encode("utf-8")
        endpoint = (
            f"{self.base_url}/api/chat" if self.provider == "ollama"
            else f"{self.base_url}/v1/responses"
        )
        headers = {"Content-Type": "application/json"}
        if self.provider == "openai":
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            endpoint,
            data=encoded,
            method="POST",
            headers=headers,
        )
        retries = int(self.config.get("maximum_retries", 3))
        timeout = int(self.config.get("timeout_seconds", 180))
        for attempt in range(retries + 1):
            try:
                with urlopen(request, timeout=timeout) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                if not self._response_complete(response_payload):
                    if attempt >= retries:
                        raise LLMNodeJudgeError(
                            "Ollama ended before completing its structured response."
                        )
                    time.sleep(min(2 ** attempt, 8))
                    continue
                return response_payload
            except HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                retryable = exc.code == 429 or exc.code >= 500
                if not retryable or attempt >= retries:
                    raise LLMNodeJudgeError(
                        f"{self.provider} LLM API failed with HTTP {exc.code}: {body[:500]}"
                    ) from exc
            except URLError as exc:
                if attempt >= retries:
                    raise LLMNodeJudgeError(f"{self.provider} LLM API connection failed: {exc}") from exc
            time.sleep(min(2 ** attempt, 8))
        raise LLMNodeJudgeError(f"{self.provider} LLM API failed after retries.")

    def _response_complete(self, response: dict) -> bool:
        return self.provider != "ollama" or response.get("done") is True

    @staticmethod
    def _output_text(response: dict) -> str:
        message = response.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
        if isinstance(response.get("output_text"), str):
            return response["output_text"]
        for item in response.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") == "refusal":
                    raise LLMNodeJudgeError(f"LLM refused node judgment: {content.get('refusal', '')}")
                if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                    return content["text"]
        raise LLMNodeJudgeError("LLM response contained no structured output text.")

    @staticmethod
    def _response_id(response: dict) -> str:
        if response.get("id"):
            return str(response["id"])
        digest = hashlib.sha256(
            json.dumps(response, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:24]
        return f"ollama_{digest}"

    @staticmethod
    def _usage(response: dict) -> dict:
        if isinstance(response.get("usage"), dict):
            return response["usage"]
        return {
            "input_tokens": response.get("prompt_eval_count", 0),
            "output_tokens": response.get("eval_count", 0),
            "total_duration_ns": response.get("total_duration", 0),
            "load_duration_ns": response.get("load_duration", 0),
            "prompt_eval_duration_ns": response.get("prompt_eval_duration", 0),
            "eval_duration_ns": response.get("eval_duration", 0),
        }

    def _validate_batch_result(
        self, parsed: dict, batch: list[dict]
    ) -> tuple[list[dict], list[dict], list[dict]]:
        if not isinstance(parsed, dict) or not isinstance(parsed.get("sentences"), list):
            raise LLMNodeJudgeError("LLM result is missing the sentences array.")
        invalid_outputs: list[dict] = []
        format_corrections: list[dict] = []
        invalid_policy = str(
            self.config.get("invalid_output_policy", "reject_and_audit")
        ).casefold()
        expected_sentence_ids = {row["sentence_id"] for row in batch}
        unique_sentences: dict[str, dict] = {}
        for result in parsed["sentences"]:
            sentence_id = result.get("sentence_id", "")
            if sentence_id not in expected_sentence_ids and len(batch) == 1:
                corrected_sentence_id = batch[0]["sentence_id"]
                format_corrections.append({
                    "sentence_id": corrected_sentence_id,
                    "correction_type": "single_sentence_id_restored",
                    "reported_sentence_id": sentence_id,
                    "corrected_sentence_id": corrected_sentence_id,
                    "reason": "The one-row batch makes the intended source sentence unambiguous.",
                })
                result["sentence_id"] = corrected_sentence_id
                sentence_id = corrected_sentence_id
            if sentence_id not in unique_sentences:
                unique_sentences[sentence_id] = result
                continue
            existing = unique_sentences[sentence_id]
            existing_candidate_ids = {
                row.get("candidate_id") for row in existing.get("candidate_judgments", [])
            }
            duplicate_candidate_ids = {
                row.get("candidate_id") for row in result.get("candidate_judgments", [])
            }
            if existing_candidate_ids & duplicate_candidate_ids:
                raise LLMNodeJudgeError(
                    f"LLM returned conflicting duplicate results for sentence {sentence_id}."
                )
            if invalid_policy == "fail":
                raise LLMNodeJudgeError(
                    f"LLM returned a duplicate result for sentence {sentence_id}."
                )
            for field in ("candidate_judgments", "discoveries", "new_class_candidates"):
                existing.setdefault(field, []).extend(result.get(field, []))
            format_corrections.append({
                "sentence_id": sentence_id,
                "correction_type": "duplicate_sentence_rows_merged",
                "reason": "Disjoint candidate judgments returned in repeated sentence rows were merged.",
            })
        parsed["sentences"] = list(unique_sentences.values())
        allowed_labels = set(self.ontology.nodes)
        candidates = {
            candidate["candidate_id"]: candidate
            for sentence in batch for candidate in sentence.get("candidates", [])
        }
        candidate_sentences = {
            candidate["candidate_id"]: sentence["sentence_id"]
            for sentence in batch for candidate in sentence.get("candidates", [])
        }
        result_by_sentence = {row["sentence_id"]: row for row in parsed["sentences"]}
        seen_candidate_ids: set[str] = set()
        for result in parsed["sentences"]:
            valid_judgments = []
            for judgment in result.get("candidate_judgments", []):
                candidate_id = judgment.get("candidate_id", "")
                if candidate_id not in candidates or candidate_id in seen_candidate_ids:
                    if invalid_policy == "fail":
                        raise LLMNodeJudgeError("LLM must return exactly one judgment for every candidate ID.")
                    invalid_outputs.append({
                        "sentence_id": result.get("sentence_id", ""),
                        "output_type": "candidate_judgment",
                        "reason": "Unknown or duplicate candidate judgment was removed.",
                        "raw_output": dict(judgment),
                    })
                    continue
                seen_candidate_ids.add(candidate_id)
                valid_judgments.append(judgment)
            result["candidate_judgments"] = valid_judgments

        missing_ids = set(candidates) - seen_candidate_ids
        if missing_ids and invalid_policy == "fail":
            raise LLMNodeJudgeError("LLM must return exactly one judgment for every candidate ID.")
        fallback_decision = "review" if invalid_policy.startswith("review") else "reject"
        for candidate_id in sorted(missing_ids):
            candidate = candidates[candidate_id]
            sentence_id = candidate_sentences[candidate_id]
            result = result_by_sentence.get(sentence_id)
            if result is None:
                result = {
                    "sentence_id": sentence_id,
                    "candidate_judgments": [],
                    "discoveries": [],
                    "new_class_candidates": [],
                }
                parsed["sentences"].append(result)
                result_by_sentence[sentence_id] = result
            label = candidate.get("suggested_label", "")
            if label not in allowed_labels:
                label = "NONE"
            fallback = {
                "candidate_id": candidate_id,
                "decision": fallback_decision,
                "ontology_label": label,
                "canonical_name": candidate.get("deterministically_cleaned_name") or candidate.get("surface_text") or "unresolved candidate",
                "definition": "The local model did not return a complete structured judgment for this grounded candidate.",
                "reason": "The candidate was quarantined for human review because its structured DeepSeek judgment was missing.",
                "confidence": 0.0,
            }
            result["candidate_judgments"].append(fallback)
            invalid_outputs.append({
                "sentence_id": sentence_id,
                "output_type": "candidate_judgment",
                "reason": "Missing candidate judgment was synthesized as a human-review decision.",
                "candidate_id": candidate_id,
            })
            format_corrections.append({
                "sentence_id": sentence_id,
                "correction_type": "missing_candidate_judgment_quarantined",
                "candidate_id": candidate_id,
                "decision": fallback_decision,
            })
        sentence_text = {row["sentence_id"]: row["text"] for row in batch}
        candidate_spans = {
            (
                sentence["sentence_id"], int(candidate["start_char"]),
                int(candidate["end_char"]), candidate["suggested_label"],
            )
            for sentence in batch for candidate in sentence.get("candidates", [])
        }
        span_corrections: list[dict] = []
        for result in parsed["sentences"]:
            text = sentence_text.get(result["sentence_id"])
            if text is None:
                raise LLMNodeJudgeError(f"Unknown sentence ID in LLM output: {result['sentence_id']}")
            for judgment in result["candidate_judgments"]:
                if judgment["decision"] == "accept" and judgment["ontology_label"] not in allowed_labels:
                    raise LLMNodeJudgeError("Accepted LLM judgment has no valid ontology class.")
                if str(judgment.get("reason", "")).strip().casefold() in {"", "none", "n/a", "na"}:
                    old_reason = judgment.get("reason", "")
                    judgment["reason"] = (
                        f"The configured LLM judge selected {judgment['ontology_label']} for "
                        f"candidate {judgment['candidate_id']}."
                    )
                    format_corrections.append({
                        "sentence_id": result["sentence_id"],
                        "correction_type": "uninformative_reason_replaced",
                        "candidate_id": judgment["candidate_id"],
                        "reported_reason": old_reason,
                        "corrected_reason": judgment["reason"],
                    })
            valid_discoveries = []
            for discovery in result["discoveries"]:
                if not bool(self.config.get("allow_llm_discovery", True)):
                    invalid_outputs.append({
                        "sentence_id": result["sentence_id"],
                        "output_type": "discovery",
                        "reason": "LLM discovery is disabled for this pipeline.",
                        "raw_output": dict(discovery),
                    })
                    continue
                try:
                    correction = self._align_and_validate_span(
                        text, discovery, result["sentence_id"]
                    )
                    if discovery["ontology_label"] not in allowed_labels:
                        raise LLMNodeJudgeError("LLM discovery has an unknown ontology class.")
                except LLMNodeJudgeError as exc:
                    if invalid_policy == "fail":
                        raise
                    invalid_outputs.append({
                        "sentence_id": result["sentence_id"],
                        "output_type": "discovery",
                        "reason": str(exc),
                        "raw_output": dict(discovery),
                    })
                    continue
                if correction:
                    span_corrections.append(correction)
                if str(discovery.get("reason", "")).strip().casefold() in {"", "none", "n/a", "na"}:
                    old_reason = discovery.get("reason", "")
                    discovery["reason"] = (
                        f"The configured LLM judge discovered {discovery['surface_text']} "
                        f"as {discovery['ontology_label']} in the exact source sentence."
                    )
                    format_corrections.append({
                        "sentence_id": result["sentence_id"],
                        "correction_type": "uninformative_reason_replaced",
                        "surface_text": discovery["surface_text"],
                        "reported_reason": old_reason,
                        "corrected_reason": discovery["reason"],
                    })
                discovery_key = (
                    result["sentence_id"], int(discovery["start_char"]),
                    int(discovery["end_char"]), discovery["ontology_label"],
                )
                if discovery_key in candidate_spans:
                    format_corrections.append({
                        "sentence_id": result["sentence_id"],
                        "correction_type": "redundant_discovery_removed",
                        "surface_text": discovery["surface_text"],
                        "reason": "The exact span and class were already judged as a supplied candidate.",
                    })
                    continue
                valid_discoveries.append(discovery)
            result["discoveries"] = valid_discoveries

            valid_class_candidates = []
            for proposal in result["new_class_candidates"]:
                if not bool(self.config.get("allow_new_class_candidates", True)):
                    invalid_outputs.append({
                        "sentence_id": result["sentence_id"],
                        "output_type": "new_class_candidate",
                        "reason": "New ontology-class proposals are disabled for this pipeline.",
                        "raw_output": dict(proposal),
                    })
                    continue
                if proposal["proposed_class_name"] in allowed_labels:
                    invalid_outputs.append({
                        "sentence_id": result["sentence_id"],
                        "output_type": "new_class_candidate",
                        "reason": "Proposed class already exists in the ontology.",
                        "raw_output": dict(proposal),
                    })
                    continue
                try:
                    correction = self._align_and_validate_span(
                        text, proposal, result["sentence_id"]
                    )
                except LLMNodeJudgeError as exc:
                    if invalid_policy == "fail":
                        raise
                    invalid_outputs.append({
                        "sentence_id": result["sentence_id"],
                        "output_type": "new_class_candidate",
                        "reason": str(exc),
                        "raw_output": dict(proposal),
                    })
                    continue
                if correction:
                    span_corrections.append(correction)
                valid_class_candidates.append(proposal)
            result["new_class_candidates"] = valid_class_candidates
        return span_corrections, invalid_outputs, format_corrections

    @staticmethod
    def _align_and_validate_span(text: str, row: dict, sentence_id: str) -> dict | None:
        start, end = int(row["start_char"]), int(row["end_char"])
        surface = row["surface_text"]
        if 0 <= start < end <= len(text) and text[start:end] == surface:
            return None
        occurrences = [match.start() for match in re.finditer(re.escape(surface), text)]
        if not occurrences:
            raise LLMNodeJudgeError(
                "LLM discovery text does not occur in the source sentence."
            )
        ranked = sorted((abs(value - start), value) for value in occurrences)
        if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
            raise LLMNodeJudgeError(
                "LLM discovery span is ambiguous because the source text occurs multiple times."
            )
        corrected_start = ranked[0][1]
        corrected_end = corrected_start + len(surface)
        row["start_char"] = corrected_start
        row["end_char"] = corrected_end
        return {
            "sentence_id": sentence_id,
            "surface_text": surface,
            "reported_start_char": start,
            "reported_end_char": end,
            "corrected_start_char": corrected_start,
            "corrected_end_char": corrected_end,
            "method": "exact_source_text_nearest_offset",
        }

    def _schema(self, batch: list[dict] | None = None) -> dict:
        labels = [*self.ontology.nodes]
        sentence_ids = [row["sentence_id"] for row in (batch or [])]
        candidate_ids = [
            candidate["candidate_id"]
            for row in (batch or []) for candidate in row.get("candidates", [])
        ]
        judgment = {
            "type": "object",
            "properties": {
                "candidate_id": {
                    "type": "string",
                    **({"enum": candidate_ids} if candidate_ids else {}),
                },
                "decision": {"type": "string", "enum": ["accept", "review", "reject"]},
                "ontology_label": {"type": "string", "enum": [*labels, "NONE"]},
                "canonical_name": {"type": "string", "minLength": 1},
                "definition": {"type": "string", "minLength": 1},
                "reason": {"type": "string", "minLength": 1},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["candidate_id", "decision", "ontology_label", "canonical_name", "definition", "reason", "confidence"],
            "additionalProperties": False,
        }
        discovery = {
            "type": "object",
            "properties": {
                "start_char": {"type": "integer", "minimum": 0},
                "end_char": {"type": "integer", "minimum": 1},
                "surface_text": {"type": "string", "minLength": 1},
                "ontology_label": {"type": "string", "enum": labels},
                "canonical_name": {"type": "string", "minLength": 1},
                "definition": {"type": "string", "minLength": 1},
                "reason": {"type": "string", "minLength": 1},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["start_char", "end_char", "surface_text", "ontology_label", "canonical_name", "definition", "reason", "confidence"],
            "additionalProperties": False,
        }
        class_proposal = {
            "type": "object",
            "properties": {
                "start_char": {"type": "integer", "minimum": 0},
                "end_char": {"type": "integer", "minimum": 1},
                "surface_text": {"type": "string", "minLength": 1},
                "proposed_class_name": {"type": "string", "minLength": 1},
                "definition": {"type": "string", "minLength": 1},
                "reason": {"type": "string", "minLength": 1},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["start_char", "end_char", "surface_text", "proposed_class_name", "definition", "reason", "confidence"],
            "additionalProperties": False,
        }
        sentence = {
            "type": "object",
            "properties": {
                "sentence_id": {
                    "type": "string",
                    **({"enum": sentence_ids} if sentence_ids else {}),
                },
                "candidate_judgments": {
                    "type": "array", "items": judgment,
                    **({"minItems": len(candidate_ids), "maxItems": len(candidate_ids)}
                       if batch is not None and len(batch) == 1 else {}),
                },
                "discoveries": {
                    "type": "array", "items": discovery,
                    "maxItems": 6 if bool(self.config.get("allow_llm_discovery", False)) else 0,
                },
                "new_class_candidates": {
                    "type": "array", "items": class_proposal,
                    "maxItems": 1 if bool(self.config.get("allow_new_class_candidates", False)) else 0,
                },
            },
            "required": ["sentence_id", "candidate_judgments", "discoveries", "new_class_candidates"],
            "additionalProperties": False,
        }
        return {
            "type": "object",
            "properties": {"sentences": {
                "type": "array", "items": sentence,
                **({"minItems": len(batch), "maxItems": len(batch)}
                   if batch is not None else {}),
            }},
            "required": ["sentences"],
            "additionalProperties": False,
        }


# Backwards-compatible import name for existing integrations and historical tests.
OpenAIResponsesNodeJudge = MandatoryLLMNodeJudge
