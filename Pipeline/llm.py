"""Small text-only local-LLM client backed by Ollama."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import time
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LLMError(RuntimeError):
    """Raised when a required structured model call cannot be completed."""


Transport = Callable[[dict], dict]


@dataclass
class OllamaClient:
    """Call a local Ollama chat model and return compact structured text."""

    model: str = "qwen3:4b-instruct"
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: int = 600
    max_tokens: int = 4096
    context_window: int = 32768
    keep_alive: str = "30m"
    retries: int = 2
    transport: Transport | None = None
    audit: list[dict] = field(default_factory=list)

    @classmethod
    def from_config(cls, config: dict, *, model: str | None = None) -> "OllamaClient":
        spec = config.get("ollama", config)
        return cls(
            model=model or spec.get("model", "qwen3:4b-instruct"),
            base_url=str(spec.get("base_url", "http://127.0.0.1:11434")).rstrip("/"),
            timeout_seconds=int(spec.get("timeout_seconds", 600)),
            max_tokens=int(spec.get("max_tokens", 4096)),
            context_window=int(spec.get("context_window", 32768)),
            keep_alive=str(spec.get("keep_alive", "30m")),
            retries=int(spec.get("retries", 2)),
        )

    def complete_text(self, stage: str, instructions: str, payload: dict | str) -> str:
        """Return model text and record timing; pipeline code parses every field."""
        request_body = {
            "model": self.model,
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": 0,
                "num_ctx": self.context_window,
                "num_predict": self.max_tokens,
            },
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)},
            ],
        }
        started = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 2):
            try:
                response = self.transport(request_body) if self.transport else self._post(request_body)
                # Direct Ollama responses use message.content. The choices
                # branch keeps injected OpenAI-shaped test transports simple.
                content = (response.get("message") or {}).get("content")
                if content is None:
                    content = response["choices"][0]["message"]["content"]
                content = str(content).strip()
                if not content:
                    raise LLMError("Ollama returned empty text")
                usage = response.get("usage") or {}
                self.audit.append({
                    "stage": stage,
                    "provider": "ollama",
                    "model": self.model,
                    "attempt": attempt,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "input_tokens": int(response.get("prompt_eval_count", usage.get("prompt_tokens", 0))),
                    "output_tokens": int(response.get("eval_count", usage.get("completion_tokens", 0))),
                    "status": "complete",
                })
                return content
            except (KeyError, IndexError, TypeError, HTTPError, URLError, LLMError) as exc:
                last_error = exc
        self.audit.append({
            "stage": stage, "provider": "ollama", "model": self.model,
            "attempt": self.retries + 1,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "status": "failed", "error": f"{type(last_error).__name__}: {last_error}",
        })
        raise LLMError(f"Ollama stage {stage!r} failed: {last_error}")

    def _post(self, body: dict) -> dict:
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
