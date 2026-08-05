from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
import unicodedata
from typing import Callable, Iterable

from .common import normalized

def build_required_ner_artifacts(
    parsed: dict, accepted: list[dict], review: list[dict], config: dict,
) -> tuple[list[dict], dict]:
    """Materialize and validate the mandatory multi-channel NER boundary."""
    spec = config.get("ner", {})
    enabled = bool(spec.get("enabled", True))
    required = bool(spec.get("required", True))
    if required and not enabled:
        raise RuntimeError("NER is required but disabled in node_extraction.json.")

    summary = parsed.get("summary", {})
    sentences = {
        sentence["id"]: sentence
        for section in parsed.get("sections", [])
        for paragraph in section.get("paragraphs", [])
        for sentence in paragraph.get("sentences", [])
    }
    if required and not sentences:
        raise RuntimeError("NER cannot run because parsing produced no narrative sentences.")
    page_count = int(summary.get("pages", 0) or 0)
    last_page = int(summary.get("last_narrative_page", page_count) or 0)
    minimum_ratio = float(spec.get("minimum_narrative_page_ratio", 0.5))
    if required and page_count >= 6 and last_page < max(1, int(page_count * minimum_ratio)):
        raise RuntimeError(
            "NER blocked: narrative parsing ended suspiciously early; inspect the PDF parser output."
        )

    rows: list[dict] = []
    errors: list[dict] = []
    channel_counts: dict[str, int] = defaultdict(int)
    trusted_channels = set(spec.get("trusted_channels", []))
    for status, candidates in (("accepted", accepted), ("review", review)):
        for candidate in candidates:
            source = candidate.get("source", {})
            channel = candidate.get("extraction_method", "") or "unknown"
            channel_counts[channel] += 1
            row = {
                "ner_candidate_id": candidate.get("candidate_id") or candidate.get("mention_id"),
                "status": status,
                "label": candidate.get("label", ""),
                "surface_text": candidate.get("surface_text", ""),
                "canonical_name": candidate.get("canonical_name", ""),
                "channel": channel,
                "confidence": candidate.get("confidence", 0.0),
                "source": source,
            }
            rows.append(row)
            if status == "accepted" and trusted_channels and channel not in trusted_channels:
                errors.append({
                    "ner_candidate_id": row["ner_candidate_id"],
                    "reason": "An untrusted candidate channel bypassed adjudication.",
                    "channel": channel,
                })
            if source.get("kind") != "sentence_span":
                continue
            sentence = sentences.get(source.get("sentence_id", ""))
            start, end = source.get("start_char"), source.get("end_char")
            valid = bool(
                sentence is not None and isinstance(start, int) and isinstance(end, int)
                and 0 <= start < end <= len(sentence["text"])
                and normalized(sentence["text"][start:end]) == normalized(candidate.get("surface_text", ""))
            )
            if not valid:
                errors.append({
                    "ner_candidate_id": row["ner_candidate_id"],
                    "sentence_id": source.get("sentence_id", ""),
                    "reason": "Candidate does not align to the declared exact source span.",
                })

    declared_channels = set(spec.get("channels", []))
    undeclared = sorted(channel for channel in channel_counts if channel not in declared_channels)
    if required and undeclared:
        errors.append({
            "reason": "NER candidate used an undeclared extraction channel.",
            "channels": undeclared,
        })
    status = {
        "schema_version": "1.0",
        "required": required,
        "enabled": enabled,
        "status": "failed" if errors else "complete",
        "architecture": spec.get("architecture", "ontology_guided_multichannel_ner_v1"),
        "config_hash": hashlib.sha256(
            json.dumps(spec, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "candidate_count": len(rows),
        "accepted_count": len(accepted),
        "trusted_channels": sorted(trusted_channels),
        "review_count": len(review),
        "channel_counts": dict(sorted(channel_counts.items())),
        "exact_span_errors": errors,
        "artifacts": ["ner_candidates.jsonl", "ner_stage.json"],
    }
    if required and errors:
        raise RuntimeError(f"Required NER validation failed with {len(errors)} error(s).")
    return rows, status
