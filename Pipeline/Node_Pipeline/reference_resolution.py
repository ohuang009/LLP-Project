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

from .common import CONFIG_PATH, ONTOLOGY_PATH, ROOT, WORKSPACE, json_read, normalized, stable_id
from .candidate_generation import flatten_sentences

REFERENCE_HEAD_LABELS: dict[str, frozenset[str]] = {
    "model": frozenset({"Model", "AISystem"}),
    "method": frozenset({"Method", "AISystem", "TreatmentProcess"}),
    "approach": frozenset({"Method", "AISystem", "TreatmentProcess"}),
    "framework": frozenset({"Method", "AISystem"}),
    "technique": frozenset({"Method", "TreatmentProcess"}),
    "technology": frozenset({"Method", "Tool", "TreatmentProcess", "Intervention"}),
    "algorithm": frozenset({"Method", "Model"}),
    "procedure": frozenset({"Method", "TreatmentProcess"}),
    "system": frozenset({"AISystem", "Model", "Tool", "WaterSystem", "TreatmentPlant", "TreatmentTrain", "TreatmentUnit"}),
    "platform": frozenset({"AISystem", "Tool", "SoftwareArtifact"}),
    "process": frozenset({"TreatmentProcess"}),
    "agent": frozenset({"Agent", "AISystem"}),
    "tool": frozenset({"Tool", "SoftwareArtifact"}),
    "dataset": frozenset({"Dataset", "DataSource"}),
    "database": frozenset({"Dataset", "DataSource"}),
    "configuration": frozenset({"TreatmentTrain", "TreatmentProcess", "Condition"}),
    "plant": frozenset({"TreatmentPlant"}),
    "reactor": frozenset({"TreatmentUnit"}),
    "result": frozenset({"Observation", "Claim", "Metric"}),
}
REFERENCE_PATTERN = re.compile(
    r"\b(?:(?P<determiner>this|that|the|our|new|current|proposed|present|previous|aforementioned|these|those)\s+"
    r"(?P<head>models?|methods?|systems?|platforms?|approaches?|frameworks?|process(?:es)?|agents?|tools?|"
    r"datasets?|databases?|techniques?|technologies?|algorithms?|procedures?|configurations?|plants?|reactors?|results?)"
    r"|(?P<pronoun>it|they|them))\b",
    re.I,
)
REFERENCE_EXCLUDED_LABELS = frozenset({"EvidenceFragment", "Mention", "Publication", "Person"})

def _reference_head(match: re.Match[str]) -> str:
    head = (match.group("head") or match.group("pronoun") or "").casefold()
    irregular = {"approaches": "approach", "processes": "process", "technologies": "technology"}
    if head in irregular:
        return irregular[head]
    return head[:-1] if head.endswith("s") and head not in {"this"} else head


def _reference_context(sentence_rows: list[dict], index: int, maximum_previous: int = 2) -> list[dict]:
    """Return an antecedent-oriented, exact 2-3 sentence window."""
    section_id = sentence_rows[index]["section"]["id"]
    positions = [index]
    position = index - 1
    while position >= 0 and len(positions) <= maximum_previous:
        if sentence_rows[position]["section"]["id"] != section_id:
            break
        positions.append(position)
        position -= 1
    positions.reverse()
    return [
        {
            "sentence_id": sentence_rows[position]["sentence"]["id"],
            "paragraph_id": sentence_rows[position]["paragraph"]["id"],
            "role": "target" if position == index else "previous",
            "text": sentence_rows[position]["sentence"]["text"],
            "pages": sentence_rows[position]["sentence"].get("pages", []),
        }
        for position in positions
    ]


def _reference_source(row: dict, context: list[dict], match: re.Match[str]) -> dict:
    sentence = row["sentence"]
    return {
        "kind": "sentence_span",
        "sentence_id": sentence["id"],
        "paragraph_id": row["paragraph"]["id"],
        "section_id": row["section"]["id"],
        "section_title": row["section"].get("title", ""),
        "pages": sentence.get("pages", []),
        "start_char": match.start(),
        "end_char": match.end(),
        "quote": sentence["text"][match.start():match.end()],
        "evidence_quote": sentence["text"],
        "context_sentence_ids": [item["sentence_id"] for item in context],
        "context_sentences": context,
        "context_policy": "target sentence plus up to two preceding narrative sentences in the same section",
    }


def _antecedent_candidate(mention: dict, distance: int) -> dict:
    source = mention["source"]
    confidence = {0: .98, 1: .93, 2: .82}.get(distance, .70)
    return {
        "target_mention_id": mention["mention_id"],
        "canonical_id": mention.get("canonical_id", ""),
        "canonical_name": mention.get("canonical_name") or mention["surface_text"],
        "label": mention["label"],
        "surface_text": mention["surface_text"],
        "antecedent_sentence_id": source["sentence_id"],
        "antecedent_evidence_quote": source.get("evidence_quote", ""),
        "antecedent_context_sentences": source.get("context_sentences", []),
        "antecedent_start_char": source.get("start_char"),
        "antecedent_end_char": source.get("end_char"),
        "antecedent_pages": source.get("pages", []),
        "sentence_distance": distance,
        "confidence": confidence,
    }


def _candidate_identity(candidate: dict) -> tuple[str, str]:
    identity = candidate.get("canonical_id") or normalized(candidate["canonical_name"])
    return candidate["label"], identity


def resolve_references(
    parsed: dict,
    mentions: list[dict],
    *,
    allow_deterministic_auto: bool = True,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Resolve generic surface references without ever making them canonical nodes.

    Automatic resolution is deliberately narrow: a compatible explicit noun phrase must have one nearest
    canonical antecedent no more than one sentence back. Bare ``it`` only auto-resolves when exactly one
    candidate precedes it in the same sentence. Ambiguous but bounded cases are queued for a person; cases
    with no defensible finite candidate set are audit-logged as ignored.
    """
    spec = json_read(CONFIG_PATH).get("reference_resolution", {})
    maximum_previous = int(spec.get("maximum_previous_sentences", 2))
    automatic_maximum = int(spec.get("automatic_maximum_sentence_distance", 1))
    maximum_review_candidates = int(spec.get("maximum_human_review_candidates", 5))
    rows = flatten_sentences(parsed)
    sentence_positions = {row["sentence"]["id"]: index for index, row in enumerate(rows)}
    sentence_sections = {row["sentence"]["id"]: row["section"]["id"] for row in rows}
    mentions_by_sentence: dict[str, list[dict]] = defaultdict(list)
    for mention in mentions:
        source = mention.get("source", {})
        sentence_id = source.get("sentence_id")
        if (
            source.get("kind") == "sentence_span"
            and sentence_id in sentence_positions
            and mention["label"] not in REFERENCE_EXCLUDED_LABELS
            and not mention.get("reference_resolution")
        ):
            mentions_by_sentence[sentence_id].append(mention)

    resolved: list[dict] = []
    review: list[dict] = []
    ignored: list[dict] = []
    document_id = parsed["document"]["id"]
    for index, row in enumerate(rows):
        sentence = row["sentence"]
        context = _reference_context(rows, index, maximum_previous)
        for match in REFERENCE_PATTERN.finditer(sentence["text"]):
            head = _reference_head(match)
            is_pronoun = bool(match.group("pronoun"))
            determiner = (match.group("determiner") or "").casefold()
            # In "results suggest that model capability...", ``that`` is a
            # complementizer and the phrase is a class-level statement, not a
            # reference to the previous named model.
            preceding = sentence["text"][max(0, match.start() - 28):match.start()]
            following = sentence["text"][match.end():match.end() + 28]
            if (
                determiner == "that"
                and re.search(r"\b(?:suggest|show|indicate|reveal|confirm)(?:s|ed)?\s*$", preceding, re.I)
                and re.match(r"\s+(?:capabilit|performance|behavior|behaviour|selection|choice|output)", following, re.I)
            ):
                continue
            strong_determiner = determiner in {"this", "that", "our", "new", "current", "proposed", "present", "aforementioned"}
            is_plural = head in {"they", "them"} or (match.group("head") or "").casefold().endswith("s")
            expected_labels = REFERENCE_HEAD_LABELS.get(head)
            candidates: list[dict] = []
            for candidate_sentence_id, candidate_mentions in mentions_by_sentence.items():
                if sentence_sections[candidate_sentence_id] != row["section"]["id"]:
                    continue
                candidate_position = sentence_positions[candidate_sentence_id]
                distance = index - candidate_position
                if distance < 0 or distance > maximum_previous:
                    continue
                for mention in candidate_mentions:
                    source = mention["source"]
                    if distance == 0 and (source.get("end_char") is None or source["end_char"] > match.start()):
                        continue
                    if expected_labels and mention["label"] not in expected_labels:
                        continue
                    candidates.append(_antecedent_candidate(mention, distance))

            # Multiple occurrences of the same canonical node are one antecedent choice; retain the nearest evidence.
            best_by_entity: dict[tuple[str, str], dict] = {}
            for candidate in sorted(candidates, key=lambda item: (item["sentence_distance"], -item["confidence"], item["antecedent_start_char"] or 0)):
                best_by_entity.setdefault(_candidate_identity(candidate), candidate)
            candidates = sorted(best_by_entity.values(), key=lambda item: (item["sentence_distance"], -item["confidence"], item["canonical_name"].casefold()))

            # In an explicit agentic-architecture sentence, "this system/platform" denotes the
            # named AI system rather than a nearby simulator/tool. This remains deterministic
            # only when one AI-system identity exists inside the traceable context window.
            contextual_type_override = False
            if head in {"system", "platform"} and re.search(
                r"\b(?:agents?|agentic|orchestrator|architecture|LLM|function-call|workflow|limited|limitation|predefined|offline)\b",
                sentence["text"], re.I,
            ):
                ai_candidates = [candidate for candidate in candidates if candidate["label"] == "AISystem"]
                ai_identities = {_candidate_identity(candidate) for candidate in ai_candidates}
                if len(ai_identities) == 1:
                    candidates = ai_candidates
                    contextual_type_override = True

            source = _reference_source(row, context, match)
            resolution_id = stable_id("reference", document_id, sentence["id"], match.start(), match.end())
            base = {
                "resolution_id": resolution_id,
                "document_id": document_id,
                "surface_text": source["quote"],
                "reference_head": head,
                "source": source,
                "candidate_targets": candidates[:maximum_review_candidates],
            }
            nearest_distance = candidates[0]["sentence_distance"] if candidates else None
            nearest = [candidate for candidate in candidates if candidate["sentence_distance"] == nearest_distance]
            # A definite reference immediately after an explicit introduction is
            # deterministic: one named candidate, one sentence away, with the
            # antecedent supplied by an introduction statement.
            definite_introduction = bool(
                determiner == "the"
                and head in {"framework", "system", "platform"}
                and nearest_distance == 1
                and len(nearest) == 1
                and re.search(
                    r"\b(?:introduc(?:e|es|ed|ing)|propos(?:e|es|ed|ing)|present(?:s|ed|ing)?)\b",
                    nearest[0].get("antecedent_evidence_quote", ""),
                    re.I,
                )
            )
            auto_target = None
            allowed_auto_distance = maximum_previous if contextual_type_override else automatic_maximum
            if allow_deterministic_auto and not is_plural and not is_pronoun and (strong_determiner or definite_introduction) and nearest_distance is not None and nearest_distance <= allowed_auto_distance and len(nearest) == 1:
                same_sentence_gap = match.start() - (nearest[0].get("antecedent_end_char") or match.start())
                if nearest_distance > 0 or same_sentence_gap <= 250:
                    auto_target = nearest[0]
            elif allow_deterministic_auto and head == "it" and len(candidates) == 1 and nearest_distance == 0:
                same_sentence_gap = match.start() - (candidates[0].get("antecedent_end_char") or match.start())
                if same_sentence_gap <= 200:
                    auto_target = candidates[0]

            if auto_target:
                confidence = .97 if not is_pronoun and nearest_distance == 0 else (.93 if not is_pronoun else .90)
                mention_id = stable_id("mention", resolution_id, auto_target["target_mention_id"])
                resolved.append({
                    "candidate_id": resolution_id,
                    "mention_id": mention_id,
                    "document_id": document_id,
                    "label": auto_target["label"],
                    "surface_text": source["quote"],
                    "canonical_name": auto_target["canonical_name"],
                    "canonical_id": auto_target.get("canonical_id", ""),
                    "confidence": confidence,
                    "extraction_method": "deterministic_reference_resolution",
                    "status": "accepted",
                    "source": source,
                    "reference_resolution": {
                        "status": "auto_resolved",
                        "resolution_id": resolution_id,
                        "confidence": confidence,
                        "target_mention_id": auto_target["target_mention_id"],
                        "target_canonical_name": auto_target["canonical_name"],
                        "antecedent_sentence_id": auto_target["antecedent_sentence_id"],
                        "antecedent_evidence_quote": auto_target["antecedent_evidence_quote"],
                        "reason": "Exactly one nearest ontology-compatible antecedent was found in the current or immediately preceding sentence.",
                    },
                    "validation": {
                        "outcome": "accept",
                        "judge": "deterministic_reference_resolver",
                        "llm_used": False,
                        "reason": "The vague surface span was retained only as a traceable mention of a specific existing node.",
                    },
                })
            elif candidates and len(candidates) <= maximum_review_candidates and nearest_distance is not None and nearest_distance <= maximum_previous:
                review.append({
                    **base,
                    "status": "pending",
                    "recommended_target_mention_id": candidates[0]["target_mention_id"],
                    "reason": "More than one plausible antecedent exists, the reference is plural/pronominal, or the only antecedent is two sentences away.",
                })
            else:
                ignored.append({
                    **base,
                    "status": "ignored",
                    "reason": f"No sufficiently small, ontology-compatible antecedent set was found within the preceding {maximum_previous} sentences.",
                })
    return resolved, review, ignored


def _llm_resolved_reference(item: dict, candidate: dict, judgment: dict) -> dict:
    audit = judgment.get("_llm_audit", {})
    confidence = float(judgment["confidence"])
    return {
        "candidate_id": item["resolution_id"],
        "mention_id": stable_id(
            "mention", item["resolution_id"], candidate["target_mention_id"], "llm_reference"
        ),
        "document_id": item["document_id"],
        "label": candidate["label"],
        # Preserve what the paper literally says. canonical_name is the semantic replacement.
        "surface_text": item["surface_text"],
        "canonical_name": candidate["canonical_name"],
        "canonical_id": candidate.get("canonical_id", ""),
        "confidence": confidence,
        "extraction_method": "contextual_llm_reference_resolution",
        "status": "accepted",
        "source": item["source"],
        "reference_resolution": {
            "status": "llm_resolved",
            "resolution_id": item["resolution_id"],
            "confidence": confidence,
            "target_mention_id": candidate["target_mention_id"],
            "target_canonical_name": candidate["canonical_name"],
            "antecedent_sentence_id": candidate["antecedent_sentence_id"],
            "antecedent_evidence_quote": candidate["antecedent_evidence_quote"],
            "decision": "SAME",
            "reason": judgment["reason"],
            "provider": audit.get("provider", ""),
            "model": audit.get("model", ""),
            "request_hash": audit.get("request_hash", ""),
            "prompt_version": audit.get("prompt_version", ""),
        },
        "validation": {
            "outcome": "accept",
            "judge": "contextual_llm_reference_judge",
            "llm_used": True,
            "reason": (
                "The model compared the vague mention and the specific candidate in context and "
                "returned SAME above the configured confidence threshold."
            ),
        },
    }


def optional_llm_reference_resolution(
    parsed: dict,
    mentions: list[dict],
    progress: Callable[[str, str, int], None],
    *,
    reference_judge=None,
) -> tuple[list[dict], list[dict], list[dict], list[dict], dict]:
    """Resolve each vague mention by explicitly comparing it with candidate nodes in context."""
    config = json_read(CONFIG_PATH)
    spec = config.get("llm_reference_judge", {})
    status = {
        "enabled": bool(spec.get("enabled")),
        "used": False,
        "provider": spec.get("provider", "ollama"),
        "model": spec.get("model", ""),
        "reason": "",
    }
    deterministic, pair_review, ignored = resolve_references(
        parsed, mentions, allow_deterministic_auto=True
    )
    status["deterministic_resolved"] = len(deterministic)
    if not pair_review or not spec.get("enabled"):
        status["reason"] = (
            "No contextual reference pairs were found."
            if not pair_review
            else "Disabled in configuration; candidate pairs require human review."
        )
        return deterministic, pair_review, ignored, [], status

    try:
        judge = reference_judge
        if judge is None:
            from Pipeline.Knowledge_Graph_Core.llm_reference_judge import MandatoryLLMReferenceJudge
            from Pipeline.Knowledge_Graph_Core.ontology import Ontology

            judge = MandatoryLLMReferenceJudge(
                Ontology(ONTOLOGY_PATH), config, ROOT / "cache" / "llm_reference_judge"
            )
        judge.verify_runtime()
    except Exception as exc:
        status["reason"] = str(exc).splitlines()[0]
        return deterministic, pair_review, ignored, [], status

    pairs: list[dict] = []
    pair_index: dict[str, tuple[dict, dict]] = {}
    for item in pair_review:
        for candidate in item["candidate_targets"]:
            pair_id = stable_id(
                "reference-pair", item["resolution_id"], candidate["target_mention_id"]
            )
            pair_index[pair_id] = (item, candidate)
            pairs.append({
                "pair_id": pair_id,
                "reference": {
                    "surface_text": item["surface_text"],
                    "reference_head": item["reference_head"],
                    "sentence_id": item["source"]["sentence_id"],
                    "evidence_quote": item["source"]["evidence_quote"],
                    "context_sentences": item["source"].get("context_sentences", []),
                },
                "candidate": {
                    "target_mention_id": candidate["target_mention_id"],
                    "surface_text": candidate["surface_text"],
                    "canonical_name": candidate["canonical_name"],
                    "ontology_label": candidate["label"],
                    "sentence_id": candidate["antecedent_sentence_id"],
                    "evidence_quote": candidate["antecedent_evidence_quote"],
                    "context_sentences": candidate.get("antecedent_context_sentences", []),
                    "sentence_distance": candidate["sentence_distance"],
                },
            })

    progress(
        "resolving",
        "Comparing each vague mention with possible specific nodes using their sentence contexts.",
        59,
    )
    try:
        judgments = judge.judge(pairs)
    except Exception as exc:
        status["reason"] = f"Contextual reference model became unavailable: {str(exc).splitlines()[0]}"
        return deterministic, pair_review, ignored, judge.audit_records, status

    threshold = float(spec.get("same_confidence_threshold", .8))
    judgments_by_resolution: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for pair_id, judgment in judgments.items():
        item, candidate = pair_index[pair_id]
        judgments_by_resolution[item["resolution_id"]].append((candidate, judgment))

    resolved: list[dict] = list(deterministic)
    remaining: list[dict] = []
    for item in pair_review:
        evaluated = judgments_by_resolution.get(item["resolution_id"], [])
        same = [
            (candidate, judgment)
            for candidate, judgment in evaluated
            if judgment["decision"] == "SAME" and float(judgment["confidence"]) >= threshold
        ]
        compact_judgments = [{
            "target_mention_id": candidate["target_mention_id"],
            "target_canonical_name": candidate["canonical_name"],
            "decision": judgment["decision"],
            "confidence": judgment["confidence"],
            "reason": judgment["reason"],
            "request_hash": judgment.get("_llm_audit", {}).get("request_hash", ""),
        } for candidate, judgment in evaluated]
        all_other_decisions_are_different = all(
            judgment["decision"] == "DIFFERENT"
            for candidate, judgment in evaluated
            if candidate["target_mention_id"] != same[0][0]["target_mention_id"]
        ) if len(same) == 1 else False
        if len(same) == 1 and all_other_decisions_are_different:
            resolved.append(_llm_resolved_reference(item, same[0][0], same[0][1]))
        elif evaluated and all(judgment["decision"] == "DIFFERENT" for _, judgment in evaluated):
            ignored.append({
                **item,
                "status": "ignored",
                "llm_pair_judgments": compact_judgments,
                "reason": "The contextual model judged every supplied candidate to be a different entity.",
            })
        else:
            remaining.append({
                **item,
                "status": "pending",
                "llm_pair_judgments": compact_judgments,
                "reason": (
                    "The contextual model was uncertain, below the SAME confidence threshold, or "
                    "identified more than one possible match; human review is required."
                ),
            })

    status.update({
        "used": True,
        "reason": "The local model completed contextual pairwise reference adjudication.",
        "pairs": len(pairs),
        "resolved": len(resolved),
    })
    return resolved, remaining, ignored, judge.audit_records, status
