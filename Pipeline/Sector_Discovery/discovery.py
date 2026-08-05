from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable


@dataclass(frozen=True)
class GroundedCandidate:
    start: int
    end: int
    label: str
    canonical_name: str
    canonical_id: str
    method: str = "grammatical_pattern"
    confidence: float = 0.94


def _normalized(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def _stable_id(prefix: str, *values: object) -> str:
    raw = "\u241f".join(str(value) for value in values)
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:20]}"


def load_sector_profile(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    vocabulary = data.get("construction_vocabulary", {})
    required = {"system_heads", "agent_heads", "model_cues", "task_patterns", "dataset_patterns"}
    missing = sorted(required - set(vocabulary))
    if missing:
        raise ValueError("Sector profile is missing construction vocabulary: " + ", ".join(missing))
    return data


_INTRODUCTION_RE = re.compile(
    r"\b(?:(?:here|(?:in\s+)?this\s+(?:study|paper|work))\s*,?\s*)?(?:we\s+)?"    r"(?:introduc(?:e|es|ed|ing)|propos(?:e|es|ed|ing)|present(?:s|ed|ing)?|"
    r"develop(?:s|ed|ing)?(?:\s+and\s+(?:validat|evaluat)(?:e|es|ed|ing))?|"
    r"propos(?:e|es|ed|ing)\s+and\s+evaluat(?:e|es|ed|ing))\s+"
    r"an?\s+(?P<name>(?-i:[a-z])[^.;]{2,115}?\b"    r"(?:system|framework|platform|assistant|agent|pipeline))"
    r"(?=\s*(?:[.,;]|that|which|in\s+which|where|designed|capable|for\s|to\s|$))",
    re.I,
)

_AGENT_IDENTIFIER_RE = re.compile(r"(?<![A-Za-z0-9])(?P<prefix>[A-Z]{2,6})[\s_-]+(?P<role>[a-z][A-Za-z0-9-]{2,24})(?:\s+Lite)?\b")
_MODEL_IDENTIFIER_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Z][A-Za-z0-9]*(?:[-:./][A-Za-z0-9.]+)+|"
    r"[A-Z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*)(?![A-Za-z0-9])"
)
_MODEL_EXCLUSIONS = {
    "ai", "dl", "ea", "ecwt", "gb", "gbr", "gguf", "gnn", "gpt", "llm", "llms", "mcdi", "ml", "pdf", "rag", "rmse", "sec", "tee",
    "fig", "table", "prompt", "python", "raspberry", "pi", "epanet", "wntr", "wds", "dwds",
    "wdn", "wdns", "wdss", "wwtp", "wwtps", "anytown", "autogen", "gps x", "lca", "prvs", "sota", "physics informed", "deepseek ai",
}
_AGENT_ROLE_EXCLUSIONS = {
    "agent", "agents", "model", "models", "system", "systems", "framework", "platform",
    "and", "are", "based", "driven", "employed", "feedback", "for", "from", "functions",
    "in", "into", "management", "of", "operations", "optimization", "parameters", "per",
    "results", "state", "trajectory", "values", "with",
}
_AGENT_ENUMERATION_RE = re.compile(
    r"\b(?:(?:two|three|four|five|six|several|multiple|\d+)\s+(?:expert(?:-level)?\s+)?agents?|"
    r"agents?\s*[:\-\u2013\u2014])\b",
    re.I,
)


def _descriptor_tokens(value: str, profile: dict) -> set[str]:
    stopwords = {_normalized(word) for word in profile["construction_vocabulary"].get("identity_stopwords", [])}
    return {token for token in _normalized(value).split() if token not in stopwords and len(token) > 1}


def _cluster_system_candidates(rows: list[dict], document_id: str, profile: dict) -> dict[str, list[GroundedCandidate]]:
    raw: list[tuple[str, GroundedCandidate]] = []
    heads = tuple(profile["construction_vocabulary"]["system_heads"])
    for row in rows:
        text = row["text"]
        for match in _INTRODUCTION_RE.finditer(text):
            surface = match.group("name").strip(" ,;:")
            if not surface.casefold().endswith(heads) or len(surface.split()) > 14:
                continue
            raw.append((row["sentence_id"], GroundedCandidate(
                match.start("name"), match.start("name") + len(surface), "AISystem", surface,
                _stable_id("entity", document_id, "AISystem", _normalized(surface)), confidence=0.96,
            )))
    if not raw:
        return {}

    parent = list(range(len(raw)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    token_sets = [_descriptor_tokens(candidate.canonical_name, profile) for _, candidate in raw]
    for left in range(len(raw)):
        for right in range(left + 1, len(raw)):
            if token_sets[left] and token_sets[right] and token_sets[left] & token_sets[right]:
                union(left, right)

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(raw)):
        groups[find(index)].append(index)

    result: dict[str, list[GroundedCandidate]] = defaultdict(list)
    for indices in groups.values():
        canonical = max(
            (raw[index][1].canonical_name for index in indices),
            key=lambda value: (
                bool(re.search(r"\b(?:artificial\s+intelligence|AI)\b", value, re.I)),
                len(_descriptor_tokens(value, profile)), len(value),
            ),
        )
        canonical_id = _stable_id("entity", document_id, "AISystem", _normalized(canonical))
        for index in indices:
            sentence_id, candidate = raw[index]
            result[sentence_id].append(replace(candidate, canonical_name=canonical, canonical_id=canonical_id))
    return result


def _discover_agent_aliases(rows: list[dict]) -> dict[tuple[str, str], str]:
    observed: dict[str, dict[str, str]] = defaultdict(dict)
    for row in rows:
        text = row["text"]
        # Only an explicit local enumeration establishes an acronym-role family.
        # Document-wide accumulation produced LLM_into and WDN_environment.
        if not _AGENT_ENUMERATION_RE.search(text):
            continue
        matches = list(_AGENT_IDENTIFIER_RE.finditer(text))
        per_prefix: dict[str, list[re.Match]] = defaultdict(list)
        for match in matches:
            per_prefix[match.group("prefix").casefold()].append(match)
        for prefix_key, family in per_prefix.items():
            valid = [
                match for match in family
                if match.group("role").casefold() not in _AGENT_ROLE_EXCLUSIONS
            ]
            if len({match.group("role").casefold() for match in valid}) < 2:
                continue
            for match in valid:
                prefix, role = match.group("prefix"), match.group("role")
                observed[prefix_key][role.casefold()] = f"{prefix}_{role}"
    return {
        (prefix, role): canonical
        for prefix, roles in observed.items()
        if len(roles) >= 2
        for role, canonical in roles.items()
    }

def _agent_candidates(rows: list[dict], document_id: str) -> dict[str, list[GroundedCandidate]]:
    aliases = _discover_agent_aliases(rows)
    result: dict[str, list[GroundedCandidate]] = defaultdict(list)
    for row in rows:
        text = row["text"]
        for match in _AGENT_IDENTIFIER_RE.finditer(text):
            canonical = aliases.get((match.group("prefix").casefold(), match.group("role").casefold()))
            if not canonical:
                continue
            result[row["sentence_id"]].append(GroundedCandidate(
                match.start(), match.end(), "Agent", canonical,
                _stable_id("entity", document_id, "Agent", _normalized(canonical)), confidence=0.97,
            ))
    return result


def _valid_model_surface(surface: str, *, allow_uppercase: bool = False) -> bool:
    normalized_surface = _normalized(surface)
    if normalized_surface in _MODEL_EXCLUSIONS:
        return False
    if re.search(r"(?:agent|framework|platform)$", surface, re.I):
        return False
    if re.search(r"(?:EPANET|WDS|WDN)based$", surface, re.I):
        return False
    if re.fullmatch(r"[A-Z][a-z]?(?:/|-)[A-Z][a-z]?", surface):
        return False
    if surface.isupper() and not any(char.isdigit() for char in surface) and not allow_uppercase:
        return False
    return not bool(re.fullmatch(
        r"(?:January|February|March|April|May|June|July|August|September|October|November|December)-\d{4}",
        surface,
        re.I,
    ))


_ENUMERATED_MODEL_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:Le\s+Chat|Claude\s+Sonnet(?:\s+\d+(?:\.\d+)*)?|"
    r"[A-Z][A-Za-z0-9]*(?:[-:./][A-Za-z0-9.]+)*)(?![A-Za-z0-9])"
)


def _model_candidates(rows: list[dict], document_id: str, profile: dict) -> dict[str, list[GroundedCandidate]]:
    cues = "|".join(re.escape(value) for value in profile["construction_vocabulary"]["model_cues"])
    result: dict[str, list[GroundedCandidate]] = defaultdict(list)
    for row in rows:
        text = row["text"]
        if not re.search(rf"\b(?:{cues})\b", text, re.I):
            continue
        cue_matches = list(re.finditer(rf"\b(?:{cues})\b", text, re.I))
        matches = list(_MODEL_IDENTIFIER_RE.finditer(text))
        enumerated_ids: set[tuple[int, int]] = set()
        for region in re.finditer(r"\b(?:include(?:s|d|ing)?|such as)\s+(?P<body>[^.;]{2,220})", text, re.I):
            prefix = text[max(0, region.start() - 55):region.start()]
            if not re.search(rf"\b(?:{cues})\b", prefix, re.I):
                continue
            for item in _ENUMERATED_MODEL_RE.finditer(text, region.start("body"), region.end("body")):
                matches.append(item)
                enumerated_ids.add((item.start(), item.end()))
        for match in matches:
            distance = min(abs(match.start() - cue.start()) for cue in cue_matches)
            enumerated = (match.start(), match.end()) in enumerated_ids
            if not enumerated and distance > 55:
                continue
            surface = match.group(0).strip(" ,;:()")
            suffix = text[match.end():match.end() + 12]
            if re.match(r"\s+et\s+al\b", suffix, re.I):
                continue
            allow_uppercase = enumerated or distance <= 25 or "-" in surface
            if not _valid_model_surface(surface, allow_uppercase=allow_uppercase):
                continue
            if re.fullmatch(r"[ST]\d+", surface) and re.search(r"\b(?:Table|Fig\.?)\s*$", text[max(0, match.start() - 15):match.start()], re.I):
                continue
            canonical = re.sub(r"\s*-\s*", "-", surface)
            result[row["sentence_id"]].append(GroundedCandidate(
                match.start(), match.start() + len(surface), "Model", canonical,
                _stable_id("entity", "Model", _normalized(canonical)), confidence=0.92,
            ))

    return result



def _profile_canonical(label: str, surface: str) -> str:
    value = _normalized(surface)
    if label == "Dataset":
        sample_match = re.search(r"(?P<count>\d{1,3}(?:,\s*\d{3})+|\d+)\s+sample\s+data", surface, re.I)
        if sample_match:
            count = re.sub(r"\D", "", sample_match.group("count"))
            return f"{count} sample data"
        if value == "eleven wdss":
            return "eleven-WDS benchmark dataset"

    canonical = {
        ("TreatmentProcess", "mcdi"): "membrane capacitive deionization",
        ("TreatmentProcess", "membrane capacitive deionization"): "membrane capacitive deionization",
        ("TreatmentProcess", "electrochemical water treatment"): "electrochemical water treatment",
        ("Model", "gradient boosting regressor"): "gradient boosting regression",
        ("Metric", "r2"): "coefficient of determination (R2)",
        ("Metric", "coefficient of determination"): "coefficient of determination (R2)",
        ("Metric", "conciseness score"): "conciseness",
        ("Metric", "factual consistency"): "factual consistency",
        ("Metric", "reference accuracy"): "reference accuracy",
        ("Metric", "thermodynamic energy efficiency"): "thermodynamic energy efficiency",
        ("Metric", "specific energy consumption"): "specific energy consumption",
        ("Task", "model calibration"): "hydraulic model calibration",
    }.get((label, value))
    if canonical:
        return canonical
    return surface


def _canonicalize_counted_datasets(result: dict[str, list[GroundedCandidate]], document_id: str) -> None:
    groups: dict[tuple[str, str], list[tuple[str, int, GroundedCandidate]]] = defaultdict(list)
    for sentence_id, candidates in result.items():
        for index, candidate in enumerate(candidates):
            if candidate.label != "Dataset":
                continue
            match = re.match(r"(?:corpus\s+of\s+)?(?P<count>\d+)\s+.*?(?P<kind>papers|studies|data\s+points)$", candidate.canonical_name, re.I)
            if match:
                kind = "paper corpus" if match.group("kind").casefold() in {"papers", "studies"} else "data points"
                groups[(match.group("count"), kind)].append((sentence_id, index, candidate))
    for (count, kind), members in groups.items():
        if kind == "paper corpus":
            canonical = f"{count}-paper corpus"
        else:
            canonical = max((candidate.canonical_name for _, _, candidate in members), key=len)
        canonical_id = _stable_id("entity", document_id, "Dataset", _normalized(canonical))
        for sentence_id, index, candidate in members:
            result[sentence_id][index] = replace(candidate, canonical_name=canonical, canonical_id=canonical_id)


def _profile_pattern_candidates(rows: list[dict], document_id: str, profile: dict) -> dict[str, list[GroundedCandidate]]:
    result: dict[str, list[GroundedCandidate]] = defaultdict(list)
    vocabulary = profile["construction_vocabulary"]
    specs = (
        ("Task", vocabulary.get("task_patterns", [])),
        ("Dataset", vocabulary.get("dataset_patterns", [])),
        ("Model", vocabulary.get("model_patterns", [])),
        ("Tool", vocabulary.get("tool_patterns", [])),
        ("Method", vocabulary.get("method_patterns", [])),
        ("Parameter", vocabulary.get("parameter_patterns", [])),
        ("Condition", vocabulary.get("condition_patterns", [])),
        ("Challenge", vocabulary.get("challenge_patterns", [])),
        ("TreatmentProcess", vocabulary.get("process_patterns", [])),
        ("Metric", vocabulary.get("metric_patterns", [])),
    )
    for row in rows:
        for label, patterns in specs:
            for source in patterns:
                for match in re.finditer(rf"\b(?:{source})\b", row["text"], re.I):
                    surface = match.group(0).strip()
                    normalized_surface = _normalized(surface)
                    if normalized_surface in {"analysis", "control", "prediction", "optimization", "monitoring", "the dataset", "a dataset"}:
                        continue
                    if label == "Dataset" and not (any(char.isdigit() for char in surface) or re.search(
                        r"\b(?:standardized|benchmark|corpus|question|eleven|WDSs?|MCDI|EPANET|WaterRAG)\b", surface, re.I
                    )):
                        continue
                    canonical = _profile_canonical(label, surface)
                    result[row["sentence_id"]].append(GroundedCandidate(
                        match.start(), match.end(), label, canonical,
                        _stable_id("entity", document_id, label, _normalized(canonical)),
                        method="environmental_profile_pattern", confidence=0.90,
                    ))
    _canonicalize_counted_datasets(result, document_id)
    return result

def discover_document_candidates(rows: Iterable[dict], document_id: str, profile: dict) -> dict[str, list[GroundedCandidate]]:
    """Discover exact, ontology-typed spans without relying on a previously seen lexicon entry."""
    materialized = list(rows)
    channels = (
        _cluster_system_candidates(materialized, document_id, profile),
        _agent_candidates(materialized, document_id),
        _model_candidates(materialized, document_id, profile),
        _profile_pattern_candidates(materialized, document_id, profile),
    )
    result: dict[str, list[GroundedCandidate]] = defaultdict(list)
    seen: set[tuple[str, int, int, str]] = set()
    for channel in channels:
        for sentence_id, candidates in channel.items():
            for candidate in candidates:
                key = (sentence_id, candidate.start, candidate.end, candidate.label)
                if key not in seen:
                    seen.add(key)
                    result[sentence_id].append(candidate)
    return result
