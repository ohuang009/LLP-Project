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

from Pipeline.Node_Pipeline.common import ROOT, WORKSPACE, json_read, jsonl_write, stable_id


CONFIG_PATH = ROOT / "config" / "relationship_extraction.json"
ONTOLOGY_PATH = ROOT / "ontology" / "ontology.json"
RELATIONSHIP_OUTPUTS = (
    "relationship_candidates.jsonl",
    "relationship_unknown_nodes.jsonl",
    "assertions.jsonl",
    "relationship_rejections.jsonl",
    "llm_relationship_judge_calls.jsonl",
    "canonical_relationships.jsonl",
)

QUANTITATIVE_RESULT_RE = re.compile(
    r"\b(?:remov(?:e|es|ed|al)|measur(?:e|es|ed)|observ(?:e|es|ed)|detect(?:s|ed)?|"
    r"predict(?:s|ed|ion)?|reduc(?:e|es|ed|tion)|increas(?:e|es|ed)|decreas(?:e|es|ed)|"
    r"achiev(?:e|es|ed)|efficien(?:cy|t)|concentration)\b",
    re.I,
)
MEASUREMENT_RE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>%|percent|mg(?:N)?/L|[Âµu]g/L|ng/L|ppm|ppb|Â°C|K|h|hr|hours?|min|minutes?|days?|gCO[^\s,.;)]*)(?=$|[^A-Za-z0-9])",
    re.I,
)
COUNT_RESULT_RE = re.compile(
    r"(?P<value>\d+)\s+(?P<unit>questions?|studies|articles|references|tasks|cases)\b",
    re.I,
)
NAMED_SCORE_RE = re.compile(
    r"(?P<metric>context recall|faithfulness|factuality|accuracy|correctness|relevance)"
    r"[^.;]{0,25}?(?P<value>0?\.\d+|\d+(?:\.\d+)?)",
    re.I,
)
CONDITION_RE = re.compile(
    r"(?P<name>pH|temperature|contact time|retention time|HRT|SRT|pressure|flow rate)\s*"
    r"(?:of|=|at|was|were)?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>Â°C|K|h|hr|hours?|min|minutes?|days?|bar|kPa|MPa|L/s|m3/day)?",
    re.I,
)

def _sentence_rows(parsed: dict) -> list[tuple[dict, dict, dict]]:
    return [
        (sentence, paragraph, section)
        for section in parsed.get("sections", [])
        for paragraph in section.get("paragraphs", [])
        for sentence in paragraph.get("sentences", [])
    ]


def _is_current_study_sentence(text: str, section_title: str = "") -> bool:
    """Conservatively avoid creating quantitative observations for plans/prior work."""
    if re.search(
        r"\b(?:future work|will aim|will be|would|could|may|might|expected to|plans? to|aims? to|potential to)\b",
        text, re.I,
    ):
        return False
    first_person = bool(re.search(
        r"\b(?:we|our|this (?:study|paper|work)|the proposed|our (?:model|method|approach))\b",
        text, re.I,
    ))
    citation = bool(re.search(
        r"\[[0-9,;\s-]+\]|\b[A-Z][A-Za-z-]+\s+et\s+al\.|"
        r"\([A-Z][A-Za-z-]+(?:\s+et\s+al\.)?,\s*\d{4}[a-z,]*\)",
        text,
    ))
    attributed_claim = bool(re.search(
        r"\b(?:proposed|developed|used|applied|demonstrated|reported|found|showed|achieved|optimized|trained|removed)\b",
        text, re.I,
    ))
    if citation and attributed_claim and not first_person:
        return False
    return not (
        re.search(r"\b(?:related work|literature review)\b", section_title, re.I)
        and not first_person
    )
