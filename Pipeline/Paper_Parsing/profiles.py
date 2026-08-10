from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ParsingProfile:
    id: str
    publisher: str | None
    inline_abstract: bool = False
    unlabelled_abstract: bool = False
    abstract_same_column: bool = False
    styled_unnumbered_headings: bool = False
    unnumbered_headings: frozenset[str] = frozenset()
    back_matter: frozenset[str] = frozenset()


COMMON_BACK_MATTER = frozenset({
    "references", "reference", "creditauthorshipcontributionstatement",
    "declarationofcompetinginterest", "conflictofinterest", "dataavailability",
    "acknowledgments", "acknowledgements", "supplementarymaterials",
})

WATER_RESEARCH = ParsingProfile(
    id="elsevier_water_research",
    publisher="Elsevier",
    abstract_same_column=True,
    back_matter=COMMON_BACK_MATTER,
)

ACS_EST = ParsingProfile(
    id="acs_est",
    publisher="American Chemical Society",
    inline_abstract=True,
    abstract_same_column=True,
    unnumbered_headings=frozenset({
        "INTRODUCTION", "DATA AND METHODS", "METHODS", "RESULTS",
        "RESULTS AND DISCUSSION", "DISCUSSION", "CONCLUSION", "CONCLUSIONS",
    }),
    back_matter=COMMON_BACK_MATTER | frozenset({
        "associatedcontent", "authorinformation", "additionalnotes",
    }),
)

FRONTIERS = ParsingProfile(
    id="frontiers_open_access",
    publisher="Frontiers",
    unlabelled_abstract=True,
    back_matter=COMMON_BACK_MATTER | frozenset({"ethicsstatement", "authorcontributions"}),
)

IEEE_ARXIV = ParsingProfile(
    id="ieee_arxiv_preprint",
    publisher="arXiv / IEEE-style preprint",
    inline_abstract=True,
    styled_unnumbered_headings=True,
    back_matter=COMMON_BACK_MATTER | frozenset({"references", "acknowledgment", "acknowledgments"}),
)

CONFERENCE_PREPRINT = ParsingProfile(
    id="conference_preprint",
    publisher="Preprint",
    styled_unnumbered_headings=True,
    back_matter=COMMON_BACK_MATTER | frozenset({"references", "acknowledgment", "acknowledgments"}),
)

GENERIC = ParsingProfile(id="generic_research_paper", publisher=None, back_matter=COMMON_BACK_MATTER)


def compact_label(text: str) -> str:
    """Compact label. It helps convert diverse PDF layouts into clean, sentence-addressable paper text."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def visible_heading(text: str) -> str:
    """Handle visible heading for this stage. It helps convert diverse PDF layouts into clean, sentence-addressable paper text."""
    visible = re.sub(r"^[^A-Za-z0-9]+\s*", "", text).strip()
    # Small-caps fonts are sometimes extracted with a space after the first
    # capital of every word: `I. I NTRODUCTION` or `R ELATED W ORK`.
    return re.sub(r"\b([A-Z])\s+(?=[A-Z]{2,}\b)", r"\1", visible)


def detect_profile(lines: list) -> ParsingProfile:
    """Detect profile. It helps convert diverse PDF layouts into clean, sentence-addressable paper text."""
    sample_lines = [line.text for line in lines[:400]]
    sample = " ".join(sample_lines).lower()
    if "environmental science & technology" in sample or "environscitechnol" in compact_label(sample):
        return ACS_EST
    if "water research" in sample:
        return WATER_RESEARCH
    if (
        "frontiers in water" in sample or "frontiersin.org" in sample
        or "doi 10.3389/" in sample or "doi: 10.3389/" in sample
        or ("open access" in sample and "edited by" in sample and "reviewed by" in sample)
    ):
        return FRONTIERS
    if any(re.match(r"^[IVX]+\.\s+[A-Z]\s+[A-Z]{2,}", text) for text in sample_lines):
        return IEEE_ARXIV
    compact_lines = {compact_label(text) for text in sample_lines}
    all_caps = [
        text for text in sample_lines
        if 1 <= len(re.findall(r"[A-Za-z]+", text)) <= 12
        and re.search(r"[A-Z]", text) and not re.search(r"[a-z]", text)
    ]
    if "abstract" in compact_lines and "introduction" in compact_lines and len(all_caps) >= 2:
        return CONFERENCE_PREPRINT
    return GENERIC


def is_profile_heading(
    text: str, profile: ParsingProfile, *, size: float | None = None,
    body_size: float | None = None, bold: bool = False, italic: bool = False,
) -> bool:
    """Check whether profile heading. It helps convert diverse PDF layouts into clean, sentence-addressable paper text."""
    clean = visible_heading(text).strip()
    if clean.upper() in profile.unnumbered_headings:
        return True
    if not profile.styled_unnumbered_headings or size is None or body_size is None:
        return False
    if not clean or len(clean) > 120 or clean.endswith((".", ",", ";", ":")):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z-]*", clean)
    if not words:
        return False
    styled = bold or italic or size >= body_size * 1.14
    roman = bool(re.match(r"^[IVX]+\.\s+", clean))
    all_caps = bool(words) and not re.search(r"[a-z]", clean)
    return bool(styled and len(words) <= 14 and (roman or all_caps or size >= body_size * 1.18))
