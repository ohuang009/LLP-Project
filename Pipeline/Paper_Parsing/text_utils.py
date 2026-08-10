from __future__ import annotations

import re
import unicodedata


SPACE_RE = re.compile(r"[ \t]+")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
SECTION_NUMBER_RE = re.compile(
    r"^(?:\d{1,2}(?:\.\d{1,2}){0,3}\.?|[IVX]+\.)\s+(?=[A-Z])", re.I
)
CAPTION_RE = re.compile(r"^(?:fig(?:ure)?|table|box|plate|map)\s*[.:]?\s*\d+", re.I)
FORMULA_RE = re.compile(r"^(?:=|[A-Za-z]\s*=|[\u2211\u222b\u221a]|\(?\d+\)?\s*$)")

ABBREVIATIONS = (
    "e.g.", "i.e.", "et al.", "fig.", "eq.", "dr.", "prof.", "vs.",
    "no.", "approx.", "ca.", "sp.", "spp.", "U.S.",
)

# These prefixes normally form genuine compounds. Other line-final hyphens are
# treated as typesetting hyphens and removed when the next line starts lowercase.
PRESERVE_HYPHEN_PREFIXES = {
    "net", "life", "real", "high", "low", "long", "short", "state",
    "data", "process", "user", "domain", "top", "large", "small", "well",
    "self", "multi", "cross", "carbon", "time", "case", "end", "of",
}

TERMINAL_RE = re.compile(
    r"[.!?][\"'\u201d\u2019)\]]*(?:\s*\d+(?:\s*[,\-\u2013\u2212]\s*\d+)*)?$"
)
LIST_ITEM_RE = re.compile(r"^(?:\(?\d{1,3}\)?[.)]|[a-z][.)]|[\u2022\u25aa])\s+", re.I)
LINE_END_HYPHENS = "-\u2010\u2011\u2012\u2013"


def normalize_text(text: str) -> str:
    """Normalize text. It helps convert diverse PDF layouts into clean, sentence-addressable paper text."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "")
    text = text.replace("\ufffd", "-")
    text = re.sub(
        r"(?<=[A-Za-z])\s*[\u2010\u2011\u2012\u2013]\s*(?=[A-Za-z])",
        "-",
        text,
    )
    text = re.sub(r"(?<=[A-Za-z])\s*['\u2019]\s*(?=[A-Za-z])", "'", text)
    text = SPACE_RE.sub(" ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(
        r"(?<=[.!?])\s+where\b",
        lambda match: match.group(0)[:-5] + "Where",
        text,
        flags=re.I,
    )
    return text.strip()


def join_lines(lines: list[str]) -> str:
    """Join lines. It helps convert diverse PDF layouts into clean, sentence-addressable paper text."""
    result = ""
    for raw in lines:
        line = normalize_text(raw)
        if not line:
            continue
        if result.endswith(tuple(LINE_END_HYPHENS)) and line[:1].islower():
            left_word = re.search(r"([A-Za-z]+)[\-\u2010\u2011\u2012\u2013]$", result)
            prefix = left_word.group(1).lower() if left_word else ""
            if prefix in PRESERVE_HYPHEN_PREFIXES:
                result = result[:-1] + "-" + line
            else:
                result = result[:-1] + line
        else:
            result = f"{result} {line}".strip()
    return normalize_text(result)


def has_terminal_punctuation(text: str) -> bool:
    """Check whether terminal punctuation. It helps convert diverse PDF layouts into clean, sentence-addressable paper text."""
    clean = normalize_text(text)
    return bool(TERMINAL_RE.search(clean)) or clean.endswith((":", ";"))


def starts_like_continuation(text: str) -> bool:
    """Check whether continuation. It helps convert diverse PDF layouts into clean, sentence-addressable paper text."""
    clean = normalize_text(text)
    clean = re.sub(r"^[\"'\u201c\u2018(\[]+", "", clean)
    return bool(clean) and (clean[0].islower() or clean[0] in ",;:)]")


def begins_list_item(text: str) -> bool:
    """Check whether list item. It helps convert diverse PDF layouts into clean, sentence-addressable paper text."""
    return bool(LIST_ITEM_RE.match(normalize_text(text)))


def split_sentences(text: str) -> list[str]:
    """Split sentences. It helps convert diverse PDF layouts into clean, sentence-addressable paper text."""
    protected = text
    period_marker = "\ue000"
    for abbreviation in ABBREVIATIONS:
        protected = re.sub(
            re.escape(abbreviation),
            lambda match: match.group(0).replace(".", period_marker),
            protected,
            flags=re.I,
        )
    sentences = SENTENCE_BOUNDARY_RE.split(protected)
    merged: list[str] = []
    citation_only = re.compile(r"^\d+(?:\s*[,\-–−]\s*\d+)*$")
    for sentence in sentences:
        if merged and citation_only.fullmatch(sentence.strip()):
            merged[-1] = f"{merged[-1]} {sentence.strip()}"
        else:
            merged.append(sentence)
    restored: list[str] = []
    for sentence in merged:
        sentence = sentence.replace(period_marker, ".")
        sentence = normalize_text(sentence)
        if sentence and has_terminal_punctuation(sentence):
            restored.append(sentence)
    return restored


def looks_like_caption(text: str) -> bool:
    """Check whether caption. It helps convert diverse PDF layouts into clean, sentence-addressable paper text."""
    return bool(CAPTION_RE.match(normalize_text(text)))


def looks_like_formula(text: str) -> bool:
    """Check whether formula. It helps convert diverse PDF layouts into clean, sentence-addressable paper text."""
    clean = normalize_text(text)
    symbol_ratio = sum(not c.isalnum() and not c.isspace() for c in clean) / max(len(clean), 1)
    return "cid:" in clean or bool(FORMULA_RE.match(clean)) or (symbol_ratio > 0.35 and len(clean) < 180)


def looks_like_heading(
    text: str, size: float, body_size: float, *, bold: bool = False, italic: bool = False
) -> bool:
    """Check whether heading. It helps convert diverse PDF layouts into clean, sentence-addressable paper text."""
    clean = normalize_text(text)
    if not clean or len(clean) > 140 or clean.endswith((".", ";", ",")):
        return False
    known = {
        "abstract", "introduction", "background", "methods", "materials and methods",
        "study area", "results", "discussion", "results and discussion", "conclusion",
        "conclusions", "limitations", "acknowledgements", "references",
    }
    compact = re.sub(r"\s+", "", clean).lower()
    if compact in {"abstract", "introduction", "methodology", "references", "conclusions"}:
        return True
    lowered = SECTION_NUMBER_RE.sub("", clean).lower()
    numbered = bool(SECTION_NUMBER_RE.match(clean))
    if lowered in known:
        return True
    if not numbered:
        return False

    prefix = SECTION_NUMBER_RE.match(clean).group(0)
    first_number = re.match(r"\d+", prefix)
    if first_number and int(first_number.group(0)) > 9:
        return False
    remainder = SECTION_NUMBER_RE.sub("", clean).strip()
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", remainder)
    if (
        not words or len(words) > 14 or size < body_size * .8
        or re.search(r"(?:https?://|www\.|[=#])", remainder, re.I)
        or not words[0][0].isupper()
    ):
        return False
    styled = bold or italic or size >= body_size * 1.08
    content_words = [
        word for word in words
        if word.lower() not in {"a", "an", "and", "as", "at", "for", "from", "in", "of", "on", "or", "the", "to", "with"}
    ]
    title_like = sum(word[0].isupper() for word in content_words) / max(1, len(content_words)) >= .55
    return bool(styled or len(words) <= 6 or (len(words) <= 10 and title_like))
