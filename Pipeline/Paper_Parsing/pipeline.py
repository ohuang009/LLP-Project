from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import math
import re

from .models import chunk_record, stable_id
from .pdf_parser import Line, extract_lines
from .profiles import (
    ParsingProfile, compact_label, detect_profile, is_profile_heading, visible_heading,
)
from .text_utils import (
    begins_list_item, has_terminal_punctuation, join_lines, looks_like_heading,
    split_sentences, starts_like_continuation,
)


def _canonical_heading(text: str) -> str:
    """Handle canonical heading for this stage. It helps convert diverse PDF layouts into clean, sentence-addressable paper text."""
    normalized = visible_heading(text)
    visible = re.sub(
        r"^(?:(?:\d+(?:\.\d+)*|[IVX]+|[A-Z])\.?)\s+", "", normalized,
    ).strip()
    compact = "".join(visible.split()).lower()
    labels = {
        "abstract": "Abstract",
        "references": "References",
        "reference": "References",
        "highlights": "Highlights",
        "graphicalabstract": "Graphical abstract",
    }
    return labels.get(compact, normalized)


def _back_matter_label(text: str) -> str:
    """Normalize both `References` and numbered forms such as `8. References`."""
    visible = re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", visible_heading(text)).strip()
    return compact_label(visible)


def _is_back_matter_heading(
    line: Line, profile: ParsingProfile, page_count: int, body_size: float,
) -> bool:
    """Reject table headers such as a lone `Reference` cell."""
    if _back_matter_label(line.text) not in profile.back_matter:
        return False
    earliest_page = max(2, math.ceil(page_count * 0.5))
    if line.page < earliest_page:
        return False
    numbered = bool(re.match(r"^\s*\d+(?:\.\d+)*\.?\s+", visible_heading(line.text)))
    return bool(
        numbered or line.bold or line.size >= body_size * 1.08
        or looks_like_heading(
            line.text, line.size, body_size, bold=line.bold, italic=line.italic
        )
    )


def _clean_authors_text(text: str) -> str:
    """Remove common preprint affiliation markers without changing author names."""
    value = text.replace("�", " ").replace("\u2020", " ").replace("\u2021", " ")
    value = re.sub(r"([A-Za-z])\s+([\u0300-\u036f])(?=\s|$)", r"\1\2", value)
    value = re.sub(r"([A-Za-z])\s+([\u0300-\u036f])([a-z])", r"\1\3\2", value)
    value = __import__("unicodedata").normalize("NFC", value)
    if "Ã" in value or "â" in value:
        try:
            value = value.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    value = value.replace("#", " ").replace("\u2217", " ").replace("*", " ")
    value = re.sub(r"\s+\d+(?=\s*(?:,|$))", "", value)
    value = re.sub(r"(?:,\s*)?\*+(?=\s*(?:,|$))", "", value)
    value = re.sub(r"\s+[a-z](?:\s*,\s*[a-z])*(?=\s*(?:,|$))", "", value)
    value = re.sub(r"(?:,\s*){2,}", ", ", value)
    value = re.sub(r"\s+,", ",", value)
    value = re.sub(r"\s+", " ", value).strip(" ,")
    return value


def _person_like(line: Line) -> bool:
    """Handle person like for this stage. It helps convert diverse PDF layouts into clean, sentence-addressable paper text."""
    return bool(re.search(r"[A-Z][A-Za-z-]+(?:\s+[A-Z][A-Za-z-]+)+", line.text))


def _affiliation_like(line: Line) -> bool:
    """Handle affiliation like for this stage. It helps convert diverse PDF layouts into clean, sentence-addressable paper text."""
    return bool(re.search(
        r"\b(?:university|department|school|institute|centre|center|laboratory|academy|faculty)\b",
        line.text, re.I,
    ))


def _author_furniture(line: Line) -> bool:
    """Handle author furniture for this stage. It helps convert diverse PDF layouts into clean, sentence-addressable paper text."""
    return bool(
        "@" in line.text
        or re.search(
            r"\b(?:published as part of|special issue|cite this|corresponding author|"
            r"e-?mail|doi|received|accepted|copyright)\b",
            line.text, re.I,
        )
    )


def _author_continuation(line: Line) -> bool:
    """Handle author continuation for this stage. It helps convert diverse PDF layouts into clean, sentence-addressable paper text."""
    clean = line.text.strip(" ,")
    return bool(
        _person_like(line)
        or re.fullmatch(r"(?:and\s+)?[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+)?", clean)
        or line.text.lstrip().startswith((",", "and "))
    )


def _is_introduction_heading(text: str) -> bool:
    """Check whether introduction heading. It helps convert diverse PDF layouts into clean, sentence-addressable paper text."""
    clean = visible_heading(text).strip()
    return bool(re.fullmatch(
        r"(?:(?:\d+(?:\.\d+)*|[IVX]+)\.?\s+)?INTRODUCTION", clean, flags=re.I,
    ))


def _is_narrative_text(text: str) -> bool:
    """Check whether narrative text. It helps convert diverse PDF layouts into clean, sentence-addressable paper text."""
    words = re.findall(r"[A-Za-z]{2,}", text)
    if "cid:" in text and len(words) < 8:
        return False
    return not (len(text) < 60 and len(words) < 4)


def _section_level(title: str) -> int:
    """Handle section level for this stage. It helps convert diverse PDF layouts into clean, sentence-addressable paper text."""
    match = re.match(r"^(\d+(?:\.\d+)*)\.", title)
    if match:
        return len(match.group(1).split("."))
    if re.match(r"^[A-Z]\.\s+", title):
        return 2
    return 1


def _apply_section_hierarchy(sections: list[dict]) -> None:
    """Apply section hierarchy. It helps convert diverse PDF layouts into clean, sentence-addressable paper text."""
    stack: list[dict] = []
    for section in sections:
        level = _section_level(section["title"])
        while stack and stack[-1]["level"] >= level:
            stack.pop()
        parent = stack[-1] if stack else None
        section["level"] = level
        section["parent_section_id"] = parent["id"] if parent else None
        section["child_section_ids"] = []
        if parent:
            parent["child_section_ids"].append(section["id"])
        stack.append(section)

    by_id = {section["id"]: section for section in sections}
    for section in reversed(sections):
        inherited = list(section["pages"])
        for child_id in section["child_section_ids"]:
            inherited.extend(by_id[child_id]["pages"])
        section["pages"] = sorted(set(inherited))


def _paragraph_groups(
    lines: list[Line], body_size: float, profile: ParsingProfile
) -> list[tuple[str | None, list[Line]]]:
    """Handle paragraph groups for this stage. It helps convert diverse PDF layouts into clean, sentence-addressable paper text."""
    if profile.id == "acs_est":
        expanded: list[Line] = []
        inline_subsection = re.compile(
            r"^(\d+(?:\.\d+){1,3}\.\s+.+?[.)])\s+([A-Z].{10,})$"
        )
        for line in lines:
            match = inline_subsection.match(line.text)
            if line.bold and match:
                expanded.append(replace(line, text=match.group(1).rstrip(".")))
                expanded.append(replace(line, text=match.group(2), bold=False, size=body_size))
            else:
                expanded.append(line)
        lines = expanded

    groups: list[tuple[str | None, list[Line]]] = []
    current_heading: str | None = None
    current: list[Line] = []
    previous: Line | None = None
    heading_line: Line | None = None

    def flush() -> None:
        """Finish the current buffered paragraph or record. It helps convert diverse PDF layouts into clean, sentence-addressable paper text."""
        nonlocal current
        if current:
            groups.append((current_heading, current))
            current = []

    for line in lines:
        profile_heading = is_profile_heading(
            line.text, profile, size=line.size, body_size=body_size,
            bold=line.bold, italic=line.italic,
        )
        wrapped_heading = bool(
            heading_line and current_heading and not current
            and line.page == heading_line.page
            and line.top - heading_line.bottom <= body_size * 1.05
            and abs(line.size - heading_line.size) <= max(0.6, body_size * 0.08)
            and abs(line.x0 - heading_line.x0) <= body_size * 4
            and line.bold == heading_line.bold and line.italic == heading_line.italic
            and (heading_line.bold or heading_line.italic or heading_line.size >= body_size * 1.12)
            and not has_terminal_punctuation(line.text)
        )
        if wrapped_heading:
            old_heading = current_heading
            current_heading = join_lines([current_heading, line.text])
            if groups and groups[-1] == (old_heading, []):
                groups[-1] = (current_heading, [])
            heading_line = line
            continue
        if (
            profile_heading
            or looks_like_heading(line.text, line.size, body_size, bold=line.bold, italic=line.italic)
        ):
            flush()
            current_heading = _canonical_heading(visible_heading(line.text))
            groups.append((current_heading, []))
            previous = None
            heading_line = line
            continue
        citation_only = bool(re.fullmatch(r"\s*\d+(?:\s*[,\-–−]\s*\d+)*\s*", line.text))
        if citation_only and current:
            current.append(line)
            continue
        same_page = bool(previous and line.page == previous.page)
        gap = line.top - previous.bottom if same_page else 0
        indent_shift = line.x0 - previous.x0 if same_page else 0
        page_break_after_terminal = bool(
            previous and not same_page and has_terminal_punctuation(previous.text)
        )
        visual_break = bool(
            gap > body_size * 0.35
            or indent_shift > body_size * 0.55
            or page_break_after_terminal
        )
        # Geometry proposes a boundary; grammar confirms it. Column changes,
        # page breaks, and formatting newlines cannot cut an unfinished
        # sentence into a new paragraph.
        prior_complete = bool(previous and has_terminal_punctuation(previous.text))
        explicit_list = begins_list_item(line.text)
        new_paragraph = bool(current and visual_break and (prior_complete or explicit_list))
        if new_paragraph:
            flush()
        current.append(line)
        previous = line
        heading_line = None
    flush()
    return groups


def _is_publisher_furniture(text: str, profile: ParsingProfile) -> bool:
    """Check whether publisher furniture. It helps convert diverse PDF layouts into clean, sentence-addressable paper text."""
    clean = " ".join(text.split())
    if profile.id == "acs_est" and re.match(
        r"^(?:Received|Revised|Accepted|Published):", clean, flags=re.I
    ):
        return True
    return False


def _reconstruct_paragraphs(
    groups: list[tuple[str | None, list[Line]]], profile: ParsingProfile,
    body_size: float,
) -> tuple[list[tuple[str | None, list[Line]]], int]:
    """Handle reconstruct paragraphs for this stage. It helps convert diverse PDF layouts into clean, sentence-addressable paper text."""
    reconstructed: list[tuple[str | None, list[Line]]] = []
    discarded = 0

    filtered_groups: list[tuple[str | None, list[Line]]] = []
    for index, (heading, lines) in enumerate(groups):
        if not lines:
            filtered_groups.append((heading, lines))
            continue
        text = join_lines([line.text for line in lines])
        next_text = ""
        for next_heading, next_lines in groups[index + 1:]:
            if next_lines:
                if next_heading == heading:
                    next_text = join_lines([line.text for line in next_lines])
                break
        numeric_only = bool(re.fullmatch(r"[\d.%\s,\-–−]+", text))
        if numeric_only:
            discarded += 1
            continue
        filtered_groups.append((heading, lines))

    for heading, lines in filtered_groups:
        if not lines:
            reconstructed.append((heading, []))
            continue
        text = join_lines([line.text for line in lines])
        if _is_publisher_furniture(text, profile):
            discarded += 1
            continue

        prior_index = next((
            i for i in range(len(reconstructed) - 1, -1, -1)
            if reconstructed[i][1] or reconstructed[i][0] != heading
        ), None)
        if prior_index is not None:
            prior_heading, prior_lines = reconstructed[prior_index]
            if prior_lines and prior_heading == heading:
                prior_text = join_lines([line.text for line in prior_lines])
                list_boundary = prior_text.rstrip().endswith(":") and re.match(r"^\(?\d+[.)]", text)
                if not list_boundary and (
                    starts_like_continuation(text)
                    or not has_terminal_punctuation(prior_text)
                ):
                    reconstructed[prior_index] = (heading, prior_lines + lines)
                    continue
        reconstructed.append((heading, lines))

    clean_groups: list[tuple[str | None, list[Line]]] = []
    for heading, lines in reconstructed:
        if not lines:
            clean_groups.append((heading, lines))
            continue
        text = join_lines([line.text for line in lines])
        if not has_terminal_punctuation(text):
            discarded += 1
            continue
        clean_groups.append((heading, lines))
    return clean_groups, discarded


def _clean_profile_lines(lines: list[Line], profile: ParsingProfile) -> list[Line]:
    """Clean profile lines. It helps convert diverse PDF layouts into clean, sentence-addressable paper text."""
    if profile.id != "acs_est":
        return lines
    date = r"[A-Za-z]+\s+\d{1,2},\s+\d{4}"
    label = rf"(?:Received|Revised|Accepted|Published):\s*{date}"
    cleaned: list[Line] = []
    for line in lines:
        text = re.sub(label, "", line.text, flags=re.I).strip()
        if not text or re.fullmatch(date, text) or re.match(
            r"^(?:Received|Revised|Accepted|Published):?\s*$", text, flags=re.I
        ):
            continue
        cleaned.append(replace(line, text=text))
    return cleaned


def _front_matter(
    lines: list[Line], body_size: float, profile: ParsingProfile
) -> tuple[dict, list[Line]]:
    """Handle front matter for this stage. It helps convert diverse PDF layouts into clean, sentence-addressable paper text."""
    first_page = [line for line in lines if line.page == 1]
    title_seed_candidates = [
        line for line in first_page
        if line.size >= body_size * 1.35 and len(line.text) >= 24
        and "journal homepage" not in line.text.lower()
    ]
    title = ""
    title_lines: list[Line] = []
    if title_seed_candidates:
        seed = max(title_seed_candidates, key=lambda line: (len(line.text), line.top))
        title_lines = [
            line for line in first_page
            if abs(line.size - seed.size) < 1.0
            and abs(line.top - seed.top) < seed.size * 3.2
            and line.x0 >= seed.x0 - 4
        ]
        title = join_lines([line.text for line in sorted(title_lines, key=lambda line: line.top)])

    # Many preprints use one font size for the entire page and distinguish the
    # title only with bold weight. Fall back to the first compact bold block.
    if not title:
        abstract_top = next((
            line.top for line in first_page if compact_label(line.text) == "abstract"
        ), float("inf"))
        bold_before_abstract = sorted(
            [line for line in first_page if line.bold and line.top < abstract_top and len(line.text) >= 6],
            key=lambda line: line.top,
        )
        if bold_before_abstract:
            seed = bold_before_abstract[0]
            title_lines = [seed]
            for line in bold_before_abstract[1:]:
                if line.top - title_lines[-1].bottom > body_size * 1.15:
                    break
                title_lines.append(line)
            candidate = join_lines([line.text for line in title_lines])
            if len(candidate) >= 24:
                title = candidate
    title = re.sub(r"(?<=[A-Za-z])-\s+(?=[A-Z][a-z])", "-", title)

    authors = ""
    author_lines: list[Line] = []
    if title:
        title_bottom = max((line.bottom for line in title_lines), default=0)
        title_x0 = min((line.x0 for line in title_lines), default=0)
        title_x1 = max((line.x1 for line in title_lines), default=float("inf"))
        author_candidates = sorted([
            line for line in first_page if line.top > title_bottom
            and line.top < title_bottom + body_size * 8
            and line.size >= body_size * .95
            and (line.text.count(",") >= 1 or " and " in line.text or _person_like(line))
            and not _affiliation_like(line)
            and not _author_furniture(line)
            and not line.italic
            and len(line.text) < 180
        ], key=lambda line: (
            line.top,
            -max(0.0, min(line.x1, title_x1) - max(line.x0, title_x0)),
        ))
        if author_candidates:
            seed = author_candidates[0]
            # Column splitting may create several author fragments on the same
            # physical baseline (for example, three separately positioned names).
            author_lines = sorted([
                line for line in first_page
                if abs(line.top - seed.top) <= body_size * .25
                and line.size >= body_size * .75 and _person_like(line)
                and not _affiliation_like(line) and not _author_furniture(line)
                and len(line.text) < 180
                and line.x1 >= title_x0 - body_size * 4
                and line.x0 <= title_x1 + body_size * 4
            ], key=lambda row: (row.top, row.x0)) or [seed]
            # A wrapped continuation may use a slightly smaller font.
            for line in sorted(first_page, key=lambda row: (row.top, row.x0)):
                if line in author_lines or line.top < max(row.top for row in author_lines) - body_size * .25:
                    continue
                if line.top > max(row.bottom for row in author_lines) + body_size * 1.35:
                    break
                if _affiliation_like(line) or _author_furniture(line):
                    break
                if re.fullmatch(r"[\d,\s*†‡]+", line.text):
                    continue
                if (
                    line.size >= body_size * .75 and _author_continuation(line)
                    and len(line.text) < 180
                    and line.x1 >= title_x0 - body_size * 4
                    and line.x0 <= title_x1 + body_size * 4
                ):
                    author_lines.append(line)
                else:
                    continue
            author_parts = ""
            ordered_authors = sorted(author_lines, key=lambda row: (row.top, row.x0))
            for index, line in enumerate(ordered_authors):
                separator = " "
                if index:
                    prior = ordered_authors[index - 1]
                    clean_line = line.text.lstrip()
                    surname_continuation = bool(re.match(
                        r"^[A-Z][A-Za-z'\-]+(?:\s*,|$)", clean_line
                    ))
                    if (
                        abs(line.top - prior.top) <= body_size * .25
                        and line.x0 - prior.x1 > body_size * 3
                        and not clean_line.startswith((",", "and "))
                    ):
                        separator = ", "
                    elif (
                        line.top > prior.top + body_size * .25
                        and not clean_line.startswith((",", "and "))
                        and not surname_continuation
                    ):
                        separator = ", "
                author_parts = f"{author_parts}{separator if author_parts else ''}{line.text}"
            authors = _clean_authors_text(author_parts)

    intro_index = next((
        i for i, line in enumerate(lines) if _is_introduction_heading(line.text)
    ), None)
    abstract_index = next((
        i for i, line in enumerate(lines) if line.page == 1
        and (
            compact_label(line.text) == "abstract"
            or (
                profile.inline_abstract
                and re.match(r"^ABSTRACT\s*(?::|[-\u2013\u2014])\s*\S", line.text, flags=re.I)
            )
        )
    ), None)
    if abstract_index is not None:
        abstract_heading = lines[abstract_index]
        unlabelled_intro_index = None
        if intro_index is None and profile.id == "elsevier_water_research":
            same_column = [
                (i, line) for i, line in enumerate(lines[abstract_index + 1:], abstract_index + 1)
                if line.page == 1 and line.x0 >= abstract_heading.x0 - 3
                and not line.text.upper().startswith("KEYWORDS")
            ]
            split_top = next((
                line.top for (_, prior), (_, line) in zip(same_column, same_column[1:])
                if line.top - prior.bottom > body_size * 2.0
                and line.size >= body_size * .95
            ), None)
            if split_top is not None:
                unlabelled_intro_index = next((
                    i for i, line in enumerate(lines)
                    if i > abstract_index and line.page == 1
                    and line.top >= split_top - body_size * .35
                    and line.size >= body_size * .95
                ), None)
        after_abstract = intro_index if intro_index is not None else next((
            i for i, line in enumerate(lines) if i > abstract_index and line.page > 1
        ), len(lines))
        if unlabelled_intro_index is not None:
            after_abstract = unlabelled_intro_index
        abstract_candidates = [
            line for line in lines[abstract_index + 1:after_abstract]
            if line.page == 1
            and not line.text.upper().startswith("KEYWORDS")
            and (
                not profile.abstract_same_column
                or line.x0 >= abstract_heading.x0 - 3
            )
        ]
        if profile.inline_abstract:
            first_abstract = replace(
                abstract_heading,
                text=re.sub(
                    r"^ABSTRACT\s*(?::|[-\u2013\u2014])\s*", "", abstract_heading.text,
                    flags=re.I,
                ),
            )
            abstract_lines = [first_abstract] + abstract_candidates
        else:
            abstract_lines = abstract_candidates
        content = [replace(abstract_heading, text="Abstract")] + abstract_lines
        if unlabelled_intro_index is not None:
            content.append(replace(lines[unlabelled_intro_index], text="Introduction"))
        content.extend(lines[after_abstract:])
    elif profile.unlabelled_abstract and intro_index is not None:
        intro_line = lines[intro_index]
        pre_intro = [
            line for line in lines[:intro_index]
            if line.page == 1 and line.top > max((row.bottom for row in title_lines), default=0)
            and line.x0 >= intro_line.x0 - 6
            and not line.text.upper().startswith("KEYWORDS")
        ]
        abstract_start = next((
            index for index, line in enumerate(pre_intro)
            if len(line.text) >= 45 and line.size <= body_size * 1.25
            and not _person_like(line) and not _affiliation_like(line)
        ), None)
        abstract_lines = pre_intro[abstract_start:] if abstract_start is not None else []
        marker = replace(abstract_lines[0] if abstract_lines else intro_line, text="Abstract")
        content = [marker] + abstract_lines + lines[intro_index:]
    else:
        start = intro_index if intro_index is not None else 0
        content = lines[start:]

    # References are useful bibliographic data, but poor narrative chunks and
    # especially noisy inputs for entity/relation extraction.
    page_count = max((line.page for line in lines), default=1)
    stop = next((
        i for i, line in enumerate(content)
        if _is_back_matter_heading(line, profile, page_count, body_size)
    ), None)
    if stop is not None:
        content = content[:stop]
    if author_lines:
        content = [line for line in content if line not in author_lines]
    content = _clean_profile_lines(content, profile)
    return {"title": title or None, "authors_text": authors or None}, content


def parse_pdf(pdf_bytes: bytes, filename: str = "document.pdf") -> dict:
    """Handle parse pdf for this stage. It helps convert diverse PDF layouts into clean, sentence-addressable paper text."""
    if not pdf_bytes.startswith(b"%PDF"):
        raise ValueError("The uploaded file does not appear to be a PDF.")
    lines, diagnostics = extract_lines(pdf_bytes)
    if not lines:
        raise ValueError("No embedded narrative text was found. This PDF may need OCR before parsing.")

    profile = detect_profile(lines)
    doc_id = stable_id("doc", filename, len(pdf_bytes))
    metadata, narrative_lines = _front_matter(lines, diagnostics["body_font_size"], profile)
    grouped = _paragraph_groups(narrative_lines, diagnostics["body_font_size"], profile)
    grouped, discarded_incomplete = _reconstruct_paragraphs(
        grouped, profile, diagnostics["body_font_size"]
    )
    diagnostics["discarded_incomplete_paragraphs"] = discarded_incomplete
    sections: list[dict] = []
    chunks: list[dict] = []
    current_section: dict | None = None

    for heading, paragraph_lines in grouped:
        section_title = heading or ("Abstract" if not sections else sections[-1]["title"])
        if current_section is None or current_section["title"] != section_title:
            section_id = stable_id("sec", doc_id, len(sections), section_title)
            current_section = {
                "id": section_id,
                "title": section_title,
                "ordinal": len(sections),
                "pages": [],
                "paragraphs": [],
            }
            sections.append(current_section)
        section = current_section
        text = join_lines([line.text for line in paragraph_lines])
        if len(text) < 2 or not _is_narrative_text(text):
            continue
        pages = sorted({line.page for line in paragraph_lines})
        section["pages"] = sorted(set(section["pages"] + pages))
        paragraph_id = stable_id("para", section["id"], len(section["paragraphs"]), text)
        sentences = []
        for sentence_text in split_sentences(text):
            sentence_id = stable_id("sent", paragraph_id, len(sentences), sentence_text)
            sentence = {
                "id": sentence_id,
                "ordinal": len(sentences),
                "text": sentence_text,
                "pages": pages,
            }
            sentences.append(sentence)
            chunks.append(chunk_record(
                chunk_id=sentence_id, chunk_type="sentence", text=sentence_text,
                ordinal=len(chunks), pages=pages, parent_id=paragraph_id,
                section_id=section["id"],
            ))
        paragraph = {
            "id": paragraph_id,
            "ordinal": len(section["paragraphs"]),
            "text": text,
            "pages": pages,
            "sentences": sentences,
        }
        section["paragraphs"].append(paragraph)
        chunks.append(chunk_record(
            chunk_id=paragraph_id, chunk_type="paragraph", text=text,
            ordinal=len(chunks), pages=pages, parent_id=section["id"],
            section_id=section["id"],
        ))

    _apply_section_hierarchy(sections)

    for section in sections:
        section_text = "\n\n".join(p["text"] for p in section["paragraphs"])
        if not section_text:
            continue
        chunks.append(chunk_record(
            chunk_id=section["id"], chunk_type="section", text=section_text,
            ordinal=len(chunks), pages=section["pages"],
            parent_id=section["parent_section_id"] or doc_id,
            section_id=section["id"],
        ))

    counts = Counter(chunk["type"] for chunk in chunks)
    warnings = []
    narrative_pages = sorted({
        page for section in sections for paragraph in section["paragraphs"]
        for page in paragraph.get("pages", [])
    })
    narrative_page_ratio = (
        len(narrative_pages) / diagnostics["page_count"] if diagnostics["page_count"] else 0.0
    )
    last_narrative_page = max(narrative_pages, default=0)
    total_chars = sum(len(line.text) for line in lines)
    if total_chars < diagnostics["page_count"] * 250:
        warnings.append("Low embedded-text density detected; OCR may improve this document.")
    if discarded_incomplete:
        warnings.append(
            f"Discarded {discarded_incomplete} incomplete or non-narrative text fragments."
        )
    if diagnostics["page_count"] >= 6 and last_narrative_page < math.ceil(diagnostics["page_count"] * 0.5):
        warnings.append(
            "Narrative extraction ended before the midpoint of the PDF; inspect for a false back-matter cutoff."
        )
    return {
        "schema_version": "1.0",
        "document": {
            "id": doc_id,
            "filename": Path(filename).name,
            "media_type": "application/pdf",
            "parsed_at": datetime.now(timezone.utc).isoformat(),
            "parser": "pdfplumber-layout",
            "parsing_profile": profile.id,
            "publisher": profile.publisher,
            "language": "und",
            "title": metadata["title"],
            "authors_text": metadata["authors_text"],
        },
        "summary": {
            "pages": diagnostics["page_count"],
            "sections": len(sections),
            "paragraphs": counts["paragraph"],
            "sentences": counts["sentence"],
            "narrative_pages": len(narrative_pages),
            "last_narrative_page": last_narrative_page,
            "narrative_page_ratio": round(narrative_page_ratio, 3),
            "warnings": warnings,
        },
        "filtering": diagnostics,
        "sections": sections,
        "chunks": chunks,
    }
