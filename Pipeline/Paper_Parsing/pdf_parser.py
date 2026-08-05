from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from statistics import median

from .text_utils import looks_like_caption, looks_like_formula, normalize_text


SECTION_PREFIX = re.compile(r"^\d{1,2}(?:\.\d{1,2}){0,3}\.?\s+[A-Za-z]")


def _infer_ruled_table_bboxes(page) -> list[tuple[float, float, float, float]]:
    """Find column-width ruled tables missed by pdfplumber's cell detector.

    Journal tables often have only a top and bottom horizontal rule. Grouping
    matching long edges recovers the full region, including its caption.
    """
    groups: dict[str, list[tuple[float, float, float]]] = {"left": [], "right": []}
    width = float(page.width)
    for edge in page.edges:
        if edge.get("orientation") != "h":
            continue
        x0, x1 = float(edge["x0"]), float(edge["x1"])
        length = x1 - x0
        if not width * 0.10 <= length <= width * 0.62:
            continue
        midpoint = (x0 + x1) / 2
        column = "left" if midpoint < width / 2 else "right"
        # Do not combine rules that cross the document gutter.
        if column == "left" and x1 > width * 0.68:
            continue
        if column == "right" and x0 < width * 0.32:
            continue
        groups[column].append((x0, x1, float(edge["top"])))

    regions: list[tuple[float, float, float, float]] = []
    for edges in groups.values():
        unique_tops = sorted({round(edge[2], 1) for edge in edges})
        if len(unique_tops) < 2:
            continue
        top, bottom = unique_tops[0], unique_tops[-1]
        if not 18 <= bottom - top <= 260:
            continue
        x0 = min(edge[0] for edge in edges)
        x1 = max(edge[1] for edge in edges)
        regions.append((x0, max(0.0, top - 34.0), x1, min(float(page.height), bottom + 3.0)))
    return regions


@dataclass
class Line:
    text: str
    page: int
    top: float
    bottom: float
    x0: float
    x1: float
    size: float
    bold: bool = False
    italic: bool = False


def _inside_bbox(word: dict, bbox: tuple[float, float, float, float]) -> bool:
    cx = (float(word["x0"]) + float(word["x1"])) / 2
    cy = (float(word["top"]) + float(word["bottom"])) / 2
    x0, top, x1, bottom = bbox
    return x0 <= cx <= x1 and top <= cy <= bottom


def _words_to_lines(words: list[dict], page_number: int) -> list[Line]:
    groups: list[list[dict]] = []
    for word in sorted(words, key=lambda w: (round(float(w["top"]), 1), float(w["x0"]))):
        center = (float(word["top"]) + float(word["bottom"])) / 2
        target = next((g for g in reversed(groups[-8:]) if abs(
            center - sum((float(x["top"]) + float(x["bottom"])) / 2 for x in g) / len(g)
        ) <= max(2.5, float(word.get("size", 10)) * 0.28)), None)
        if target is None:
            groups.append([word])
        else:
            target.append(word)

    lines: list[Line] = []
    horizontal_groups: list[list[dict]] = []
    for group in groups:
        ordered = sorted(group, key=lambda w: float(w["x0"]))
        current: list[dict] = []
        for word in ordered:
            typical_size = median(float(w.get("size", 10)) for w in current) if current else float(word.get("size", 10))
            # Words sharing a baseline can belong to opposing columns. A real
            # inter-column gap is much wider than ordinary word spacing.
            if current and float(word["x0"]) - float(current[-1]["x1"]) > max(8.0, typical_size * 0.8):
                horizontal_groups.append(current)
                current = []
            current.append(word)
        if current:
            horizontal_groups.append(current)

    for group in horizontal_groups:
        ordered = sorted(group, key=lambda w: float(w["x0"]))
        text = normalize_text(" ".join(str(w["text"]) for w in ordered))
        if text:
            fonts = [str(w.get("fontname", "")).lower() for w in group]
            lines.append(Line(
                text=text, page=page_number,
                top=min(float(w["top"]) for w in group),
                bottom=max(float(w["bottom"]) for w in group),
                x0=min(float(w["x0"]) for w in group),
                x1=max(float(w["x1"]) for w in group),
                size=median(float(w.get("size", 10)) for w in group),
                bold=sum("bold" in font or "semibold" in font for font in fonts) >= max(1, len(fonts) // 2),
                italic=sum("italic" in font or "oblique" in font for font in fonts) >= max(1, len(fonts) // 2),
            ))
    return lines


def _reading_order(lines: list[Line], width: float) -> list[Line]:
    left = [line for line in lines if line.x1 < width * 0.56]
    right = [line for line in lines if line.x0 > width * 0.44]
    if len(left) + len(right) < len(lines) * 0.58 or min(len(left), len(right)) < 4:
        return sorted(lines, key=lambda line: (line.top, line.x0))
    spanning = [line for line in lines if line not in left and line not in right]
    ordered: list[Line] = []
    bands = sorted(spanning, key=lambda line: line.top)
    cursor = -1.0
    for boundary in bands + [Line("", 0, float("inf"), 0, 0, 0, 0)]:
        ordered.extend(sorted((x for x in left if cursor < x.top < boundary.top), key=lambda x: x.top))
        ordered.extend(sorted((x for x in right if cursor < x.top < boundary.top), key=lambda x: x.top))
        if boundary.text:
            ordered.append(boundary)
        cursor = boundary.top
    return ordered


def extract_lines(pdf_bytes: bytes) -> tuple[list[Line], dict]:
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("PDF support is missing. Install dependencies with: pip install -r requirements.txt") from exc

    from io import BytesIO
    pages: list[tuple[list[Line], float, float]] = []
    excluded_tables = 0
    excluded_figures = 0
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            # A tighter tolerance is important for ACS PDFs, whose justified
            # text otherwise collapses whole phrases into a single word.
            words = page.extract_words(
                x_tolerance=1, y_tolerance=3,
                extra_attrs=["size", "fontname"], keep_blank_chars=False,
            )
            table_bboxes: list[tuple[float, float, float, float]] = []
            try:
                table_bboxes = [tuple(table.bbox) for table in page.find_tables()]
            except Exception:
                table_bboxes = []
            inferred_tables = _infer_ruled_table_bboxes(page) if page_index > 1 else []
            table_bboxes.extend(
                bbox for bbox in inferred_tables
                if not any(
                    abs(bbox[0] - known[0]) < 5 and abs(bbox[1] - known[1]) < 8
                    and abs(bbox[2] - known[2]) < 5 and abs(bbox[3] - known[3]) < 8
                    for known in table_bboxes
                )
            )
            excluded_tables += len(table_bboxes)
            figure_bboxes = []
            for image in page.images:
                bbox = (float(image["x0"]), float(image["top"]), float(image["x1"]), float(image["bottom"]))
                area = max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
                if area > float(page.width) * float(page.height) * 0.025:
                    figure_bboxes.append(bbox)
            excluded_figures += len(figure_bboxes)
            excluded_regions = table_bboxes + figure_bboxes
            words = [w for w in words if not any(_inside_bbox(w, bbox) for bbox in excluded_regions)]
            lines = _words_to_lines(words, page_index)
            pages.append((lines, float(page.width), float(page.height)))

    furniture = Counter()
    for lines, _, height in pages:
        for line in lines:
            if line.top < height * 0.09 or line.bottom > height * 0.91:
                key = normalize_text(line.text).lower()
                if len(key) > 2:
                    furniture[key] += 1
    repeated = {text for text, count in furniture.items() if count >= max(2, len(pages) // 3)}

    kept: list[Line] = []
    excluded_furniture = excluded_captions = excluded_formulae = 0
    for lines, width, height in pages:
        page_kept: list[Line] = []
        caption_region: tuple[float, float, float, float] | None = None
        caption_line_count = 0
        for line in lines:
            key = normalize_text(line.text).lower()
            margin_artifact = bool(
                line.x0 < width * .045 and line.x1 < width * .14
                and len(line.text) < 20
                and len(re.findall(r"[A-Za-z]+", line.text)) <= 3
            )
            caption_continuation = False
            if caption_region is not None:
                cap_x0, cap_x1, cap_bottom, cap_size = caption_region
                overlap = max(0.0, min(line.x1, cap_x1) - max(line.x0, cap_x0))
                line_width = max(1.0, line.x1 - line.x0)
                caption_continuation = (
                    caption_line_count < 6
                    and line.top >= cap_bottom
                    and line.top - cap_bottom <= cap_size * 1.2
                    and overlap / line_width >= 0.55
                    and line.size <= cap_size * 1.08
                )
                if line.top - cap_bottom > cap_size * 3:
                    caption_region = None
            if margin_artifact or key in repeated or ((line.top < height * 0.045 or line.bottom > height * 0.955) and len(line.text) < 90):
                excluded_furniture += 1
            elif looks_like_caption(line.text):
                excluded_captions += 1
                caption_region = (line.x0, line.x1, line.bottom, line.size)
                caption_line_count = 1
            elif caption_continuation:
                excluded_captions += 1
                cap_x0, cap_x1, _, cap_size = caption_region
                caption_region = (min(cap_x0, line.x0), max(cap_x1, line.x1), line.bottom, cap_size)
                caption_line_count += 1
            elif "cid:" in line.text:
                excluded_formulae += 1
            elif looks_like_formula(line.text):
                excluded_formulae += 1
            elif line.italic and len(line.text) < 48 and not SECTION_PREFIX.match(line.text):
                excluded_formulae += 1
            else:
                page_kept.append(line)
        kept.extend(_reading_order(page_kept, width))

    sizes = [line.size for line in kept if len(line.text) > 35]
    body_size = median(sizes) if sizes else 10.0
    diagnostics = {
        "page_count": len(pages),
        "body_font_size": round(body_size, 2),
        "excluded_table_regions": excluded_tables,
        "excluded_figure_regions": excluded_figures,
        "excluded_header_footer_lines": excluded_furniture,
        "excluded_caption_lines": excluded_captions,
        "excluded_formula_lines": excluded_formulae,
    }
    return kept, diagnostics
