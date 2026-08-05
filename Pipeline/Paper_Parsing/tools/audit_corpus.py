from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path


PARSER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PARSER_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Pipeline.Paper_Parsing.pipeline import parse_pdf  # noqa: E402


DEFAULT_INPUTS = (
    PROJECT_ROOT / "SamplePapers",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_stem(path: Path) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", path.stem).strip("-.")
    return value[:96] or "paper"


def _flatten_paragraphs(parsed: dict) -> list[dict]:
    return [
        paragraph
        for section in parsed.get("sections", [])
        for paragraph in section.get("paragraphs", [])
    ]


def _issues(parsed: dict) -> list[str]:
    issues: list[str] = []
    document = parsed["document"]
    summary = parsed["summary"]
    paragraphs = _flatten_paragraphs(parsed)
    text = "\n".join(paragraph["text"] for paragraph in paragraphs)
    section_titles = [section["title"] for section in parsed.get("sections", [])]

    if not document.get("title"):
        issues.append("missing_title")
    if not document.get("authors_text"):
        issues.append("missing_authors")
    authors = document.get("authors_text") or ""
    if re.search(r"(?:published as part of|special issue|cite this|@)", authors, re.I):
        issues.append("author_furniture_leakage")
    if re.search(r"\b(?:university|department|institute|laboratory)\b", authors, re.I):
        issues.append("author_affiliation_leakage")
    if summary.get("sections", 0) < 3:
        issues.append("too_few_sections")
    if summary.get("paragraphs", 0) < 5:
        issues.append("too_few_paragraphs")
    if summary.get("sentences", 0) < 20:
        issues.append("too_few_sentences")
    if summary.get("narrative_page_ratio", 0) < 0.45:
        issues.append("low_narrative_page_coverage")
    if summary.get("pages", 0) >= 6 and summary.get("last_narrative_page", 0) < summary["pages"] * 0.5:
        issues.append("early_narrative_cutoff")
    if not any("abstract" in title.lower() for title in section_titles):
        issues.append("missing_abstract_section")
    if not any("intro" in title.lower() for title in section_titles):
        issues.append("missing_introduction_section")
    if any(len(title) > 160 or title.endswith((".", ",", ";")) for title in section_titles):
        issues.append("suspicious_section_title")
    if any(len(paragraph["text"]) > 6000 for paragraph in paragraphs):
        issues.append("oversized_paragraph")
    if any(not paragraph.get("sentences") for paragraph in paragraphs):
        issues.append("paragraph_without_sentences")
    if "cid:" in text.lower() or "\ufffd" in text:
        issues.append("encoding_or_formula_artifact")
    if re.search(r"\b(?:references|bibliography)\b.{0,80}\bet al\.\b", text[-8000:], re.I | re.S):
        issues.append("possible_reference_leakage")
    if re.search(r"\b(?:journal homepage|available online at|contents lists available)\b", text, re.I):
        issues.append("publisher_furniture_leakage")

    coverage_mismatches = 0
    for paragraph in paragraphs:
        joined = " ".join(sentence["text"] for sentence in paragraph.get("sentences", []))
        if joined != paragraph["text"]:
            coverage_mismatches += 1
    if coverage_mismatches:
        issues.append("sentence_coverage_mismatch")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse and audit a multi-format PDF corpus")
    parser.add_argument("inputs", nargs="*", type=Path, default=list(DEFAULT_INPUTS))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "PipelineAudits" / "parser")
    args = parser.parse_args()

    candidates: list[Path] = []
    for root in args.inputs:
        if root.is_file() and root.suffix.lower() == ".pdf":
            candidates.append(root.resolve())
        elif root.is_dir():
            candidates.extend(path.resolve() for path in sorted(root.rglob("*.pdf")))
    unique: dict[str, Path] = {}
    locations: dict[str, list[str]] = {}
    for path in candidates:
        digest = _sha256(path)
        unique.setdefault(digest, path)
        locations.setdefault(digest, []).append(str(path.relative_to(PROJECT_ROOT)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir = args.output_dir / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for index, (digest, path) in enumerate(sorted(unique.items(), key=lambda item: item[1].name.lower()), start=1):
        print(f"[{index:02d}/{len(unique):02d}] {path.name}", flush=True)
        try:
            output_name = f"{index:02d}-{_safe_stem(path)}-{digest[:8]}.json"
            output_path = parsed_dir / output_name
            if output_path.exists():
                parsed = json.loads(output_path.read_text(encoding="utf-8"))
                print("  reused completed parse", flush=True)
            else:
                parsed = parse_pdf(path.read_bytes(), path.name)
                output_path.write_text(
                    json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            summary = parsed["summary"]
            document = parsed["document"]
            paragraphs = _flatten_paragraphs(parsed)
            issues = _issues(parsed)
            rows.append({
                "status": "parsed",
                "sha256": digest,
                "filename": path.name,
                "locations": locations[digest],
                "profile": document.get("parsing_profile"),
                "publisher": document.get("publisher"),
                "title": document.get("title"),
                "authors_text": document.get("authors_text"),
                "pages": summary.get("pages"),
                "sections": summary.get("sections"),
                "paragraphs": summary.get("paragraphs"),
                "sentences": summary.get("sentences"),
                "narrative_pages": summary.get("narrative_pages"),
                "last_narrative_page": summary.get("last_narrative_page"),
                "narrative_page_ratio": summary.get("narrative_page_ratio"),
                "max_paragraph_chars": max((len(item["text"]) for item in paragraphs), default=0),
                "warning_count": len(summary.get("warnings", [])),
                "warnings": summary.get("warnings", []),
                "issues": issues,
                "parsed_json": str(output_path.relative_to(args.output_dir)),
            })
        except Exception as exc:  # preserve failures for corpus-level diagnosis
            rows.append({
                "status": "failed",
                "sha256": digest,
                "filename": path.name,
                "locations": locations[digest],
                "error": f"{type(exc).__name__}: {exc}",
                "issues": ["parse_failure"],
            })

    report = {
        "pdf_files": len(candidates),
        "unique_pdfs": len(unique),
        "parsed": sum(row["status"] == "parsed" for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
        "profiles": dict(Counter(row.get("profile", "failed") for row in rows)),
        "issue_counts": dict(Counter(issue for row in rows for issue in row.get("issues", []))),
        "papers": rows,
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    columns = (
        "status", "filename", "profile", "publisher", "pages", "sections",
        "paragraphs", "sentences", "narrative_page_ratio", "last_narrative_page",
        "max_paragraph_chars", "title", "authors_text", "issues", "parsed_json",
    )
    with (args.output_dir / "audit.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            item = dict(row)
            item["issues"] = ";".join(row.get("issues", []))
            writer.writerow(item)

    print(json.dumps({key: report[key] for key in ("unique_pdfs", "parsed", "failed", "profiles", "issue_counts")}, indent=2))


if __name__ == "__main__":
    main()
