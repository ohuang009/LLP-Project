from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from Pipeline.Paper_Parsing.text_utils import has_terminal_punctuation


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit pipeline parser JSON structure")
    parser.add_argument("json_file", type=Path)
    args = parser.parse_args()
    result = json.loads(args.json_file.read_text(encoding="utf-8"))

    print("DOCUMENT", result["document"])
    print("SUMMARY", result["summary"])
    print("\nSECTIONS")
    ids: list[str] = []
    ordinal_errors: list[tuple] = []
    grammar_errors: list[tuple] = []
    coverage_errors: list[tuple] = []
    suspicious: list[tuple] = []

    for section in result["sections"]:
        paragraphs = section["paragraphs"]
        sentence_count = sum(len(paragraph["sentences"]) for paragraph in paragraphs)
        lengths = [len(paragraph["text"]) for paragraph in paragraphs]
        print(
            f'{section["ordinal"]:02d} | {section["title"]} | '
            f'pages={section["pages"]} | paragraphs={len(paragraphs)} '
            f'sentences={sentence_count} | chars={sum(lengths)} | '
            f'range={min(lengths) if lengths else 0}-{max(lengths) if lengths else 0}'
        )
        ids.append(section["id"])
        if [p["ordinal"] for p in paragraphs] != list(range(len(paragraphs))):
            ordinal_errors.append(("paragraphs", section["title"]))

        for paragraph in paragraphs:
            ids.append(paragraph["id"])
            if not has_terminal_punctuation(paragraph["text"]):
                grammar_errors.append(("paragraph", section["title"], paragraph["ordinal"]))
            sentences = paragraph["sentences"]
            if [s["ordinal"] for s in sentences] != list(range(len(sentences))):
                ordinal_errors.append(("sentences", section["title"], paragraph["ordinal"]))
            joined = " ".join(sentence["text"] for sentence in sentences)
            if joined != paragraph["text"]:
                coverage_errors.append((
                    section["title"], paragraph["ordinal"],
                    len(paragraph["text"]), len(joined),
                    paragraph["text"][-120:], joined[-120:],
                ))
            for sentence in sentences:
                ids.append(sentence["id"])
                if not has_terminal_punctuation(sentence["text"]):
                    grammar_errors.append(("sentence", section["title"], paragraph["ordinal"]))

            alpha = len(re.findall(r"[A-Za-z]", paragraph["text"]))
            digits = len(re.findall(r"\d", paragraph["text"]))
            if len(paragraph["text"]) < 80 or digits > alpha * 0.22 or "cid:" in paragraph["text"]:
                suspicious.append((
                    section["title"], paragraph["ordinal"], len(paragraph["text"]),
                    round(digits / max(alpha, 1), 2), paragraph["text"][:220],
                ))

    print("\nINTEGRITY", {
        "ids": len(ids),
        "duplicate_ids": len(ids) - len(set(ids)),
        "ordinal_errors": ordinal_errors,
        "grammar_errors": grammar_errors,
        "sentence_coverage_mismatches": len(coverage_errors),
        "suspicious_paragraphs": len(suspicious),
    })
    print("\nCOVERAGE ERRORS")
    for item in coverage_errors:
        print(item)
    print("\nSUSPICIOUS PARAGRAPHS")
    for item in suspicious:
        print(item)


if __name__ == "__main__":
    main()
