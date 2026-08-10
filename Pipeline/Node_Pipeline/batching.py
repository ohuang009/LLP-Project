"""Build paragraph-scoped node candidates from raw SVO output."""
from __future__ import annotations

from Pipeline.core import stable_id


def _sentences(parsed: dict) -> list[dict]:
    return [{
        "document_id": parsed["document"]["id"],
        "section_id": section["id"],
        "section_title": section.get("title", ""),
        "paragraph_id": paragraph["id"],
        "sentence_id": sentence["id"],
        "text": sentence["text"],
        "pages": sentence.get("pages", []),
    } for section in parsed.get("sections", [])
      for paragraph in section.get("paragraphs", [])
      for sentence in paragraph.get("sentences", [])]


def build_paragraph_batches(parsed: dict, raw: dict) -> list[dict]:
    """Batch by paragraph while keeping SVO arguments and context separate."""
    ordered = _sentences(parsed)
    positions = {row["sentence_id"]: index for index, row in enumerate(ordered)}
    analyses = {row["sentence_id"]: row for row in raw["sentences"]}
    arguments: dict[str, dict] = {}
    for triple in raw["triples"]:
        for role in ("subject", "object"):
            span = triple[role]
            candidate_id = stable_id(
                "node-candidate", triple["sentence_id"], span["start"], span["end"]
            )
            row = arguments.setdefault(candidate_id, {
                "candidate_id": candidate_id,
                "document_id": triple["document_id"],
                "section_id": triple["section_id"],
                "section_title": triple["section_title"],
                "paragraph_id": triple["paragraph_id"],
                "sentence_id": triple["sentence_id"],
                "sentence_text": triple["sentence_text"],
                "surface_text": span["text"],
                "start": span["start"], "end": span["end"],
                "roles": [], "triple_ids": [],
            })
            if role not in row["roles"]:
                row["roles"].append(role)
            if triple["triple_id"] not in row["triple_ids"]:
                row["triple_ids"].append(triple["triple_id"])

    batches: list[dict] = []
    for section in parsed.get("sections", []):
        for paragraph in section.get("paragraphs", []):
            sentence_rows = []
            for sentence in paragraph.get("sentences", []):
                sentence_id = sentence["id"]
                index = positions[sentence_id]
                analysis = analyses.get(sentence_id, {})
                sentence_rows.append({
                    "sentence_id": sentence_id,
                    "text": sentence["text"],
                    "previous_sentence": ordered[index - 1]["text"] if index else "",
                    "next_sentence": ordered[index + 1]["text"] if index + 1 < len(ordered) else "",
                    "subjects": [
                        row for row in arguments.values()
                        if row["sentence_id"] == sentence_id and "subject" in row["roles"]
                    ],
                    "objects": [
                        row for row in arguments.values()
                        if row["sentence_id"] == sentence_id and "object" in row["roles"]
                    ],
                    "context": analysis.get("context", []),
                    "main": analysis.get("main", []),
                })
            batches.append({
                "document_id": parsed["document"]["id"],
                "section_id": section["id"],
                "section_title": section.get("title", ""),
                "paragraph_id": paragraph["id"],
                "paragraph_text": paragraph.get("text") or " ".join(row["text"] for row in sentence_rows),
                "sentences": sentence_rows,
            })
    return batches


def candidate_rows(batch: dict) -> list[dict]:
    """Return each unique subject/object candidate in a paragraph once."""
    rows: dict[str, dict] = {}
    for sentence in batch["sentences"]:
        for row in [*sentence["subjects"], *sentence["objects"]]:
            rows[row["candidate_id"]] = row
    return list(rows.values())
