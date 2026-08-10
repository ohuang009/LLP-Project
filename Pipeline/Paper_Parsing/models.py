from __future__ import annotations

import hashlib


def stable_id(prefix: str, *parts: object) -> str:
    """Handle stable id for this stage. It helps convert diverse PDF layouts into clean, sentence-addressable paper text."""
    material = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(material.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def chunk_record(
    *, chunk_id: str, chunk_type: str, text: str, ordinal: int,
    pages: list[int], parent_id: str | None, section_id: str | None,
) -> dict:
    """Handle chunk record for this stage. It helps convert diverse PDF layouts into clean, sentence-addressable paper text."""
    return {
        "id": chunk_id,
        "type": chunk_type,
        "ordinal": ordinal,
        "text": text,
        "pages": sorted(set(pages)),
        "parent_id": parent_id,
        "section_id": section_id,
        "character_count": len(text),
    }

