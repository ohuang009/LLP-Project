from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT.parents[1] / "SamplePapers" / "evaluation"


def main() -> int:
    manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    papers = manifest["papers"]
    errors: list[str] = []
    observed = Counter(row["split"] for row in papers)
    expected = {"refinement": 8, "testing": 5, "heldout": 2}
    if dict(observed) != expected:
        errors.append(f"split counts {dict(observed)} != {expected}")
    for row in papers:
        path = CORPUS / row["file"]
        if not path.is_file():
            errors.append(f"missing: {row['file']}")
            continue
        payload = path.read_bytes()
        if not payload.startswith(b"%PDF-"):
            errors.append(f"not a PDF: {row['file']}")
            continue
        digest = hashlib.sha256(payload).hexdigest()
        if digest != row["sha256"]:
            errors.append(f"checksum mismatch: {row['file']}")
        try:
            pages = len(PdfReader(path).pages)
            if pages < 1:
                errors.append(f"no readable pages: {row['file']}")
        except Exception as exc:  # pragma: no cover - diagnostic CLI
            errors.append(f"unreadable: {row['file']}: {exc}")
    report = {
        "status": "PASS" if not errors else "FAIL",
        "papers": len(papers),
        "splits": dict(observed),
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
