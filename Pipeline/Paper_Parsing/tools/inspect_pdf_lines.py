from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


PARSER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PARSER_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Pipeline.Paper_Parsing.pdf_parser import extract_lines  # noqa: E402
from Pipeline.Paper_Parsing.profiles import detect_profile  # noqa: E402
from Pipeline.Paper_Parsing.text_utils import looks_like_heading  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect layout lines used by the pipeline parser")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--pattern", default=r"abstract|intro|method|result|discussion|conclu|reference")
    parser.add_argument("--page", type=int)
    parser.add_argument("--all", action="store_true", help="Print every line in the selected page(s)")
    parser.add_argument("--limit", type=int, help="Stop after this many matching lines")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    lines, diagnostics = extract_lines(args.pdf.read_bytes())
    profile = detect_profile(lines)
    body_size = diagnostics["body_font_size"]
    print(f"profile={profile.id} body_size={body_size} pages={diagnostics['page_count']}")
    pattern = re.compile(args.pattern, re.I)
    printed = 0
    for index, line in enumerate(lines):
        if args.page is not None and line.page != args.page:
            continue
        if args.all or pattern.search(line.text) or looks_like_heading(
            line.text, line.size, body_size, bold=line.bold, italic=line.italic
        ):
            print(
                f"{index:04d} p{line.page:02d} top={line.top:7.1f} x={line.x0:6.1f} "
                f"size={line.size:4.1f} bold={int(line.bold)} italic={int(line.italic)} | {line.text}"
            )
            printed += 1
            if args.limit is not None and printed >= args.limit:
                break


if __name__ == "__main__":
    main()
