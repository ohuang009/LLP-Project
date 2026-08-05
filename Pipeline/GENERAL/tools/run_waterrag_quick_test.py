from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Pipeline.Node_Pipeline.node_only_runner import run_node_only_extraction


def main() -> None:
    pdf = PROJECT_ROOT / "SamplePapers" / "waterrag-a-multiagent-retrieval-augmented-generation-framework-to-support-water-industry-transitions-to-net-zero.pdf"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = PROJECT_ROOT / "PipelineAudits" / "runs" / f"waterrag_quick_abstract_{stamp}"

    def progress(stage: str, message: str, value: int) -> None:
        print(f"[{value:>3}%] {stage}: {message}", flush=True)

    summary = run_node_only_extraction(
        pdf,
        run_dir,
        progress,
        include_section_titles={"Abstract"},
    )
    print(json.dumps({"run_dir": str(run_dir), "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
