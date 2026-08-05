from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_regression_corpus import DEFAULT_CONFIG, evaluate_case, read_json


RUN_PREFIX = {
    "waterrag": "01_",
    "llm_epanet": "02_",
    "epanet_agentic": "03_",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply trusted regression gates to a fixed-corpus run directory.")
    parser.add_argument("runs_root", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = read_json(args.config)
    results = []
    for spec in config["papers"]:
        prefix = RUN_PREFIX[spec["id"]]
        matches = sorted(path for path in args.runs_root.iterdir() if path.is_dir() and path.name.startswith(prefix))
        results.append(evaluate_case(spec, matches[0] if len(matches) == 1 else None))
    payload = {
        "status": "PASS" if all(row["status"] == "PASS" for row in results) else "FAIL",
        "runs_root": str(args.runs_root),
        "papers": results,
    }
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
