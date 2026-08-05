from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from Pipeline.Knowledge_Graph_Core.llm_relationship_judge import MandatoryLLMRelationshipJudge
from Pipeline.Knowledge_Graph_Core.ontology import Ontology


def main() -> int:
    config = json.loads(
        (ROOT / "config" / "relationship_extraction.json").read_text(encoding="utf-8")
    )
    model = config["llm_relationship_judge"]["model"]
    judge = MandatoryLLMRelationshipJudge(
        Ontology(ROOT / "ontology" / "ontology.json"),
        config,
        ROOT / "cache" / "deepseek_smoke_test",
    )
    judge.verify_runtime()
    candidate = {
        "candidate_id": "deepseek-smoke-uses-model",
        "document_id": "deepseek-smoke-document",
        "sentence_id": "deepseek-smoke-sentence",
        "paragraph_id": "deepseek-smoke-paragraph",
        "section_id": "deepseek-smoke-section",
        "section_title": "Methods",
        "pages": [1],
        "subject_mention_id": "deepseek-smoke-system",
        "subject_label": "AISystem",
        "subject_text": "WaterRAG",
        "subject_start_char": 0,
        "subject_end_char": 8,
        "object_mention_id": "deepseek-smoke-model",
        "object_label": "Model",
        "object_text": "GPT-4",
        "object_start_char": 14,
        "object_end_char": 19,
        "allowed_predicates": ["USES_MODEL"],
        "endpoint_candidate_count": 1,
        "trigger_text": "uses",
        "trigger_start_char": 9,
        "trigger_end_char": 13,
        "evidence_quote": "WaterRAG uses GPT-4 for question answering.",
        "context_quotes": [],
        "attribution": "source_statement",
        "flags": {
            "negated": False,
            "modal": False,
            "hypothetical": False,
            "comparative": False,
            "causal": False,
        },
    }
    started = time.perf_counter()
    result = judge.judge([candidate])[candidate["candidate_id"]]
    elapsed = time.perf_counter() - started
    audit = result.pop("_llm_audit")
    passed = (
        result["predicate"] == "USES_MODEL"
        and audit["provider"] == "ollama"
        and audit["model"] == model
        and audit["llm_used"] is True
        and audit["status"] in {"complete", "complete_with_safe_repairs"}
    )
    payload = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "provider": "ollama",
        "model": model,
        "elapsed_seconds": round(elapsed, 3),
        "candidate": candidate,
        "judgment": result,
        "audit": audit,
    }
    output = ROOT / "output" / "deepseek_smoke_test.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
