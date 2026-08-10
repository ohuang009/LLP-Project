"""Read-only environment check for the maintained pipeline."""
from __future__ import annotations

from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import sys
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent


def _version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return ""


def main() -> int:
    checks: list[tuple[str, bool, str]] = []
    checks.append(("Python 3.11/3.12", (3, 11) <= sys.version_info[:2] <= (3, 12), sys.version.split()[0]))
    for package in ("pdfplumber", "numpy", "spacy", "en-core-web-sm", "sentence-transformers", "torch", "transformers"):
        version = _version(package)
        checks.append((package, bool(version), version or "missing"))
    try:
        import spacy
        spacy.load("en_core_web_sm")
        checks.append(("spaCy dependency parser", True, "en_core_web_sm"))
    except Exception as exc:
        checks.append(("spaCy dependency parser", False, str(exc).splitlines()[0]))

    config = json.loads((ROOT / "GENERAL" / "config" / "pipeline.json").read_text(encoding="utf-8"))
    try:
        from Pipeline.Grammatical_Parsing.candidates import preload_grammatical_models

        readiness = preload_grammatical_models(
            ROOT / "GENERAL" / "ontology" / "ontology.json", config,
        )
        checks.append((
            "SciBERT ontology model", bool(readiness.get("ready")),
            str(readiness.get("scibert_model", config["grammar"]["scibert_model"])),
        ))
    except Exception as exc:
        checks.append(("SciBERT ontology model", False, str(exc).splitlines()[0]))
    try:
        from Pipeline.lexicon import load_lexicon

        checks.append(("Persistent lexicon", True, f"{len(load_lexicon())} node/type pairs"))
    except Exception as exc:
        checks.append(("Persistent lexicon", False, str(exc).splitlines()[0]))

    ollama = config["ollama"]
    model = ollama["model"]
    tags_url = f"{str(ollama['base_url']).rstrip('/')}/api/tags"
    try:
        with urlopen(tags_url, timeout=5) as response:
            names = {row.get("name") for row in json.loads(response.read().decode("utf-8")).get("models", [])}
        checks.append(("Ollama + Qwen", model in names, f"{model} installed" if model in names else f"missing {model}"))
    except (OSError, URLError, ValueError, json.JSONDecodeError) as exc:
        checks.append(("Ollama + Qwen", False, f"Ollama unavailable: {exc}"))
    checks.append(("Java", bool(shutil.which("java")), shutil.which("java") or "missing"))
    neo4j = Path(os.environ.get("NEO4J_HOME", "")) / "bin" / "neo4j.bat" if os.environ.get("NEO4J_HOME") else None
    checks.append(("Neo4j", bool(neo4j and neo4j.is_file()), str(neo4j or "set NEO4J_HOME")))

    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL':4}  {name}: {detail}")
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
