"""Read-only setup audit for node-only and full pipeline modes."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys
from urllib import error, request


WORKSPACE = Path(__file__).resolve().parent.parent
PIPELINE = WORKSPACE / "Pipeline"
GENERAL = PIPELINE / "GENERAL"


def package_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def url_available(url: str, timeout: float = 3.0) -> tuple[bool, str]:
    try:
        with request.urlopen(url, timeout=timeout) as response:
            return True, str(response.status)
    except error.HTTPError as exc:
        # An HTTP response proves that the service is reachable, even if auth rejects it.
        return True, str(exc.code)
    except (error.URLError, TimeoutError, OSError) as exc:
        return False, str(exc.reason if isinstance(exc, error.URLError) else exc)


def ollama_models(base_url: str) -> tuple[bool, set[str], str]:
    try:
        with request.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=4) as response:
            payload = json.loads(response.read().decode("utf-8"))
        names = {
            value
            for row in payload.get("models", [])
            for value in (row.get("name", ""), row.get("model", ""))
            if value
        }
        return True, names, "API reachable"
    except (error.URLError, TimeoutError, OSError, ValueError) as exc:
        return False, set(), str(exc.reason if isinstance(exc, error.URLError) else exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("node-only", "full"), default="full")
    args = parser.parse_args()

    checks: list[tuple[str, str, bool, bool]] = []

    python_ok = (3, 11) <= sys.version_info[:2] <= (3, 12)
    checks.append(("Python", sys.version.split()[0], True, python_ok))

    required_packages = (
        "pdfplumber", "numpy", "spacy", "en-core-web-sm",
        "sentence-transformers", "torch", "transformers", "huggingface-hub",
    )
    for distribution in required_packages:
        version = package_version(distribution)
        checks.append((f"Python package: {distribution}", version or "missing", True, bool(version)))
    pypdf_version = package_version("pypdf")
    checks.append(("Optional tool package: pypdf", pypdf_version or "missing", False, bool(pypdf_version)))

    try:
        import spacy
        spacy.load("en_core_web_sm")
        spacy_detail, spacy_ok = "en_core_web_sm loads", True
    except Exception as exc:  # setup checker must report all failures in one pass
        spacy_detail, spacy_ok = str(exc).splitlines()[0], False
    checks.append(("spaCy dependency model", spacy_detail, True, spacy_ok))

    scibert_cache = GENERAL / "cache" / "scibert" / "models--allenai--scibert_scivocab_uncased"
    scibert_snapshots = list((scibert_cache / "snapshots").glob("*")) if scibert_cache.exists() else []
    checks.append((
        "SciBERT local model",
        str(scibert_cache.relative_to(WORKSPACE)) if scibert_snapshots else "not downloaded",
        True,
        bool(scibert_snapshots),
    ))

    node_config = json.loads((GENERAL / "config" / "node_extraction.json").read_text(encoding="utf-8"))
    llm_spec = node_config["llm_node_judge"]
    ollama_ok, model_names, ollama_detail = ollama_models(llm_spec["base_url"])
    checks.append(("Ollama API", ollama_detail, args.mode == "full", ollama_ok))
    model = llm_spec["model"]
    model_ok = model in model_names or f"{model}:latest" in model_names
    checks.append((f"Ollama model: {model}", "downloaded" if model_ok else "missing", args.mode == "full", model_ok))

    try:
        java = subprocess.run(
            ["java", "-version"], capture_output=True, text=True, timeout=5, check=False,
        )
        java_line = (java.stderr or java.stdout).splitlines()[0]
        java_ok = java.returncode == 0
    except (OSError, subprocess.SubprocessError) as exc:
        java_line, java_ok = str(exc), False
    checks.append(("Java for Neo4j", java_line, True, java_ok))

    runtimes = sorted((WORKSPACE / "LocalNeo4j" / "runtime").glob("neo4j-community-*"))
    checks.append(("Bundled Neo4j runtime", runtimes[-1].name if runtimes else "missing", True, bool(runtimes)))
    neo4j_ok, neo4j_detail = url_available("http://127.0.0.1:7477/")
    checks.append(("Neo4j HTTP endpoint", neo4j_detail, True, neo4j_ok))

    print(f"Pipeline environment check ({args.mode})")
    print("=" * 78)
    for name, detail, required, passed in checks:
        status = "PASS" if passed else ("FAIL" if required else "WARN")
        requirement = "required" if required else "optional"
        print(f"{status:4}  {name:36} [{requirement}]  {detail}")

    failures = [name for name, _, required, passed in checks if required and not passed]
    if failures:
        print("\nMissing mandatory requirements:")
        for name in failures:
            print(f"- {name}")
        print("\nFollow Pipeline/README.md, then rerun this command.")
        return 1
    print("\nEnvironment is ready for this mode.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

