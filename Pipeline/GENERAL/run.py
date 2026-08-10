from __future__ import annotations

from datetime import datetime, timezone
from email.parser import BytesParser
from email.policy import default
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import mimetypes
from pathlib import Path
import re
import sys
import threading
import traceback
from urllib.parse import parse_qs, unquote, urlparse
from urllib.error import URLError
from urllib.request import urlopen
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Pipeline.Node_Pipeline import ROOT, RUNS_DIR, ONTOLOGY_PATH, apply_node_review, json_read, json_write, jsonl_read, run_extraction, stable_id
from Pipeline.Graph_Persistence.neo4j_writer import graph_summary, remove_document, reset_graph, upsert_run
from Pipeline.lexicon import load_lexicon


STATIC = ROOT / "static"
UPLOADS = ROOT / "uploads"
WORKSPACE = ROOT.parents[1]
SAMPLES = WORKSPACE / "SamplePapers"
NEO4J_BROWSER_URL = "http://127.0.0.1:7477/"
HOST = "127.0.0.1"
PORT = 8767
MAX_UPLOAD = 80 * 1024 * 1024
jobs: dict[str, dict] = {}
lock = threading.Lock()
graph_actions: set[str] = set()
GRAPH_RESET_ACTION = "__reset_all__"
STAGE_ORDER = ("parsing", "ner", "adjudicating", "resolving", "relationships")
NODE_CONFIG = ROOT / "config" / "pipeline.json"
MODEL_READINESS: dict = {
    "ready": False,
    "status": "not_started",
    "warmup_completed": False,
}


def preload_pipeline_models() -> dict:
    """Preload required local NLP models before the HTTP server starts."""
    global MODEL_READINESS
    MODEL_READINESS = {
        "ready": False,
        "status": "loading",
        "warmup_completed": False,
    }
    print("Preloading spaCy and SciBERT; the website will open when they are ready...")
    try:
        from Pipeline.Grammatical_Parsing.candidates import preload_grammatical_models

        MODEL_READINESS = preload_grammatical_models(
            ONTOLOGY_PATH, json_read(NODE_CONFIG)
        )
    except Exception as exc:
        MODEL_READINESS = {
            "ready": False,
            "status": "failed",
            "warmup_completed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(f"Model preload failed: {MODEL_READINESS['error']}")
        raise
    print(
        "Model preload complete in "
        f"{MODEL_READINESS.get('elapsed_seconds', 0.0):.1f}s: "
        f"{MODEL_READINESS.get('parser_model', 'spaCy')} + "
        f"{MODEL_READINESS.get('scibert_model', 'SciBERT')}"
    )
    return dict(MODEL_READINESS)


def configured_llm_model() -> str:
    """Return the default model used when a run does not select one explicitly."""
    return str(json_read(NODE_CONFIG)["ollama"]["model"])


def _ollama_catalog(payload: dict, default_model: str) -> dict:
    """Normalize Ollama's model list for the existing model selector UI."""
    rows = []
    for model in payload.get("models", []):
        name = str(model.get("name", "")).strip()
        if not name:
            continue
        details = model.get("details") or {}
        rows.append({
            "name": name,
            "parameter_size": str(details.get("parameter_size", "")),
            "quantization": str(details.get("quantization_level", "")),
            "size_bytes": int(model.get("size", 0)),
        })
    rows.sort(key=lambda row: (row["name"] != default_model, row["name"].casefold()))
    names = {row["name"] for row in rows}
    available = default_model in names
    return {
        "available": available,
        "default_model": default_model,
        "models": rows,
        **({} if available else {
            "error": f"Ollama is running, but {default_model} is not installed. Pull it, then reload the website."
        }),
    }


def ollama_model_catalog() -> dict:
    """Return locally installed Ollama models and require the configured Qwen model."""
    config = json_read(NODE_CONFIG)["ollama"]
    default_model = configured_llm_model()
    url = f"{str(config.get('base_url', 'http://127.0.0.1:11434')).rstrip('/')}/api/tags"
    try:
        with urlopen(url, timeout=5) as response:
            return _ollama_catalog(json.loads(response.read().decode("utf-8")), default_model)
    except (OSError, URLError, ValueError, json.JSONDecodeError) as exc:
        return {
            "available": False,
            "default_model": default_model,
            "models": [],
            "error": f"Start Ollama and make sure {default_model} is installed ({type(exc).__name__}).",
        }


def selected_llm_model(value: str | None) -> str:
    """Return the requested local Ollama model or the configured default."""
    return str(value or configured_llm_model()).strip()


def advance_stage_reports(
    current_rows: list[dict], stage: str, message: str, value: int, report: dict | None = None,
) -> list[dict]:
    """Keep completed/running stage state accurate for live and finished runs."""
    stage_rows = {
        row["stage"]: dict(row)
        for row in current_rows
        if isinstance(row, dict) and row.get("stage") in STAGE_ORDER
    }
    if stage in STAGE_ORDER:
        current_index = STAGE_ORDER.index(stage)
        for name in STAGE_ORDER[:current_index]:
            if name in stage_rows:
                stage_rows[name]["status"] = "complete"
        stage_rows[stage] = {
            "stage": stage,
            "status": "complete" if report is not None else "running",
            "message": message,
            "progress": value,
            **({"output": report} if report is not None else {}),
        }
    elif stage == "complete":
        for row in stage_rows.values():
            row["status"] = "complete"
    return [stage_rows[name] for name in STAGE_ORDER if name in stage_rows]


def utcnow() -> str:
    """Handle utcnow for this stage. It supports the local interface used to launch and inspect pipeline runs."""
    return datetime.now(timezone.utc).isoformat()


def safe_name(value: str) -> str:
    """Handle safe name for this stage. It supports the local interface used to launch and inspect pipeline runs."""
    name = Path(value).name
    clean = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")
    return clean or "paper.pdf"


def load_existing_jobs() -> None:
    """Load existing jobs. It supports the local interface used to launch and inspect pipeline runs."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    for manifest in RUNS_DIR.glob("*/manifest.json"):
        try:
            summary = json_read(manifest)
        except Exception:
            continue
        publication = graph_publication(summary)
        jobs[manifest.parent.name] = {
            "job_id": manifest.parent.name, "status": "complete", "stage": "complete", "progress": 100,
            "message": "The paper is fully evaluated and ready for a graph decision.", "created_at": summary.get("created_at", ""),
            "summary": summary, "stages": summary.get("stages", []), "error": "",
            "llm_model": (summary.get("llm") or {}).get("model", ""),
            "processing_unit": "paragraph",
            "graph_publication": publication,
        }


def graph_publication(summary: dict) -> dict:
    """Return the durable one-time graph publication state, including legacy runs."""
    stored = summary.get("graph_publication")
    if isinstance(stored, dict) and stored.get("state") in {"not_added", "added", "removed"}:
        return {
            "state": stored["state"],
            "add_count": int(stored.get("add_count", 1 if stored["state"] != "not_added" else 0)),
            "added_at": stored.get("added_at", ""),
            "removed_at": stored.get("removed_at", ""),
            "graph_url": stored.get("graph_url", NEO4J_BROWSER_URL),
        }
    neo4j = summary.get("neo4j") or {}
    state = "removed" if neo4j.get("status") == "removed" else (
        "added" if neo4j.get("status") == "upserted" else "not_added"
    )
    return {
        "state": state,
        "add_count": 1 if state in {"added", "removed"} else 0,
        "added_at": summary.get("created_at", "") if state in {"added", "removed"} else "",
        "removed_at": "",
        "graph_url": NEO4J_BROWSER_URL,
    }


def mark_manifest_removed_by_reset(summary: dict, reset_at: str) -> bool:
    """Mark a previously published run removed after a database-wide reset."""
    publication = graph_publication(summary)
    if publication["state"] != "added":
        return False
    publication.update({"state": "removed", "removed_at": reset_at})
    summary["graph_publication"] = publication
    summary["neo4j"] = {
        "status": "removed", "reason": "graph_reset", "removed_at": reset_at,
    }
    return True


def job_payload(job_id: str, *, details: bool = False) -> dict:
    """Handle job payload for this stage. It supports the local interface used to launch and inspect pipeline runs."""
    with lock:
        state = dict(jobs[job_id])
    state["graph_publication"] = graph_publication(state.get("summary") or {})
    if not details or state.get("status") != "complete":
        return state
    run_dir = RUNS_DIR / job_id
    state["mentions"] = jsonl_read(run_dir / "mentions.jsonl")
    state["entities"] = jsonl_read(run_dir / "canonical_entities_merged.jsonl")
    state["node_review_candidates"] = jsonl_read(run_dir / "node_review_candidates.jsonl")
    state["ontology_labels"] = sorted(json_read(ONTOLOGY_PATH).get("nodes", {}))
    state["relationship_candidates"] = jsonl_read(run_dir / "relationship_candidates.jsonl")
    state["relationship_unknown_nodes"] = jsonl_read(run_dir / "relationship_unknown_nodes.jsonl")
    state["relationships"] = jsonl_read(run_dir / "canonical_relationships.jsonl")
    state["assertions"] = jsonl_read(run_dir / "assertions.jsonl")
    question_path = run_dir / "question_answerability.json"
    state["question_answerability"] = json_read(question_path) if question_path.is_file() else {"questions": []}
    state["validation"] = json_read(run_dir / "validation_report.json")
    state["stages"] = state.get("stages") or state["summary"].get("stages", [])
    state["neo4j"] = state["summary"].get("neo4j", {})
    state["downloads"] = state["summary"].get("outputs", [])
    return state


def launch(
    filename: str,
    pdf_bytes: bytes,
    *,
    llm_model: str | None = None,
) -> dict:
    """Launch this stage. It supports the local interface used to launch and inspect pipeline runs."""
    llm_model = selected_llm_model(llm_model)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    job_id = f"run_{stamp}_{stable_id('p', filename, len(pdf_bytes))[-6:]}"
    upload = UPLOADS / job_id / safe_name(filename)
    upload.parent.mkdir(parents=True, exist_ok=True)
    upload.write_bytes(pdf_bytes)
    state = {
        "job_id": job_id, "status": "queued", "stage": "queued", "progress": 2,
        "message": "Paper received. Preparing the combined pipeline.", "created_at": utcnow(),
        "stages": [], "error": "", "llm_model": llm_model,
        "processing_unit": "paragraph",
    }
    with lock:
        jobs[job_id] = state

    def progress(stage: str, message: str, value: int, *details: object) -> None:
        """Record progress for the current background extraction job. It supports the local interface used to launch and inspect pipeline runs."""
        report = next((item for item in reversed(details) if isinstance(item, dict)), None)
        with lock:
            jobs[job_id].update({
                "status": "running" if stage != "complete" else "complete",
                "stage": stage, "message": message, "progress": value,
                "stages": advance_stage_reports(jobs[job_id].get("stages", []), stage, message, value, report),
            })

    def worker() -> None:
        """Run extraction in the background and record its final state. It supports the local interface used to launch and inspect pipeline runs."""
        try:
            summary = run_extraction(
                upload, RUNS_DIR / job_id, progress, llm_model=llm_model,
            )
            with lock:
                jobs[job_id].update({"status": "complete", "summary": summary, "progress": 100})
        except Exception as exc:
            traceback.print_exc()
            with lock:
                jobs[job_id].update({
                    "status": "failed", "stage": "failed",
                    "message": "The run stopped before producing reviewable output.",
                    "error": f"{type(exc).__name__}: {exc}",
                })

    threading.Thread(target=worker, daemon=True, name=job_id).start()
    return state


class Handler(BaseHTTPRequestHandler):
    server_version = "EngineeredWaterWorkbench/1.0"

    def log_message(self, fmt: str, *args: object) -> None:
        """Suppress the default noisy HTTP request logging. It supports the local interface used to launch and inspect pipeline runs."""
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def send_json(self, payload: object, status: int = 200) -> None:
        """Send a JSON response with the correct HTTP headers. It supports the local interface used to launch and inspect pipeline runs."""
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_bytes(self, data: bytes, content_type: str, filename: str | None = None) -> None:
        """Send a binary artifact with the correct HTTP headers. It supports the local interface used to launch and inspect pipeline runs."""
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{safe_name(filename)}"')
        self.end_headers()
        self.wfile.write(data)

    def read_json(self) -> dict:
        """Read and decode the request body as JSON. It supports the local interface used to launch and inspect pipeline runs."""
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1024 * 1024:
            raise ValueError("Request is too large.")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:
        """Handle a read request from the local pipeline interface. It supports the local interface used to launch and inspect pipeline runs."""
        path = unquote(urlparse(self.path).path)
        if path == "/neo4j-browser" or path.startswith("/neo4j-browser/"):
            self.send_response(302)
            self.send_header("Location", NEO4J_BROWSER_URL)
            self.end_headers()
            return
        if path == "/api/status":
            self.send_json({
                "ok": True, "mode": "node_and_relationship", "relationship_extraction": True,
                "model_readiness": dict(MODEL_READINESS),
                "lexicon_entries": len(load_lexicon()),
                "relationship_processing_unit": "paragraph",
                "job_count": len(jobs),
                "neo4j_browser": NEO4J_BROWSER_URL,
                "neo4j_http": "http://localhost:7477/", "neo4j_bolt": "bolt://localhost:7690",
                "water_ontology_browser": "http://127.0.0.1:7478/",
                "water_ontology_http": "http://localhost:7478/", "water_ontology_bolt": "bolt://localhost:7691",
            })
            return
        if path == "/api/models":
            self.send_json(ollama_model_catalog())
            return
        if path == "/api/graph/summary":
            try:
                self.send_json(graph_summary())
            except (OSError, ValueError, RuntimeError) as exc:
                self.send_json({"error": str(exc)}, 503)
            return
        if path == "/api/jobs":
            with lock:
                rows = sorted((dict(value) for value in jobs.values()), key=lambda row: row.get("created_at", ""), reverse=True)
            self.send_json({"jobs": rows})
            return
        match = re.fullmatch(r"/api/jobs/([^/]+)", path)
        if match:
            job_id = match.group(1)
            if job_id not in jobs:
                self.send_json({"error": "Unknown run."}, 404)
            else:
                self.send_json(job_payload(job_id, details=True))
            return
        match = re.fullmatch(r"/api/jobs/([^/]+)/download/(.+)", path)
        if match:
            job_id, filename = match.groups()
            run_dir = (RUNS_DIR / job_id).resolve()
            target = (run_dir / Path(filename).name).resolve()
            if run_dir.parent != RUNS_DIR.resolve() or target.parent != run_dir or not target.is_file():
                self.send_json({"error": "File not found."}, 404)
                return
            self.send_bytes(target.read_bytes(), mimetypes.guess_type(target.name)[0] or "application/octet-stream", target.name)
            return
        match = re.fullmatch(r"/api/jobs/([^/]+)/bundle.zip", path)
        if match:
            job_id = match.group(1)
            run_dir = (RUNS_DIR / job_id).resolve()
            if run_dir.parent != RUNS_DIR.resolve() or not (run_dir / "manifest.json").is_file():
                self.send_json({"error": "Run not found."}, 404)
                return
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                for target in sorted(run_dir.iterdir()):
                    if target.is_file():
                        archive.writestr(target.name, target.read_bytes())
            self.send_bytes(buffer.getvalue(), "application/zip", f"{job_id}-knowledge-extraction.zip")
            return
        if path == "/" or path == "/index.html":
            target = STATIC / "index.html"
        else:
            target = (STATIC / path.lstrip("/")).resolve()
            if target.parent != STATIC.resolve():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_bytes(target.read_bytes(), mimetypes.guess_type(target.name)[0] or "application/octet-stream")

    def do_POST(self) -> None:
        """Handle a write or pipeline-launch request from the local interface. It supports the local interface used to launch and inspect pipeline runs."""
        parsed_url = urlparse(self.path)
        path = unquote(parsed_url.path)
        query = parse_qs(parsed_url.query)
        if path == "/api/graph/reset":
            try:
                body = self.read_json()
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                self.send_json({"error": f"Invalid reset request: {exc}"}, 400)
                return
            if not isinstance(body, dict) or body.get("confirmation") != "RESET":
                self.send_json({"error": "Type RESET exactly to confirm the graph reset."}, 400)
                return
            with lock:
                if graph_actions:
                    self.send_json({"error": "Another graph action is still running. Try again when it finishes."}, 409)
                    return
                graph_actions.add(GRAPH_RESET_ACTION)
            try:
                receipt = reset_graph()
                reset_at = utcnow()
                updated_runs = 0
                manifest_update_errors: list[str] = []
                for manifest_path in sorted(RUNS_DIR.glob("*/manifest.json")):
                    try:
                        manifest = json_read(manifest_path)
                        if not mark_manifest_removed_by_reset(manifest, reset_at):
                            continue
                        json_write(manifest_path, manifest)
                        job_id = manifest_path.parent.name
                        with lock:
                            if job_id in jobs:
                                jobs[job_id]["summary"] = manifest
                                jobs[job_id]["graph_publication"] = graph_publication(manifest)
                        updated_runs += 1
                    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                        manifest_update_errors.append(f"{manifest_path.parent.name}: {exc}")
                receipt.update({
                    "reset_at": reset_at,
                    "published_runs_marked_removed": updated_runs,
                    "manifest_update_errors": manifest_update_errors,
                })
                audit_path = RUNS_DIR / "graph_reset_history.jsonl"
                with audit_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(receipt, ensure_ascii=False) + "\n")
                self.send_json({"ok": True, "receipt": receipt})
            except (OSError, ValueError, RuntimeError) as exc:
                self.send_json({"error": str(exc)}, 500)
            finally:
                with lock:
                    graph_actions.discard(GRAPH_RESET_ACTION)
            return
        if path == "/api/extract/sample":
            sample = SAMPLES / "evaluation" / "refinement" / "01_waterrag.pdf"
            if not sample.is_file():
                self.send_json({"error": "WaterRAG sample paper is not available."}, 404)
                return
            try:
                self.send_json(launch(
                    sample.name, sample.read_bytes(), llm_model=query.get("model", [None])[0],
                ), 202)
            except (OSError, RuntimeError, ValueError) as exc:
                self.send_json({"error": str(exc)}, 400)
            return
        if path == "/api/extract/sample/epanet-agentic":
            sample = SAMPLES / "evaluation" / "refinement" / "03_epanet_agentic.pdf"
            if not sample.is_file():
                self.send_json({"error": "The EPANET-Agentic sample paper is not available."}, 404)
                return
            try:
                self.send_json(launch(
                    sample.name, sample.read_bytes(), llm_model=query.get("model", [None])[0],
                ), 202)
            except (OSError, RuntimeError, ValueError) as exc:
                self.send_json({"error": str(exc)}, 400)
            return
        if path == "/api/extract":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_UPLOAD:
                    raise ValueError("Upload must be a PDF smaller than 80 MB.")
                content_type = self.headers.get("Content-Type", "")
                raw = self.rfile.read(length)
                message = BytesParser(policy=default).parsebytes(
                    f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + raw
                )
                part = next((item for item in message.iter_attachments() if item.get_filename()), None)
                if part is None:
                    raise ValueError("No PDF file was supplied.")
                filename = safe_name(part.get_filename() or "paper.pdf")
                data = part.get_payload(decode=True) or b""
                if not data.startswith(b"%PDF"):
                    raise ValueError("The uploaded file does not appear to be a PDF.")
                model_part = next((
                    item for item in message.walk()
                    if item.get_param("name", header="content-disposition") == "model"
                ), None)
                model = None
                if model_part is not None:
                    model = (model_part.get_payload(decode=True) or b"").decode("utf-8").strip()
                self.send_json(launch(
                    filename, data, llm_model=model,
                ), 202)
            except Exception as exc:
                self.send_json({"error": str(exc)}, 400)
            return
        match = re.fullmatch(r"/api/jobs/([^/]+)/graph/(add|remove)", path)
        if match:
            job_id, action = match.groups()
            run_dir = (RUNS_DIR / job_id).resolve()
            manifest_path = run_dir / "manifest.json"
            if run_dir.parent != RUNS_DIR.resolve() or not manifest_path.is_file():
                self.send_json({"error": "Run not found."}, 404)
                return
            with lock:
                if job_id in graph_actions or GRAPH_RESET_ACTION in graph_actions:
                    self.send_json({"error": "A graph action is already running for this paper."}, 409)
                    return
                manifest = json_read(manifest_path)
                publication = graph_publication(manifest)
                allowed = publication["state"] == ("not_added" if action == "add" else "added")
                if not allowed:
                    if action == "add" and publication["add_count"] >= 1:
                        message = "This paper has already been added once and cannot be added again."
                    else:
                        message = f"This paper cannot be {action}ed while its graph state is {publication['state']}."
                    self.send_json({"error": message, "graph_publication": publication}, 409)
                    return
                graph_actions.add(job_id)
            try:
                if action == "add":
                    parsed = json_read(run_dir / "parsed.json")
                    entities = jsonl_read(run_dir / "canonical_entities_merged.jsonl")
                    relationships = jsonl_read(run_dir / "canonical_relationships.jsonl")
                    receipt = upsert_run(run_dir, parsed, entities, relationships)
                    json_write(run_dir / "neo4j_upsert.json", receipt)
                    publication.update({
                        "state": "added", "add_count": 1, "added_at": utcnow(), "removed_at": "",
                    })
                    outputs = manifest.setdefault("outputs", [])
                    if "neo4j_upsert.json" not in outputs:
                        outputs.append("neo4j_upsert.json")
                else:
                    document_id = (manifest.get("document") or {}).get("id", "")
                    receipt = remove_document(document_id, job_id)
                    json_write(run_dir / "neo4j_removal.json", receipt)
                    publication.update({"state": "removed", "removed_at": utcnow()})
                    outputs = manifest.setdefault("outputs", [])
                    if "neo4j_removal.json" not in outputs:
                        outputs.append("neo4j_removal.json")
                manifest["neo4j"] = receipt
                manifest["graph_publication"] = publication
                json_write(manifest_path, manifest)
                with lock:
                    if job_id in jobs:
                        jobs[job_id]["summary"] = manifest
                        jobs[job_id]["graph_publication"] = publication
                self.send_json({"ok": True, "receipt": receipt, "graph_publication": publication})
            except (OSError, ValueError, RuntimeError) as exc:
                self.send_json({"error": str(exc), "graph_publication": publication}, 400)
            finally:
                with lock:
                    graph_actions.discard(job_id)
            return
        match = re.fullmatch(r"/api/jobs/([^/]+)/node-review", path)
        if match:
            job_id = match.group(1)
            run_dir = (RUNS_DIR / job_id).resolve()
            if run_dir.parent != RUNS_DIR.resolve() or not run_dir.is_dir():
                self.send_json({"error": "Run not found."}, 404)
                return
            try:
                body = self.read_json()
                result = apply_node_review(
                    run_dir,
                    body["candidate_id"],
                    body["decision"],
                    ontology_label=body.get("ontology_label", ""),
                    canonical_name=body.get("canonical_name", ""),
                    reason=body.get("reason", ""),
                )
                manifest = result["manifest"]
                if graph_publication(manifest)["state"] == "added":
                    relationships = []
                    if manifest.get("relationship_extraction") != "disabled_by_node_only_boundary":
                        from Pipeline.Edge_Extraction import materialize_canonical_relationships
                        relationships = materialize_canonical_relationships(
                            run_dir, result["entities"], jsonl_read(run_dir / "assertions.jsonl")
                        )
                        from Pipeline.Question_Evaluation.evaluator import evaluate_question_answerability
                        question_evaluation = evaluate_question_answerability(
                            run_dir, result["entities"], relationships, jsonl_read(run_dir / "assertions.jsonl")
                        )
                        manifest["counts"]["relationships"] = len(relationships)
                        manifest["counts"]["answerable_questions"] = question_evaluation["answerable_questions"]
                        manifest["counts"]["total_questions"] = question_evaluation["total_questions"]
                    graph = upsert_run(run_dir, json_read(run_dir / "parsed.json"), result["entities"], relationships)
                    json_write(run_dir / "neo4j_upsert.json", graph)
                    manifest["neo4j"] = graph
                    json_write(run_dir / "manifest.json", manifest)
                with lock:
                    jobs[job_id]["summary"] = manifest
                self.send_json({
                    "item": result["item"],
                    "decision": result["effective_decision"],
                    "counts": manifest["counts"],
                    "canonical_entities": len(result["entities"]),
                })
            except (KeyError, ValueError, RuntimeError) as exc:
                self.send_json({"error": str(exc)}, 400)
            return
        self.send_json({"error": "Unknown endpoint."}, 404)


def main() -> None:
    """Start the local full-pipeline server."""
    # Load expensive NLP/LLM dependencies before accepting requests, then restore
    # summaries for run directories that already exist on disk.
    preload_pipeline_models()
    load_existing_jobs()
    # Bind the threaded HTTP server used by the local browser workbench.
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Engineered Water Systems Extraction Workbench: http://{HOST}:{PORT}/")
    print("Combined node, predicate, and Neo4j upsert pipeline is active.")
    # Serve until interrupted and always release the listening socket on exit.
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    # Start the workbench only when this module is run as a script.
    main()
