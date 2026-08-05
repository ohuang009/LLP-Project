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
from urllib.parse import unquote, urlparse
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Pipeline.Node_Pipeline import ROOT, RUNS_DIR, LEXICON_PATH, ONTOLOGY_PATH, apply_node_review, apply_reference_resolution, apply_resolution, json_read, json_write, jsonl_read, run_extraction, stable_id
from Pipeline.Graph_Persistence.neo4j_writer import upsert_run


STATIC = ROOT / "static"
UPLOADS = ROOT / "uploads"
WORKSPACE = ROOT.parents[1]
SAMPLES = WORKSPACE / "SamplePapers"
NEO4J_BROWSER_ZIP = next(
    iter(sorted((WORKSPACE / "LocalNeo4j" / "runtime").glob("neo4j-community-*/web/neo4j-browser-*.zip"), reverse=True)),
    None,
)
HOST = "127.0.0.1"
PORT = 8767
MAX_UPLOAD = 80 * 1024 * 1024
jobs: dict[str, dict] = {}
lock = threading.Lock()
STAGE_ORDER = ("parsing", "ner", "adjudicating", "resolving", "relationships", "canonicalizing", "matching", "neo4j")


def advance_stage_reports(
    current_rows: list[dict], stage: str, message: str, value: int, report: dict | None = None,
) -> list[dict]:
    """Keep completed/running stage state accurate for live and finished runs."""
    stage_rows = {row["stage"]: dict(row) for row in current_rows}
    if stage in STAGE_ORDER:
        current_index = STAGE_ORDER.index(stage)
        for name in STAGE_ORDER[:current_index]:
            if name in stage_rows:
                stage_rows[name]["status"] = "complete"
        stage_rows[stage] = report or {
            "stage": stage, "status": "running", "message": message, "progress": value,
        }
    elif stage == "complete":
        for row in stage_rows.values():
            row["status"] = "complete"
    return [stage_rows[name] for name in STAGE_ORDER if name in stage_rows]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_name(value: str) -> str:
    name = Path(value).name
    clean = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")
    return clean or "paper.pdf"


def load_existing_jobs() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    for manifest in RUNS_DIR.glob("*/manifest.json"):
        try:
            summary = json_read(manifest)
        except Exception:
            continue
        jobs[manifest.parent.name] = {
            "job_id": manifest.parent.name, "status": "complete", "stage": "complete", "progress": 100,
            "message": "The paper is fully extracted and upserted into Neo4j.", "created_at": summary.get("created_at", ""),
            "summary": summary, "stages": summary.get("stages", []), "error": "",
        }


def job_payload(job_id: str, *, details: bool = False) -> dict:
    with lock:
        state = dict(jobs[job_id])
    if not details or state.get("status") != "complete":
        return state
    run_dir = RUNS_DIR / job_id
    state["mentions"] = jsonl_read(run_dir / "mentions.jsonl")
    state["entities"] = jsonl_read(run_dir / "canonical_entities_merged.jsonl")
    state["node_review_candidates"] = jsonl_read(run_dir / "node_review_candidates.jsonl")
    state["ontology_labels"] = sorted(json_read(ONTOLOGY_PATH).get("nodes", {}))
    state["reference_resolutions"] = jsonl_read(run_dir / "reference_resolution_review.jsonl")
    state["similar_nodes"] = jsonl_read(run_dir / "similar_nodes_review.jsonl")
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


def launch(filename: str, pdf_bytes: bytes) -> dict:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    job_id = f"run_{stamp}_{stable_id('p', filename, len(pdf_bytes))[-6:]}"
    upload = UPLOADS / job_id / safe_name(filename)
    upload.parent.mkdir(parents=True, exist_ok=True)
    upload.write_bytes(pdf_bytes)
    state = {
        "job_id": job_id, "status": "queued", "stage": "queued", "progress": 2,
        "message": "Paper received. Preparing the combined pipeline.", "created_at": utcnow(),
        "stages": [], "error": "",
    }
    with lock:
        jobs[job_id] = state

    def progress(stage: str, message: str, value: int, report: dict | None = None) -> None:
        with lock:
            jobs[job_id].update({
                "status": "running" if stage != "complete" else "complete",
                "stage": stage, "message": message, "progress": value,
                "stages": advance_stage_reports(jobs[job_id].get("stages", []), stage, message, value, report),
            })

    def worker() -> None:
        try:
            summary = run_extraction(upload, RUNS_DIR / job_id, progress)
            with lock:
                jobs[job_id].update({"status": "complete", "summary": summary, "progress": 100})
        except Exception as exc:
            traceback.print_exc()
            with lock:
                jobs[job_id].update({
                    "status": "failed", "stage": "failed", "progress": 100,
                    "message": "The run stopped before producing reviewable output.",
                    "error": f"{type(exc).__name__}: {exc}",
                })

    threading.Thread(target=worker, daemon=True, name=job_id).start()
    return state


class Handler(BaseHTTPRequestHandler):
    server_version = "EngineeredWaterWorkbench/1.0"

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def send_json(self, payload: object, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_bytes(self, data: bytes, content_type: str, filename: str | None = None) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{safe_name(filename)}"')
        self.end_headers()
        self.wfile.write(data)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1024 * 1024:
            raise ValueError("Request is too large.")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path == "/neo4j-browser":
            self.send_response(302)
            self.send_header("Location", "/neo4j-browser/")
            self.end_headers()
            return
        if path.startswith("/neo4j-browser/"):
            if NEO4J_BROWSER_ZIP is None or not NEO4J_BROWSER_ZIP.is_file():
                self.send_json({"error": "The bundled Neo4j Browser is unavailable."}, 503)
                return
            relative = path.removeprefix("/neo4j-browser/") or "index.html"
            parts = Path(relative).parts
            if any(part in {"", ".", ".."} for part in parts):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            member = "browser/" + "/".join(parts)
            try:
                with zipfile.ZipFile(NEO4J_BROWSER_ZIP) as archive:
                    data = archive.read(member)
            except (KeyError, zipfile.BadZipFile):
                # Browser is a single-page application; client-side routes use its index.
                if "." not in Path(relative).name:
                    with zipfile.ZipFile(NEO4J_BROWSER_ZIP) as archive:
                        data = archive.read("browser/index.html")
                    member = "browser/index.html"
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
            self.send_bytes(data, mimetypes.guess_type(member)[0] or "application/octet-stream")
            return
        if path == "/api/status":
            self.send_json({
                "ok": True, "mode": "node_and_relationship", "relationship_extraction": True,
                "lexicon_entries": len(json_read(LEXICON_PATH)["entries"]),
                "job_count": len(jobs),
                "neo4j_browser": f"http://localhost:{PORT}/neo4j-browser/?connectURL=bolt%3A%2F%2Flocalhost%3A7690&db=neo4j",
                "neo4j_http": "http://localhost:7477/", "neo4j_bolt": "bolt://localhost:7690",
                "water_ontology_browser": f"http://localhost:{PORT}/neo4j-browser/?connectURL=bolt%3A%2F%2Flocalhost%3A7691&db=neo4j",
                "water_ontology_http": "http://localhost:7478/", "water_ontology_bolt": "bolt://localhost:7691",
            })
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
        path = unquote(urlparse(self.path).path)
        if path == "/api/extract/sample":
            sample = next(iter(SAMPLES.glob("waterrag*.pdf")), None)
            if sample is None:
                self.send_json({"error": "WaterRAG sample paper is not available."}, 404)
                return
            self.send_json(launch(sample.name, sample.read_bytes()), 202)
            return
        if path == "/api/extract/sample/epanet-agentic":
            sample = SAMPLES / "1-s2.0-S0043135426001156-main.pdf"
            if not sample.is_file():
                self.send_json({"error": "The EPANET-Agentic sample paper is not available."}, 404)
                return
            self.send_json(launch(sample.name, sample.read_bytes()), 202)
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
                self.send_json(launch(filename, data), 202)
            except Exception as exc:
                self.send_json({"error": str(exc)}, 400)
            return
        match = re.fullmatch(r"/api/jobs/([^/]+)/reference-resolution", path)
        if match:
            job_id = match.group(1)
            run_dir = (RUNS_DIR / job_id).resolve()
            if run_dir.parent != RUNS_DIR.resolve() or not run_dir.is_dir():
                self.send_json({"error": "Run not found."}, 404)
                return
            try:
                body = self.read_json()
                result = apply_reference_resolution(
                    run_dir, body["resolution_id"], body["decision"], body.get("target_mention_id", "")
                )
                manifest = result["manifest"]
                if manifest.get("relationship_extraction") != "disabled_by_node_only_boundary":
                    from Pipeline.Edge_Extraction import materialize_canonical_relationships
                    relationships = materialize_canonical_relationships(
                        run_dir, result["entities"], jsonl_read(run_dir / "assertions.jsonl")
                    )
                    from Pipeline.Question_Evaluation.evaluator import evaluate_question_answerability
                    question_evaluation = evaluate_question_answerability(
                        run_dir, result["entities"], relationships, jsonl_read(run_dir / "assertions.jsonl")
                    )
                    graph = upsert_run(run_dir, json_read(run_dir / "parsed.json"), result["entities"], relationships)
                    json_write(run_dir / "neo4j_upsert.json", graph)
                    manifest["counts"]["relationships"] = len(relationships)
                    manifest["counts"]["answerable_questions"] = question_evaluation["answerable_questions"]
                    manifest["counts"]["total_questions"] = question_evaluation["total_questions"]
                    manifest["neo4j"] = graph
                    json_write(run_dir / "manifest.json", manifest)
                with lock:
                    jobs[job_id]["summary"] = manifest
                self.send_json({
                    "item": result["item"], "counts": manifest["counts"],
                    "canonical_entities": len(result["entities"]),
                })
            except (KeyError, ValueError, RuntimeError) as exc:
                self.send_json({"error": str(exc)}, 400)
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
                if manifest.get("relationship_extraction") != "disabled_by_node_only_boundary":
                    from Pipeline.Edge_Extraction import materialize_canonical_relationships
                    relationships = materialize_canonical_relationships(
                        run_dir, result["entities"], jsonl_read(run_dir / "assertions.jsonl")
                    )
                    from Pipeline.Question_Evaluation.evaluator import evaluate_question_answerability
                    question_evaluation = evaluate_question_answerability(
                        run_dir, result["entities"], relationships, jsonl_read(run_dir / "assertions.jsonl")
                    )
                    graph = upsert_run(run_dir, json_read(run_dir / "parsed.json"), result["entities"], relationships)
                    json_write(run_dir / "neo4j_upsert.json", graph)
                    manifest["counts"]["relationships"] = len(relationships)
                    manifest["counts"]["answerable_questions"] = question_evaluation["answerable_questions"]
                    manifest["counts"]["total_questions"] = question_evaluation["total_questions"]
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
        match = re.fullmatch(r"/api/jobs/([^/]+)/resolution", path)
        if match:
            job_id = match.group(1)
            run_dir = (RUNS_DIR / job_id).resolve()
            if run_dir.parent != RUNS_DIR.resolve() or not run_dir.is_dir():
                self.send_json({"error": "Run not found."}, 404)
                return
            try:
                body = self.read_json()
                result = apply_resolution(run_dir, body["resolution_id"], body["decision"], body.get("chosen_canonical_name", ""))
                graph = upsert_run(
                    run_dir, json_read(run_dir / "parsed.json"),
                    result["merged_entities"], result["canonical_relationships"],
                )
                json_write(run_dir / "neo4j_upsert.json", graph)
                result["neo4j"] = graph
                with lock:
                    manifest = json_read(run_dir / "manifest.json")
                    manifest["counts"]["merged_entities"] = len(result["merged_entities"])
                    manifest["counts"]["relationships"] = len(result["canonical_relationships"])
                    manifest["counts"]["answerable_questions"] = result["question_answerability"]["answerable_questions"]
                    manifest["counts"]["total_questions"] = result["question_answerability"]["total_questions"]
                    manifest["neo4j"] = graph
                    json_write(run_dir / "manifest.json", manifest)
                    jobs[job_id]["summary"] = manifest
                self.send_json(result)
            except (KeyError, ValueError, RuntimeError) as exc:
                self.send_json({"error": str(exc)}, 400)
            return
        self.send_json({"error": "Unknown endpoint."}, 404)


def main() -> None:
    load_existing_jobs()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Engineered Water Systems Extraction Workbench: http://{HOST}:{PORT}/")
    print("Combined node, predicate, and Neo4j upsert pipeline is active.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
