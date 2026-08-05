from __future__ import annotations

from http.server import ThreadingHTTPServer

try:
    from . import run as base
except ImportError:  # Support direct script launch as well as package execution.
    import run as base
from Pipeline.Graph_Persistence.neo4j_writer import graph_summary


PORT = 8767


class NodeOnlyHandler(base.Handler):
    def do_GET(self) -> None:
        path = base.unquote(base.urlparse(self.path).path)
        if path == "/api/graph/summary":
            try:
                self.send_json(graph_summary())
            except RuntimeError as exc:
                self.send_json({"ok": False, "error": str(exc)}, 503)
            return
        if path == "/api/status":
            self.send_json({
                "ok": True, "mode": "paper_graph", "relationship_extraction": True,
                "execution_boundary": "pipeline.node_pipeline.run_extraction",
                "lexicon_entries": len(base.json_read(base.LEXICON_PATH)["entries"]),
                "job_count": len(base.jobs),
                "neo4j_browser": "http://localhost:8767/neo4j-browser/?connectURL=bolt%3A%2F%2Flocalhost%3A7690&db=neo4j",
                "neo4j_http": "http://localhost:7477/", "neo4j_bolt": "bolt://localhost:7690",
            })
            return
        if path in {"/", "/index.html"}:
            target = base.ROOT / "static" / "index.html"
            self.send_bytes(target.read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/engineered-water.grass":
            target = base.ROOT / "static" / "engineered-water.grass"
            self.send_bytes(target.read_bytes(), "text/plain; charset=utf-8")
            return
        super().do_GET()


def main() -> None:
    base.load_existing_jobs()
    server = ThreadingHTTPServer((base.HOST, PORT), NodeOnlyHandler)
    print(f"Engineered Water Systems Paper Graph Workbench: http://{base.HOST}:{PORT}/")
    print("LLM context-window relationship extraction is enabled.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
