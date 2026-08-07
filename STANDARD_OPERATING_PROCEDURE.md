# Engineered Water Systems Pipeline — Standard Operating Procedure

## 1. Purpose and scope

This SOP explains how to set up and operate the local paper-to-knowledge-graph project. It covers:

1. starting the complete website/workbench;
2. running the pipeline directly, without the website;
3. starting the local Neo4j graph database; and
4. opening, querying, and confirming the graph.

Run all commands from the repository root (`LLP Project`) in PowerShell unless stated otherwise.
The checked-in local configuration is for development only: Neo4j authentication is disabled and
all services listen on localhost.

## 2. Service map

| Component | Address | Purpose |
|---|---|---|
| Workbench website | `http://127.0.0.1:8767/` | Upload papers, monitor extraction, review results, and publish/remove papers |
| Workbench-hosted Neo4j Browser | `http://127.0.0.1:8767/neo4j-browser/?connectURL=bolt%3A%2F%2Flocalhost%3A7690&db=neo4j` | Visually explore the graph and run Cypher |
| Neo4j HTTP | `http://127.0.0.1:7477/` | Database HTTP endpoint |
| Neo4j Bolt | `bolt://127.0.0.1:7690` | Neo4j Browser/database connection |
| Ollama API | `http://127.0.0.1:11434/` | Local DeepSeek inference |

## 3. One-time workstation setup

### 3.1 Install system prerequisites

Install:

- Python 3.11 or 3.12 (64-bit);
- Java 21 or 25; and
- Ollama.

Confirm that the Python launcher and Java are available:

```powershell
py -3.12 --version
java -version
```

If `py` is not recognized, install Python from python.org and enable the Python launcher during
installation. Do not use Python 3.13 for this project unless the supported range is updated.

### 3.2 Create the Python environment

Using the environment's Python executable directly avoids PowerShell activation-policy issues:

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
& .\.venv\Scripts\python.exe -m spacy validate
```

Expected result: `en_core_web_sm` is installed and compatible. If it is missing:

```powershell
& .\.venv\Scripts\python.exe -m spacy download en_core_web_sm
```

### 3.3 Download SciBERT

The first pipeline run downloads SciBERT automatically. Pre-download it before an offline run:

```powershell
& .\.venv\Scripts\python.exe -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('allenai/scibert_scivocab_uncased', cache_folder='Pipeline/GENERAL/cache/scibert')"
```

Expected cache location:

```text
Pipeline/GENERAL/cache/scibert/models--allenai--scibert_scivocab_uncased/
```

### 3.4 Download the configured local LLMs

Start the Ollama desktop application, then run:

```powershell
ollama pull qwen3:4b-instruct
ollama pull deepseek-r1:7b
ollama list
```

Both models should appear in the list. The current node configuration uses `qwen3:4b-instruct`
for automated node adjudication, while full relationship extraction requires `deepseek-r1:7b`.
The configuration files are authoritative if these names change. If the desktop application is
not running the API, keep this command open in its own terminal:

```powershell
ollama serve
```

### 3.5 Check the Neo4j configuration after moving the repository

The development instance configuration is:

```text
LocalNeo4j/instances/engineered-water-ontology/conf/neo4j.conf
```

Its `server.directories.*` settings contain absolute Windows paths. If the repository is copied
or moved, update those paths before starting Neo4j. Keep the database, logs, run, import, plugins,
and temporary directories under the same `engineered-water-ontology` instance folder.

## 4. Daily full-stack startup

Use three PowerShell terminals. Leave each foreground process running.

### Terminal 1 — start Neo4j

```powershell
$env:NEO4J_CONF = (Resolve-Path ".\LocalNeo4j\instances\engineered-water-ontology\conf").Path
& ".\LocalNeo4j\runtime\neo4j-community-2026.06.0\bin\neo4j-admin.bat" server console
```

Wait until the terminal reports that Neo4j has started. A normal start exposes HTTP on `7477`
and Bolt on `7690`. Do not start a second copy: a `store_lock` error usually means the configured
database is already running.

### Terminal 2 — confirm or start Ollama

First check the API:

```powershell
Invoke-RestMethod http://127.0.0.1:11434/api/tags
```

If it is unavailable, start the Ollama desktop application or run:

```powershell
ollama serve
```

### Terminal 3 — verify the environment and start the website

```powershell
& .\.venv\Scripts\python.exe .\Pipeline\check_environment.py --mode full
& .\.venv\Scripts\python.exe -m Pipeline.GENERAL.node_only_server
```

Despite the historical `node_only_server` module name, this is the current full workbench and
runs node and relationship extraction. Expected startup output includes:

```text
Engineered Water Systems Paper Graph Workbench: http://127.0.0.1:8767/
LLM context-window relationship extraction is enabled.
```

Open `http://127.0.0.1:8767/` in a browser.

## 5. Run a paper through the website

1. Confirm the page's **Shared graph maintenance** section does not say Neo4j is unavailable.
2. Select or drag in a PDF smaller than 80 MB.
3. Start extraction and keep Terminal 3 open.
4. Wait for every stage to complete. Full-paper DeepSeek processing can be slow on CPU.
5. Review pending nodes, references, and possible duplicate entities where applicable.
6. Inspect the validation and downloaded artifacts.
7. To persist the evaluated paper, click **Save to shared graph**.
8. After the save completes, click **View saved results** or **Open shared graph**.

An evaluated run is not added to Neo4j automatically. Saving is an explicit, one-time action.
Once a run is saved and later removed, that same run cannot be saved again. **Reset shared graph**
deletes every node and relationship and marks all previously saved runs as removed; use it only
when a complete graph reset is intended.

Generated run evidence is stored under:

```text
PipelineAudits/runs/<run_id>/
```

The main completion checks are `manifest.json` and `validation_report.json`. A full run also
contains `canonical_entities_merged.jsonl` and `canonical_relationships.jsonl`.

## 6. Run the pipeline locally without the website

The direct commands below execute the pipeline in the foreground and write the same audit-style
run directory. They do not start an HTTP server and do not automatically publish to Neo4j.

### 6.1 Full nodes-and-relationships run

Requirements: Python environment, SciBERT, the Ollama API, and both configured models
(`qwen3:4b-instruct` for node adjudication and `deepseek-r1:7b` for relationships). Neo4j is not
required merely to produce and evaluate the files.

Set an input PDF and unique output directory, then run the public full-pipeline entry point:

```powershell
$pdf = (Resolve-Path ".\SamplePapers\evaluation\refinement\01_waterrag.pdf").Path
$runDir = Join-Path ".\PipelineAudits\runs" ("manual_full_" + (Get-Date -Format "yyyyMMddTHHmmss"))
& .\.venv\Scripts\python.exe -c "import sys; from pathlib import Path; from Pipeline import run_extraction; summary=run_extraction(Path(sys.argv[1]), Path(sys.argv[2]), lambda stage,message,percent,*rest: print(f'[{percent:3}%] {stage}: {message}', flush=True)); print(summary['counts'])" "$pdf" "$runDir"
```

Replace the value of `$pdf` with any local PDF. Success criteria:

- the command reaches `100% complete`;
- `$runDir\manifest.json` exists;
- `$runDir\validation_report.json` reports `"status": "PASS"`; and
- `$runDir\canonical_relationships.jsonl` exists.

### 6.2 Node-only run

Use this when relationships are intentionally out of scope. Ollama is optional in this mode;
without it, untrusted node candidates remain in the review artifacts. Neo4j is not required.

```powershell
$pdf = (Resolve-Path ".\SamplePapers\evaluation\refinement\01_waterrag.pdf").Path
$runDir = Join-Path ".\PipelineAudits\runs" ("manual_nodes_" + (Get-Date -Format "yyyyMMddTHHmmss"))
& .\.venv\Scripts\python.exe -c "import sys; from pathlib import Path; from Pipeline import run_node_only_extraction; summary=run_node_only_extraction(Path(sys.argv[1]), Path(sys.argv[2]), lambda stage,message,percent: print(f'[{percent:3}%] {stage}: {message}', flush=True)); print(summary['counts'])" "$pdf" "$runDir"
```

A node-only run should have `"relationship_extraction": "disabled_by_node_only_boundary"` in
`manifest.json` and will not create semantic relationships.

### 6.3 Review or publish a command-line run later

The website loads completed runs from `PipelineAudits/runs/` when it starts. To use the normal
review and one-time publication controls for a command-line run, start or restart the website,
open that run, review it, and click **Save to shared graph**. This keeps the manifest and Neo4j
publication receipts consistent.

## 7. Start and verify the graph only

Start Neo4j using the Terminal 1 command in section 4. Then verify both ports:

```powershell
Test-NetConnection 127.0.0.1 -Port 7477 -InformationLevel Quiet
Test-NetConnection 127.0.0.1 -Port 7690 -InformationLevel Quiet
```

Both commands should return `True`. The website provides a more useful graph-health check once it
is running:

```powershell
Invoke-RestMethod http://127.0.0.1:8767/api/graph/summary
```

This reports canonical entity, relationship, evidence, duplicate, and latest-run counts.

## 8. Open and view the graph

1. Keep both Neo4j and the workbench website running.
2. Open the [local Neo4j Browser](http://127.0.0.1:8767/neo4j-browser/?connectURL=bolt%3A%2F%2Flocalhost%3A7690&db=neo4j).
3. Use `bolt://localhost:7690` and database `neo4j` if a connection form appears. Authentication
   is disabled in this localhost-only development configuration.
4. Paste a Cypher query into the command bar and press `Ctrl+Enter`.

Show a sample of the complete paper graph:

```cypher
MATCH p=(source)-[relationship]-(target)
WHERE source.nodeKind = 'ontology_entity'
  AND target.nodeKind = 'ontology_entity'
RETURN p
LIMIT 150
```

Show node and relationship totals:

```cypher
MATCH (n)
WITH count(n) AS nodes
MATCH ()-[r]->()
RETURN nodes, count(r) AS relationships
```

Show ontology-type counts for saved paper entities:

```cypher
MATCH (entity)
WHERE entity.nodeKind = 'ontology_entity'
RETURN entity.ontologyClass AS ontologyClass, count(*) AS entities
ORDER BY entities DESC
```

For a specific paper, the workbench generates a document-scoped query under **Manual database
query**. Copy that query, open Neo4j Browser, paste it, and press `Ctrl+Enter`.

If the Browser displays no nodes, first confirm that a completed paper was explicitly saved to
the shared graph. Pipeline evaluation by itself leaves Neo4j unchanged.

## 9. Normal shutdown

1. Stop the workbench in Terminal 3 with `Ctrl+C`.
2. Stop `ollama serve` with `Ctrl+C` if it was started manually. The desktop Ollama application
   may remain running for later use.
3. Stop Neo4j in Terminal 1 with `Ctrl+C` and wait for shutdown to complete.

Do not terminate Java while Neo4j is writing. A controlled `Ctrl+C` shutdown reduces the risk of
database recovery or lock issues on the next start.

## 10. Troubleshooting checklist

| Symptom | Check and corrective action |
|---|---|
| `.venv\Scripts\python.exe` is missing | Repeat section 3.2 with Python 3.11 or 3.12. |
| `check_environment.py` reports SciBERT missing | Run the explicit download command in section 3.3 while online. |
| Ollama API/model failure | Start Ollama, run `ollama list`, and pull `qwen3:4b-instruct` and `deepseek-r1:7b` if absent. |
| `store_lock` when starting Neo4j | Test ports `7477` and `7690`; another copy is normally already running. Do not delete the lock file while a Java process owns the database. |
| Neo4j fails after the repository was moved | Update the absolute `server.directories.*` paths in `neo4j.conf`. |
| Website cannot bind to port `8767` | Test the port; an existing workbench process may already be running. Stop that process cleanly before restarting. |
| Website says Neo4j is unavailable | Start Neo4j and confirm both configured ports. Refresh the page afterward. |
| Graph is empty after a successful extraction | The paper has only been evaluated. Open its results and click **Save to shared graph**. |
| Full direct run stops during relationship extraction | Confirm Ollama is reachable and the relationship model configured in `relationship_extraction.json` (`deepseek-r1:7b` currently) is installed. |
| Browser assets are unavailable | Confirm the bundled runtime contains `LocalNeo4j/runtime/neo4j-community-2026.06.0/web/neo4j-browser-*.zip` and keep the workbench running. |

## 11. POSIX command substitutions

The project is currently configured for its Windows location. On Linux/macOS, after updating
`neo4j.conf` paths, activate the environment with `. .venv/bin/activate`, replace
`.venv\Scripts\python.exe` with `python`, and start Neo4j with:

```bash
export NEO4J_CONF="$PWD/LocalNeo4j/instances/engineered-water-ontology/conf"
"$PWD/LocalNeo4j/runtime/neo4j-community-2026.06.0/bin/neo4j-admin" server console
```
