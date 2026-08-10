# Standard operating procedure: install and run the paper-to-graph pipeline

## 1. Purpose

This SOP covers a clean Windows installation, acquisition of every required local model and database asset, environment verification, routine paper processing, graph publication, and synchronization of the persistent node/type lexicon through GitHub.

The maintained stack is:

- Python 3.11 or 3.12 for the pipeline and local website.
- spaCy `en_core_web_sm` for grammatical parsing.
- SciBERT `allenai/scibert_scivocab_uncased` for advisory ontology typing.
- Ollama with `qwen3:4b-instruct` for local structured LLM extraction.
- Neo4j Community Edition with Java 21 for the shared graph.

All commands below assume PowerShell is open at the repository root.

## 2. Required assets

Install or obtain the following before processing a paper:

1. Git and access to the project repository.
2. Python 3.11 or 3.12 from the official Python distribution.
3. [Ollama for Windows](https://docs.ollama.com/windows). Ollama serves its local API at `http://127.0.0.1:11434` by default.
4. The `qwen3:4b-instruct` Ollama model.
5. A Java 21 JDK.
6. The Windows ZIP distribution of [Neo4j Community Edition](https://neo4j.com/docs/operations-manual/current/installation/windows/).
7. Internet access during initial Python package, Qwen, and SciBERT downloads. Routine extraction is local after these assets are cached.
8. One or more text-based scientific PDFs. Scanned image-only PDFs require OCR before this pipeline can extract their text.

The repository tracks configuration, ontology files, source code, and `Pipeline/GENERAL/lexicon/nodes.json`. It does not track the virtual environment, model caches, Ollama models, Neo4j installation, Neo4j data, uploaded PDFs, or generated run artifacts.

## 3. Clone and install Python dependencies

```powershell
git clone https://github.com/ohuang009/LLP-Project.git '.\LLP Project'
Set-Location '.\LLP Project'
python -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Confirm `python --version` reports Python 3.11 or 3.12 before creating the environment. If the installer exposes only the Windows launcher, use `py -3.12 -m venv .venv` instead. Activate `.venv` in every new PowerShell session before running pipeline commands.

## 4. Install and verify Ollama/Qwen

Install Ollama from its official Windows installer, then download the configured model. The model name must match `Pipeline/GENERAL/config/pipeline.json` unless `--model` is supplied for a run.

```powershell
ollama pull qwen3:4b-instruct
ollama list
Invoke-RestMethod http://127.0.0.1:11434/api/tags
```

The Ollama Windows application normally runs in the background after launch. If the API is not responding, launch Ollama from the Start menu or run `ollama serve` in a separate terminal. The official [model-pull API documentation](https://docs.ollama.com/api/pull) describes the same download operation programmatically.

## 5. Download and verify the SciBERT asset

The production configuration uses `scibert_local_files_only: true`, so the model must be downloaded once into the repository-local ignored cache before the pipeline starts.

```powershell
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('allenai/scibert_scivocab_uncased', cache_folder=r'Pipeline/GENERAL/cache/scibert'); print('SciBERT downloaded')"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('allenai/scibert_scivocab_uncased', cache_folder=r'Pipeline/GENERAL/cache/scibert', local_files_only=True); print('SciBERT local cache OK')"
```

Do not commit `Pipeline/GENERAL/cache/`; each workstation keeps its own model cache.

## 6. Install and configure Neo4j

Follow Neo4j's official [Windows ZIP installation procedure](https://neo4j.com/docs/operations-manual/current/installation/windows/): install Java 21, extract Neo4j Community Edition outside the repository, and set `NEO4J_HOME` to the extracted directory.

For each PowerShell session:

```powershell
$env:NEO4J_HOME = 'C:\Tools\neo4j-community-<VERSION>'
$env:NEO4J_CONF = (Resolve-Path '.\LocalNeo4j\config').Path
java -version
& "$env:NEO4J_HOME\bin\neo4j-admin.bat" server validate-config
```

Before first startup on a new clone, open `LocalNeo4j/config/neo4j.conf` and change every absolute `server.directories.*` path and the `java.io.tmpdir` path to the current repository location. Neo4j documents its directory settings in [File locations](https://neo4j.com/docs/operations-manual/current/configuration/file-locations/).

Start Neo4j in a dedicated terminal and leave it running:

```powershell
& "$env:NEO4J_HOME\bin\neo4j.bat" console
```

This repository's tracked configuration exposes only localhost:

- Browser/HTTP: `http://127.0.0.1:7477/`
- Bolt: `bolt://127.0.0.1:7690`

Authentication is disabled in this local development configuration. Do not change the listen address to a network-accessible interface without enabling authentication and reviewing Neo4j security settings.

## 7. Return to the project after closing PowerShell

Closing PowerShell does not remove the installed dependencies, downloaded models, virtual environment, lexicon, or Neo4j data. It only clears the current directory, virtual-environment activation, and session-only environment variables.

In every new PowerShell window that you will use for pipeline or website commands, run:

```powershell
Set-Location 'C:\Users\ohuan\Downloads\LLP Project'
Set-ExecutionPolicy -Scope Process Bypass
& .\.venv\Scripts\Activate.ps1
python --version
python -m Pipeline --help
```

When activation succeeds, the prompt normally begins with `(.venv)`. That window is now ready to run the website or any `python -m Pipeline ...` command. Do not run `pip install` again during routine startup.

Ollama usually continues running in the Windows background. Check it with:

```powershell
Invoke-RestMethod http://127.0.0.1:11434/api/tags
```

If that request fails, launch Ollama from the Windows Start menu. Alternatively, run `ollama serve` in another PowerShell window and leave it open.

Neo4j is needed only when using graph features such as publishing, viewing graph counts, removing a paper, or resetting the graph. When needed, open a separate PowerShell window and run:

```powershell
$env:NEO4J_HOME = 'C:\Tools\neo4j-community-<VERSION>'
$env:NEO4J_CONF = (Resolve-Path 'C:\Users\ohuan\Downloads\LLP Project\LocalNeo4j\config').Path

& "$env:NEO4J_HOME\bin\neo4j.bat" console
```

Replace the `NEO4J_HOME` example with the directory where Neo4j was actually extracted. The Neo4j console occupies that window until it is stopped with `Ctrl+C`; run website or pipeline commands in the first window.

To run the complete environment check in the first window, set the same Neo4j variables there without running `neo4j console`:

```powershell
$env:NEO4J_HOME = 'C:\Tools\neo4j-community-<VERSION>'
$env:NEO4J_CONF = (Resolve-Path '.\LocalNeo4j\config').Path
python -m Pipeline.check_environment
```

If the repository is moved or cloned elsewhere, replace `C:\Users\ohuan\Downloads\LLP Project` in these commands and update the absolute paths in `LocalNeo4j/config/neo4j.conf` as described in section 6. You do not need to reinstall packages or redownload models every time PowerShell is reopened.

## 8. Verify the complete environment

With Ollama and Neo4j running and the virtual environment activated:

```powershell
python -m Pipeline.check_environment
```

Do not start a production run until every line reports `PASS`. This check loads spaCy and SciBERT, validates the persistent lexicon, confirms the configured Qwen model through the Ollama API, and verifies Java and `NEO4J_HOME`.

Run the automated tests after initial setup and after code changes:

```powershell
python -m unittest discover -s .\Pipeline\GENERAL\tests -p 'test_*.py'
python -m unittest discover -s .\Pipeline\Paper_Parsing\tests -p 'test_*.py'
```

## 9. Choose one of two operating paths

Both paths use the same pipeline and persistent lexicon. Choose the website when you want the complete guided extraction, review, and graph workflow. Choose command lines when you want to run individual stages or automate runs.

### Path 1: open the website and run the pipeline there

1. Prepare the command window using section 7.
2. Confirm Ollama is running.
3. Start Neo4j in its separate window if you plan to publish or inspect the graph.
4. Start the website:

```powershell
python -m Pipeline --stage server
```

5. Leave that PowerShell window open and open `http://127.0.0.1:8767/` in a browser.
6. Select or upload a PDF, select the installed Ollama model, and start extraction.
7. Review queued nodes and inferred relationship endpoints in the website.
8. Use **Save to shared graph** only after review. Neo4j must be running for this step.

The website preloads spaCy and SciBERT before accepting work and displays the number of prior node/type pairs in the footer. Stop the website with `Ctrl+C`.

### Path 2: run pipeline stages from command lines

Prepare the command window using section 7, then run whichever stage is required. Use a unique output directory for every new paper.

| Stage | Purpose | Additional service required |
|---|---|---|
| `parse` | Extract PDF structure and source text only | None |
| `nodes` | Parse and extract typed nodes | Ollama |
| `relationships` | Add relationships to an existing node run | Ollama |
| `full` | Parse, extract nodes, and extract relationships | Ollama |
| `publish` | Publish a completed run to the graph | Neo4j |
| `graph-summary`, `remove-paper`, `reset-graph` | Maintain the graph | Neo4j |

#### Full command-line extraction

```powershell
python -m Pipeline --stage full --input '.\papers\paper.pdf' --output '.\PipelineAudits\runs\paper-001'
```

This parses the PDF, extracts and types nodes, updates the persistent lexicon after node extraction succeeds, extracts relationships, and writes auditable artifacts. It does not publish to Neo4j.

#### Node-only extraction

```powershell
python -m Pipeline --stage nodes --input '.\papers\paper.pdf' --output '.\PipelineAudits\runs\paper-001-nodes'
```

The node-only stage also updates the lexicon after successful completion.

#### Other supported stages

```powershell
# Parse without node extraction; this does not update the lexicon.
python -m Pipeline --stage parse --input '.\papers\paper.pdf' --output '.\PipelineAudits\runs\paper-001-parse'

# Add relationships to an existing node run; this does not update the lexicon.
python -m Pipeline --stage relationships --output '.\PipelineAudits\runs\paper-001-nodes'

# Inspect graph counts.
python -m Pipeline --stage graph-summary --json
```

Command-line extraction writes the same run artifacts used by the website. If a command-line run produces review candidates, start the website afterward to perform human review before publishing.

## 10. Review and publish

Nodes marked `needs_review` do not enter accepted mentions until a reviewer accepts them. A human acceptance through the website immediately adds that canonical node/type pair to the lexicon. Rejected items are never added.

Publishing a completed run to Neo4j is always a separate action:

```powershell
python -m Pipeline --stage publish --output '.\PipelineAudits\runs\paper-001'
```

Use the website for the preferred review-and-publish workflow. Do not publish a run until its node and relationship review queues have been checked.

Graph removal commands are destructive. Record the document and run identifiers before removal:

```powershell
python -m Pipeline --stage remove-paper --document-id '<DOCUMENT-ID>' --run-id '<RUN-ID>'
python -m Pipeline --stage reset-graph --confirm RESET
```

`reset-graph` removes the entire local graph and should be used only when that is the intended outcome.

## 11. Maintain the persistent lexicon in GitHub

The lexicon is `Pipeline/GENERAL/lexicon/nodes.json`. It stores only previously accepted pairs in this form:

```json
[
  {
    "name": "EPANET",
    "type": "Tool"
  }
]
```

Its rules are:

- Exact normalized node names retrieve prior types for the next paper.
- Prior types are advisory; current paper evidence can produce a different type.
- Different valid types for the same name are preserved as separate pairs.
- No alias, definition, source, timestamp, paper identifier, confidence, or count is stored.
- A node pass that fails before producing accepted entities does not update the file; later relationship-stage failure does not erase node progress already recorded for that paper.

After each reviewed paper, inspect, commit, and push the lexicon so it persists across machines and uses:

```powershell
git diff -- Pipeline/GENERAL/lexicon/nodes.json
git add Pipeline/GENERAL/lexicon/nodes.json
git commit -m 'data: update persistent node type lexicon'
git push
```

The pipeline saves the file locally; GitHub persistence requires this normal commit-and-push step. Before starting a new batch on another workstation, pull the latest repository changes.

Avoid processing papers concurrently in separate clones without coordinating lexicon updates. If Git reports a conflict, retain the union of unique `{name, type}` pairs, keep the file as a valid JSON array, then rerun the lexicon and pipeline tests before committing.

## 12. Routine shutdown and backup

1. Wait for the current paper run to finish.
2. Review and commit the changed lexicon.
3. Stop the website with `Ctrl+C`.
4. Stop the Neo4j console with `Ctrl+C` and allow shutdown to complete.
5. Exit Ollama from the system tray when local model service is no longer needed.
6. Back up `LocalNeo4j/storage/` separately if the local graph must be recoverable. It is intentionally ignored by Git.

## 13. Troubleshooting

- **Ollama unavailable or model missing:** start Ollama, run `ollama pull qwen3:4b-instruct`, and confirm `/api/tags` lists the exact configured name.
- **SciBERT cannot be loaded offline:** repeat both commands in section 5 while internet access is available and confirm the second local-only command succeeds.
- **spaCy model missing:** reactivate `.venv` and rerun `pip install -r requirements.txt`.
- **Neo4j fails validation:** confirm Java 21, `NEO4J_HOME`, `NEO4J_CONF`, and every absolute path in `LocalNeo4j/config/neo4j.conf`.
- **Port already in use:** stop the process using 7477 or 7690, or update both the tracked Neo4j configuration and the pipeline's Neo4j connection settings together.
- **Lexicon validation fails:** ensure the root is a JSON array and every row contains exactly `name` and a valid ontology `type`.
