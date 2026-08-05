# Paper-to-knowledge-graph pipeline

This is the canonical installation and operations guide for the complete pipeline. Start here
when setting up a new machine. The code is organized into importable stage packages; generated
runs and review evidence are written to `PipelineAudits/runs/`.

## What is required

The supported local setup uses Python 3.11 or 3.12 and Java 21 or 25. The commands below assume
a POSIX-compatible shell.

| Component | Required? | Used for | Installation source |
|---|---|---|---|
| Python 3.11 or 3.12 | Always | Every stage | [python.org](https://www.python.org/downloads/) |
| Packages in `requirements.txt` | Always | Parsing, grammar, SciBERT, and verification | Repository root |
| `en_core_web_sm` | Always | POS tags, dependency parsing, and SVO extraction | Installed by `requirements.txt` |
| `allenai/scibert_scivocab_uncased` | Always | Ontology typing of subject/object candidates | [Hugging Face model](https://huggingface.co/allenai/scibert_scivocab_uncased) |
| Ollama | Full mode; optional in node-only mode | Local LLM service | [Ollama downloads](https://ollama.com/download) |
| `deepseek-r1:7b` | Full mode; optional in node-only mode | Node, reference, and edge judgments | [Ollama model page](https://registry.ollama.com/library/deepseek-r1) |
| Java 21 or 25 | Always | Bundled Neo4j 2026 runtime | Oracle JDK or Zulu JDK |
| Local Neo4j | Always in current orchestrators | Canonical graph persistence | Bundled under `LocalNeo4j/runtime/` |
| `all-MiniLM-L6-v2` | Optional | Embedding-based entity-resolution experiments | [Hugging Face model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| Human review | Conditional | Review candidates and fuzzy duplicate decisions | Workbench review queue |

Neo4j 2026 supports Java 21 and 25; see the official
[Neo4j requirements](https://neo4j.com/docs/operations-manual/current/installation/requirements/).

## Install the Python environment

From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m spacy validate
```

The canonical dependency file is `requirements.txt` at the repository root.
`Pipeline/requirements.txt` redirects to it. The complete environment contains:

| Package | Why it is present |
|---|---|
| `pdfplumber` | Layout-aware PDF text extraction |
| `numpy` | SciBERT similarity matrices and entity-resolution calculations |
| `spacy` and `en-core-web-sm` | Complete token/POS/dependency analysis and SVO extraction |
| `sentence-transformers` | Local SciBERT and optional MiniLM encoders |
| `torch`, `transformers`, `huggingface-hub` | Explicit local model runtime and downloads |
| `pypdf` | Optional corpus-integrity and PDF verification utilities |

No `ollama`, `neo4j`, `openai`, or `requests` Python package is required. Ollama and Neo4j
are called through their local HTTP APIs using Python's standard library.

If the direct spaCy model dependency cannot be installed, use:

```bash
python -m spacy download en_core_web_sm
```

## Download the local encoder models

### SciBERT — mandatory

SciBERT is pretrained on scientific papers and types every grammatical subject/object candidate
against the ontology. It is not yet the planned environment-specific fine-tune.

The first pipeline run downloads it automatically. To download it explicitly before an offline
run and place it in the configured cache:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('allenai/scibert_scivocab_uncased', cache_folder='Pipeline/GENERAL/cache/scibert')"
```

Expected cache:

```text
Pipeline/GENERAL/cache/scibert/models--allenai--scibert_scivocab_uncased/
```

After downloading all Hugging Face models, offline mode can be enforced with:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Hugging Face documents its version-aware cache in the
[Hub download guide](https://huggingface.co/docs/huggingface_hub/guides/download).

### MiniLM — optional

The shared entity-resolution library can use `sentence-transformers/all-MiniLM-L6-v2` for
embedding similarity. The current production canonicalization path does not require it.

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

## Install Ollama and DeepSeek

Ollama is an external local runtime, not a pip dependency. Install it from the
[official download page](https://ollama.com/download). Its API runs at `http://127.0.0.1:11434`.

Download the exact production model configured in `Pipeline/GENERAL/config/node_extraction.json`:

```bash
ollama pull deepseek-r1:7b
ollama list
```

Verify the API:

```bash
curl --fail --silent http://127.0.0.1:11434/api/tags
```

The 7B model download is approximately 4.7 GB. The node judge can expand from a 32,768-token
context to 65,536 tokens for candidate-heavy sentences; large contexts increase RAM use and
latency. A GPU is optional, but CPU-only full-paper adjudication is substantially slower.

Production uses DeepSeek for node cleanup and accept/review/reject decisions, contextual
reference resolution, and relationship judgment. Node-only mode can finish without Ollama:
untrusted candidates remain in review and unresolved references are audited. Full mode requires
the relationship LLM, so Ollama and `deepseek-r1:7b` are mandatory. The standalone shared-core
config mentions Llama 3.1, but production configuration overrides it; do not download Llama 3.1
unless intentionally running the shared core by itself.

## Start Neo4j

Install Java 21 or 25 and configure a Neo4j Community instance to use the checked-in settings
under `LocalNeo4j/`. The local runtime and database state are intentionally excluded from Git.
With `NEO4J_HOME` pointing to an extracted Neo4j distribution and `NEO4J_CONF` pointing to the
instance configuration, start the server in the foreground:

```bash
java -version
export NEO4J_CONF="$PWD/LocalNeo4j/instances/engineered-water-ontology/conf"
"$NEO4J_HOME/bin/neo4j-admin" server console
```

Endpoints:

- HTTP transaction API: `http://127.0.0.1:7477/`
- Bolt: `bolt://127.0.0.1:7690`
- Workbench-hosted browser: `http://127.0.0.1:8767/neo4j-browser/`

Stop it with `Ctrl-C` in the server terminal.

The checked-in development instance disables authentication and listens only on localhost.
Do not expose these ports or reuse this configuration for production.

## Check the setup

Run the read-only checker after installing packages and downloading models:

```bash
python Pipeline/check_environment.py --mode full
```

For node-only mode, where Ollama is optional:

```bash
python Pipeline/check_environment.py --mode node-only
```

It verifies Python, declared packages, the spaCy model, the SciBERT cache, Ollama and DeepSeek,
Java, the bundled Neo4j runtime, and the live Neo4j endpoint.

## Run the pipeline

### Full workbench: nodes and edges

Start Neo4j and Ollama first, then:

```bash
python -m Pipeline.GENERAL.node_only_server
```

Open `http://127.0.0.1:8767/`, select a PDF, and start extraction. The workbench executes the
full node-and-relationship pipeline.

### Programmatic node-only run

This mode parses and creates nodes but deliberately does not extract relationships:

```python
from pathlib import Path
from Pipeline import run_node_only_extraction

summary = run_node_only_extraction(
    Path("SamplePapers/your-paper.pdf"),
    Path("PipelineAudits/runs/my-node-only-run"),
    lambda stage, message, percent: print(percent, stage, message),
)
print(summary["counts"])
```

Neo4j remains mandatory because the current node-only orchestrator persists final canonical
nodes. For a section-limited development run, pass `include_section_titles={"Abstract"}`.

## Stage requirements and failure behavior

| Stage | Required? | Configuration | If unavailable or disabled |
|---|---|---|---|
| Narrative PDF parsing | Required | `GENERAL/config/project.json` | Run stops |
| spaCy POS/dependency parsing | Required | `grammatical_candidate_generation.parser_model` | Run stops |
| Subject/predicate/object extraction | Required | `grammatical_candidate_generation.required` | Run stops |
| SciBERT ontology typing | Required | `grammatical_candidate_generation.scibert_model` | Run stops |
| Structural metadata extraction | Required | Built in | Run stops if parsing is invalid |
| Persistent lexicon | Required identity input | `GENERAL/lexicon/lexicon.json` | Known aliases cannot map directly |
| DeepSeek node judgment | Optional in node-only; needed for automated acceptance | `llm_node_judge` | Untrusted candidates remain review/audit |
| DeepSeek reference resolution | Optional | `llm_reference_judge` | Deterministic matches remain; ambiguous references go to review/ignore audit |
| DeepSeek relationship judgment | Required in full; absent in node-only | `relationship_extraction.json` | Full semantic edges cannot be produced |
| Exact canonicalization | Required | Built in | Run stops |
| Fuzzy duplicate review | Optional for completion | `similarity` | Possible duplicates stay separate |
| Human node review | Conditional | Workbench queue | Review nodes are not added until accepted |
| Question evaluation | Full mode | Built in | Node-only mode omits it |
| Semantic/generalization validation | Full mode | Built in | Node-only uses a smaller validation report |
| Neo4j upsert | Required in current orchestrators | `Graph_Persistence/neo4j_writer.py` | Run stops before completion |

## Configuration files

| File | Controls |
|---|---|
| `Pipeline/GENERAL/config/node_extraction.json` | Required grammar/NER, SciBERT, node/reference LLMs, context, and similarity |
| `Pipeline/GENERAL/config/relationship_extraction.json` | Relationship LLM, endpoint policy, and thresholds |
| `Pipeline/GENERAL/config/node_validity_rubric.json` | Accept/review/reject rubric |
| `Pipeline/GENERAL/config/project.json` | Parser exclusions and full-pipeline behavior |
| `Pipeline/GENERAL/ontology/ontology.json` | Runtime ontology and valid node/predicate types |
| `Pipeline/GENERAL/lexicon/lexicon.json` | Persistent cross-paper identity aliases |

Configuration is authoritative: changing a model name also changes what must be pulled or
downloaded locally.

## Execution order and outputs

1. `Paper_Parsing` creates narrative sections, paragraphs, sentences, pages, and stable IDs.
2. `Grammatical_Parsing` records every token and generates predicate-first SVO triples.
3. `Node_Pipeline` creates exact-span candidates, types and adjudicates them, resolves references,
   and canonicalizes accepted mentions.
4. `Edge_Extraction` builds and judges relationship candidates between accepted node mentions.
5. `Question_Evaluation` and `Validation` calculate answerability and completeness.
6. `Graph_Persistence` validates and upserts canonical nodes, evidence, and relationships.

Every mention and relationship must retain sentence evidence. Major artifacts include:

- `parsed.json`
- `grammatical_analysis.jsonl`, `grammatical_triples.jsonl`, `scibert_typings.jsonl`
- `node_candidates.jsonl`, `candidate_judgments.jsonl`, `mentions.jsonl`
- `node_review_candidates.jsonl`, `node_rejections.jsonl`
- `canonical_entities.jsonl`, `canonical_entities_merged.jsonl`
- `relationship_candidates.jsonl`, `assertions.jsonl`, `canonical_relationships.jsonl`
- `validation_report.json`, `neo4j_upsert.json`, `manifest.json`

## Tests

From the repository root:

```bash
python -m unittest discover -s Pipeline/GENERAL/tests -p "test_*.py"
python -m unittest discover -s Pipeline/Knowledge_Graph_Core/tests -p "test_*.py"
python -m unittest discover -s Pipeline/Paper_Parsing/tests -p "test_*.py"
```

Use Hugging Face offline variables during tests only after the required model is cached.

## Folder map

| Folder | Responsibility |
|---|---|
| `Paper_Parsing` | Narrative PDF parsing and publisher/layout profiles |
| `Grammatical_Parsing` | Token/POS/dependency analysis, SVO extraction, and SciBERT typing |
| `Node_Pipeline` | Node candidates, LLM decisions, reference resolution, and canonicalization |
| `Edge_Extraction` | Observation assembly, edge extraction, and relationship materialization |
| `Knowledge_Graph_Core` | Reusable ontology, LLM judge, extraction, and resolution components |
| `Sector_Discovery` | Generalized discovery of domain entities and local system names |
| `Question_Evaluation` | Graph answerability evaluation |
| `Validation` | Semantic completeness, LLM usage, and generalization diagnostics |
| `Graph_Persistence` | Neo4j validation and persistence |
| `GENERAL` | Workbench, configuration, ontology, lexicon, tests, tools, and local caches |
