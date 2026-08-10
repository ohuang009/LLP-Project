# LLM paper-to-knowledge-graph pipeline

This repository turns scientific PDFs into reviewed Neo4j nodes and relationships. The parser, website, ontology, and Neo4j integration remain; the extraction path is intentionally compact and LLM-focused.

## Pipeline

1. `Paper_Parsing/` extracts sections, paragraphs, sentences, pages, and exact source spans from a PDF.
2. `Grammatical_Parsing/candidates.py` extracts the raw main-clause subject, verb, and object. It does **no cleanup**. Subordinate and non-core fragments are stored separately as sentence `context`.
3. `Node_Pipeline/pipeline.py` batches sentences by paragraph and runs:
   - Qwen pass 1: choose a more specific node name using the previous, target, and next sentence.
   - SciBERT: produce an advisory ontology type or `NONE` for every raw subject/object.
   - Qwen pass 2: assign an ontology type or `NONE`; SciBERT and exact-name types from prior papers are evidence, not authority.
   - Qwen pass 3: judge the node as `accepted`, `needs_review`, or `disregard`.
4. `Edge_Extraction/pipeline.py` processes each paragraph containing at least one accepted node and runs:
   - Qwen generation: request 5-10 paragraph-grounded relationships.
   - Qwen refinement: correct direction, predicate choice, endpoint specificity, duplicates, and evidence without adding facts.
   - Qwen validation: apply the grounding, specificity, accuracy, usefulness, ontology, and provenance rubric.
   - Dynamic token-budget batching lets several independent paragraphs share an Ollama call while
     keeping their paragraph IDs, sentence IDs, and evidence separate.
5. `Graph_Persistence/neo4j_writer.py` publishes only accepted entities and relationships when a user explicitly chooses to add the run.

An inferred relationship endpoint goes into `review_queue.jsonl`. It does not become a node or reach Neo4j until a human accepts it.

## Repository map

```text
Pipeline/
  core.py                         shared paths, IDs, and JSON I/O
  lexicon.py                      minimal persistent node/type memory
  llm.py                          structured local Ollama client
  cli.py                          command-line entry point
  Paper_Parsing/                  PDF parser (unchanged extraction role)
  Grammatical_Parsing/candidates.py
  Node_Pipeline/pipeline.py
  Edge_Extraction/pipeline.py
  ontology.py
  Graph_Persistence/neo4j_writer.py
  GENERAL/
    config/pipeline.json          the only extraction configuration
    lexicon/nodes.json            version-controlled prior node/type pairs
    ontology/                     ontology JSON and maintained workbook
    run.py + static/              existing local website
    tests/                        pipeline, website-state, and Neo4j tests
LocalNeo4j/                       existing local Neo4j setup
SamplePapers/                     two PDFs used by the website's sample buttons
PipelineAudits/                   generated run output; never pipeline state
```

## Persistent node/type lexicon

`Pipeline/GENERAL/lexicon/nodes.json` is the pipeline's only cross-paper extraction memory. Each row has exactly two fields: `name` and `type`. It intentionally stores no aliases, definitions, source text, provenance, timestamps, or occurrence counts.

The pipeline reads exact normalized-name matches as advisory typing evidence. It updates the file after every successful `full` or `nodes` paper run and after a reviewer accepts a queued node. Commit and push changes to this file so that other clones and later GitHub checkouts share the same progress.

## Setup and run

Follow [STANDARD_OPERATING_PROCEDURE.md](STANDARD_OPERATING_PROCEDURE.md) for a clean-machine setup, all model and database assets, verification, routine operation, and lexicon synchronization.

```powershell
& .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m Pipeline --stage full --input .\paper.pdf --output .\PipelineAudits\runs\my-run
```

Start Ollama before running the pipeline. The default model is the locally installed `qwen3:4b-instruct`; it can be changed in `Pipeline/GENERAL/config/pipeline.json` or with `--model`.

Batching is configured in the same file. It reserves context headroom and caps expected text output,
so a large paragraph automatically runs alone instead of overflowing a combined request. Qwen returns
compact tab-separated lines; deterministic Python code validates them and creates every JSON/JSONL artifact.

Start the existing website with:

```powershell
python -m Pipeline --stage server
```

Publishing remains a separate, explicit action:

```powershell
python -m Pipeline --stage publish --output .\PipelineAudits\runs\my-run
```

## Review and provenance rules

- `surface_text`, offsets, sentence ID, paragraph ID, and evidence quote are immutable source truth.
- Qwen may change `specific_name`; it may not rewrite the raw subject/object.
- A SciBERT `NONE` candidate still goes to Qwen ontology typing.
- A node marked `needs_review` is absent from `mentions.jsonl` until a human accepts it.
- Every relationship must have at least one accepted endpoint and paragraph-local evidence sentence IDs.
- Inferred endpoints stay queued; relationships that depend on them stay out of `assertions.jsonl` until review.
- Neo4j publication validates entity types, predicate domain/range, mention evidence, and canonical IDs again.

## Tests

```powershell
python -m unittest discover -s .\Pipeline\GENERAL\tests -p "test_*.py"
python -m unittest discover -s .\Pipeline\Paper_Parsing\tests -p "test_*.py"
```

The focused tests use injected fake Ollama and SciBERT transports, so they verify batching, call order, review gates, relationship counts, and provenance without invoking the local model.
