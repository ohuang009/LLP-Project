# Edge Extraction

| Order | File | Responsibility |
|---:|---|---|
| 1 | `common.py` | Relationship configuration, patterns, and sentence utilities |
| 2 | `observation_mentions.py` | Provenance-local quantitative observation assembly |
| 3 | `extractor.py` | Candidate edge generation and mandatory LLM relationship adjudication |
| 4 | `materialization.py` | Deduplication into canonical entity-to-entity relationships |

The reusable graph utilities live in `Pipeline/Knowledge_Graph_Core`; this folder owns the
paper relationship workflow and its artifacts.

Full edge extraction requires the configured local Ollama/DeepSeek relationship judge. See
`../README.md` for installation, model download, and failure behavior.
