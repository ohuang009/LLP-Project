# Knowledge graph core

Reusable ontology, LLM adjudication, relationship extraction, entity-resolution,
review-queue, model, and validation components used by the production stages.

- Python modules in this folder are imported as `Pipeline.Knowledge_Graph_Core`.
- `config/` contains focused library test fixtures and shared ontology snapshots.
- `tests/` contains the core-library unit suite.

This is an internal library, not a second runnable pipeline.

Install the complete environment from the repository-level `requirements.txt`. This folder's
requirements file redirects to that canonical specification; model/runtime instructions are in
`../README.md`.
