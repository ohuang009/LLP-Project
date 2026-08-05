# General pipeline assets

This folder contains cross-cutting assets used by the production stages:

- `run.py` and `node_only_server.py` serve the workbench.
- `config/` contains extraction, relationship, sector, and schema settings.
- `ontology/`, `lexicon/`, and `review/` contain the graph contract and reviewed identity state.
- `static/` contains the workbench UI.
- `tests/` contains end-to-end and contract regression tests.
- `tools/` contains current corpus, model, evaluation, export, and rematerialization utilities.
- `cache/` contains reusable local model results.

The stage implementations are sibling folders under `Pipeline/`. Source papers, generated audits,
and the Neo4j runtime remain at repository level because they are data/runtime rather than pipeline code.

Run `python -m Pipeline.GENERAL.node_only_server` from the repository root to launch the
workbench. See `../README.md` for the complete mandatory and optional dependency, model-download,
service, configuration, and verification guide.
