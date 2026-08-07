# Engineered Water Systems knowledge graph

This repository contains one production paper-to-knowledge-graph pipeline, its source papers,
generated audit boundary, and local Neo4j runtime.

## Start here

The complete dependency, local-model, configuration, and operating guide is
[`Pipeline/README.md`](Pipeline/README.md).

For a Windows-first, step-by-step operating runbook covering the website, direct pipeline runs,
Neo4j startup, and graph viewing, use
[`STANDARD_OPERATING_PROCEDURE.md`](STANDARD_OPERATING_PROCEDURE.md).

Quick sequence after following that setup guide and starting the configured Neo4j instance:

```bash
python Pipeline/check_environment.py --mode full
python -m Pipeline.GENERAL.node_only_server
```

Open `http://127.0.0.1:8767/`.

## Folder map

| Folder | Purpose |
|---|---|
| `Pipeline/` | Canonical implementation, runtime assets, requirements, tests, and setup guide |
| `SamplePapers/` | Workbench, evaluation, and regression source PDFs |
| `PipelineAudits/` | Retained generated runs and review evidence |
| `LocalNeo4j/` | Local Neo4j distribution and paper-graph instance |

Do not create new root-level output, temporary, or report folders. Pipeline output belongs under
`PipelineAudits/`.
