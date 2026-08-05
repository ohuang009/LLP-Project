# Pipeline audits

This is the single location for generated evidence from the current production pipeline.

- `runs/` contains complete immutable pipeline runs. Only the latest authoritative run for a source/configuration should be retained.
- `exports/` contains human-readable or compact exports derived from a retained run.
- `reference/` contains hand-reviewed reference graphs used to evaluate pipeline output.
- `parser/` is the default destination for generated corpus-parsing audits when they are run.

Current authoritative evidence: `runs/waterrag_full_node_pipeline_chunked_20260805T101654Z/`.
Its `validation_report.json` has status `PASS`; its source, manifest, node decisions, review queues,
model-call receipts, grammatical analysis, Neo4j receipt, and validation evidence are colocated.

Temporary retries, smoke tests, parser audits, report renders, and superseded configurations do not belong here.
