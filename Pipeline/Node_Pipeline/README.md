# Node pipeline

Each file owns one explicit node-extraction step:

| Order | File | Responsibility |
|---:|---|---|
| 1 | `common.py` | Paths, shared constants, stable IDs, normalization, and JSON/JSONL I/O |
| 2 | `span_detection.py` | Exact-span rules and the `Span` contract |
| 3 | `lexicon.py` | Persistent lexicon matching and prior-node lookup by node type |
| 4 | `candidate_generation.py` | Metadata and grammatical subject/object candidate generation |
| 5 | `ner_boundary.py` | Required NER artifact construction and exact-span validation |
| 6 | `adjudication.py` | DeepSeek accept/review/reject decisions and grounding checks |
| 7 | `reference_resolution.py` | Contextual resolution of vague references to specific accepted nodes |
| 8 | `canonicalization.py` | Canonical entities, similarity candidates, and reviewed merges |
| 9 | `review_workflows.py` | Human node, reference, and duplicate-resolution decisions |
| 10 | `orchestrator.py` | Full node-and-edge workflow coordination |
| — | `node_only_runner.py` | Stable node-only execution boundary |
| — | `contracts.py` | Candidate and judgment artifact schemas |

The public API is re-exported from `__init__.py`. New code should import from the focused
module that owns the behavior or from `Pipeline.Node_Pipeline`.

Runtime dependencies, local models, optional services, and setup checks are documented in
`../README.md`.

