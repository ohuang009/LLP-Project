# Engineered Water Systems evaluation corpus

This folder contains 15 publicly findable papers in which AI or machine learning is used for an engineered-water task. The split is deliberately staged:

- `refinement/`: eight papers used to diagnose and improve the pipeline;
- `testing/`: five papers reserved for the post-change generalization check;
- `heldout/`: two papers for the user's final run. The pipeline does not run these during development.
- `gold/`: source-grounded sentence, node, relationship, and ontology-gap annotations used to score refinement runs.

`manifest.json` records titles, authors, water and AI roles, landing pages, direct PDF URLs, and SHA-256 checksums. A paper is considered accessible when a public landing page or direct full-text URL is supplied. Publisher availability can change; the local copies and checksums make this evaluation set reproducible.

Important: the corpus is an evaluation asset, not redistribution guidance. Follow each publisher's license and terms when sharing the PDFs outside this workspace.

## Reproducibility checks

From the project root, run:

```bash
python Pipeline/GENERAL/tools/verify_corpus.py
```

The verifier checks split counts, file existence, PDF signatures, page readability, and SHA-256 values. It does not contact publisher sites.

Validate the human annotations independently with:

```bash
python SamplePapers/evaluation/gold/validate_gold.py
```
