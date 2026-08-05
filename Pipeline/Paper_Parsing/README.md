# Paper parsing

The consolidated layout-aware PDF parser used by the production pipeline. It preserves sections,
paragraphs, sentences, pages, author metadata, and stable provenance IDs while excluding tables,
captions, references, and repeated page furniture from narrative extraction.

- Core implementation: `pipeline.py`, `pdf_parser.py`, `profiles.py`, `models.py`, and `text_utils.py`.
- Public stage boundary: `parser.py`.
- Parser regression tests: `tests/`.
- Corpus and layout diagnostics: `tools/` (generated audits default to `../../PipelineAudits/parser/`).

Install the complete environment from the repository-level `requirements.txt`. This folder's
requirements file redirects to that canonical specification; setup instructions are in `../README.md`.
