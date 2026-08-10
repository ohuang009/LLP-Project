# Human gold annotations

This directory contains source-grounded annotations for evaluating extraction quality. Gold files are review data, not pipeline input and not reusable lexicon state.

## Record contract

Each JSONL row represents one deliberately selected sentence:

- `required_nodes`: `[exact source span, ontology type, canonical name]`.
- `required_relationships`: `[subject name, subject type, predicate, object name, object type]`.
- `forbidden_relationships`: plausible but unsupported triples that must not be emitted.
- `ontology_gaps`: entailed facts that the current ontology cannot encode without violating a domain/range signature. These are diagnostic and excluded from extraction recall.
- `notes`: adjudication guidance for pronouns, aliases, scope, or parser defects.

The relationship evidence is the row's `sentence_id`. A relationship may use a contextual endpoint named in a preceding sentence, but every row states the canonical endpoint explicitly. Nodes are strict exact-span targets; canonical names may resolve an abbreviation or local referring expression without changing the source span.

## Scoring policy

1. Validate node spans against the pinned `parsed.json` before scoring.
2. Match nodes by ontology type plus normalized canonical name or an explicitly listed exact alias.
3. Match relationships by normalized endpoints, predicate, direction, and cited sentence ID.
4. Score required nodes and required relationships separately. Report micro precision, recall, and F1, plus predicate-level recall.
5. Treat `ontology_gaps` as ontology-design findings, not model false negatives.
6. Count every emitted forbidden relationship as a precision error.
7. Report competency coverage for `PROPOSES`, `HAS_AGENT`, `PERFORMS`, `EVALUATED_ON`, `EVALUATED_BY`, and `REPORTS` independently; aggregate scores can hide total failure of one predicate.

The first set, `EWS-R07_sentence_gold.jsonl`, covers 50 high-information sentences from the refinement split. It intentionally emphasizes contribution, architecture, tasks, evaluation, quantitative results, and limitations rather than sampling sentences uniformly.

See `EWS-R07_findings.md` for the strict baseline, ontology limitations exposed by annotation, and the resulting prompt-policy decisions.

Validate the gold set and score a completed run with:

```powershell
python .\SamplePapers\evaluation\gold\validate_gold.py
python .\SamplePapers\evaluation\gold\score_run.py --run .\PipelineAudits\runs\RUN_NAME
```

`score_run.py` is deliberately strict: node matches require the same sentence, exact span, and type; relationship matches additionally require normalized endpoint names, direction, predicate, and the gold evidence sentence. It reports unresolved endpoints separately so canonicalization misses cannot masquerade as edge-extractor misses.
