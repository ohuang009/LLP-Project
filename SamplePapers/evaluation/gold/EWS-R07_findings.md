# EWS-R07 gold-set findings

## What the gold set measures

The 50 selected sentences contain 229 required exact-span nodes and 131 required, ontology-valid relationships. They emphasize the paper's contribution, component agents, models and tools, performed tasks, evaluation data and metrics, quantitative results, and limitations. Seven tempting but unsupported relationships are explicitly forbidden.

The annotations also record 29 ontology-gap notes. These notes are not model false negatives and are excluded from relationship recall.

## Strict baseline

Scoring `evaluation_07_edge_iter04_20260810` with `score_run.py` yields:

| Measure | Strict result |
|---|---:|
| Exact sentence + span + type node recall | 40 / 229 (17.47%) |
| Gold relationships with resolvable endpoints | 9 / 131 |
| Exact endpoint + predicate + direction + evidence relationship recall | 0 / 131 |

This is an intentionally strict regression baseline, not a semantic-similarity score. It exposes canonicalization and endpoint failures instead of granting partial credit for nearby concepts. The run predates the active DeepSeek refactor and should be replaced by a fresh score after that pipeline completes.

## Highest-impact findings

1. **Publication-dependent facts are unreachable without a structural Publication node.** `PROPOSES` and `REPORTS` require `Publication` as their domain. The active raw-SVO node path currently produces only sentence-derived mentions, so a title-derived Publication mention must be added deterministically or these competency predicates cannot materialize.

2. **`Agent` is too narrow for facts stated about the named expert agents.** The ontology excludes `Agent` from `USES_DATASET`, `USES_METHOD`, `EVALUATED_ON`, `EVALUATED_BY`, `OUTPERFORMS`, and `HAS_LIMITATION`. The paper explicitly states each of these fact patterns about EA know, EA data, or EA hybrid. Prompts must not evade this by inconsistently retyping the same component as `AISystem`; the signatures or modeling policy need a deliberate ontology decision.

3. **Main-clause SVO spans miss nested scientific entities.** Enumerated agents, metric lists, model lists, quantitative result clauses, and parenthetical evaluation data are central to this paper but often are not the main subject or object. A rubric can improve precision, but it cannot judge candidates that were never generated.

4. **Minimum relationship quotas create hallucination pressure.** Many scientific paragraphs contain fewer than five ontology-worthy facts. The active configuration now allows zero relationships and the prompt explicitly prohibits padding.

5. **Automatic completeness was over-optimistic.** The earlier run reported 100% semantic coverage while answering only 3/10 competency questions and leaving 91.4% of entities orphaned. Gold predicate recall should be the primary refinement signal; heuristic semantic coverage should remain a warning layer.

6. **Parser defects need separate tracking.** The selected sentences include joined words such as `visionenabled` and `peerreviewed`, plus several section/title fragments elsewhere in the paper. Gold spans preserve parser output exactly so extraction tests remain reproducible, but parser quality should be scored independently.

## Prompt-policy consequences

- Node decisions now use explicit accept/review/disregard boundaries; review is reserved for one genuine human-resolvable ambiguity.
- Type instructions distinguish the high-confusion pairs observed in the annotations: `Agent`/`AISystem`, `Tool`/`SoftwareArtifact`, `Dataset`/`DataSource`, `Metric`/`Observation`, and `Task`/`Method`.
- Relationship generation scans the competency predicates first but treats priority as search order, never as evidence.
- Validation separates semantic entailment from deterministic ontology, ID, and provenance gates.
- Inferred endpoint evidence must be an exact substring of its cited sentence.
