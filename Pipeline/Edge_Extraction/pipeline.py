"""Compatibility facade for the relationship pipeline's separated modules.

New code should import from ``Pipeline.Edge_Extraction`` or the focused module
that owns the behavior. Existing direct imports from this module remain valid.
"""

from .context import (
    PRIORITY_PREDICATES, endpoint_payload as _endpoint_payload,
    endpoint_type as _endpoint_type, nodes_for_paragraph as _nodes_for_paragraph,
    paragraphs as _paragraphs, predicate_payload as _predicate_payload,
)
from .extraction import run_relationship_extraction
from .materialization import materialize_canonical_relationships
from .passes import (
    RELATIONSHIP_RUBRIC, generate_relationships as _generate,
    generation_payload as _generation_payload, refine_relationships as _refine,
    refinement_payload as _refinement_payload,
    validate_relationships as _validate_with_llm,
    validation_payload as _validation_payload,
)
from .validation import (
    deterministic_gates as _deterministic_gates,
    inferred_candidate as _inferred_candidate,
)


__all__ = [
    "RELATIONSHIP_RUBRIC", "materialize_canonical_relationships",
    "run_relationship_extraction",
]
