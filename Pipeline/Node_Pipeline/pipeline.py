"""Compatibility facade for the node pipeline's separated modules.

New code should import from ``Pipeline.Node_Pipeline`` or the focused module
that owns the behavior. Existing direct imports from this module remain valid.
"""

from .batching import (
    _sentences, build_paragraph_batches, candidate_rows as _candidate_rows,
)
from .entities import (
    _context_for, canonical_entities, mention_from_candidate as _mention,
)
from .extraction import Progress, _emit, extract_nodes
from .passes import (
    NODE_RUBRIC, cleanup_candidates as _cleanup_pass, cleanup_payload as _cleanup_payload,
    judge_candidates as _judgment_pass, judgment_payload as _judgment_payload,
    type_candidates as _typing_pass, typing_payload as _typing_payload,
)
from .runners import run_extraction, run_node_only_extraction
from .provenance import validate_provenance
from .review import apply_node_review


__all__ = [
    "NODE_RUBRIC", "apply_node_review", "build_paragraph_batches",
    "canonical_entities", "extract_nodes", "run_extraction",
    "run_node_only_extraction", "validate_provenance",
]
