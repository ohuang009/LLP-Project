from __future__ import annotations

from typing import Sequence


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceEmbeddingBackend:
    """Lazy local sentence-transformer backend used by entity resolution."""

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 64, model=None):
        self.model_name = model_name
        self.batch_size = batch_size
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Embedding-based entity resolution requires sentence-transformers. "
                    "Install the repository requirements.txt."
                ) from exc
            model = SentenceTransformer(model_name)
        self.model = model

    def encode(self, texts: Sequence[str]):
        return self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=len(texts) >= self.batch_size * 2,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
