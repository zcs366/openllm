"""EmbeddingEngine — thin wrapper around sentence-transformers for dense vector encoding.

Provides a consistent interface for encoding single texts and batches into
fixed-dimensional embeddings suitable for retrieval, clustering, and similarity.

Default model: all-MiniLM-L6-v2 (384-dim, fast, good general-purpose quality).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingEngine:
    """Encode text(s) into dense embedding vectors using sentence-transformers.

    Parameters
    ----------
    model_name : str
        HuggingFace sentence-transformers model identifier.
        Defaults to ``'all-MiniLM-L6-v2'`` (384-dimensional).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model: Optional[SentenceTransformer] = None  # lazy load

    # -- lazy loading -------------------------------------------------------

    def _ensure_model(self) -> SentenceTransformer:
        """Load the underlying SentenceTransformer on first use."""
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    # -- public API ---------------------------------------------------------

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Return the embedding dimension for the current model.

        Raises
        ------
        RuntimeError
            If the model does not expose a known embedding dimension.
        """
        model = self._ensure_model()
        dim = model.get_embedding_dimension()
        if dim is None:
            raise RuntimeError(
                f"Model '{self._model_name}' does not report an embedding dimension."
            )
        return dim

    def encode(self, text: str) -> list[float]:
        """Encode a single text string into an embedding vector.

        Parameters
        ----------
        text : str
            The text to encode.

        Returns
        -------
        list[float]
            A list of floats representing the embedding.
        """
        model = self._ensure_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of text strings into embedding vectors.

        Parameters
        ----------
        texts : list[str]
            List of texts to encode.

        Returns
        -------
        list[list[float]]
            A list of embedding vectors, one per input text.
        """
        if not texts:
            return []

        model = self._ensure_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return [vec.tolist() for vec in embeddings]
