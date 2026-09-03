"""EmbeddingEngine — thin wrapper around sentence-transformers for dense vector encoding.

Provides a consistent interface for encoding single texts and batches into
fixed-dimensional embeddings suitable for retrieval, clustering, and similarity.

Default model: all-MiniLM-L6-v2 (384-dim, fast, good general-purpose quality).
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class EmbeddingEngine:
    """Encode text(s) into dense embedding vectors using sentence-transformers.

    Parameters
    ----------
    model_name : str
        HuggingFace sentence-transformers model identifier.
        Defaults to ``'all-MiniLM-L6-v2'`` (384-dimensional).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 local_files_only: bool = True) -> None:
        self._model_name = model_name
        self._local_files_only = local_files_only  # 离线优先：防每次联网 HEAD 检查拖死
        self._model: Optional["SentenceTransformer"] = None  # lazy load

    # -- lazy loading -------------------------------------------------------

    def _ensure_model(self) -> "SentenceTransformer":
        """Load the underlying SentenceTransformer on first use."""
        if self._model is None:
            # 惰性import（DR-20260828-01）：sentence_transformers是重依赖链
            # （torch/transformers），顶层import会阻断整个openllm包的导入。
            # 模型本身是懒加载的，import与之保持同一惰性语义。
            import os
            if self._local_files_only:
                # 离线强制（2026-09-04 实测）：ST 5.5.1 即使 local_files_only=True
                # 仍无条件联网 HEAD 检查 processor_config.json，网络不通时拖死加载。
                # env 离线模式是唯一可靠拦截；local_files_only=False 时不设（允许下载）。
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name,
                                               local_files_only=self._local_files_only)
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
