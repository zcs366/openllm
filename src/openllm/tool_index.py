"""ToolIndex — Hybrid search index for tool definitions.

Combines BM25 keyword search with embedding semantic search to find
the most relevant tools for a given query.

Usage::

    from openllm.embedding import EmbeddingEngine
    from openllm.tool_index import ToolIndex

    engine = EmbeddingEngine()
    index = ToolIndex(engine)
    index.add({"name": "read_file", "description": "Read file contents", "tags": ["file", "io"]})
    results = index.search("read a file", top_k=3)
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from .embedding import EmbeddingEngine


class ToolIndex:
    """Hybrid search index over tool definitions.

    Combines BM25 keyword matching with embedding cosine similarity
    for semantic search. Results are re-ranked with configurable weights.

    Parameters
    ----------
    embedding_engine : EmbeddingEngine
        Engine for computing dense embeddings.
    bm25_weight : float
        Weight for BM25 keyword score in the final ranking (default 0.5).
    embed_weight : float
        Weight for embedding similarity score (default 0.5).
    """

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        bm25_weight: float = 0.5,
        embed_weight: float = 0.5,
    ) -> None:
        self._engine = embedding_engine
        self._bm25_weight = bm25_weight
        self._embed_weight = embed_weight

        # Storage
        self._tools: list[dict[str, Any]] = []
        self._embeddings: list[list[float]] = []

        # BM25 corpus stats (recomputed on add)
        self._corpus_tokens: list[list[str]] = []
        self._df: Counter[str] = Counter()
        self._N: int = 0
        self._avg_dl: float = 0.0

    @property
    def size(self) -> int:
        """Number of tools in the index."""
        return len(self._tools)

    def add(self, tool_def: dict) -> None:
        """Add a tool definition to the index.

        Parameters
        ----------
        tool_def : dict
            Tool definition. Must contain ``'name'`` and ``'description'``.
            Optional keys: ``'schema'``, ``'tags'``, ``'effects'``, etc.

        Raises
        ------
        ValueError
            If ``'name'`` or ``'description'`` is missing.
        """
        if "name" not in tool_def:
            raise ValueError("tool_def must contain 'name'")
        if "description" not in tool_def:
            raise ValueError("tool_def must contain 'description'")

        # Store tool def (shallow copy)
        self._tools.append(dict(tool_def))

        # Build searchable text: name + description + tags
        text = self._build_text(tool_def)

        # Tokenize for BM25
        tokens = _tokenize(text)
        self._corpus_tokens.append(tokens)

        # Update corpus statistics
        self._rebuild_corpus_stats()

        # Compute and store embedding
        embedding = self._engine.encode(text)
        self._embeddings.append(embedding)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search for tools matching the query.

        Combines BM25 keyword matching with embedding similarity.
        Returns up to ``top_k`` results sorted by descending score.

        Parameters
        ----------
        query : str
            Search query text.
        top_k : int
            Maximum number of results to return.

        Returns
        -------
        list[dict]
            List of tool definitions with an added ``'score'`` field.
        """
        if not self._tools:
            return []

        q_tokens = _tokenize(query)
        q_embedding = self._engine.encode(query)

        # Score each tool
        scored: list[tuple[float, dict]] = []
        for i, tool in enumerate(self._tools):
            # BM25 score
            bm25 = _bm25_score(
                q_tokens,
                self._corpus_tokens[i],
                self._df,
                self._N,
                self._avg_dl,
            )

            # Embedding cosine similarity
            embed_sim = _cosine_similarity(q_embedding, self._embeddings[i])

            # Combined score
            combined = self._bm25_weight * bm25 + self._embed_weight * embed_sim
            scored.append((combined, tool))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Return top_k with score attached
        results = []
        for score, tool in scored[:top_k]:
            result = dict(tool)
            result["score"] = round(score, 6)
            results.append(result)

        return results

    # ── Internal helpers ────────────────────────────────────────

    @staticmethod
    def _build_text(tool_def: dict) -> str:
        """Build searchable text from a tool definition."""
        parts = [
            tool_def.get("name", ""),
            tool_def.get("description", ""),
        ]
        tags = tool_def.get("tags", [])
        if tags:
            parts.append(" ".join(tags))
        return " ".join(parts)

    def _rebuild_corpus_stats(self) -> None:
        """Rebuild BM25 corpus statistics (DF, N, avg_dl)."""
        self._N = len(self._corpus_tokens)
        self._df = Counter()
        for tokens in self._corpus_tokens:
            self._df.update(set(tokens))
        if self._N > 0:
            self._avg_dl = sum(len(t) for t in self._corpus_tokens) / self._N
        else:
            self._avg_dl = 0.0


# ── BM25 helpers ──────────────────────────────────────────────


def _tokenize(text: str) -> list[str]:
    """Tokenize text: English words (lowercase, >= 2 chars) + Chinese bigrams."""
    tokens = []
    tokens.extend(w.lower() for w in re.findall(r"[a-zA-Z_]{2,}", text.lower()))
    ch = [c for c in text if "\u4e00" <= c <= "\u9fff"]
    tokens.extend(ch[i] + ch[i + 1] for i in range(len(ch) - 1))
    return tokens


def _bm25_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    df: Counter,
    N: int,
    avg_dl: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Compute BM25 score for a query against a single document."""
    if not query_tokens or N == 0:
        return 0.0
    score = 0.0
    dl = len(doc_tokens)
    tf = Counter(doc_tokens)
    for q in query_tokens:
        if q not in tf:
            continue
        doc_freq = df.get(q, 0)
        if doc_freq == 0:
            continue
        idf = math.log((N - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
        tf_norm = (tf[q] * (k1 + 1)) / (
            tf[q] + k1 * (1 - b + b * dl / max(avg_dl, 1))
        )
        score += idf * tf_norm
    return score


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
