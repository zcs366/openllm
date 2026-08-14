"""Tests for EmbeddingEngine (T-CC-21).

Three tests:
  1. Single encode — a single text produces a float list of correct length.
  2. Batch encode — multiple texts produce the expected number of vectors.
  3. Dimension check — the engine reports the correct embedding dimension.
"""
from __future__ import annotations

import pytest

from openllm.embedding import EmbeddingEngine

# ── Shared fixture ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def engine() -> EmbeddingEngine:
    """Create a shared EmbeddingEngine using the default model."""
    return EmbeddingEngine("all-MiniLM-L6-v2")


# ── Test 1: Single encode ──────────────────────────────────────

class TestSingleEncode:
    """Verify encode() returns a correct embedding for a single text."""

    def test_returns_list_of_floats(self, engine: EmbeddingEngine) -> None:
        result = engine.encode("Hello, world!")
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_dimension_matches(self, engine: EmbeddingEngine) -> None:
        result = engine.encode("The quick brown fox.")
        assert len(result) == engine.dimension

    def test_deterministic(self, engine: EmbeddingEngine) -> None:
        """Same input should produce identical embeddings."""
        a = encode_deterministic(engine, "deterministic test")
        b = encode_deterministic(engine, "deterministic test")
        assert a == b


# ── Test 2: Batch encode ───────────────────────────────────────

class TestBatchEncode:
    """Verify encode_batch() handles lists of texts correctly."""

    def test_returns_correct_count(self, engine: EmbeddingEngine) -> None:
        texts = ["apple", "banana", "cherry"]
        results = engine.encode_batch(texts)
        assert len(results) == 3

    def test_each_vector_has_correct_length(self, engine: EmbeddingEngine) -> None:
        texts = ["hello", "world"]
        results = engine.encode_batch(texts)
        for vec in results:
            assert len(vec) == engine.dimension

    def test_empty_list_returns_empty(self, engine: EmbeddingEngine) -> None:
        results = engine.encode_batch([])
        assert results == []

    def test_results_match_single_encode(self, engine: EmbeddingEngine) -> None:
        """Batch encode of a single element should match encode()."""
        text = "consistency check"
        single = engine.encode(text)
        batch = engine.encode_batch([text])[0]
        assert single == batch


# ── Test 3: Dimension check ────────────────────────────────────

class TestDimension:
    """Verify the reported embedding dimension is correct."""

    def test_default_model_dimension(self, engine: EmbeddingEngine) -> None:
        assert engine.dimension == 384

    def test_model_name_property(self, engine: EmbeddingEngine) -> None:
        assert engine.model_name == "all-MiniLM-L6-v2"


# ── Helpers ─────────────────────────────────────────────────────

def encode_deterministic(engine: EmbeddingEngine, text: str) -> list[float]:
    """Encode with determinism guaranteed (no dropout/noise)."""
    return engine.encode(text)
