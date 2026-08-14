"""Tests for the ContextEngine ABC (T-CC-16).

Two tests:
  1. Interface completeness — the ABC has exactly the four required abstract
     methods and cannot be instantiated directly.
  2. Mock implementation — a concrete subclass works correctly and respects
     the contract.
"""
from __future__ import annotations

import pytest

from openllm.context_engine import ContextEngine, ContextEntry, SearchResult


# ── Test 1: Interface completeness ─────────────────────────────

class TestContextEngineInterface:
    """Verify the ABC defines the expected contract."""

    def test_has_four_abstract_methods(self) -> None:
        abstracts = ContextEngine.__abstractmethods__
        assert abstracts == frozenset({"compress", "persist", "restore", "search"})

    def test_cannot_instantiate_abc_directly(self) -> None:
        with pytest.raises(TypeError, match="abstract method"):
            ContextEngine()  # type: ignore[abstract]

    def test_dataclasses_are_importable(self) -> None:
        entry = ContextEntry(role="user", content="hello", token_count=1)
        assert entry.role == "user"
        result = SearchResult(entry=entry, score=0.9)
        assert result.score == 0.9


# ── Test 2: Mock implementation verification ──────────────────

class InMemoryContextEngine(ContextEngine):
    """Minimal concrete implementation for testing."""

    def __init__(self) -> None:
        self._store: list[ContextEntry] = []

    def compress(self, entries: list[ContextEntry], target_tokens: int) -> list[ContextEntry]:
        budget = 0
        kept: list[ContextEntry] = []
        for e in entries:
            if budget + e.token_count <= target_tokens:
                kept.append(e)
                budget += e.token_count
        return kept

    def persist(self, entries: list[ContextEntry]) -> int:
        self._store = list(entries)
        return len(self._store)

    def restore(self) -> list[ContextEntry]:
        return list(self._store)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        results = []
        for e in self._store:
            if query.lower() in e.content.lower():
                results.append(SearchResult(entry=e, score=1.0))
        return results[:top_k]


class TestMockImplementation:
    """Concrete subclass must honour the contract."""

    def setup_method(self) -> None:
        self.engine = InMemoryContextEngine()

    def test_compress_respects_budget(self) -> None:
        entries = [
            ContextEntry(role="user", content="a", token_count=10),
            ContextEntry(role="assistant", content="b", token_count=20),
            ContextEntry(role="user", content="c", token_count=5),
        ]
        compressed = self.engine.compress(entries, target_tokens=15)
        assert sum(e.token_count for e in compressed) <= 15

    def test_persist_and_restore_roundtrip(self) -> None:
        entries = [
            ContextEntry(role="system", content="You are helpful."),
            ContextEntry(role="user", content="Hello!"),
        ]
        count = self.engine.persist(entries)
        assert count == 2
        restored = self.engine.restore()
        assert len(restored) == 2
        assert restored[0].content == "You are helpful."

    def test_search_returns_matches(self) -> None:
        entries = [
            ContextEntry(role="user", content="python is great"),
            ContextEntry(role="user", content="rust is fast"),
            ContextEntry(role="user", content="Python has many libs"),
        ]
        self.engine.persist(entries)
        hits = self.engine.search("python")
        assert len(hits) == 2
        assert all("python" in h.entry.content.lower() for h in hits)

    def test_search_top_k_limits_results(self) -> None:
        entries = [
            ContextEntry(role="user", content=f"match {i}") for i in range(10)
        ]
        self.engine.persist(entries)
        hits = self.engine.search("match", top_k=3)
        assert len(hits) == 3
