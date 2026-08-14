"""Tests for ToolIndex (T-CC-22).

Four test classes covering:
  1. Add tool — adding tools increases index size, validates required fields.
  2. Search by keyword — BM25 finds tools with matching keywords.
  3. Search by embedding — semantic search finds related tools.
  4. Empty index — searching an empty index returns empty results.
"""
from __future__ import annotations

import pytest

from openllm.embedding import EmbeddingEngine
from openllm.tool_index import ToolIndex


# ── Shared fixtures ────────────────────────────────────────────


@pytest.fixture(scope="module")
def engine() -> EmbeddingEngine:
    """Shared EmbeddingEngine using the default model."""
    return EmbeddingEngine("all-MiniLM-L6-v2")


@pytest.fixture
def populated_index(engine: EmbeddingEngine) -> ToolIndex:
    """Index pre-populated with 5 sample tools."""
    idx = ToolIndex(engine)
    idx.add({"name": "read_file", "description": "Read file contents from disk"})
    idx.add({"name": "write_file", "description": "Write content to a file on disk"})
    idx.add({"name": "shell", "description": "Execute a shell command"})
    idx.add({"name": "search", "description": "Search files by regex pattern"})
    idx.add({"name": "python_exec", "description": "Execute Python code"})
    return idx


# ── Test 1: Add tool ──────────────────────────────────────────


class TestAddTool:
    """Verify add() correctly indexes tool definitions."""

    def test_add_increases_size(self, engine: EmbeddingEngine) -> None:
        idx = ToolIndex(engine)
        assert idx.size == 0
        idx.add({"name": "test_tool", "description": "A test tool"})
        assert idx.size == 1

    def test_add_multiple_tools(self, engine: EmbeddingEngine) -> None:
        idx = ToolIndex(engine)
        idx.add({"name": "tool_a", "description": "Alpha tool"})
        idx.add({"name": "tool_b", "description": "Beta tool"})
        idx.add({"name": "tool_c", "description": "Gamma tool"})
        assert idx.size == 3

    def test_add_requires_name(self, engine: EmbeddingEngine) -> None:
        idx = ToolIndex(engine)
        with pytest.raises(ValueError, match="name"):
            idx.add({"description": "Missing name"})

    def test_add_requires_description(self, engine: EmbeddingEngine) -> None:
        idx = ToolIndex(engine)
        with pytest.raises(ValueError, match="description"):
            idx.add({"name": "no_desc"})


# ── Test 2: Search by keyword ─────────────────────────────────


class TestSearchKeyword:
    """Verify BM25 keyword matching finds the right tools."""

    def test_exact_keyword_match(self, populated_index: ToolIndex) -> None:
        results = populated_index.search("read file")
        assert len(results) > 0
        top = results[0]
        assert top["name"] == "read_file"

    def test_keyword_returns_results(self, populated_index: ToolIndex) -> None:
        results = populated_index.search("shell execute")
        assert len(results) > 0
        names = [r["name"] for r in results]
        assert "shell" in names

    def test_score_field_present(self, populated_index: ToolIndex) -> None:
        results = populated_index.search("python")
        assert all("score" in r for r in results)
        assert all(isinstance(r["score"], float) for r in results)

    def test_results_sorted_by_score(self, populated_index: ToolIndex) -> None:
        results = populated_index.search("write content to file")
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)


# ── Test 3: Search by embedding ───────────────────────────────


class TestSearchEmbedding:
    """Verify semantic (embedding) search finds related tools."""

    def test_semantic_search_finds_related(
        self, populated_index: ToolIndex
    ) -> None:
        """A query semantically related to file operations should find them."""
        results = populated_index.search("load data from a document")
        assert len(results) > 0
        top_names = [r["name"] for r in results[:2]]
        assert "read_file" in top_names or "write_file" in top_names

    def test_unrelated_query_has_low_scores(
        self, populated_index: ToolIndex
    ) -> None:
        """A completely unrelated query should have low scores."""
        results = populated_index.search("cook pasta recipe")
        assert len(results) > 0
        # All scores should be low for an unrelated query
        assert results[0]["score"] < 1.0


# ── Test 4: Empty index ───────────────────────────────────────


class TestEmptyIndex:
    """Verify behavior on an empty index."""

    def test_search_empty_returns_empty(self, engine: EmbeddingEngine) -> None:
        idx = ToolIndex(engine)
        results = idx.search("anything at all")
        assert results == []

    def test_search_empty_respects_top_k(
        self, engine: EmbeddingEngine
    ) -> None:
        idx = ToolIndex(engine)
        results = idx.search("query", top_k=10)
        assert results == []

    def test_size_is_zero(self, engine: EmbeddingEngine) -> None:
        idx = ToolIndex(engine)
        assert idx.size == 0
