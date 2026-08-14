"""Context Engineering Acceptance Tests (T-CC-26)
================================================

End-to-end acceptance tests for the openLLM context engineering system.
Verifies the complete lifecycle: message → compress → persist → restore → search.

Three acceptance tests:
  1. Full pipeline: message → compress → persist → restore → search
  2. Token budget compliance under stress
  3. Performance benchmark (latency bounds)
"""
from __future__ import annotations

import time
import tempfile
import json
from pathlib import Path

import pytest

from openllm.context_engine import ContextEngine, ContextEntry, SearchResult
from openllm.message import Message
from openllm.compaction_control import (
    CompactionController,
    CompactionConfig,
    CompactionStrategy,
    estimate_tokens,
)
from openllm.compaction_strategy import RollingWindowStrategy
from openllm.persistence import DiskPersistence


# ── In-memory ContextEngine for acceptance testing ──────────────


class AcceptanceContextEngine(ContextEngine):
    """Full-featured in-memory ContextEngine for acceptance tests.

    Implements compress/persist/restore/search with keyword-based relevance.
    """

    def __init__(self) -> None:
        self._store: list[ContextEntry] = []
        self._compress_log: list[dict] = []

    def compress(self, entries: list[ContextEntry], target_tokens: int) -> list[ContextEntry]:
        budget = 0
        kept: list[ContextEntry] = []
        for e in entries:
            if budget + e.token_count <= target_tokens:
                kept.append(e)
                budget += e.token_count
        self._compress_log.append({
            "input_count": len(entries),
            "output_count": len(kept),
            "input_tokens": sum(e.token_count for e in entries),
            "output_tokens": budget,
            "target_tokens": target_tokens,
        })
        return kept

    def persist(self, entries: list[ContextEntry]) -> int:
        self._store = list(entries)
        return len(self._store)

    def restore(self) -> list[ContextEntry]:
        return list(self._store)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_lower = query.lower()
        results = []
        for e in self._store:
            content_lower = e.content.lower()
            if query_lower in content_lower:
                # Simple keyword relevance score
                score = content_lower.count(query_lower) / max(len(content_lower), 1)
                results.append(SearchResult(entry=e, score=score))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


# ══════════════════════════════════════════════════════════════
# Test 1: Full Pipeline — message → compress → persist → restore → search
# ══════════════════════════════════════════════════════════════


class TestFullPipeline:
    """Acceptance: full lifecycle from message creation through search.

    Steps:
      1. Create Message objects → convert to ContextEntry
      2. Compress entries to fit a token budget
      3. Persist compressed entries
      4. Restore from persistence
      5. Search restored entries for relevant content
    """

    def test_full_pipeline_message_to_search(self) -> None:
        engine = AcceptanceContextEngine()

        # Step 1: Create messages
        messages = [
            Message(role="system", content="You are a helpful assistant specializing in Python."),
            Message(role="user", content="What is the difference between a list and a tuple in Python?"),
            Message(role="assistant", content="A list is mutable while a tuple is immutable. Lists use [] and tuples use ()."),
            Message(role="user", content="How do decorators work in Python?"),
            Message(role="assistant", content="Decorators are functions that modify other functions. They use the @ syntax."),
            Message(role="user", content="What about context managers?"),
            Message(role="assistant", content="Context managers use the 'with' statement to manage resources like file handles."),
        ]

        # Step 2: Convert to ContextEntry
        entries = []
        for msg in messages:
            tc = estimate_tokens(msg.content)
            entries.append(ContextEntry(
                role=msg.role,
                content=msg.content,
                token_count=tc,
                metadata={"type": "conversation"},
            ))

        assert len(entries) == 7
        total_input_tokens = sum(e.token_count for e in entries)
        assert total_input_tokens > 0

        # Step 3: Compress to fit budget (aggressive: target < half of total)
        budget = total_input_tokens // 2
        compressed = engine.compress(entries, target_tokens=budget)
        compressed_tokens = sum(e.token_count for e in compressed)
        assert compressed_tokens <= budget, (
            f"Compressed tokens {compressed_tokens} exceed budget {budget}"
        )
        assert len(compressed) <= len(entries)

        # Step 4: Persist compressed entries
        persisted_count = engine.persist(compressed)
        assert persisted_count == len(compressed)

        # Step 5: Restore from persistence
        restored = engine.restore()
        assert len(restored) == len(compressed)

        # Step 6: Search restored entries
        hits = engine.search("python", top_k=10)
        assert len(hits) >= 2, "Should find at least 2 entries mentioning 'python'"
        for hit in hits:
            assert "python" in hit.entry.content.lower()
            assert hit.score > 0

        # Verify full pipeline log
        assert len(engine._compress_log) == 1
        log = engine._compress_log[0]
        assert log["input_count"] == 7
        assert log["output_count"] < 7
        assert log["input_tokens"] > log["output_tokens"]

    def test_persistence_survives_disk_write(self) -> None:
        """Acceptance: DiskPersistence round-trip preserves data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dp = DiskPersistence()
            path = str(Path(tmpdir) / "state" / "context.json")

            state = {
                "version": 1,
                "entries": [
                    {"role": "user", "content": "test message", "token_count": 3},
                    {"role": "assistant", "content": "test response", "token_count": 3},
                ],
                "metadata": {"session_id": "test-001"},
            }

            dp.save(state, path)
            loaded = dp.load(path)

            assert loaded is not None
            assert loaded["version"] == 1
            assert len(loaded["entries"]) == 2
            assert loaded["metadata"]["session_id"] == "test-001"


# ══════════════════════════════════════════════════════════════
# Test 2: Token Budget Compliance
# ══════════════════════════════════════════════════════════════


class TestTokenBudgetCompliance:
    """Acceptance: compression respects token budgets under various conditions.

    Tests:
      - Under budget: no compression
      - At budget: no compression needed
      - Over budget: truncate strategy
      - Way over budget: aggressive strategy
      - RollingWindow strategy budget compliance
    """

    def test_under_budget_no_compression(self) -> None:
        messages = [
            Message(role="user", content="short"),
            Message(role="assistant", content="reply"),
        ]
        config = CompactionConfig(max_tokens=8192)
        controller = CompactionController(config)

        result = controller.process(messages)
        assert result == messages, "Under budget should return messages unchanged"

    def test_truncate_strategy_respects_keep_recent(self) -> None:
        """When token usage hits caution threshold, truncate drops oldest."""
        # Create messages that exceed caution threshold
        long_content = "x" * 1000  # ~250 tokens
        messages = [
            Message(role="system", content="System prompt"),
            Message(role="user", content=f"Message 1: {long_content}"),
            Message(role="assistant", content=f"Reply 1: {long_content}"),
            Message(role="user", content=f"Message 2: {long_content}"),
            Message(role="assistant", content=f"Reply 2: {long_content}"),
            Message(role="user", content=f"Message 3: {long_content}"),
            Message(role="assistant", content=f"Reply 3: {long_content}"),
        ]

        # Total tokens ≈ 1522. Caution=0.7, Critical=0.85.
        # For truncate: need ratio in (0.7, 0.85) → max_tokens in (1791, 2174)
        config = CompactionConfig(max_tokens=2000, keep_recent=3)
        controller = CompactionController(config)

        result = controller.process(messages)

        # Should have system + 3 recent messages
        system_msgs = [m for m in result if m.role == "system"]
        assert len(system_msgs) == 1, "System message should be preserved"
        assert len(result) <= 5, "Should drop oldest messages"

    def test_aggressive_strategy极端压缩(self) -> None:
        """Way over budget triggers aggressive compression."""
        huge_content = "y" * 4000  # ~1000 tokens each
        messages = [
            Message(role="system", content="sys"),
            Message(role="user", content=f"msg1 {huge_content}"),
            Message(role="user", content=f"msg2 {huge_content}"),
            Message(role="user", content=f"msg3 {huge_content}"),
            Message(role="user", content=f"msg4 {huge_content}"),
        ]

        # max_tokens=500, total ~4000+tokens, ratio > 1.0 → aggressive
        config = CompactionConfig(max_tokens=500, keep_recent=2)
        controller = CompactionController(config)

        result = controller.process(messages)

        # Aggressive: system + marker + recent 2
        non_system = [m for m in result if m.role != "system"]
        assert len(non_system) <= 3, (
            f"Aggressive should keep few messages, got {len(result)} total"
        )

    def test_rolling_window_budget_compliance(self) -> None:
        """RollingWindowStrategy always produces output within budget."""
        strategy = RollingWindowStrategy(min_keep=2)

        messages = [
            Message(role="user", content=f"Message {i}: " + "a" * 200)
            for i in range(20)
        ]

        total_tokens = strategy._estimate_tokens(messages)
        target = total_tokens // 3  # Aggressive target

        compressed = strategy.compress(messages, target)
        compressed_tokens = strategy._estimate_tokens(compressed)

        assert compressed_tokens <= target * 1.1, (  # Allow 10% overshoot for min_keep
            f"RollingWindow over budget: {compressed_tokens} > {target * 1.1}"
        )

    def test_stats_track_compression(self) -> None:
        """CompactionController stats are updated after compression."""
        messages = [
            Message(role="user", content="x" * 500)
            for _ in range(10)
        ]

        config = CompactionConfig(max_tokens=500, keep_recent=3)
        controller = CompactionController(config)

        controller.process(messages)
        stats = controller.stats()

        assert stats["total_compressions"] == 1
        assert stats["tokens_before"] > 0
        assert stats["tokens_after"] > 0
        assert stats["tokens_after"] <= stats["tokens_before"]
        assert stats["last_strategy"] in ("truncate", "summarize", "aggressive")


# ══════════════════════════════════════════════════════════════
# Test 3: Performance Benchmark
# ══════════════════════════════════════════════════════════════


class TestPerformanceBenchmark:
    """Acceptance: operations complete within reasonable time bounds.

    Benchmarks:
      - Compress 1000 messages < 100ms
      - Persist + restore 500 entries < 50ms (in-memory)
      - Search 500 entries < 50ms
      - Full pipeline 100 messages < 200ms
    """

    def test_compress_1000_messages_under_100ms(self) -> None:
        """Compressing 1000 messages must complete under 100ms."""
        messages = [
            Message(role="user", content=f"Message {i}: " + "z" * 50)
            for i in range(1000)
        ]

        config = CompactionConfig(max_tokens=2000, keep_recent=10)
        controller = CompactionController(config)

        start = time.perf_counter()
        result = controller.process(messages)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.1, f"Compress 1000 messages took {elapsed:.3f}s, limit 0.1s"
        assert len(result) <= len(messages)

    def test_persist_restore_500_entries_under_50ms(self) -> None:
        """Persist + restore 500 entries must complete under 50ms."""
        engine = AcceptanceContextEngine()
        entries = [
            ContextEntry(role="user", content=f"Entry {i}: " + "a" * 100, token_count=25)
            for i in range(500)
        ]

        start = time.perf_counter()
        engine.persist(entries)
        restored = engine.restore()
        elapsed = time.perf_counter() - start

        assert elapsed < 0.05, f"Persist+restore 500 took {elapsed:.3f}s, limit 0.05s"
        assert len(restored) == 500

    def test_search_500_entries_under_50ms(self) -> None:
        """Search over 500 entries must complete under 50ms."""
        engine = AcceptanceContextEngine()
        entries = [
            ContextEntry(role="user", content=f"Entry {i} about python testing", token_count=10)
            if i % 5 == 0
            else ContextEntry(role="user", content=f"Entry {i} about rust performance", token_count=10)
            for i in range(500)
        ]
        engine.persist(entries)

        start = time.perf_counter()
        hits = engine.search("python", top_k=10)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.05, f"Search 500 took {elapsed:.3f}s, limit 0.05s"
        assert len(hits) == 10

    def test_full_pipeline_100_messages_under_200ms(self) -> None:
        """Full pipeline (message→compress→persist→restore→search) < 200ms."""
        engine = AcceptanceContextEngine()

        start = time.perf_counter()

        # Create
        entries = []
        for i in range(100):
            tc = estimate_tokens(f"Message {i} " + "content " * 10)
            entries.append(ContextEntry(
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i} " + "content " * 10,
                token_count=tc,
            ))

        # Compress
        total_tokens = sum(e.token_count for e in entries)
        compressed = engine.compress(entries, target_tokens=total_tokens // 2)

        # Persist
        engine.persist(compressed)

        # Restore
        restored = engine.restore()

        # Search
        hits = engine.search("content", top_k=20)

        elapsed = time.perf_counter() - start

        assert elapsed < 0.2, f"Full pipeline 100 msgs took {elapsed:.3f}s, limit 0.2s"
        assert len(restored) > 0
        assert len(hits) > 0
