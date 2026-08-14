"""Integration tests for openLLM context engineering modules (T-CC-25).

Exercises cross-module interactions between components created in T-CC-16
through T-CC-24:

  1. ContextEngine + CompactionController — compress/persist round-trip
  2. UnifiedMemory + ContextDriftDetector — memory storage and drift check
  3. End-to-end flow — ContextPressureMonitor → CompactionController → Memory
  4. ToolRegistryBridge + CompactionController — bridge config + compression
  5. Degradation — graceful handling of invalid inputs across all modules
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest

from openllm.context_engine import ContextEngine, ContextEntry, SearchResult
from openllm.compaction_control import (
    CompactionConfig,
    CompactionController,
    CompactionStrategy,
    estimate_tokens,
)
from openllm.message import Message
from openllm.memory.context_pressure import (
    ContextPressureMonitor,
    THRESHOLD_CAUTION,
    THRESHOLD_CRITICAL,
)
from openllm.core.context_drift_detector import ContextDriftDetector
from openllm.memory.unified_memory import UnifiedMemory, MemoryEntry
from openllm.isn.tool_registry_bridge import (
    ToolConfig,
    ToolRegistryBridge,
    BridgeResult,
)


# ── Helpers ──────────────────────────────────────────────────────

class InMemoryContextEngine(ContextEngine):
    """Minimal ContextEngine for integration testing."""

    def __init__(self):
        self._store: list[ContextEntry] = []

    def compress(self, entries, target_tokens):
        budget = 0
        kept = []
        for e in entries:
            if budget + e.token_count <= target_tokens:
                kept.append(e)
                budget += e.token_count
        return kept

    def persist(self, entries):
        self._store = list(entries)
        return len(self._store)

    def restore(self):
        return list(self._store)

    def search(self, query, top_k=5):
        results = []
        for e in self._store:
            if query.lower() in e.content.lower():
                results.append(SearchResult(entry=e, score=1.0))
        return results[:top_k]


class MockToolRegistry:
    """Lightweight stand-in for ToolRegistry in integration tests."""

    def __init__(self):
        self._tools = {}
        self._descriptions = {}
        self._schemas = {}
        self._requires_verify = set()

    def register(self, name, func, description="", schema=None):
        self._tools[name] = func
        self._descriptions[name] = description
        self._schemas[name] = schema or {}
        return True


def _make_messages(n: int, token_each: int = 200) -> list[Message]:
    """Create N messages with roughly token_each tokens each."""
    chars = "x" * (token_each * 4)  # ~4 chars per token
    return [Message(role="user", content=f"msg{i} {chars}") for i in range(n)]


def _sample_tool_fn(path: str, limit: int = 10) -> str:
    """Dummy tool function for bridge registration tests."""
    return f"scanned {path} limit={limit}"


# ── Test 1: ContextEngine + CompactionController ─────────────────

class TestContextEnginePlusCompaction:
    """Verify ContextEngine and CompactionController cooperate.

    - Entries compressed by the engine fit the same budget used by
      the controller's strategy selection.
    - Persisted + restored entries survive a compress cycle.
    """

    def test_compress_then_persist_roundtrip(self) -> None:
        engine = InMemoryContextEngine()

        entries = [
            ContextEntry(role="system", content="You are helpful.", token_count=20),
            ContextEntry(role="user", content="Hello!", token_count=5),
            ContextEntry(role="assistant", content="Hi there.", token_count=5),
        ]

        # Compress to a tight budget
        compressed = engine.compress(entries, target_tokens=25)
        assert sum(e.token_count for e in compressed) <= 25

        # Persist compressed entries, restore, and verify
        count = engine.persist(compressed)
        assert count == len(compressed)
        restored = engine.restore()
        assert restored == compressed

    def test_compaction_controller_selects_correct_strategy(self) -> None:
        config = CompactionConfig(
            max_tokens=1000,
            caution_ratio=0.70,
            critical_ratio=0.85,
        )
        ctrl = CompactionController(config)

        # Under caution → no compression needed
        msgs = [Message(role="user", content="x" * 100)]  # ~25 tokens
        result = ctrl.process(msgs)
        assert result == msgs  # unchanged

        # Above caution (70%) but below critical (85%) → TRUNCATE
        # 500 tokens out of 1000 → 50% — still under caution; use 750 instead
        # Each msg with token_each=50 → ~50 tokens. 15 msgs = ~750 tokens → 75%
        msgs = _make_messages(15, token_each=50)
        result = ctrl.process(msgs)
        assert len(result) < len(msgs)
        assert ctrl.stats()["last_strategy"] == "truncate"

    def test_engine_and_controller_share_token_estimate(self) -> None:
        """Both modules should produce compatible token estimates."""
        text = "hello world 你好世界"
        engine_est = len(text) // 4  # RollingWindow-style
        ctrl_est = estimate_tokens(text)
        # They use different heuristics but should be in the same ballpark
        assert 1 <= engine_est <= ctrl_est + 5


# ── Test 2: UnifiedMemory + ContextDriftDetector ────────────────

class TestMemoryAndDriftIntegration:
    """Verify UnifiedMemory and ContextDriftDetector can work together.

    A conversation stored in memory can be checked for drift, and the
    drift report accurately reflects the stored sequence.
    """

    def test_drift_detection_on_stored_messages(self, tmp_path: Path) -> None:
        mem = UnifiedMemory(memory_dir=tmp_path / "mem")

        # Store a short conversation as memory entries
        messages = [
            "Let's discuss the Python GIL and concurrency",
            "The GIL prevents true parallelism in CPython",
            "What about asyncio and cooperative multitasking?",
            "Asyncio uses a single thread with an event loop",
        ]
        for i, text in enumerate(messages):
            mem.store(
                key=f"conv-turn-{i}",
                value={"text": text},
                importance=0.6,
                layer="hot",
                tags=["conversation"],
            )

        # Retrieve stored messages
        results = mem.retrieve("Python GIL", layer="hot")
        assert len(results) >= 1

        # Run drift detection on the original message sequence
        detector = ContextDriftDetector()
        drift_points = detector.detect_drift_in_conversation(
            messages, threshold=0.5
        )
        # Topic stays consistent → no high-drift points expected
        assert all(dp["drift"] < 0.9 for dp in drift_points)

    def test_memory_temperature_increases_with_access(self, tmp_path: Path) -> None:
        mem = UnifiedMemory(memory_dir=tmp_path / "mem2")
        entry = mem.store(
            key="key-decision-1",
            value={"decision": "use async"},
            importance=0.8,
            layer="warm",
            tags=["insight"],
        )
        t1 = entry.temperature()
        # Access it a few times
        for _ in range(5):
            mem.get("key-decision-1")
        t2 = entry.temperature()
        assert t2 >= t1  # access increases heat → temperature


# ── Test 3: End-to-end flow ─────────────────────────────────────

class TestEndToEndFlow:
    """PressureMonitor → CompactionController → Memory pipeline.

    Simulates a real session: record usage, detect pressure,
    compress, store the compressed context in memory.
    """

    def test_pressure_drives_compaction_and_memory_store(
        self, tmp_path: Path
    ) -> None:
        # 1. Set up a pressure monitor (in-memory state)
        monitor = ContextPressureMonitor(state_path=tmp_path / "pressure.json")

        # 2. Record normal usage
        session_id = "sess-e2e-001"
        monitor.record_usage(session_id, tokens_used=5000, tokens_budget=10000)
        assert monitor.get_pressure_level(session_id) == "normal"

        # 3. Ramp up to critical
        monitor.record_usage(session_id, tokens_used=9000, tokens_budget=10000)
        level = monitor.get_pressure_level(session_id)
        assert level == "critical"

        should, strategy = monitor.should_compress(session_id)
        assert should is True
        assert strategy == "standard"

        # 4. Use CompactionController to compress
        config = CompactionConfig(max_tokens=10000)
        ctrl = CompactionController(config)
        messages = _make_messages(20, token_each=600)  # ~12000 tokens
        compressed = ctrl.process(messages)
        assert len(compressed) < len(messages)

        # 5. Store compressed result in memory
        mem = UnifiedMemory(memory_dir=tmp_path / "mem")
        mem.store(
            key=f"compressed-context-{session_id}",
            value={
                "original_count": len(messages),
                "compressed_count": len(compressed),
                "strategy": "standard",
            },
            importance=0.7,
            layer="cold",
            tags=["compaction", "context"],
        )

        results = mem.retrieve("compressed-context", layer="cold")
        assert len(results) >= 1
        assert results[0].value["strategy"] == "standard"


# ── Test 4: ToolRegistryBridge + CompactionController ───────────

class TestBridgeAndCompactionIntegration:
    """ToolRegistryBridge config validation + CompactionController.

    Verifies that tool configs pass validation through the bridge
    and that compression metadata can carry tool registration info.
    """

    def test_bridge_register_and_compression_metadata(self) -> None:
        registry = MockToolRegistry()
        bridge = ToolRegistryBridge(registry)

        # Register a tool
        result = bridge.register_from_config({
            "name": "file_scanner",
            "description": "Scan files",
            "module": "builtins",
            "function": "len",  # use built-in as stand-in
            "params": {},
        })
        assert result.success is True

        # Verify it's listed
        tools = bridge.list_registered()
        assert any(t["name"] == "file_scanner" for t in tools)

        # Compress a conversation that references the tool
        config = CompactionConfig(max_tokens=200)
        ctrl = CompactionController(config)
        messages = [
            Message(role="system", content="You have a tool called file_scanner."),
            Message(role="user", content="Scan my project directory for Python files"),
            Message(role="assistant", content="Calling file_scanner with path=/project"),
            Message(role="tool", content="Found 42 .py files"),
            Message(role="user", content="Great, now summarize the results"),
        ]
        compressed = ctrl.process(messages)
        # At least the tool registration reference should survive compression
        assert any("file_scanner" in m.content for m in compressed)

    def test_bridge_validation_prevents_bad_config(self) -> None:
        registry = MockToolRegistry()
        bridge = ToolRegistryBridge(registry)

        # Missing required fields
        result = bridge.register_from_config({
            "name": "",
            "description": "broken tool",
            "module": "",
            "function": "",
        })
        assert result.success is False
        assert len(result.errors) > 0


# ── Test 5: Degradation ─────────────────────────────────────────

class TestDegradationGracefulHandling:
    """Verify all modules handle edge cases and invalid inputs gracefully.

    Integration tests must prove the system degrades without crashing
    when faced with adversarial or unexpected inputs.
    """

    def test_context_engine_handles_empty_list(self) -> None:
        engine = InMemoryContextEngine()
        compressed = engine.compress([], target_tokens=100)
        assert compressed == []

        count = engine.persist([])
        assert count == 0

        restored = engine.restore()
        assert restored == []

        results = engine.search("anything", top_k=5)
        assert results == []

    def test_compaction_controller_handles_single_message(self) -> None:
        config = CompactionConfig(max_tokens=10000)
        ctrl = CompactionController(config)
        msgs = [Message(role="user", content="one message")]
        result = ctrl.process(msgs)
        assert len(result) == 1  # single message kept

    def test_pressure_monitor_handles_zero_budget(self, tmp_path: Path) -> None:
        monitor = ContextPressureMonitor(state_path=tmp_path / "press_zero.json")
        monitor.record_usage("s1", tokens_used=100, tokens_budget=0)
        # Zero budget is rejected — should stay unknown
        assert monitor.get_pressure_level("s1") == "unknown"

    def test_drift_detector_handles_identical_messages(self) -> None:
        detector = ContextDriftDetector()
        drift = detector.compute_drift(
            "hello world how are you",
            "hello world how are you",
        )
        assert drift == 0.0  # identical → zero drift

    def test_memory_handles_duplicate_keys(self, tmp_path: Path) -> None:
        mem = UnifiedMemory(memory_dir=tmp_path / "mem_dup")
        mem.store(key="k1", value={"v": 1}, layer="warm")
        mem.store(key="k1", value={"v": 2}, layer="warm")  # overwrite
        entry = mem.get("k1")
        assert entry is not None
        assert entry.value["v"] == 2  # latest wins

    def test_estimate_tokens_edge_cases(self) -> None:
        assert estimate_tokens("") == 0
        assert estimate_tokens("x") >= 1
        # Pure Chinese text
        cn_tokens = estimate_tokens("你好世界欢迎来到开放的大模型时代")
        assert cn_tokens >= 1
        # Pure ASCII
        ascii_tokens = estimate_tokens("a" * 100)
        assert ascii_tokens >= 1

    def test_compaction_config_validation(self) -> None:
        with pytest.raises(ValueError):
            CompactionConfig(max_tokens=0)
        with pytest.raises(ValueError):
            CompactionConfig(max_tokens=1000, caution_ratio=0.0)
        with pytest.raises(ValueError):
            CompactionConfig(
                max_tokens=1000, caution_ratio=0.9, critical_ratio=0.8
            )  # caution > critical

    def test_tool_config_validation_errors(self) -> None:
        bad = ToolConfig(name="", description="", module="", function="")
        errors = bad.validate()
        assert len(errors) == 3  # name, module, function all empty
