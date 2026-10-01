"""Nail test — MemoryBus silent-degradation visibility (audit P1-1, 2026-10-01).

Before: provider query/write exceptions and builtin-register failures were only
logger calls — invisible to the operator. Now each also goes through the
existing visible channel core.degradation_trace.trace_degradation (main_loop
announces "近24h降级N次" at startup), plus per-round failure rosters exposed
via health_check() -> _meta.

Invariants pinned here (控制流不得改变):
1. query() with an always-raising provider must NOT raise; other providers'
   results still come back.
2. _last_query_failures contains the failing provider's name.
3. A degradation record lands on disk (LOG_PATH monkeypatched to tmp —
   the real ~/.openllm is never touched by these tests).
4. Same for write() (_last_write_failures) and _register_builtin_providers
   (_register_failures).
"""
import json

import pytest

from openllm.isa.memory_bus import (
    MemoryBus, MemoryRecord, Query, WriteRequest, WriteResult,
)


# ── fake providers ──

class GoodProvider:
    def __init__(self, name="good", priority=10):
        self._name, self._priority = name, priority

    @property
    def name(self): return self._name

    @property
    def priority(self): return self._priority

    def search(self, query):
        return [MemoryRecord(
            record_id=f"{self._name}:1", content=f"hello from {self._name}",
            source=self._name, record_type="event", importance=0.5,
            temperature=0.5, trust_level="internal", tags=[],
            timestamp=0.0, score=1.0, provider=self._name)]

    def store(self, request):
        return WriteResult(success=True, record_id=f"{self._name}:w1",
                           provider=self._name)

    def count(self): return 1

    def health(self): return {"status": "ok"}


class ExplodingProvider(GoodProvider):
    """search() 与 store() 必抛——模拟坏 provider。"""

    def search(self, query):
        raise RuntimeError(f"{self._name} search boom")

    def store(self, request):
        raise RuntimeError(f"{self._name} store boom")


@pytest.fixture()
def bus():
    """Fresh bus with builtin providers detached (hermetic, no real ~/.hermes reads)."""
    b = MemoryBus()
    b._providers.clear()
    b._register_failures.clear()
    return b


@pytest.fixture()
def trace_log(tmp_path, monkeypatch):
    """Redirect degradation_trace.LOG_PATH into tmp — real ~/.openllm untouched."""
    import openllm.core.degradation_trace as dt
    log = tmp_path / "degradation_log.jsonl"
    monkeypatch.setattr(dt, "LOG_PATH", log)
    return log


def _read_records(log_path):
    if not log_path.exists():
        return []
    return [json.loads(l) for l in
            log_path.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_query_survives_exploding_provider_and_traces(bus, trace_log):
    """① query() 不抛且好 provider 结果照旧；② _last_query_failures 含坏名；
    ③ 降级记录落盘（body/phase/detail 对得上）。"""
    bus.register(GoodProvider("good", priority=10))
    bus.register(ExplodingProvider("bad", priority=1))  # tried first

    results = bus.query(Query(text="hello", top_k=5))

    # 不抛 + 好provider结果照旧
    assert any(r.provider == "good" for r in results), \
        f"good provider results must still come back, got {[r.provider for r in results]}"

    # 名单
    assert "bad" in bus._last_query_failures
    assert "boom" in bus._last_query_failures["bad"]

    # 落盘
    recs = _read_records(trace_log)
    q_recs = [r for r in recs if r["body"] == "MemoryBus"
              and r["phase"] == "provider_query" and r["detail"] == "bad"]
    assert len(q_recs) == 1, f"exactly one trace record expected, got {q_recs}"
    assert q_recs[0]["error"] == "RuntimeError"

    # health_check 暴露
    h = bus.health_check()
    assert "_meta" in h
    assert h["_meta"]["status"] == "degraded"
    assert "bad" in h["_meta"]["last_query_failures"]


def test_query_round_resets_roster(bus, trace_log):
    """名单语义：每轮 query 重置——上一轮的失败不得冒充本轮的。"""
    bus.register(GoodProvider("good", priority=10))
    bus.query(Query(text="hello", top_k=5))
    assert bus._last_query_failures == {}

    bus.register(ExplodingProvider("bad", priority=1))
    bus.query(Query(text="hello", top_k=5))
    assert "bad" in bus._last_query_failures

    # 坏 provider 撤下后，下一轮名单必须清空
    bus.unregister("bad")
    bus.query(Query(text="hello", top_k=5))
    assert bus._last_query_failures == {}
    assert bus.health_check()["_meta"]["status"] == "ok"


def test_write_survives_exploding_provider_and_traces(bus, trace_log):
    """write(): 坏 provider 的 store 异常被记账，好 provider 照常接写入。"""
    bus.register(GoodProvider("good", priority=10))
    bus.register(ExplodingProvider("bad", priority=1))

    result = bus.write(WriteRequest(content="x", source="t"))

    assert result.success is True
    assert result.provider == "good"
    assert "bad" in bus._last_write_failures

    recs = _read_records(trace_log)
    w_recs = [r for r in recs if r["body"] == "MemoryBus"
              and r["phase"] == "provider_write" and r["detail"] == "bad"]
    assert len(w_recs) == 1


def test_register_failure_traced(monkeypatch, tmp_path, trace_log):
    """内置 provider 注册失败：logger 之外同步进 _register_failures + trace，
    且不得阻断 MemoryBus 构造（控制流不变）。"""
    import importlib as _il
    real_import = _il.import_module

    def fake_import(name, package=None):
        if name.endswith("recall_provider"):
            raise ImportError("simulated recall_provider breakage")
        return real_import(name, package)

    monkeypatch.setattr(_il, "import_module", fake_import)

    b = MemoryBus()  # 不得抛

    assert "RecallProvider" in b._register_failures
    assert "recall" not in b._providers           # 其余 provider 照常
    assert "jiak" in b._providers                 # 至少一个兄弟注册成功

    h = b.health_check()
    assert h["_meta"]["status"] == "degraded"
    assert "RecallProvider" in h["_meta"]["register_failures"]

    recs = _read_records(trace_log)
    r_recs = [r for r in recs if r["body"] == "MemoryBus"
              and r["phase"] == "provider_register"]
    assert len(r_recs) == 1
    assert r_recs[0]["detail"] == "RecallProvider"


def test_all_providers_healthy_meta_ok(bus, trace_log):
    """无失败时 _meta.status == ok——health()的老消费者 .get('status') 不读到 None。"""
    bus.register(GoodProvider("good", priority=10))
    bus.query(Query(text="hello", top_k=5))
    h = bus.health_check()
    for name, prov_health in h.items():
        assert prov_health.get("status") in ("ok", "degraded", "error"), \
            f"{name}: every entry must carry a status"
    assert h["_meta"]["status"] == "ok"
    assert _read_records(trace_log) == []
