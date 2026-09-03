"""test_event_bus.py — IAI 事件总线测试"""
import json
import time
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from openllm.iai.event_bus import (
    EventBus, Event, Subscriber, BaseEventEmitter, SignalBridge,
)


# ── Event 数据类测试 ──

class TestEvent:
    def test_create_event(self):
        e = Event(source="IAX", type="heartbeat.ping", timestamp=time.time(),
                  entropy_score=0.1, payload={"seq": 1})
        assert e.source == "IAX"
        assert e.type == "heartbeat.ping"
        assert e.event_id.startswith("evt-")

    def test_event_to_dict(self):
        e = Event(source="ISA", type="memory.write", timestamp=1000.0,
                  entropy_score=0.5, payload={"key": "val"})
        d = e.to_dict()
        assert d["source"] == "ISA"
        assert d["timestamp"] == 1000.0
        assert d["payload"] == {"key": "val"}
        assert "event_id" in d

    def test_event_from_dict(self):
        d = {"source": "IOS", "type": "decision.request", "timestamp": 2000.0,
             "entropy_score": 0.3, "payload": {}, "event_id": "evt-test123"}
        e = Event.from_dict(d)
        assert e.source == "IOS"
        assert e.event_id == "evt-test123"

    def test_blocked_event_type(self):
        with pytest.raises(PermissionError, match="赫尔墨斯红线"):
            Event(source="ISN", type="terminate", timestamp=time.time(),
                  entropy_score=0.0, payload={})

    def test_blocked_shutdown(self):
        with pytest.raises(PermissionError, match="赫尔墨斯红线"):
            Event(source="IKO", type="shutdown", timestamp=time.time(),
                  entropy_score=0.0, payload={})

    def test_blocked_kill(self):
        with pytest.raises(PermissionError):
            Event(source="IAI", type="kill", timestamp=time.time(),
                  entropy_score=0.0, payload={})

    def test_blocked_abort(self):
        with pytest.raises(PermissionError):
            Event(source="IAX", type="abort", timestamp=time.time(),
                  entropy_score=0.0, payload={})

    def test_case_insensitive_block(self):
        with pytest.raises(PermissionError):
            Event(source="IAX", type="TERMINATE", timestamp=time.time(),
                  entropy_score=0.0, payload={})


# ── EventBus 核心测试 ──

class TestEventBus:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.bus = EventBus(log_dir=Path(self.tmp), ttl_hours=24)

    def test_publish_subscribe(self):
        received = []
        self.bus.subscribe(lambda e: received.append(e))
        e = Event(source="IAX", type="heartbeat.ping", timestamp=time.time(),
                  entropy_score=0.1, payload={})
        notified = self.bus.publish(e)
        assert notified == 1
        assert len(received) == 1
        assert received[0].source == "IAX"

    def test_source_filter(self):
        received = []
        self.bus.subscribe(lambda e: received.append(e), source_filter="IAX")
        self.bus.publish(Event("IAX", "ping", time.time(), 0.0, {}))
        self.bus.publish(Event("ISA", "write", time.time(), 0.0, {}))
        assert len(received) == 1

    def test_type_filter(self):
        received = []
        self.bus.subscribe(lambda e: received.append(e), type_filter="ping")
        self.bus.publish(Event("IAX", "ping", time.time(), 0.0, {}))
        self.bus.publish(Event("IAX", "pong", time.time(), 0.0, {}))
        assert len(received) == 1

    def test_dual_filter(self):
        received = []
        self.bus.subscribe(lambda e: received.append(e),
                           source_filter="IAX", type_filter="ping")
        self.bus.publish(Event("IAX", "ping", time.time(), 0.0, {}))
        self.bus.publish(Event("IAX", "pong", time.time(), 0.0, {}))
        self.bus.publish(Event("ISA", "ping", time.time(), 0.0, {}))
        assert len(received) == 1

    def test_unsubscribe(self):
        received = []
        sid = self.bus.subscribe(lambda e: received.append(e))
        self.bus.publish(Event("IAX", "ping", time.time(), 0.0, {}))
        assert len(received) == 1
        assert self.bus.unsubscribe(sid) is True
        self.bus.publish(Event("IAX", "ping", time.time(), 0.0, {}))
        assert len(received) == 1  # 不再收到

    def test_unsubscribe_nonexistent(self):
        assert self.bus.unsubscribe("nonexistent") is False

    def test_subscriber_count(self):
        assert self.bus.subscriber_count() == 0
        s1 = self.bus.subscribe(lambda e: None)
        s2 = self.bus.subscribe(lambda e: None)
        assert self.bus.subscriber_count() == 2
        self.bus.unsubscribe(s1)
        assert self.bus.subscriber_count() == 1

    def test_get_history(self):
        ts = time.time()
        self.bus.publish(Event("IAX", "ping", ts, 0.1, {}))
        self.bus.publish(Event("ISA", "write", ts, 0.2, {}))
        self.bus.publish(Event("IAX", "pong", ts, 0.3, {}))
        # 全部
        h = self.bus.get_history()
        assert len(h) == 3
        # source 过滤
        h = self.bus.get_history(source_filter="IAX")
        assert len(h) == 2
        # type 过滤
        h = self.bus.get_history(type_filter="write")
        assert len(h) == 1
        # limit
        h = self.bus.get_history(limit=2)
        assert len(h) == 2

    def test_callback_exception_handled(self):
        def bad_cb(e):
            raise RuntimeError("boom")
        self.bus.subscribe(bad_cb)
        # 不应抛出异常
        notified = self.bus.publish(
            Event("IAX", "ping", time.time(), 0.0, {}))
        assert notified == 0  # 异常不算成功通知


# ── JSONL 持久化测试 ──

class TestJsonlPersistence:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.bus = EventBus(log_dir=Path(self.tmp), ttl_hours=24)

    def test_jsonl_append(self):
        self.bus.publish(Event("IAX", "ping", time.time(), 0.1, {"a": 1}))
        self.bus.publish(Event("ISA", "write", time.time(), 0.2, {"b": 2}))
        log_file = Path(self.tmp) / "events.jsonl"
        assert log_file.exists()
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 2
        d = json.loads(lines[0])
        assert d["source"] == "IAX"

    def test_load_from_jsonl(self):
        # 先写入
        self.bus.publish(Event("IAX", "ping", time.time(), 0.1, {}))
        self.bus.publish(Event("ISA", "write", time.time(), 0.2, {}))
        # 新建 bus，从 JSONL 加载
        bus2 = EventBus(log_dir=Path(self.tmp), ttl_hours=24)
        loaded = bus2.load_from_jsonl()
        assert loaded == 2
        h = bus2.get_history()
        assert len(h) == 2

    def test_cleanup_expired(self):
        # 写一个过期事件（时间戳为0）
        self.bus.publish(Event("IAX", "old", 1.0, 0.0, {}))
        self.bus.publish(Event("IAX", "new", time.time(), 0.0, {}))
        cleaned = self.bus.cleanup_expired()
        assert cleaned >= 1
        h = self.bus.get_history()
        assert all(e["type"] != "old" for e in h)

    def test_no_log_file_returns_zero(self):
        empty_bus = EventBus(log_dir=Path(self.tmp) / "nonexistent", ttl_hours=24)
        assert empty_bus.load_from_jsonl() == 0


# ── BaseEventEmitter 测试 ──

class TestBaseEventEmitter:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.bus = EventBus(log_dir=Path(self.tmp), ttl_hours=24)

    def test_five_body_emit(self):
        """五体适配器：IAX/IAI/ISA/IOS/ISN/IKO 都能发射事件。"""
        bodies = ["IAX", "IAI", "ISA", "IOS", "ISN", "IKO"]
        received = []
        self.bus.subscribe(lambda e: received.append(e))
        for name in bodies:
            emitter = type(f"Emitter{name}", (BaseEventEmitter,),
                           {"BODY_NAME": name})(self.bus)
            emitter.emit_event("test.event", {"body": name}, entropy_score=0.5)
        assert len(received) == 6
        sources = {e.source for e in received}
        assert sources == set(bodies)

    def test_emit_event_returns_event(self):
        emitter = type("X", (BaseEventEmitter,), {"BODY_NAME": "IAX"})(self.bus)
        e = emitter.emit_event("ping", {"seq": 1}, entropy_score=0.1)
        assert isinstance(e, Event)
        assert e.source == "IAX"
        assert e.payload == {"seq": 1}


# ── SignalBridge 测试 ──

class TestSignalBridge:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.bus = EventBus(log_dir=Path(self.tmp), ttl_hours=24)

    def test_bridge_does_not_crash_without_signal(self):
        """signal.py 不可用时，桥接不应崩溃。"""
        bridge = SignalBridge(self.bus, poll_interval=0.1)
        bridge._poll()  # 直接调用 poll
        # 无信号导入时静默通过

    def test_bridge_converts_signal(self):
        """模拟 signal_recv 返回数据时桥接正确转换。"""
        mock_sigs = [{"signal_id": "sig-1", "type": "message",
                       "from": "A", "to": "B", "payload": {"x": 1}}]
        import openllm.iai.event_bus as eb
        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__
        # 用 mock 替代 signal_recv
        mock_signal = MagicMock()
        mock_signal.signal_recv = MagicMock(return_value=mock_sigs)
        import sys
        sys.modules["openllm.core.signal"] = mock_signal
        try:
            bridge = SignalBridge(self.bus, poll_interval=0.1)
            bridge._poll()
            h = self.bus.get_history()
            assert len(h) == 1
            assert h[0]["source"] == "SIGNAL"
            assert h[0]["type"] == "message"
        finally:
            del sys.modules["openllm.core.signal"]


# ── 语义路由测试（P0-2 · ISA通信体战役）──

class _StubMatcher:
    """可编程语义 stub：固定相似度；raise_error=True 时 is_match 抛异常。"""
    def __init__(self, sim: float = 0.0, raise_error: bool = False):
        self.sim = sim
        self.raise_error = raise_error
        self.calls: list[tuple] = []
    def is_match(self, interest, text, threshold):
        self.calls.append((interest, text, threshold))
        if self.raise_error:
            raise RuntimeError("embedding 故障模拟")
        return self.sim >= threshold


class _FakeEngine:
    """假 embedding 引擎：固定向量，供 SemanticMatcher 数学单测。"""
    def __init__(self, vec): self._vec = vec
    def encode(self, text): return self._vec


class TestSemanticRouting:
    """语义路由：精确=硬门槛，语义=相关性层，纯精确订阅行为不变。"""

    def _make_bus(self, matcher):
        return EventBus(log_dir=Path(tempfile.mkdtemp()), semantic_matcher=matcher)

    def _mk_event(self, source="IAI", etype="insight", payload=None):
        return Event(source=source, type=etype, timestamp=time.time(),
                     entropy_score=0.0,
                     payload=payload or {"text": "模型训练进展报告"},
                     brain_id="default")

    def test_semantic_hit_receives(self):
        m = _StubMatcher(sim=0.9)
        bus = self._make_bus(m)
        got = []
        bus.subscribe(lambda e: got.append(e), semantic_filter="模型训练",
                      semantic_threshold=0.5)
        bus.publish(self._mk_event())
        assert len(got) == 1
        assert m.calls[0][0] == "模型训练"  # 兴趣文本传入匹配器

    def test_semantic_miss_ignored(self):
        bus = self._make_bus(_StubMatcher(sim=0.2))
        got = []
        bus.subscribe(lambda e: got.append(e), semantic_filter="模型训练")
        bus.publish(self._mk_event())
        assert got == []

    def test_threshold_boundary(self):
        # 相似度 == 阈值 → 命中（>= 语义）
        bus = self._make_bus(_StubMatcher(sim=0.5))
        got = []
        bus.subscribe(lambda e: got.append(e), semantic_filter="x",
                      semantic_threshold=0.5)
        bus.publish(self._mk_event())
        assert len(got) == 1
        # 阈值抬高 → 不命中
        bus2 = self._make_bus(_StubMatcher(sim=0.5))
        got2 = []
        bus2.subscribe(lambda e: got2.append(e), semantic_filter="x",
                       semantic_threshold=0.6)
        bus2.publish(self._mk_event())
        assert got2 == []

    def test_pure_precise_subscription_unchanged(self):
        """无 semantic_filter：原精确行为不变（向后兼容回归）。"""
        bus = self._make_bus(_StubMatcher(sim=0.9))  # 即使 sim 高也不触发语义
        got = []
        bus.subscribe(lambda e: got.append(e), source_filter="IAI")
        bus.subscribe(lambda e: got.append(e), source_filter="ISN")
        bus.publish(self._mk_event(source="IAI"))
        assert len(got) == 1

    def test_source_hard_gate_overrides_semantic(self):
        """精确过滤器是硬门槛：source 不匹配时语义再高也不通知。"""
        bus = self._make_bus(_StubMatcher(sim=0.99))
        got = []
        bus.subscribe(lambda e: got.append(e), source_filter="ISN",
                      semantic_filter="模型训练")
        bus.publish(self._mk_event(source="IAI"))  # 语义相关但 source 不符
        assert got == []

    def test_semantic_exception_degrades_safely(self):
        """语义判定抛异常 → publish 不崩溃、订阅者不被通知。"""
        bus = self._make_bus(_StubMatcher(sim=0.9, raise_error=True))
        got = []
        bus.subscribe(lambda e: got.append(e), semantic_filter="模型训练")
        bus.publish(self._mk_event())  # 不应抛
        assert got == []

    def test_payload_text_extraction_priority(self):
        from openllm.iai.event_semantic import event_payload_to_text
        assert event_payload_to_text({"text": "a", "body": "b"}) == "a"
        assert event_payload_to_text({"body": "b", "content": "c"}) == "b"
        assert event_payload_to_text({"message": "m"}) == "m"
        assert event_payload_to_text({"note": "n"}) == '{"note": "n"}'

    def test_semantic_matcher_math_and_degrade(self):
        """SemanticMatcher：cosine 数学 + 零向量 + encode 异常降级 None。"""
        from openllm.iai.event_semantic import SemanticMatcher
        m = SemanticMatcher()
        m._engine = _FakeEngine([1.0, 0.0, 0.0])
        assert m.similarity("a", "b") == pytest.approx(1.0)
        assert m.is_match("a", "b", 0.5) is True

        m2 = SemanticMatcher()
        m2._engine = _FakeEngine([0.0, 0.0])
        assert m2.similarity("a", "b") == 0.0

        class _Boom:
            def encode(self, text): raise RuntimeError("no model")
        m3 = SemanticMatcher()
        m3._engine = _Boom()
        assert m3.similarity("a", "b") is None
        assert m3.is_match("a", "b", 0.5) is False

    def test_matcher_import_missing_fails_closed(self):
        """SemanticMatcher 不可用（import 失败）→ 语义订阅永不命中但不崩。"""
        import openllm.iai.event_bus as eb_module
        orig = eb_module.SemanticMatcher
        eb_module.SemanticMatcher = None
        try:
            bus = EventBus(log_dir=Path(tempfile.mkdtemp()))
            got = []
            bus.subscribe(lambda e: got.append(e), semantic_filter="模型训练")
            bus.publish(self._mk_event())
            assert got == []
        finally:
            eb_module.SemanticMatcher = orig
