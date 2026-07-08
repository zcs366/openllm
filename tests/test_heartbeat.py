"""heartbeat.py 单元测试

覆盖：
  - 状态机转换合法性
  - 事件分发
  - 超时监控（看门狗）
  - 红线验证（心跳无终止对话权限）
  - 生命周期（start/stop）
  - 边界条件
"""
import time
import pytest
from openllm.core.heartbeat import (
    Heartbeat,
    HeartbeatState,
    HeartbeatEvent,
)


# ── 基础状态机 ──────────────────────────────────────

class TestStateMachine:
    def test_initial_state(self):
        hb = Heartbeat()
        assert hb.state == HeartbeatState.IDLE
        assert hb.tick_count == 0
        assert hb.uptime is None

    def test_tick_advances_through_states(self):
        hb = Heartbeat()
        hb.start()
        # IDLE → tick() cycles through CHECKING→DISPATCHING→MONITORING→IDLE
        result = hb.tick()
        assert result["tick"] == 1
        assert hb.state == HeartbeatState.IDLE  # returns to IDLE
        assert result["events_dispatched"] == 0
        assert result["timeouts"] == []

    def test_invalid_transition_raises(self):
        hb = Heartbeat()
        with pytest.raises(ValueError, match="Invalid transition"):
            hb._transition(HeartbeatState.DISPATCHING)  # IDLE→DISPATCHING illegal

    def test_stopped_cannot_tick(self):
        hb = Heartbeat()
        hb.start()
        hb.stop()
        result = hb.tick()
        assert result["state"] == "STOPPED"
        assert hb.state == HeartbeatState.STOPPED

    def test_multiple_ticks(self):
        hb = Heartbeat()
        hb.start()
        for i in range(5):
            hb.tick()
        assert hb.tick_count == 5

    def test_uptime(self):
        hb = Heartbeat()
        assert hb.uptime is None
        hb.start()
        time.sleep(0.01)
        assert hb.uptime is not None
        assert hb.uptime > 0


# ── 事件分发 ────────────────────────────────────────

class TestEventDispatch:
    def test_enqueue_and_dispatch(self):
        hb = Heartbeat()
        hb.start()
        received = []
        hb.register_handler("analysis", lambda e: received.append(e))

        hb.enqueue(HeartbeatEvent("analysis", "IAI", {"result": "ok"}))
        result = hb.tick()
        assert result["events_dispatched"] == 1
        assert len(received) == 1
        assert received[0].payload["result"] == "ok"

    def test_unhandled_event_no_crash(self):
        hb = Heartbeat()
        hb.start()
        hb.enqueue(HeartbeatEvent("unknown", "X"))
        result = hb.tick()
        assert result["events_dispatched"] == 0

    def test_handler_error_does_not_crash(self):
        hb = Heartbeat()
        hb.start()
        def bad_handler(e):
            1 / 0
        hb.register_handler("bad", bad_handler)
        hb.enqueue(HeartbeatEvent("bad", "X"))
        result = hb.tick()
        assert result["events_dispatched"] == 0  # error counted but not dispatched

    def test_fIFO_order(self):
        hb = Heartbeat()
        hb.start()
        order = []
        hb.register_handler("e", lambda e: order.append(e.source))
        hb.enqueue(HeartbeatEvent("e", "first"))
        hb.enqueue(HeartbeatEvent("e", "second"))
        hb.tick()
        assert order == ["first", "second"]

    def test_multiple_event_types(self):
        hb = Heartbeat()
        hb.start()
        results = {"a": [], "b": []}
        hb.register_handler("a", lambda e: results["a"].append(e))
        hb.register_handler("b", lambda e: results["b"].append(e))
        hb.enqueue(HeartbeatEvent("a", "X"))
        hb.enqueue(HeartbeatEvent("b", "X"))
        hb.enqueue(HeartbeatEvent("a", "X"))
        hb.tick()
        assert len(results["a"]) == 2
        assert len(results["b"]) == 1


# ── 超时监控 ────────────────────────────────────────

class TestTimeout:
    def test_no_timeout_when_fresh(self):
        hb = Heartbeat(timeout_ms=100)
        hb.start()
        hb.register_body("ISA")
        result = hb.tick()
        assert result["timeouts"] == []

    def test_timeout_detected(self):
        hb = Heartbeat(timeout_ms=1)  # 1ms timeout
        hb.start()
        hb.register_body("ISA")
        time.sleep(0.01)  # wait > 1ms
        result = hb.tick()
        assert "ISA" in result["timeouts"]

    def test_heartbeat_from_refreshes(self):
        hb = Heartbeat(timeout_ms=50)
        hb.start()
        hb.register_body("ISA")
        time.sleep(0.02)
        hb.heartbeat_from("ISA")  # refresh
        result = hb.tick()
        assert "ISA" not in result["timeouts"]

    def test_timeout_does_not_terminate(self):
        """红线验证：超时后系统仍在运行。"""
        hb = Heartbeat(timeout_ms=1)
        hb.start()
        hb.register_body("IO-S")
        time.sleep(0.01)
        hb.tick()
        assert hb.is_running  # still alive


# ── 红线验证 ────────────────────────────────────────

class TestRedLine:
    def test_no_terminate_permission(self):
        """心跳API中没有任何终止对话的方法。"""
        hb = Heartbeat()
        public = [m for m in dir(hb) if not m.startswith("_")]
        terminate_names = {"terminate", "kill", "abort", "shutdown",
                           "end_conversation", "stop_conversation"}
        exposed = set(public) & terminate_names
        assert not exposed, f"Heartbeat exposes terminate-like methods: {exposed}"

    def test_stop_only_stops_heartbeat(self):
        """stop()只停心跳，不终止对话。"""
        hb = Heartbeat()
        hb.start()
        hb.stop()
        assert hb.state == HeartbeatState.STOPPED
        assert hb.tick_count == 0  # no side effects


# ── 生命周期 ────────────────────────────────────────

class TestLifecycle:
    def test_start_idempotent(self):
        hb = Heartbeat()
        hb.start()
        hb.start()  # no-op
        assert hb.is_running

    def test_stop_idempotent(self):
        hb = Heartbeat()
        hb.stop()
        hb.stop()  # no-op
        assert hb.state == HeartbeatState.STOPPED

    def test_stopped_restart_raises(self):
        hb = Heartbeat()
        hb.start()
        hb.stop()
        with pytest.raises(RuntimeError, match="stopped"):
            hb.start()

    def test_is_running(self):
        hb = Heartbeat()
        assert hb.is_running  # IDLE is running
        hb.stop()
        assert not hb.is_running


# ── 边界条件 ────────────────────────────────────────

class TestEdgeCases:
    def test_invalid_interval(self):
        with pytest.raises(ValueError, match="positive"):
            Heartbeat(interval_ms=0)
        with pytest.raises(ValueError, match="positive"):
            Heartbeat(interval_ms=-10)

    def test_event_queue_cleared_after_tick(self):
        hb = Heartbeat()
        hb.start()
        hb.enqueue(HeartbeatEvent("x", "X"))
        hb.tick()
        # Queue is empty after tick
        result = hb.tick()
        assert result["events_dispatched"] == 0

    def test_elapsed_ms_recorded(self):
        hb = Heartbeat()
        hb.start()
        result = hb.tick()
        assert result["elapsed_ms"] >= 0
