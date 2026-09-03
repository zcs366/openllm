"""test_session.py — 可靠会话状态机（P1-3 · ISA 工程宝石平移）测试

覆盖：完整生命周期 · 非法转移拒绝 · 超时重试升级 · 快照恢复 ·
event_bus 适配（会话事件发布）· 无 bus 纯本地行为。
"""
import json
import time

import pytest

from openllm.iai.session import (
    SessionManager, AgentSession, SessionStatus,
)
from openllm.iai.event_bus import EventBus, Event


def make_manager(tmp_path, **kw):
    kw.setdefault("sessions_dir", tmp_path / "sessions")
    return SessionManager(**kw)


def _g(mgr: SessionManager, sid: str) -> AgentSession:
    """get + 非 None 断言（类型收窄）。"""
    s = mgr.get(sid)
    assert s is not None
    return s


class TestLifecycle:
    def test_create_initiated(self, tmp_path):
        mgr = make_manager(tmp_path)
        s = mgr.create("A", "B", "sig-1")
        assert s.status == SessionStatus.INITIATED
        assert s.session_id.startswith("ses-")
        assert s.initiator == "A" and s.target == "B"
        assert _g(mgr, s.session_id) is s

    def test_full_lifecycle(self, tmp_path):
        mgr = make_manager(tmp_path)
        s = mgr.create("A", "B", "sig-1")
        mgr.set_ack(s.session_id, "ack-1")
        mgr.set_response(s.session_id, "resp-1")
        mgr.set_closed(s.session_id, "close-1")
        s = _g(mgr, s.session_id)
        assert s.status == SessionStatus.CLOSED
        assert s.ack_signal_id == "ack-1"
        assert s.response_signal_id == "resp-1"
        assert s.close_signal_id == "close-1"

    def test_illegal_transition_rejected(self, tmp_path):
        mgr = make_manager(tmp_path)
        s = mgr.create("A", "B", "sig-1")
        mgr.set_ack(s.session_id, "ack-1")
        mgr.set_response(s.session_id, "resp-1")
        mgr.set_closed(s.session_id, "close-1")
        # CLOSED 是终态——任何转移都非法
        with pytest.raises(ValueError, match="非法状态转移"):
            mgr.update_status(s.session_id, SessionStatus.ACKED)
        # ACKED 不能直接 CLOSED（跳步：须先 RESPONDED）
        s2 = mgr.create("A", "B", "sig-2")
        mgr.set_ack(s2.session_id, "ack-2")
        with pytest.raises(ValueError, match="非法状态转移"):
            mgr.update_status(s2.session_id, SessionStatus.CLOSED)
        # INITIATED 不能直接 RESPONDED（跳步：须先 ACK）
        s3 = mgr.create("A", "B", "sig-3")
        with pytest.raises(ValueError, match="非法状态转移"):
            mgr.update_status(s3.session_id, SessionStatus.RESPONDED)

    def test_update_nonexistent_raises(self, tmp_path):
        mgr = make_manager(tmp_path)
        with pytest.raises(KeyError):
            mgr.update_status("ses-nope", SessionStatus.ACKED)

    def test_increment_retry_resets_then_escalates(self, tmp_path):
        mgr = make_manager(tmp_path)
        s = mgr.create("A", "B", "sig-1")
        mgr.increment_retry(s.session_id)  # retry=1 < 3 → 重置 INITIATED
        assert _g(mgr, s.session_id).status == SessionStatus.INITIATED
        assert _g(mgr, s.session_id).retry_count == 1
        mgr.increment_retry(s.session_id)  # retry=2
        mgr.increment_retry(s.session_id)  # retry=3 >= 3 → ESCALATED
        assert _g(mgr, s.session_id).status == SessionStatus.ESCALATED


class TestTimeout:
    def test_timeout_retry_then_escalate(self, tmp_path):
        """sla=-1 → 立即超时：3 轮 timeout_check 后 ESCALATED。"""
        mgr = make_manager(tmp_path)
        s = mgr.create("A", "B", "sig-1", sla_seconds=-1)
        # 第 1 轮：retry 0→1，重置 INITIATED
        mgr.timeout_check()
        assert _g(mgr, s.session_id).retry_count == 1
        # 第 2 轮：retry 1→2
        mgr.timeout_check()
        assert _g(mgr, s.session_id).retry_count == 2
        # 第 3 轮：retry 2→3 → ESCALATED
        mgr.timeout_check()
        assert _g(mgr, s.session_id).status == SessionStatus.ESCALATED
        # 终态不再超时
        assert mgr.timeout_check() == []

    def test_timeout_returns_timed_out(self, tmp_path):
        mgr = make_manager(tmp_path)
        mgr.create("A", "B", "sig-1", sla_seconds=-1)
        out = mgr.timeout_check()
        assert len(out) == 1

    def test_acknowledged_not_timed_out_before_sla(self, tmp_path):
        mgr = make_manager(tmp_path)
        s = mgr.create("A", "B", "sig-1", sla_seconds=3600)
        mgr.set_ack(s.session_id, "ack-1")
        assert mgr.timeout_check() == []  # 未超时


class TestPersistence:
    def test_snapshot_reload(self, tmp_path):
        d = tmp_path / "sessions"
        mgr1 = SessionManager(sessions_dir=d)
        s = mgr1.create("A", "B", "sig-1")
        mgr1.set_ack(s.session_id, "ack-1")
        mgr1.set_response(s.session_id, "resp-1")
        # 重建 manager → 从快照恢复
        mgr2 = SessionManager(sessions_dir=d)
        restored = mgr2.get(s.session_id)
        assert restored is not None
        assert restored.status == SessionStatus.RESPONDED
        assert restored.response_signal_id == "resp-1"

    def test_snapshot_file_written(self, tmp_path):
        d = tmp_path / "sessions"
        mgr = SessionManager(sessions_dir=d)
        mgr.create("A", "B", "sig-1")
        lines = [l for l in (d / "sessions.jsonl").read_text().splitlines() if l.strip()]
        assert len(lines) == 1
        assert json.loads(lines[0])["status"] == "initiated"

    def test_stats(self, tmp_path):
        mgr = make_manager(tmp_path)
        mgr.create("A", "B", "sig-1")
        s2 = mgr.create("C", "D", "sig-2")
        mgr.set_ack(s2.session_id, "a2")
        mgr.set_response(s2.session_id, "r2")
        mgr.set_closed(s2.session_id, "c2")
        stats = mgr.get_stats()
        assert stats["total"] == 2
        assert stats["active"] == 1
        assert stats["by_status"]["initiated"] == 1
        assert stats["by_status"]["closed"] == 1


class TestEventBusAdapter:
    """平移新增：会话状态变更 → event_bus 事件（source=SESSION）。"""

    def _make_bus_and_mgr(self, tmp_path):
        bus = EventBus(log_dir=tmp_path / "events")
        mgr = SessionManager(sessions_dir=tmp_path / "sessions", bus=bus)
        return bus, mgr

    def test_create_publishes_event(self, tmp_path):
        bus, mgr = self._make_bus_and_mgr(tmp_path)
        got = []
        bus.subscribe(lambda e: got.append(e), source_filter="SESSION")
        s = mgr.create("A", "B", "sig-1")
        assert len(got) == 1
        assert got[0].type == "session.created"
        assert got[0].payload["session_id"] == s.session_id

    def test_state_changes_publish_events(self, tmp_path):
        bus, mgr = self._make_bus_and_mgr(tmp_path)
        got = []
        bus.subscribe(lambda e: got.append(e), source_filter="SESSION")
        s = mgr.create("A", "B", "sig-1")
        mgr.set_ack(s.session_id, "a-1")
        mgr.set_response(s.session_id, "r-1")
        mgr.set_closed(s.session_id, "c-1")
        types = [e.type for e in got]
        assert types == ["session.created", "session.acked",
                         "session.responded", "session.closed"]

    def test_increment_retry_publishes(self, tmp_path):
        bus, mgr = self._make_bus_and_mgr(tmp_path)
        got = []
        bus.subscribe(lambda e: got.append(e), source_filter="SESSION")
        s = mgr.create("A", "B", "sig-1")
        got.clear()
        mgr.increment_retry(s.session_id)
        assert got[-1].type == "session.initiated"  # 重置重发
        assert got[-1].payload["retry_count"] == 1

    def test_no_bus_pure_local(self, tmp_path):
        """bus=None → 行为与 ISA 原版一致（不发事件不崩）。"""
        mgr = make_manager(tmp_path)
        s = mgr.create("A", "B", "sig-1")
        mgr.set_ack(s.session_id, "a-1")
        assert _g(mgr, s.session_id).status == SessionStatus.ACKED

    def test_bus_failure_silent(self, tmp_path):
        """bus.publish 抛异常 → 会话操作不崩（事件是增强）。"""
        class BadBus:
            def publish(self, ev):
                raise RuntimeError("bus down")
        mgr = SessionManager(sessions_dir=tmp_path / "sessions", bus=BadBus())
        s = mgr.create("A", "B", "sig-1")  # 不应抛
        assert s.status == SessionStatus.INITIATED
