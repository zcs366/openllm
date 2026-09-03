"""test_comm.py — AgentComm 通信栈（多agent可靠通信落地）测试

覆盖：tell 送达 · ask/reply 可靠会话全状态流 · 裁决 BLOCK 拦截 ·
broadcast 语义路由（embedding stub / 关键词降级）· 事件发布 ·
目标缺失 · 超时检查 · 统计。
"""
import time
from pathlib import Path

import pytest

from openllm.comm import Registry, CommAgent, CommMessage, SendResult
from openllm.iai.session import SessionManager, SessionStatus
from openllm.iai.event_bus import EventBus


class FakeSemantic:
    """broadcast 语义路由 stub：按兴趣判定（interest 匹配 → 0.9，否则 0.1）。"""
    def similarity(self, interest: str, body: str):
        return 0.9 if interest in ("模型训练", "模型") else 0.1


def make_env(tmp_path) -> tuple[Registry, SessionManager, EventBus]:
    """共享基础设施的通信环境（sessions/bus 恒建，全测试共享）。"""
    sessions = SessionManager(sessions_dir=tmp_path / "sessions")
    bus = EventBus(log_dir=tmp_path / "events")
    reg = Registry(sessions=sessions, bus=bus)
    return reg, sessions, bus


def make_agent(reg, aid, tmp_path, **kw):
    kw.setdefault("log_dir", tmp_path / "agents" / aid)
    return CommAgent(aid, registry=reg, **kw)


class TestTell:
    def test_send_receive(self, tmp_path):
        reg, _, _ = make_env(tmp_path)
        a = make_agent(reg, "IAI", tmp_path)
        b = make_agent(reg, "ISN", tmp_path)
        got = []
        b.on_message(lambda agent, msg: (got.append(msg.body), None)[1])
        r = a.send("ISN", "技能库更新了", kind="tell")
        assert r.ok and r.verdict_action == "PASS"
        assert b.stats()["inbox"] == 1
        b.run_once()
        assert got == ["技能库更新了"]
        assert b.stats()["received"] == 1

    def test_target_missing(self, tmp_path):
        reg, _, _ = make_env(tmp_path)
        a = make_agent(reg, "IAI", tmp_path)
        r = a.send("不存在", "hi")
        assert not r.ok
        assert "不存在" in r.reason

    def test_registry_duplicate_rejected(self, tmp_path):
        reg, _, _ = make_env(tmp_path)
        make_agent(reg, "X", tmp_path)
        with pytest.raises(ValueError):
            make_agent(reg, "X", tmp_path)


class TestReliableSession:
    def test_ask_reply_session_flow(self, tmp_path):
        """ask → ack → handler reply → respond → 发起方 close，全状态流。"""
        reg, sessions, _ = make_env(tmp_path)
        a = make_agent(reg, "IAI", tmp_path)
        b = make_agent(reg, "ISN", tmp_path)
        got = []
        a.on_message(lambda agent, msg: (got.append(msg.body), None)[1])
        b.on_message(lambda agent, msg: f"回答:{msg.body}")
        r = a.send("ISN", "今天天气如何", kind="ask")
        assert r.ok and r.session_id
        # 发方会话 INITIATED
        s0 = sessions.get(r.session_id)
        assert s0 is not None and s0.status == SessionStatus.INITIATED
        # B 收：auto ack → handler → auto reply
        b.run_once()
        # 此时 session ACKED（B ack 后 handler 触发 reply → respond）
        s1 = sessions.get(r.session_id)
        assert s1 is not None and s1.status == SessionStatus.RESPONDED
        assert b.stats()["replied"] == 1
        # A 收 reply → 发起方 close
        a.run_once()
        s2 = sessions.get(r.session_id)
        assert s2 is not None and s2.status == SessionStatus.CLOSED
        # A 已收到回复内容（handler 在 send 前注册）
        assert got == ["回答:今天天气如何"]

    def test_tell_no_reply_no_close(self, tmp_path):
        """tell 无 handler 回复：会话停在 ACKED（不误 close）。"""
        reg, sessions, _ = make_env(tmp_path)
        a = make_agent(reg, "IAI", tmp_path)
        b = make_agent(reg, "ISN", tmp_path)
        r = a.send("ISN", "纯通知", kind="tell")
        b.run_once()
        s = sessions.get(r.session_id)
        assert s is not None and s.status == SessionStatus.ACKED


class TestAdjudication:
    def test_send_blocked_by_blacklist(self, tmp_path):
        """裁决 BLOCK：黑名单禁止 A→B → send 拒绝。"""
        from openllm.governance.adjudication import TransmissionAdjudicator
        adj = TransmissionAdjudicator(log_dir=tmp_path / "adj")
        adj.block_pair("IAI", "ISN")
        sessions = SessionManager(sessions_dir=tmp_path / "sessions")
        bus = EventBus(log_dir=tmp_path / "events")
        reg = Registry(sessions=sessions, bus=bus, adjudicator=adj)
        a = make_agent(reg, "IAI", tmp_path)
        make_agent(reg, "ISN", tmp_path)
        r = a.send("ISN", "hi")
        assert not r.ok
        assert r.verdict_action == "BLOCK"
        assert a.stats()["blocked"] == 1

    def test_redline_body_blocked(self, tmp_path):
        """内容含红线/凭据 → BLOCK。"""
        reg, _, _ = make_env(tmp_path)
        a = make_agent(reg, "IAI", tmp_path)
        make_agent(reg, "ISN", tmp_path)
        r = a.send("ISN", "给你个 key: sk-abcdef1234567890ghijklmnopq")
        assert not r.ok
        assert r.verdict_action == "BLOCK"


class TestBroadcast:
    def test_semantic_routing_embedding(self, tmp_path):
        """broadcast 语义路由：兴趣匹配者收，无关者不收（embedding stub）。"""
        reg, _, _ = make_env(tmp_path)
        a = make_agent(reg, "IO", tmp_path)
        b = make_agent(reg, "IAI", tmp_path, interests=["模型训练"])
        c = make_agent(reg, "IKO", tmp_path, interests=["输出排版"])
        # 注入语义 stub（避免真实模型加载 ~7s）
        a._semantic_engine = FakeSemantic()
        bgot, cgot = [], []
        b.on_message(lambda ag, m: (bgot.append(m.body), None)[1])
        c.on_message(lambda ag, m: (cgot.append(m.body), None)[1])
        results = a.broadcast("今天用LoRA微调了7B模型，效果不错")
        # 只发给兴趣含"模型"语义的 IAI
        assert len(results) == 1 and results[0].ok
        assert results[0].message is not None and results[0].message.to_id == "IAI"
        b.run_once(); c.run_once()
        assert bgot and not cgot

    def test_broadcast_keyword_fallback(self, tmp_path, monkeypatch):
        """无 embedding（引擎不可用）→ 关键词降级匹配。"""
        reg, _, _ = make_env(tmp_path)
        a = make_agent(reg, "IO", tmp_path)
        make_agent(reg, "IAI", tmp_path, interests=["模型"])
        make_agent(reg, "IKO", tmp_path, interests=["排版"])
        # 模拟语义引擎不可用（property 返回 None → 关键词降级）
        monkeypatch.setattr(CommAgent, "_semantic", None)
        results = a.broadcast("模型训练进展报告")
        assert len(results) == 1 and results[0].ok
        assert results[0].message is not None and results[0].message.to_id == "IAI"

    def test_broadcast_all_when_not_semantic(self, tmp_path):
        reg, _, _ = make_env(tmp_path)
        a = make_agent(reg, "IO", tmp_path)
        make_agent(reg, "IAI", tmp_path)
        make_agent(reg, "IKO", tmp_path)
        results = a.broadcast("全员通知", semantic=False)
        assert len(results) == 2


class TestEventsAndWatchdog:
    def test_comm_events_published(self, tmp_path):
        reg, sessions, bus = make_env(tmp_path)
        a = make_agent(reg, "IAI", tmp_path)
        make_agent(reg, "ISN", tmp_path)
        got = []
        bus.subscribe(lambda e: got.append(e), source_filter="IAI")
        a.send("ISN", "事件测试")
        types = [e.type for e in got]
        assert "comm.sent" in types

    def test_timeout_check(self, tmp_path):
        """sla=-1 → check_timeouts 触发超时。"""
        reg, sessions, _ = make_env(tmp_path)
        a = make_agent(reg, "IAI", tmp_path)
        make_agent(reg, "ISN", tmp_path)
        r = a.send("ISN", "hi")
        # 直接建 sla=-1 的会话
        s = sessions.get(r.session_id)
        assert s is not None
        s.sla_seconds = -1
        s.last_update = 0  # 强制超时
        assert a.check_timeouts() >= 1

    def test_stats(self, tmp_path):
        reg, _, _ = make_env(tmp_path)
        a = make_agent(reg, "IAI", tmp_path)
        assert a.stats()["sent"] == 0
        a.send("NOPE", "x")  # 目标缺失不计 sent
        assert a.stats()["sent"] == 0
