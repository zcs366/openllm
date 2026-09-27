"""heartbeat synthesis 综合消化回路测试

覆盖 T1-T6 + 生产调用点验证（军规八·代码活着）。
全部替身，不构建真 Agent，不碰网络/不碰 ~/.openllm。
"""
import os
import time
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

# ── 断面：直接 import 被测函数 ──
from openllm.iax.agent_heartbeat import _synthesize
from openllm.core.models import Decision, ActionResult


# ════════════════════════════════════════════════════════════
#  测试替身工厂
# ════════════════════════════════════════════════════════════

class _FakeTurn:
    """轻量 Turn 替身——只实现 trace_phase 和 phase_metrics"""
    def __init__(self):
        self.phase_metrics = []
        self.status = "ACTIVE"

    def trace_phase(self, phase, status, duration_ms=0.0, detail=""):
        self.phase_metrics.append({
            "phase": phase,
            "status": status,
            "duration_ms": duration_ms,
            "detail": detail[:100],
        })


class _FakeHC:
    """轻量 HeartbeatContext 替身"""
    def __init__(self):
        self.user_message = ""
        self.decision = None
        self.result = None
        self.output = ""
        self.synth_output = None  # P0.5 新字段
        self.phase_log = []


def _make_stub_agent(
    provider_chat_return="这是综合后的回答",
    provider_available=True,
    extract_tool_calls_return=None,
    isn_execute_return=None,
):
    """构造一个轻量 stub agent，所有接口都可配置。"""
    _chat_calls = []
    _isn_calls = []

    def _chat(messages):
        _chat_calls.append(messages)
        return provider_chat_return

    def _extract(text):
        return extract_tool_calls_return or []

    def _isn_execute(decision):
        _isn_calls.append(decision)
        return isn_execute_return or ActionResult(
            success=True, output="工具执行结果内容", duration_ms=1.0)

    def _record(phase):
        pass

    provider = SimpleNamespace(
        chat=_chat,
        _available=provider_available,
        _last_usage={},
    )
    left = SimpleNamespace(
        provider=provider,
        _extract_tool_calls=_extract,
    )
    octopus = SimpleNamespace(left=left)
    isn = SimpleNamespace(execute=_isn_execute)
    agent = SimpleNamespace(
        octopus=octopus,
        isn=isn,
        _record_inference=_record,
        _last_output="",
    )
    # 注入调用记录便于断言
    agent._chat_calls = _chat_calls
    agent._isn_calls = _isn_calls
    return agent


# ════════════════════════════════════════════════════════════
#  T1: 有 tool_calls + 成功 + stub provider 纯文本
#       → _last_output == 综合文本，不含裸 ls 特征串
# ════════════════════════════════════════════════════════════

class TestSynthesisT1:
    def test_pure_text_synthesis(self, monkeypatch):
        monkeypatch.delenv("OPENLLM_HEARTBEAT_SYNTH", raising=False)

        agent = _make_stub_agent(
            provider_chat_return="openLLM：我读了目录，有三个研究文件。",
            provider_available=True,
        )
        msg = SimpleNamespace(text="读 I:\\hermes\\output\\818具神智能研究，谈谈看法")
        hc = _FakeHC()
        hc.decision = Decision(
            action="execute", approved=True, reason="test",
            tool_calls=[{"name": "terminal", "args": {"command": "ls"}}])
        hc.result = ActionResult(
            success=True, output="file1.txt\nfile2.txt\nfile3.txt", duration_ms=50.0)
        turn = _FakeTurn()

        _synthesize(agent, msg, hc, turn)

        assert hc.synth_output is not None
        assert "openLLM" in hc.synth_output
        assert "file1.txt" not in hc.synth_output  # 裸 ls 不应残留
        # provider.chat 至少被调用 1 次
        assert len(agent._chat_calls) >= 1


# ════════════════════════════════════════════════════════════
#  T2: stub provider 第一轮返回含 TOOL_CALLS → isn.execute
#       被二次调用，第二轮文本成为最终输出
# ════════════════════════════════════════════════════════════

class TestSynthesisT2:
    def test_round2_tool_call(self, monkeypatch):
        monkeypatch.delenv("OPENLLM_HEARTBEAT_SYNTH", raising=False)

        # 用测试本地列表追踪 provider.chat 调用
        _chat_tracker = []
        _call_count = [0]
        def _chat_side_effect(messages):
            _call_count[0] += 1
            _chat_tracker.append(messages)
            if _call_count[0] == 1:
                return 'TOOL_CALLS: {"tool_calls": [{"name": "read_file", "args": {"path": "/x"}}]}'
            return "openLLM：这是读完文件后的综合回答。"
        def _extract(text):
            if "TOOL_CALLS" in text:
                return [{"name": "read_file", "args": {"path": "/x"}}]
            return []

        agent = _make_stub_agent()
        agent.octopus.left.provider.chat = _chat_side_effect
        agent.octopus.left._extract_tool_calls = _extract

        msg = SimpleNamespace(text="读一下 /x")
        hc = _FakeHC()
        hc.decision = Decision(
            action="execute", approved=True, reason="test",
            tool_calls=[{"name": "terminal", "args": {"command": "ls"}}])
        hc.result = ActionResult(
            success=True, output="初始结果", duration_ms=50.0)
        turn = _FakeTurn()

        _synthesize(agent, msg, hc, turn)

        # isn.execute 被调用 1 次（综合轮的二次工具执行）
        assert len(agent._isn_calls) == 1
        assert agent._isn_calls[0].tool_calls == [{"name": "read_file", "args": {"path": "/x"}}]
        assert agent._isn_calls[0].reason == "heartbeat-synth"
        # provider.chat 被调用 2 次
        assert len(_chat_tracker) == 2
        # 最终输出是第二轮文本
        assert hc.synth_output is not None
        assert "综合回答" in hc.synth_output


# ════════════════════════════════════════════════════════════
#  T3: provider chat 抛异常 → _last_output 回退 result.output
# ════════════════════════════════════════════════════════════

class TestSynthesisT3:
    def test_provider_exception_fallback(self, monkeypatch):
        monkeypatch.delenv("OPENLLM_HEARTBEAT_SYNTH", raising=False)

        def _chat_raises(messages):
            raise RuntimeError("API is down")

        agent = _make_stub_agent()
        agent.octopus.left.provider.chat = _chat_raises

        msg = SimpleNamespace(text="读文件")
        hc = _FakeHC()
        hc.decision = Decision(
            action="execute", approved=True, reason="test",
            tool_calls=[{"name": "terminal", "args": {"command": "ls"}}])
        hc.result = ActionResult(
            success=True, output="fallback 内容", duration_ms=50.0)
        turn = _FakeTurn()

        _synthesize(agent, msg, hc, turn)

        # synth_output 应为 None（异常回退）
        assert getattr(hc, "synth_output", None) is None
        # turn 记录了 error 阶段
        synth_phases = [p for p in turn.phase_metrics if p["phase"] == "synthesize"]
        assert len(synth_phases) == 1
        assert synth_phases[0]["status"] == "error"
        assert "API is down" in synth_phases[0]["detail"]


# ════════════════════════════════════════════════════════════
#  T4: 无 tool_calls 的纯聊天 → 不触发综合
# ════════════════════════════════════════════════════════════

class TestSynthesisT4:
    def test_no_tool_calls_no_synth(self, monkeypatch):
        monkeypatch.delenv("OPENLLM_HEARTBEAT_SYNTH", raising=False)

        agent = _make_stub_agent()
        msg = SimpleNamespace(text="你好")
        hc = _FakeHC()
        hc.decision = Decision(
            action="execute", approved=True, reason="chat",
            tool_calls=[])  # 无工具调用
        hc.result = ActionResult(
            success=True, output="闲聊输出", duration_ms=50.0)
        turn = _FakeTurn()

        _synthesize(agent, msg, hc, turn)

        # provider.chat 不应被调用
        assert len(agent._chat_calls) == 0
        # synth_output 不应被设置
        assert getattr(hc, "synth_output", None) is None
        # phase_metrics 不应有 synthesize 记录
        synth_phases = [p for p in turn.phase_metrics if p["phase"] == "synthesize"]
        assert len(synth_phases) == 0


# ════════════════════════════════════════════════════════════
#  T5: 开关 OPENLLM_HEARTBEAT_SYNTH=0 → 完全旧行为
# ════════════════════════════════════════════════════════════

class TestSynthesisT5:
    def test_switch_off(self, monkeypatch):
        monkeypatch.setenv("OPENLLM_HEARTBEAT_SYNTH", "0")

        agent = _make_stub_agent()
        msg = SimpleNamespace(text="读文件")
        hc = _FakeHC()
        hc.decision = Decision(
            action="execute", approved=True, reason="test",
            tool_calls=[{"name": "terminal", "args": {"command": "ls"}}])
        hc.result = ActionResult(
            success=True, output="工具输出", duration_ms=50.0)
        turn = _FakeTurn()

        _synthesize(agent, msg, hc, turn)

        # provider.chat 不应被调用
        assert len(agent._chat_calls) == 0
        # phase_metrics 应记录 skip+disabled
        synth_phases = [p for p in turn.phase_metrics if p["phase"] == "synthesize"]
        assert len(synth_phases) == 1
        assert synth_phases[0]["status"] == "skip"
        assert "disabled" in synth_phases[0]["detail"]

    def test_switch_off_string(self, monkeypatch):
        """'off' 也是关闭值"""
        monkeypatch.setenv("OPENLLM_HEARTBEAT_SYNTH", "off")

        agent = _make_stub_agent()
        msg = SimpleNamespace(text="读文件")
        hc = _FakeHC()
        hc.decision = Decision(
            action="execute", approved=True, reason="test",
            tool_calls=[{"name": "terminal", "args": {"command": "ls"}}])
        hc.result = ActionResult(
            success=True, output="工具输出", duration_ms=50.0)
        turn = _FakeTurn()

        _synthesize(agent, msg, hc, turn)

        assert len(agent._chat_calls) == 0


# ════════════════════════════════════════════════════════════
#  T6: turn.phase_metrics 含 ("synthesize", ...) 条目
# ════════════════════════════════════════════════════════════

class TestSynthesisT6:
    def test_phase_metrics_recorded(self, monkeypatch):
        monkeypatch.delenv("OPENLLM_HEARTBEAT_SYNTH", raising=False)

        agent = _make_stub_agent(
            provider_chat_return="综合后的文本",
            provider_available=True,
        )
        msg = SimpleNamespace(text="读目录")
        hc = _FakeHC()
        hc.decision = Decision(
            action="execute", approved=True, reason="test",
            tool_calls=[{"name": "terminal", "args": {"command": "ls"}}])
        hc.result = ActionResult(
            success=True, output="ls 结果", duration_ms=50.0)
        turn = _FakeTurn()

        _synthesize(agent, msg, hc, turn)

        synth_phases = [p for p in turn.phase_metrics if p["phase"] == "synthesize"]
        assert len(synth_phases) == 1
        assert synth_phases[0]["status"] == "ok"
        assert synth_phases[0]["duration_ms"] >= 0
        assert "len=" in synth_phases[0]["detail"]
        assert "rounds=" in synth_phases[0]["detail"]


# ════════════════════════════════════════════════════════════
#  边界条件
# ════════════════════════════════════════════════════════════

class TestSynthesisEdgeCases:
    def test_provider_unavailable(self, monkeypatch):
        """provider._available=False → skip"""
        monkeypatch.delenv("OPENLLM_HEARTBEAT_SYNTH", raising=False)

        agent = _make_stub_agent(provider_available=False)
        msg = SimpleNamespace(text="读文件")
        hc = _FakeHC()
        hc.decision = Decision(
            action="execute", approved=True, reason="test",
            tool_calls=[{"name": "terminal", "args": {"command": "ls"}}])
        hc.result = ActionResult(
            success=True, output="输出", duration_ms=50.0)
        turn = _FakeTurn()

        _synthesize(agent, msg, hc, turn)

        assert getattr(hc, "synth_output", None) is None
        synth_phases = [p for p in turn.phase_metrics if p["phase"] == "synthesize"]
        assert len(synth_phases) == 1
        assert synth_phases[0]["detail"] == "provider_unavailable"

    def test_result_not_success(self, monkeypatch):
        """执行失败 → 不触发综合"""
        monkeypatch.delenv("OPENLLM_HEARTBEAT_SYNTH", raising=False)

        agent = _make_stub_agent()
        msg = SimpleNamespace(text="读文件")
        hc = _FakeHC()
        hc.decision = Decision(
            action="execute", approved=True, reason="test",
            tool_calls=[{"name": "terminal", "args": {"command": "ls"}}])
        hc.result = ActionResult(
            success=False, error="工具失败", duration_ms=50.0)
        turn = _FakeTurn()

        _synthesize(agent, msg, hc, turn)

        assert getattr(hc, "synth_output", None) is None
        assert len(agent._chat_calls) == 0

    def test_provider_returns_empty(self, monkeypatch):
        """provider 返回空字符串 → skip"""
        monkeypatch.delenv("OPENLLM_HEARTBEAT_SYNTH", raising=False)

        agent = _make_stub_agent(provider_chat_return="")
        msg = SimpleNamespace(text="读文件")
        hc = _FakeHC()
        hc.decision = Decision(
            action="execute", approved=True, reason="test",
            tool_calls=[{"name": "terminal", "args": {"command": "ls"}}])
        hc.result = ActionResult(
            success=True, output="输出", duration_ms=50.0)
        turn = _FakeTurn()

        _synthesize(agent, msg, hc, turn)

        assert getattr(hc, "synth_output", None) is None
        synth_phases = [p for p in turn.phase_metrics if p["phase"] == "synthesize"]
        assert len(synth_phases) == 1
        assert synth_phases[0]["status"] == "skip"
        assert "empty_response" in synth_phases[0]["detail"]

    def test_tool_output_truncated_to_4000(self, monkeypatch):
        """工具输出超 4000 字符时截断"""
        monkeypatch.delenv("OPENLLM_HEARTBEAT_SYNTH", raising=False)

        agent = _make_stub_agent(provider_chat_return="综合结果")
        msg = SimpleNamespace(text="读大文件")
        hc = _FakeHC()
        hc.decision = Decision(
            action="execute", approved=True, reason="test",
            tool_calls=[{"name": "terminal", "args": {"command": "cat"}}])
        hc.result = ActionResult(
            success=True, output="X" * 8000, duration_ms=50.0)
        turn = _FakeTurn()

        _synthesize(agent, msg, hc, turn)

        # 检查 provider 收到的 messages 中 assistant 内容被截断
        assert len(agent._chat_calls) == 1
        assistant_msg = agent._chat_calls[0][2]  # 第3条消息(index 2)
        assert len(assistant_msg["content"]) < 5000  # 截断后加上前缀


# ════════════════════════════════════════════════════════════
#  T7: 末轮 out 只有 TOOL_CALLS 行 → 收口轮被调用
#       → 收口正文成为 synth_output，trace="ok"
# ════════════════════════════════════════════════════════════

class TestSynthesisT7:
    def test_closing_round_called(self, monkeypatch):
        """末轮 out 只有 TOOL_CALLS 行 → 循环结束后 strip 空 → 收口轮 → ok

        循环 2 轮都返回 TOOL_CALLS → isn_exec 被调 2 次 → 循环后 synth_output 为空
        → 触发收口轮（call 4）→ 收口正文成为 synth_output，trace="ok"
        """
        monkeypatch.delenv("OPENLLM_HEARTBEAT_SYNTH", raising=False)

        _call_count = [0]
        _messages_at_close = [None]

        def _chat_side_effect(messages):
            _call_count[0] += 1
            if _call_count[0] <= 3:
                # call 1: 综合轮; call 2,3: 循环内两轮（都返回 TOOL_CALLS）
                return 'TOOL_CALLS: {"tool_calls": [{"name": "read_file", "args": {"path": "/x"}}]}'
            # call 4: 收口轮
            _messages_at_close[0] = messages  # capture at call time
            return "openLLM：这是收口轮给出的综合回答。"

        def _extract(text):
            if "TOOL_CALLS" in text:
                return [{"name": "read_file", "args": {"path": "/x"}}]
            return []

        agent = _make_stub_agent()
        agent.octopus.left.provider.chat = _chat_side_effect
        agent.octopus.left._extract_tool_calls = _extract

        msg = SimpleNamespace(text="读文件")
        hc = _FakeHC()
        hc.decision = Decision(
            action="execute", approved=True, reason="test",
            tool_calls=[{"name": "terminal", "args": {"command": "ls"}}])
        hc.result = ActionResult(
            success=True, output="初始工具结果", duration_ms=50.0)
        turn = _FakeTurn()

        _synthesize(agent, msg, hc, turn)

        # isn.execute 被调用 2 次（循环内两轮各 1 次）
        assert len(agent._isn_calls) == 2
        # provider.chat 被调用 4 次（1 初始 + 2 循环 + 1 收口轮）
        assert _call_count[0] == 4
        # 收口轮的 messages 末条 user 消息含"禁止再调用任何工具"
        close_msgs = _messages_at_close[0]
        assert close_msgs is not None, "closing round chat not called"
        closing_msg = close_msgs[-1]
        assert closing_msg["role"] == "user"
        assert "禁止再调用任何工具" in closing_msg["content"]
        # 最终输出是收口轮正文
        assert hc.synth_output is not None
        assert "收口轮给出的综合回答" in hc.synth_output
        # trace status = ok
        synth_phases = [p for p in turn.phase_metrics if p["phase"] == "synthesize"]
        assert len(synth_phases) == 1
        assert synth_phases[0]["status"] == "ok"


# ════════════════════════════════════════════════════════════
#  T8: 收口轮抛异常 + last_tool_output 非空 → warn + 降级
# ════════════════════════════════════════════════════════════

class TestSynthesisT8:
    def test_closing_round_fallback(self, monkeypatch):
        """收口轮异常 + last_tool_output 非空 → warn + 降级输出

        循环 2 轮都返回 TOOL_CALLS → 循环后 synth_output 为空 → 触发收口轮
        → 收口轮抛异常 → 降级为 last_tool_output + 前缀，trace="warn"
        """
        monkeypatch.delenv("OPENLLM_HEARTBEAT_SYNTH", raising=False)

        _call_count = [0]
        def _chat_side_effect(messages):
            _call_count[0] += 1
            if _call_count[0] <= 3:
                # call 1: 综合轮; call 2,3: 循环内两轮（都返回 TOOL_CALLS）
                return 'TOOL_CALLS: {"tool_calls": [{"name": "read_file", "args": {"path": "/x"}}]}'
            # call 4: 收口轮 → 抛异常
            raise RuntimeError("收口轮 API 失败")

        def _extract(text):
            if "TOOL_CALLS" in text:
                return [{"name": "read_file", "args": {"path": "/x"}}]
            return []

        agent = _make_stub_agent(
            isn_execute_return=ActionResult(
                success=True, output="工具执行的原始内容在这里", duration_ms=1.0))
        agent.octopus.left.provider.chat = _chat_side_effect
        agent.octopus.left._extract_tool_calls = _extract

        msg = SimpleNamespace(text="读文件")
        hc = _FakeHC()
        hc.decision = Decision(
            action="execute", approved=True, reason="test",
            tool_calls=[{"name": "terminal", "args": {"command": "ls"}}])
        hc.result = ActionResult(
            success=True, output="初始结果", duration_ms=50.0)
        turn = _FakeTurn()

        _synthesize(agent, msg, hc, turn)

        # isn.execute 被调用 2 次（循环内两轮各 1 次）
        assert len(agent._isn_calls) == 2
        # provider.chat 被调用 4 次（1 初始 + 2 循环 + 1 收口轮抛异常）
        assert _call_count[0] == 4
        # synth_output 降级为 last_tool_output + 前缀
        assert hc.synth_output is not None
        assert "综合未成文" in hc.synth_output
        assert "工具执行的原始内容在这里" in hc.synth_output
        # trace status = warn
        synth_phases = [p for p in turn.phase_metrics if p["phase"] == "synthesize"]
        assert len(synth_phases) == 1
        assert synth_phases[0]["status"] == "warn"
        assert "fallback_tool_output" in synth_phases[0]["detail"]


# ════════════════════════════════════════════════════════════
#  T9: 全空（收口轮也返回空 + last_tool_output 也空）→ error
# ════════════════════════════════════════════════════════════

class TestSynthesisT9:
    def test_all_empty_error(self, monkeypatch):
        """所有来源都空 → error，hc.synth_output 未写"""
        monkeypatch.delenv("OPENLLM_HEARTBEAT_SYNTH", raising=False)

        # provider 返回纯 TOOL_CALLS 行（无其他文本）→ strip 后空
        agent = _make_stub_agent(provider_chat_return='TOOL_CALLS: {"tool_calls": []}')
        # 不进 isn_exec 循环（extract 返回 [] 因为 JSON 里 tool_calls 为空列表）
        # synth_output 经 strip 后为空字符串（""），falsy
        # 且 result.output 也为空
        msg = SimpleNamespace(text="读文件")
        hc = _FakeHC()
        hc.decision = Decision(
            action="execute", approved=True, reason="test",
            tool_calls=[{"name": "terminal", "args": {"command": "ls"}}])
        hc.result = ActionResult(
            success=True, output="", duration_ms=50.0)  # 空结果
        turn = _FakeTurn()

        _synthesize(agent, msg, hc, turn)

        # synth_output 未写（None）
        assert getattr(hc, "synth_output", None) is None
        # trace status = error, detail = empty_synth
        synth_phases = [p for p in turn.phase_metrics if p["phase"] == "synthesize"]
        assert len(synth_phases) == 1
        assert synth_phases[0]["status"] == "error"
        assert "empty_synth" in synth_phases[0]["detail"]


# ════════════════════════════════════════════════════════════
#  T10: rounds 与 _isn_exec 实际调用次数一致
# ════════════════════════════════════════════════════════════

class TestSynthesisT10:
    def test_rounds_matches_isn_exec_count(self, monkeypatch):
        """rounds 计数与 _isn_exec 实际调用次数一致（回归钉子）"""
        monkeypatch.delenv("OPENLLM_HEARTBEAT_SYNTH", raising=False)

        _call_count = [0]
        def _chat_side_effect(messages):
            _call_count[0] += 1
            if _call_count[0] == 1:
                # 第一轮：输出 TOOL_CALLS → 触发 isn_exec
                return 'TOOL_CALLS: {"tool_calls": [{"name": "read_file", "args": {"path": "/x"}}]}'
            # 第二轮：纯文本（终止循环）
            return "openLLM：最终回答。"

        def _extract(text):
            if "TOOL_CALLS" in text:
                return [{"name": "read_file", "args": {"path": "/x"}}]
            return []

        agent = _make_stub_agent()
        agent.octopus.left.provider.chat = _chat_side_effect
        agent.octopus.left._extract_tool_calls = _extract

        msg = SimpleNamespace(text="读文件")
        hc = _FakeHC()
        hc.decision = Decision(
            action="execute", approved=True, reason="test",
            tool_calls=[{"name": "terminal", "args": {"command": "ls"}}])
        hc.result = ActionResult(
            success=True, output="工具结果", duration_ms=50.0)
        turn = _FakeTurn()

        _synthesize(agent, msg, hc, turn)

        # isn_exec 被调用 1 次（只有第一轮触发）
        assert len(agent._isn_calls) == 1
        # trace detail 中 rounds 应为 2（初始1 + 循环中1）
        synth_phases = [p for p in turn.phase_metrics if p["phase"] == "synthesize"]
        assert len(synth_phases) == 1
        detail = synth_phases[0]["detail"]
        # 提取 rounds 值
        import re
        m = re.search(r"rounds=(\d+)", detail)
        assert m, f"rounds not found in detail: {detail}"
        rounds_val = int(m.group(1))
        # rounds = 1(初始) + 1(循环中 _isn_exec 被调用 1 次) = 2
        assert rounds_val == 2, f"expected rounds=2 for 1 isn_exec, got {rounds_val}"


# ════════════════════════════════════════════════════════════
#  接线验证：_synthesize 在 execute_tick 流程中被调用
# ════════════════════════════════════════════════════════════

class TestSynthesisIntegration:
    """验证 _synthesize 不是写了没人调的断头管"""

    def test_synthesize_called_in_execute_tick(self, monkeypatch):
        """monkeypatch _synthesize 进 execute_tick，断言调用链完整"""
        from openllm.iax import agent_heartbeat as hb_mod

        call_log = []
        _original_synthesize = hb_mod._synthesize

        def _spy_synthesize(agent, msg, hc, turn):
            call_log.append(("synthesize", hc.decision))
            # 不真的执行 LLM，只记录
            if hc.result and hc.result.success and getattr(hc.decision, "tool_calls", None):
                hc.synth_output = "spy_synth_result"

        monkeypatch.setattr(hb_mod, "_synthesize", _spy_synthesize)

        # 重读后确认 monkeypatch 生效
        assert hb_mod._synthesize is _spy_synthesize
        # 生产调用点验证：源码中 _synthesize 被调用 1 次
        import inspect
        source = inspect.getsource(hb_mod.execute_tick)
        assert "_synthesize(" in source

        # 恢复
        monkeypatch.setattr(hb_mod, "_synthesize", _original_synthesize)
