"""
SALIENCE_GATE 事件 + 连续性门控测试（十律⑥·蜻蜓门控·PAL T-F-2）

验证目标：
  1. IAI.gate() 发布 salience.gate 事件，payload 字段完整
  2. 订阅者能收到 salience.gate 事件
  3. blocked=True/False 切换正确
  4. 不破坏现有5类事件流（回归）
  5. 心跳自动发布门控事件（任务切换时 blocked=True，完成时 blocked=False）
"""
import time
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from openllm.iai.core import IAI
from openllm.iai.event_bus import Event


# ── IAI.gate() 单元测试 ──────────────────────────────────

class TestIAIGate:
    """IAI.gate() 方法的基础测试。"""

    def test_gate_publishes_event(self):
        """gate() 发布 salience.gate 事件。"""
        iai = IAI()
        count = iai.gate("task_switch", "main_task_1")
        assert count >= 0  # 无订阅者也返回0
        history = iai.bus.get_history(type_filter="salience.gate")
        assert len(history) == 1
        assert history[0]["type"] == "salience.gate"

    def test_gate_payload_fields(self):
        """gate() 的 payload 包含所有必填字段。"""
        iai = IAI()
        iai.gate("user_message", "user_msg_target")
        history = iai.bus.get_history(type_filter="salience.gate")
        payload = history[0]["payload"]

        assert payload["interruption_type"] == "user_message"
        assert payload["target"] == "user_msg_target"
        assert payload["blocked"] is True  # default
        assert isinstance(payload["timestamp"], float)
        assert payload["timestamp"] > 0

    def test_gate_blocked_true(self):
        """blocked=True（默认）：阻断干扰。"""
        iai = IAI()
        iai.gate("signal", "heartbeat_task", blocked=True)
        history = iai.bus.get_history(type_filter="salience.gate")
        assert history[0]["payload"]["blocked"] is True

    def test_gate_blocked_false(self):
        """blocked=False：允许切换。"""
        iai = IAI()
        iai.gate("task_switch", "heartbeat_task", blocked=False)
        history = iai.bus.get_history(type_filter="salience.gate")
        assert history[0]["payload"]["blocked"] is False

    def test_gate_source_brain_optional(self):
        """source_brain 可选——不传时 payload 中无此字段。"""
        iai = IAI()
        iai.gate("task_switch", "task1")
        payload = iai.bus.get_history(type_filter="salience.gate")[0]["payload"]
        assert "source_brain" not in payload

    def test_gate_source_brain_present(self):
        """传 source_brain 时 payload 包含该字段。"""
        iai = IAI()
        iai.gate("task_switch", "task1", source_brain="octopus_left")
        payload = iai.bus.get_history(type_filter="salience.gate")[0]["payload"]
        assert payload["source_brain"] == "octopus_left"

    def test_gate_source_is_iai_gate(self):
        """事件 source 字段为 'iai.gate'。"""
        iai = IAI()
        iai.gate("task_switch", "task1")
        history = iai.bus.get_history(type_filter="salience.gate")
        assert history[0]["source"] == "iai.gate"


# ── 订阅者接收测试 ──────────────────────────────────────

class TestSalienceGateSubscriber:
    """订阅者能正确接收 salience.gate 事件。"""

    def test_subscriber_receives_gate(self):
        """订阅 type_filter='salience.gate' 的回调被触发。"""
        iai = IAI()
        received = []
        iai.subscribe(lambda e: received.append(e),
                      type_filter="salience.gate")
        iai.gate("user_message", "task_a")

        assert len(received) == 1
        assert received[0].type == "salience.gate"
        assert received[0].payload["interruption_type"] == "user_message"
        assert received[0].payload["target"] == "task_a"

    def test_unrelated_subscriber_not_triggered(self):
        """type_filter 不匹配的订阅者不会收到 salience.gate。"""
        iai = IAI()
        received = []
        iai.subscribe(lambda e: received.append(e),
                      type_filter="context.built")
        iai.gate("task_switch", "task1")
        assert len(received) == 0

    def test_unfiltered_subscriber_receives_gate(self):
        """无 type_filter 的订阅者也能收到 salience.gate。"""
        iai = IAI()
        received = []
        iai.subscribe(lambda e: received.append(e))  # no filter
        iai.gate("signal", "heartbeat")
        assert len(received) == 1
        assert received[0].type == "salience.gate"


# ── blocked 切换测试 ─────────────────────────────────────

class TestBlockedSwitching:
    """blocked=True/False 切换正确反映任务状态。"""

    def test_blocked_true_then_false(self):
        """先 blocked=True 阻断，再 blocked=False 放行。"""
        iai = IAI()
        received = []
        iai.subscribe(lambda e: received.append(e),
                      type_filter="salience.gate")

        iai.gate("task_switch", "task_alpha", blocked=True)
        iai.gate("task_switch", "task_alpha", blocked=False)

        assert len(received) == 2
        assert received[0].payload["blocked"] is True
        assert received[1].payload["blocked"] is False

    def test_multiple_tasks_independent(self):
        """不同任务的门控事件独立。"""
        iai = IAI()
        received = []
        iai.subscribe(lambda e: received.append(e),
                      type_filter="salience.gate")

        iai.gate("task_switch", "task_a", blocked=True)
        iai.gate("user_message", "task_b", blocked=False)
        iai.gate("signal", "task_c", blocked=True)

        assert len(received) == 3
        targets = [e.payload["target"] for e in received]
        assert targets == ["task_a", "task_b", "task_c"]
        blocked_vals = [e.payload["blocked"] for e in received]
        assert blocked_vals == [True, False, True]


# ── 不破坏现有事件流（回归）─────────────────────────────

class TestSalienceGateCoexistence:
    """salience.gate 不干扰现有5类事件。"""

    def test_gate_does_not_pollute_5_events(self):
        """gate() 调用不会在5类事件中产生误报。"""
        iai = IAI()
        iai.gate("task_switch", "task1")
        iai.gate("user_message", "task2", blocked=False)

        history = iai.bus.get_history()
        all_types = [e["type"] for e in history]
        assert all(t == "salience.gate" for t in all_types)
        # 确保没有5类事件被意外触发
        for evt_type in ["context.built", "reasoning.proposed",
                         "reasoning.critiqued", "decision.made",
                         "action.executed"]:
            assert evt_type not in all_types, (
                f"gate() 不应触发 {evt_type} 事件"
            )

    def test_bus_history_mixed_events(self):
        """混合5类事件 + salience.gate 的 history 一致性。"""
        iai = IAI()
        # 手动发布一些5类事件
        iai.emit("context.built", {"user_message": "hi"})
        iai.gate("task_switch", "task1")
        iai.emit("decision.made", {"action": "run"})
        iai.gate("task_switch", "task1", blocked=False)

        history = iai.bus.get_history()
        types_seen = [e["type"] for e in history]
        assert "context.built" in types_seen
        assert "decision.made" in types_seen
        assert types_seen.count("salience.gate") == 2


# ── 心跳集成测试（需 mock Agent）─────────────────────────

def _make_mock_agent():
    """创建 mock Agent 用于心跳集成测试。"""
    from openllm.core.models import (
        Message, Context, Prediction, RiskAssessment,
        Proposal, Critique, Decision, ActionResult, CausalDelta,
    )

    with patch("openllm.core.main_loop.Agent._print_startup_health"):
        from openllm.core.main_loop import Agent
        agent = Agent(mode="console")

    # Mock 章鱼I
    mock_proposal = Proposal(content="test", confidence=0.9, evidence=[])
    mock_critique = Critique(content="test", verdict="approve", concerns=[])
    agent.octopus.reason = MagicMock(return_value=(mock_proposal, mock_critique))
    agent.octopus.predict_consequences = MagicMock(return_value=Prediction(
        summary="test", consequences=[], confidence=0.7))
    agent.octopus.d0_snapshot = MagicMock(return_value={})
    agent.octopus.compare = MagicMock(return_value=CausalDelta(
        prediction_match=True, delta_summary="ok", learned=[]))
    mock_tentacles = MagicMock()
    mock_tentacles.get.return_value = None
    agent.octopus.tentacles = mock_tentacles

    # Mock ISN/IOS
    agent.isn.execute = MagicMock(return_value=ActionResult(
        success=True, output="ok", duration_ms=1.0))
    agent.ios.risk_check = MagicMock(return_value=RiskAssessment(level="low", blocked=False))
    agent.ios.arbitrate = MagicMock(return_value=Decision(
        action="execute", approved=True, reason="test"))
    agent.ios.learn_causal = MagicMock()
    agent.ios.evolve = MagicMock()
    agent.ios.recover = MagicMock(return_value=False)
    agent.ios.cap_check = MagicMock(return_value=True)
    agent.ios.governance_engine = MagicMock()
    agent.ios.governance_engine.heartbeat_trace = MagicMock()

    # Mock ISA/Session/IKO
    mock_ctx = Context(user_message="test")
    agent.isa.build_context = MagicMock(return_value=mock_ctx)
    agent.isa.respond = MagicMock()
    agent.session.compact_if_needed = MagicMock()
    agent.session.checkpoint = MagicMock()
    agent.iko.process_output = MagicMock(return_value="processed")

    # Mock misc
    agent.memory_evaluator = MagicMock()
    agent.research.to_heartbeat_context = MagicMock(return_value={})
    agent.feedback_loop.collect_feedback = MagicMock(return_value=[])
    agent.feedback_loop.apply_feedback = MagicMock(return_value={})
    agent._budget_manager = MagicMock()
    agent._wanderer.tick_active = MagicMock()
    agent._wanderer.tick_idle = MagicMock(return_value=False)

    if agent.clock:
        agent.clock.tick = MagicMock()
        agent.clock.now_status = MagicMock(return_value={
            "last_wall_time": time.time(), "gap_since_last": 0,
            "epoch": 1, "awakening_count": 0,
        })

    # Mock evidence replay
    mock_evidence = types.ModuleType("openllm.memory.evidence_replay")
    mock_evidence.create_replay_for_context = MagicMock(return_value=None)
    sys.modules["openllm.memory.evidence_replay"] = mock_evidence

    return agent


class TestHeartbeatSalienceGate:
    """心跳自动发布 salience.gate 事件。"""

    def test_heartbeat_emits_gate_on_tick(self):
        """完整心跳：tick 开始和结束各发布一次 salience.gate。"""
        agent = _make_mock_agent()
        result = agent.run_once("你好")
        assert isinstance(result, str)

        history = agent.iai.bus.get_history(type_filter="salience.gate")
        # 完成时 blocked=False（任务完成允许切换）
        assert len(history) >= 1
        gate_events = [e["payload"] for e in history]
        # 最后一个 gate 应该是 blocked=False（tick 完成）
        assert gate_events[-1]["blocked"] is False

    def test_heartbeat_task_switch_emits_blocked_true(self):
        """任务切换时：先 blocked=True 阻断，再 blocked=False 放行。"""
        agent = _make_mock_agent()

        # 第一次心跳
        agent.run_once("任务A")
        history_after_a = agent.iai.bus.get_history(type_filter="salience.gate")
        # 完成后 blocked=False
        assert any(not e["payload"]["blocked"] for e in history_after_a)

        # 清空 history 便于观察
        agent.iai.bus._history.clear()

        # 第二次心跳（不同任务）→应先发 blocked=True 阻断旧任务
        agent.run_once("任务B")
        history_after_b = agent.iai.bus.get_history(type_filter="salience.gate")

        blocked_true_events = [e for e in history_after_b
                               if e["payload"]["blocked"] is True
                               and e["payload"]["interruption_type"] == "task_switch"]
        blocked_false_events = [e for e in history_after_b
                                if e["payload"]["blocked"] is False
                                and e["payload"]["interruption_type"] == "task_switch"]

        assert len(blocked_true_events) >= 1, (
            f"任务切换应触发 blocked=True, got: {history_after_b}")
        assert len(blocked_false_events) >= 1, (
            f"任务完成应触发 blocked=False, got: {history_after_b}")

    def test_same_task_no_gate_event(self):
        """同一任务连续两次心跳：不发 blocked=True（无切换）。"""
        agent = _make_mock_agent()
        agent.run_once("同一任务")
        agent.iai.bus._history.clear()

        agent.run_once("同一任务")
        history = agent.iai.bus.get_history(type_filter="salience.gate")

        # 同任务不应有 task_switch+blocked=True 的事件
        blocked_switch_events = [e for e in history
                                 if e["payload"]["blocked"] is True
                                 and e["payload"]["interruption_type"] == "task_switch"]
        assert len(blocked_switch_events) == 0, (
            f"同任务不应有 blocked=True gate 事件: {blocked_switch_events}")

    def test_5_events_still_present(self):
        """回归：心跳仍产出完整5类事件。"""
        agent = _make_mock_agent()
        agent.run_once("你好")
        history = agent.iai.bus.get_history()
        all_types = [e["type"] for e in history]
        for evt in ["context.built", "reasoning.proposed",
                    "reasoning.critiqued", "decision.made",
                    "action.executed"]:
            assert evt in all_types, f"回归失败：缺少 {evt}"

    def test_gate_event_source(self):
        """心跳发布的 gate 事件 source 为 'iai.gate'。"""
        agent = _make_mock_agent()
        agent.run_once("你好")
        history = agent.iai.bus.get_history(type_filter="salience.gate")
        for evt in history:
            assert evt["source"] == "iai.gate"
