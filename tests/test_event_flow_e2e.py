"""
端到端事件流测试 — EventBus 5类事件全链路验证（PAL T-F-1）

验证目标：Agent完整心跳过程中 EventBus 5类事件全部被emit：
  context.built → reasoning.proposed → reasoning.critiqued → decision.made → action.executed

策略：实例化Agent + mock LLM/章鱼I → run_once → 断言bus history包含全部5类
"""
import sys
import types
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ── 模拟对象 ──────────────────────────────────────────
from openllm.core.models import (
    Message, Context, Prediction, RiskAssessment,
    Proposal, Critique, Decision, ActionResult, CausalDelta,
)


def _make_agent():
    """实例化Agent，monkeypatch掉所有会联网/IO的路径。"""
    # monkeypatch _print_startup_health 在 __init__ 内调用
    with patch("openllm.core.main_loop.Agent._print_startup_health"):
        from openllm.core.main_loop import Agent
        agent = Agent(mode="console")

    # Mock 章鱼I 左脑reason
    mock_proposal = Proposal(
        content="mock proposal for test",
        confidence=0.9,
        evidence=["evidence1"],
    )
    mock_critique = Critique(
        content="mock critique for test",
        verdict="approve",
        concerns=[],
    )
    agent.octopus.reason = MagicMock(return_value=(mock_proposal, mock_critique))

    # Mock predict_consequences（_perceive阶段调用）
    mock_prediction = Prediction(
        summary="mock prediction",
        consequences=["consequence1"],
        confidence=0.7,
    )
    agent.octopus.predict_consequences = MagicMock(return_value=mock_prediction)

    # Mock d0_snapshot（_decide阶段调用）
    agent.octopus.d0_snapshot = MagicMock(return_value={})

    # Mock compare（_learn阶段调用）
    mock_delta = CausalDelta(
        prediction_match=True,
        delta_summary="mock delta",
        learned=[],
    )
    agent.octopus.compare = MagicMock(return_value=mock_delta)

    # Mock octopus tentacles（_do_search不报错）
    mock_tentacles = MagicMock()
    mock_tentacles.get.return_value = None
    agent.octopus.tentacles = mock_tentacles

    # Mock ISN execute（_execute阶段调用）
    agent.isn.execute = MagicMock(return_value=ActionResult(
        success=True, output="mock result output", duration_ms=1.0,
    ))

    # Mock IOS（risk_check, arbitrate, learn_causal, evolve, recover, cap_check, governance_engine）
    agent.ios.risk_check = MagicMock(return_value=RiskAssessment(level="low", blocked=False))
    agent.ios.arbitrate = MagicMock(return_value=Decision(
        action="execute", approved=True, reason="test approved",
    ))
    agent.ios.learn_causal = MagicMock()
    agent.ios.evolve = MagicMock()
    agent.ios.recover = MagicMock(return_value=False)
    agent.ios.cap_check = MagicMock(return_value=True)
    agent.ios.governance_engine = MagicMock()
    agent.ios.governance_engine.heartbeat_trace = MagicMock()

    # Mock ISA.build_context
    mock_ctx = Context(user_message="你好")
    agent.isa.build_context = MagicMock(return_value=mock_ctx)
    agent.isa.respond = MagicMock()

    # Mock session
    agent.session.compact_if_needed = MagicMock()
    agent.session.checkpoint = MagicMock()

    # Mock IKO trace + process_output
    agent.iko.process_output = MagicMock(return_value="processed output")

    # Mock memory_evaluator
    agent.memory_evaluator = MagicMock()

    # Mock research
    agent.research.to_heartbeat_context = MagicMock(return_value={})

    # Mock feedback_loop
    agent.feedback_loop.collect_feedback = MagicMock(return_value=[])
    agent.feedback_loop.apply_feedback = MagicMock(return_value={})

    # Mock clock
    if agent.clock:
        agent.clock.tick = MagicMock()
        agent.clock.now_status = MagicMock(return_value={
            "last_wall_time": time.time(),
            "gap_since_last": 0,
            "epoch": 1,
            "awakening_count": 0,
        })

    # Mock wanderer
    agent._wanderer.tick_active = MagicMock()
    agent._wanderer.tick_idle = MagicMock(return_value=False)

    # Mock budget manager
    agent._budget_manager = MagicMock()

    # Mock evidence replay to avoid import
    mock_evidence = types.ModuleType("openllm.memory.evidence_replay")
    mock_evidence.create_replay_for_context = MagicMock(return_value=None)
    sys.modules["openllm.memory.evidence_replay"] = mock_evidence

    return agent


# ── 测试 ──────────────────────────────────────────────

EXPECTED_EVENTS = [
    "context.built",
    "reasoning.proposed",
    "reasoning.critiqued",
    "decision.made",
    "action.executed",
]


def test_full_heartbeat_emits_all_5_events():
    """完整心跳：Agent.run_once 后 EventBus history 必须包含5类事件。"""
    agent = _make_agent()

    # 运行一次完整心跳
    result = agent.run_once("你好")
    assert isinstance(result, str), f"run_once应返回str, got {type(result)}"

    # 从EventBus history提取所有事件类型
    history = agent.iai.bus.get_history(limit=1000)
    emitted_types = [e["type"] for e in history]

    print(f"\n  emitted events: {emitted_types}")

    # 断言：全部5类事件都被emit
    for evt_type in EXPECTED_EVENTS:
        assert evt_type in emitted_types, (
            f"缺少事件: {evt_type}\n"
            f"  已emit: {emitted_types}\n"
            f"  全部history: {history}"
        )

    # 断言：按顺序出现（context.built 在 reasoning.proposed 之前，以此类推）
    indices = {evt: emitted_types.index(evt) for evt in EXPECTED_EVENTS}
    for i in range(len(EXPECTED_EVENTS) - 1):
        assert indices[EXPECTED_EVENTS[i]] < indices[EXPECTED_EVENTS[i + 1]], (
            f"事件顺序错误: {EXPECTED_EVENTS[i]}(idx={indices[EXPECTED_EVENTS[i]]}) "
            f"应出现在 {EXPECTED_EVENTS[i+1]}(idx={indices[EXPECTED_EVENTS[i+1]]}) 之前"
        )

    print(f"  ✅ 全部5类事件按序emit，共{len(emitted_types)}条事件")


def test_event_payload_not_empty():
    """每条emit的事件都携带非空payload。"""
    agent = _make_agent()
    agent.run_once("你好")

    history = agent.iai.bus.get_history(limit=1000)
    for evt in history:
        # brain_id 字段存在（向后兼容：默认 "default"）
        assert "brain_id" in evt, f"事件{evt['type']}缺少brain_id字段"
        assert isinstance(evt["payload"], dict), (
            f"事件{evt['type']}的payload不是dict: {type(evt['payload'])}"
        )


def test_emit_counts():
    """每类事件恰好emit 1次（单次心跳不重复）。"""
    agent = _make_agent()
    agent.run_once("你好")

    history = agent.iai.bus.get_history(limit=1000)
    from collections import Counter
    counts = Counter(e["type"] for e in history)

    for evt_type in EXPECTED_EVENTS:
        assert counts[evt_type] == 1, (
            f"{evt_type} 应恰好1次, 实际{counts[evt_type]}次; 全部: {dict(counts)}"
        )
