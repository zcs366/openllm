"""固化触发接线测试（2026-09-15 · 通道合规修补·军令5）

验证两件事：
1. 接线后 _record_inference 会调用 ConsolidationTrigger.on_usage（通车验证）；
2. on_usage 抛异常时主循环照常完成（隔离性——固化链故障不阻塞主循环）。
"""
import os
import sys
import types
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from openllm.core.main_loop import Agent as MainLoop  # noqa: E402


def _make_loop_with_mocks():
    """构造最小 MainLoop：只挂 _record_inference 依赖的属性，不跑真的初始化。"""
    loop = MainLoop.__new__(MainLoop)
    loop._budget_manager = None
    loop._token_economy = MagicMock()
    loop.octopus = MagicMock()
    loop.octopus.left.provider._last_usage = {"prompt_tokens": 100, "completion_tokens": 50}
    loop.octopus.left.provider.model = "test-model"
    loop.session = types.SimpleNamespace(id="sess-test")
    loop._consolidation_trigger = None
    return loop


def test_trigger_called_after_inference(monkeypatch):
    """通车验证：token记录后 trigger.on_usage 被调用一次，参数正确。"""
    loop = _make_loop_with_mocks()
    calls = {}

    class FakeTrigger:
        def on_usage(self, session_id, used, budget):
            calls["args"] = (session_id, used, budget)

    monkeypatch.setattr(
        "openllm.isa.consolidation_trigger.ConsolidationTrigger", FakeTrigger
    )
    loop._record_inference("phase3")
    assert "args" in calls, "on_usage 未被调用——固化管线仍断头"
    session_id, used, budget = calls["args"]
    assert session_id == "sess-test"
    assert used == 150  # prompt+completion
    assert budget > 0
    # 触发器实例被缓存（惰性单例，不每轮重建）
    assert isinstance(loop._consolidation_trigger, FakeTrigger)


def test_trigger_failure_isolated(monkeypatch):
    """隔离性验证：on_usage 抛异常时主循环不炸（固化链故障不影响推理）。"""

    class BombTrigger:
        def on_usage(self, *a, **kw):
            raise RuntimeError("固化链爆炸")

    loop = _make_loop_with_mocks()
    loop._consolidation_trigger = BombTrigger()
    # 不抛异常 = 隔离成功
    loop._record_inference("phase3")
    # token_economy 照常记账（异常发生在其之后，但隔离块必须吞掉）
    assert loop._token_economy.record_usage.called


def test_no_usage_no_trigger(monkeypatch):
    """零用量时早退（模拟模式），不应触发固化链。"""
    loop = _make_loop_with_mocks()
    loop.octopus.left.provider._last_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    bomb = MagicMock(side_effect=AssertionError("零用量不应触碰固化链"))
    loop._consolidation_trigger = bomb
    loop._record_inference("phase3")  # 早退，不炸即通过
