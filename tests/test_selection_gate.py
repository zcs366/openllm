"""
SelectionGate测试 — 不依赖真实V，全部mock compute_viability。

覆盖：单validator通过/拒绝、多validator一票否决、monotone safety、
      decisions日志append-only、make_v_gate降级场景。
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import patch, MagicMock

import pytest

from openllm.core.selection_gate import Proposal, SelectionGate, make_v_gate


# ── 辅助 ──

def _make_proposal(pid: str = "p1", desc: str = "test") -> Proposal:
    return Proposal(proposal_id=pid, description=desc)


def _passing_validator(proposal) -> tuple:
    """总是通过的validator。"""
    return True, 1.0, {"reason": "ok"}


def _failing_validator(proposal) -> tuple:
    """总是拒绝的validator。"""
    return False, 0.3, {"reason": "V下降"}


# ── 测试：单validator通过 ──

def test_single_validator_pass():
    """单validator通过→accepted，status更新，decisions记录一条。"""
    gate = SelectionGate(validators=[_passing_validator])
    p = _make_proposal("p_pass")
    result = gate.evaluate(p)

    assert result["accepted"] is True
    assert result["proposal"].status == "accepted"
    assert len(result["results"]) == 1
    assert result["results"][0]["passed"] is True

    assert len(gate.decisions) == 1
    assert gate.decisions[0]["proposal_id"] == "p_pass"
    assert gate.decisions[0]["accepted"] is True


# ── 测试：单validator拒绝 ──

def test_single_validator_reject():
    """单validator拒绝→rejected（V下降场景）。"""
    gate = SelectionGate(validators=[_failing_validator])
    p = _make_proposal("p_reject")
    result = gate.evaluate(p)

    assert result["accepted"] is False
    assert result["proposal"].status == "rejected"
    assert result["results"][0]["passed"] is False


# ── 测试：多validator一票否决 ──

def test_multi_validator_one_rejects():
    """2过1拒→整体rejected（一票否决）。"""
    gate = SelectionGate(validators=[_passing_validator, _passing_validator, _failing_validator])
    p = _make_proposal("p_multi")
    result = gate.evaluate(p)

    assert result["accepted"] is False
    assert len(result["results"]) == 3
    passed_count = sum(1 for r in result["results"] if r["passed"])
    assert passed_count == 2


# ── 测试：monotone safety ──

def test_monotone_pass_equal():
    """threshold=0.0时，V持平（ΔV=0）应通过。"""
    def v_equal(proposal) -> tuple:
        # V before=1.0, V after=1.0 → ΔV=0 ≥ 0.0 → pass
        return True, 1.0, {"v_before": 1.0, "v_after": 1.0, "delta_v": 0.0}

    gate = SelectionGate(validators=[v_equal])
    p = _make_proposal("p_mono_eq")
    result = gate.evaluate(p)
    assert result["accepted"] is True


def test_monotone_reject_decrease():
    """threshold=0.0时，V微降（ΔV=-0.01）应拒绝。"""
    def v_decrease(proposal) -> tuple:
        # V before=1.0, V after=0.99 → ΔV=-0.01 < 0.0 → reject
        return False, 0.99, {"v_before": 1.0, "v_after": 0.99, "delta_v": -0.01}

    gate = SelectionGate(validators=[v_decrease])
    p = _make_proposal("p_mono_dec")
    result = gate.evaluate(p)
    assert result["accepted"] is False


# ── 测试：decisions日志append-only ──

def test_decisions_log_append_only():
    """连续3个proposal后，len==3，顺序正确，内容含proposal_id和scores。"""
    gate = SelectionGate(validators=[_passing_validator])
    for i in range(3):
        p = _make_proposal(pid=f"p_{i}")
        gate.evaluate(p)

    assert len(gate.decisions) == 3
    for i, d in enumerate(gate.decisions):
        assert d["proposal_id"] == f"p_{i}"
        assert "scores" in d
        assert "timestamp" in d


# ── 测试：make_v_gate降级 ──

def test_make_v_gate_import_error():
    """signal_bus不可导入时抛清晰ImportError（mock ImportError场景）。"""
    # 临时从sys.path中移除io-s路径，模拟导入失败
    io_s_path = str(__import__("pathlib").Path.home() / "io-s")
    original_path = sys.path.copy()
    sys.path = [p for p in sys.path if io_s_path != p]

    # 同时把signal_bus从sys.modules中清除，确保真实模块不会被缓存命中
    original_modules = dict(sys.modules)
    sys.modules.pop("signal_bus", None)

    try:
        # 只拦截signal_bus的导入，不碰其他模块
        import builtins
        real_import = builtins.__import__
        def _selective_import(name, *args, **kwargs):
            if name == "signal_bus":
                raise ImportError("no signal_bus")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=_selective_import):
            with pytest.raises(ImportError, match="无法导入signal_bus"):
                make_v_gate(threshold=0.0)
    finally:
        sys.path = original_path
        sys.modules.update(original_modules)
