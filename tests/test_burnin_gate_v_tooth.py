"""test_burnin_gate_v_tooth.py — B1-retest 可注入重测能力验收（2026-09-07）
=====================================================================

覆盖（四验收项）：
1. test_default_off_regression — retest_fn未提供，行为与改造前完全一致
2. test_enforce_retest_pass — enforce+retest_fn：V_new≥baseline−ε→passed
3. test_enforce_retest_reject — V_new<baseline−ε→rejected
4. test_enforce_retest_exception — retest_fn抛异常→not_evaluable不崩溃
5. test_enforce_retest_v_none_fallback — v_snapshot缺失+retest_fn→走retest通道
6. test_epsilon_history_lt3 — ΔV历史<3样本→ε=∞→不拒
7. test_evidence_fields — h["retest"]包含retest_score/source/duration_s
8. test_shadow_retest_fn_ignored — shadow模式下retest_fn不被调用
"""
import math
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, os.path.expanduser("~/projects/openllm/src"))

from openllm.iai.burnin_gate import (
    BurnInGate,
    GateEvidence,
    compute_epsilon,
    EPSILON_MIN_SAMPLES,
)


# ── 辅助 ───────────────────────────────────────────────

def _ev(degradations=0, questions=10, retest=None, v=None, comps=None,
        run_id="v1"):
    """构造 GateEvidence（同 test_burnin_gate.py 模式）。"""
    v_snap = None
    if v is not None:
        v_snap = {"V": v, "components": comps or {"M": 0.5, "S": 0.3, "B": 0.8}}
    return GateEvidence(
        run_id=run_id, adapter_sha="sha_x", snapshot_id="snap_x",
        degradations=degradations, questions=questions,
        retest_consistency=retest, v_snapshot=v_snap,
    )


def _gate(**kw):
    tmp = Path(tempfile.mkdtemp()) / "gate.json"
    return BurnInGate(state_path=tmp, **kw)


def _seed_gate(gate, n=5, v_base=1.0, comps=None):
    """向 gate 注入 n 条 V 历史（passed），建立基线与 ΔV 序列。

    非均匀步进产生变化的 ΔV → 非零 MAD → 有限 ε。
    示例（n=5, v_base=1.0）：V=[1.0, 1.01, 1.005, 1.015, 1.008]
    delta_v=[0.01, -0.005, 0.01, -0.007], ε≈0.0225, baseline≈1.01
    注意：evaluate()是纯函数不记账，必须用 gate()（含 _record）才能写入状态。
    """
    comps = comps or {"M": 0.5, "S": 0.3, "B": 0.8}
    # 非均匀偏移（避免恒定 ΔV → MAD=0）
    _offsets = [0.0, 0.01, 0.005, 0.015, 0.008]
    for i in range(n):
        offset = _offsets[i % len(_offsets)]
        ev = _ev(run_id=f"seed_{i}", v=v_base + offset, comps=comps)
        gate.gate(ev, rollback_fn=None)  # gate()含记账，evaluate()不记账


# ── 验收项 1：默认 off 回归 ────────────────────────────

def test_default_off_regression():
    """retest_fn=None → evaluate 行为与改造前完全一致（无 retest 证据注入）。"""
    g = _gate(shadow=True)
    # B0 通过，shadow 无基线 → passed-by-b0
    v = g.evaluate(_ev(v=0.9))
    assert v.verdict == "passed"
    assert "retest" not in v.h or v.h["retest"] == {}  # 未调 retest_fn → 无有效 retest 证据

    # B0 拒绝 → rejected（不受 retest 影响）
    v2 = g.evaluate(_ev(degradations=2, questions=10))
    assert v2.verdict == "rejected"
    assert "retest" not in v2.h


def test_default_off_enforce_no_retest_fn():
    """enforce 模式(shadow=False)但 retest_fn=None → 不调 retest，走原路径。"""
    g = _gate(shadow=False)
    v = g.evaluate(_ev(v=None))  # V=None，无 retest_fn
    assert v.verdict == "not_evaluable"
    assert "retest" not in v.h


# ── 验收项 2：enforce + retest_fn 合格 → passed ───────

def test_enforce_retest_pass():
    """V_new(retest) ≥ baseline − ε → passed。

    seed: V=[1.0, 1.01, 1.005, 1.015, 1.008]
    baseline≈1.01, MAD(deltas)≈0.0075, ε≈0.0225
    阈值 ≈ 0.9875 → V_new=1.0 ≥ 0.9875 通过
    """
    g = _gate(shadow=False)
    _seed_gate(g, n=5, v_base=1.0)

    retest_score = 1.0  # ≥ 0.9875 阈值 → passed
    v = g.evaluate(_ev(v=0.5), retest_fn=lambda: retest_score)
    assert v.verdict == "passed"
    assert v.h["retest"]["retest_score"] == retest_score
    assert v.h["retest"]["retest_source"] == "retest_fn"
    assert "retest_duration_s" in v.h["retest"]
    assert v.h["retest"]["retest_duration_s"] >= 0


# ── 验收项 2b：enforce + retest_fn 低于 baseline → rejected ──

def test_enforce_retest_reject():
    """V_new(retest) < baseline − ε → rejected。

    阈值 ≈ 0.9875 → V_new=0.98 < 0.9875 → rejected
    """
    g = _gate(shadow=False)
    _seed_gate(g, n=5, v_base=1.0)

    retest_score = 0.98  # < 0.9875 阈值 → rejected
    v = g.evaluate(_ev(v=1.0), retest_fn=lambda: retest_score)
    assert v.verdict == "rejected"
    assert "b1-v-drift" in v.reason
    assert v.h["retest"]["retest_score"] == retest_score


# ── 验收项 3：retest_fn 抛异常 → not_evaluable 不崩溃 ──

def test_enforce_retest_exception():
    """retest_fn 抛异常 → not_evaluable（试管爆裂≠没病），不崩溃。"""
    g = _gate(shadow=False)
    _seed_gate(g, n=5, v_base=1.0)

    def _explode():
        raise RuntimeError("GPU OOM — retest 无法执行")

    v = g.evaluate(_ev(v=1.0), retest_fn=_explode)
    assert v.verdict == "not_evaluable"
    assert "retest-failed" in v.reason
    assert "retest_error" in v.h
    assert "GPU OOM" in v.h["retest_error"]


def test_enforce_retest_v_none_fallback():
    """v_snapshot 缺失(V=None) + retest_fn → 走 retest 通道获取 V_new。

    retest_score=1.0 ≥ 阈值≈0.9875 → passed
    """
    g = _gate(shadow=False)
    _seed_gate(g, n=5, v_base=1.0)

    retest_score = 1.0  # ≥ 阈值≈0.9875 → passed
    v = g.evaluate(_ev(v=None), retest_fn=lambda: retest_score)
    assert v.verdict == "passed"
    assert v.h["retest"]["retest_score"] == retest_score
    # retest 从 V=None 通道修复了 v_new
    assert f"V_new={retest_score}" in v.reason


# ── 验收项 4：ε 历史 < 3 样本 → 不拒 ────────────────

def test_epsilon_history_lt3():
    """ΔV 历史样本 < 3 → ε=∞ → passed-by-b0（不拒）。"""
    g = _gate(shadow=False)
    # 只 seed 2 条（ΔV 只有 1 个，< 3）—— 用 gate() 记账
    for i in range(2):
        ev = _ev(run_id=f"seed_{i}", v=1.0 + i * 0.01,
                 comps={"M": 0.5, "S": 0.3, "B": 0.8})
        g.gate(ev, rollback_fn=None)  # gate()含记账

    eps = compute_epsilon(g.state["delta_v"], g.sigma_repeat)
    assert math.isinf(eps), f"ε 应为 ∞（样本<3），实际 {eps}"

    # retest_fn 返回低值，但 ε=∞ → 不拒绝
    v = g.evaluate(_ev(v=None), retest_fn=lambda: 0.5)
    assert v.verdict == "passed"
    assert "passed-by-b0" in v.reason


# ── 验收项 5：evidence 字段完整性 ────────────────────

def test_evidence_fields():
    """h["retest"] 包含 retest_score / retest_source / retest_duration_s。"""
    g = _gate(shadow=False)
    _seed_gate(g, n=5, v_base=1.0)

    v = g.evaluate(_ev(v=None), retest_fn=lambda: 1.0)
    retest_ev = v.h["retest"]
    assert "retest_score" in retest_ev
    assert "retest_source" in retest_ev
    assert "retest_duration_s" in retest_ev
    assert isinstance(retest_ev["retest_score"], float)
    assert isinstance(retest_ev["retest_source"], str)
    assert isinstance(retest_ev["retest_duration_s"], float)


# ── 验收项 6：shadow 模式下 retest_fn 不被调用 ──────

def test_shadow_retest_fn_ignored():
    """shadow 模式下即使传入 retest_fn 也不调用（保持 shadow 语义）。"""
    g = _gate(shadow=True)
    _seed_gate(g, n=5, v_base=1.0)

    called = MagicMock()
    v = g.evaluate(_ev(v=None), retest_fn=called)
    called.assert_not_called()  # shadow 模式不调 retest_fn
    assert v.verdict == "not_evaluable"  # V=None，无 retest 修复
