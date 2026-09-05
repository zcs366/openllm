"""
BurnInGate 测试 — 权重通道门禁（三元矛盾授权拓扑·生理门禁落位）
=================================================================

覆盖：
1. test_b0_degradations_reject —— B0硬门：degradations>D_max→rejected
2. test_b0_pass —— degradations=0→B0通过
3. test_b0_evidence_missing_hang —— 证据缺失→not_evaluable（拒绝挂起）
4. test_b0_retest —— retest一致性下限（r_min>0才启用）
5. test_b1_v_none —— V=None→not_evaluable（试管爆裂≠没病）
6. test_b1_cold_start —— 无基线/ΔV样本<3→ε=∞→passed-by-b0
7. test_b1_component_shift —— 分量基数跳变→退回B0（自欺容差防御）
8. test_b1_v_drift_shadow_vs_enforce —— V漂移：shadow记录不拒/enforce拒
9. test_epsilon_formula —— ε=max(3MAD,2σ)
10. test_rollback_on_reject —— rejected→调rollback_fn真回滚
11. test_rollback_fail_escalate —— 回滚失败→晨报升级
12. test_bus_emit —— gate.*事件入总线+降级静默
13. test_morning_line —— 晨报一行（阿佛洛狄忒）
14. test_state_append_only —— v_history/delta_v账本累积
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/projects/openllm/src"))

from openllm.iai.burnin_gate import (
    BurnInGate,
    GateEvidence,
    compute_epsilon,
    EPSILON_MIN_SAMPLES,
)


def _ev(degradations=0, questions=10, retest=None, v=None, comps=None, run_id="v3"):
    # 类型注解放宽：None 用于测证据缺失路径（B0 hang）
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


class TestB0(unittest.TestCase):
    def test_b0_degradations_reject(self):
        g = _gate(d_max=0)
        v = g.evaluate(_ev(degradations=2, questions=10))
        self.assertEqual(v.verdict, "rejected")
        self.assertIn("b0-degradations", v.reason)

    def test_b0_pass(self):
        g = _gate(d_max=0)
        v = g.evaluate(_ev(degradations=0, questions=10))
        self.assertIn(v.verdict, ("passed", "not_evaluable"))  # B0过；B1看V
        self.assertNotEqual(v.verdict, "rejected")

    def test_b0_evidence_missing_hang(self):
        g = _gate()
        v = g.evaluate(_ev(degradations=None, questions=None))
        self.assertEqual(v.verdict, "not_evaluable")
        self.assertIn("b0-evidence-missing", v.reason)

    def test_b0_retest_disabled_by_default(self):
        """r_min=0默认不启用retest门（⑥-j死阈：无分布前不定阈值）"""
        g = _gate(r_min=0.0)
        v = g.evaluate(_ev(degradations=0, retest=0.1))  # 低retest但r_min=0
        self.assertNotEqual(v.verdict, "rejected")

    def test_b0_retest_enabled(self):
        g = _gate(r_min=0.8)
        v = g.evaluate(_ev(degradations=0, retest=0.5))
        self.assertEqual(v.verdict, "rejected")
        self.assertIn("b0-retest", v.reason)


class TestB1(unittest.TestCase):
    def test_v_none_not_evaluable(self):
        g = _gate()
        v = g.evaluate(_ev(degradations=0, v=None))  # v_snapshot None
        self.assertEqual(v.verdict, "not_evaluable")
        self.assertIn("b1-v-none", v.reason)

    def test_cold_start_passed_by_b0(self):
        """无V基线（首轮）→ε=∞→B0过即passed"""
        g = _gate()
        v = g.evaluate(_ev(degradations=0, v=0.7))
        self.assertEqual(v.verdict, "passed")
        self.assertIn("passed-by-b0", v.reason)

    def test_component_shift_falls_back(self):
        """分量基数跳变（S缺失）→not_evaluable退回B0"""
        g = _gate()
        # 先攒3轮正常分量基线（M/S/B）
        for i in range(3):
            g.gate(_ev(degradations=0, v=0.7, comps={"M": 0.5, "S": 0.3, "B": 0.8},
                       run_id=f"v{i}"))
        # 跳变：S缺失
        v = g.evaluate(_ev(degradations=0, v=0.7,
                           comps={"M": 0.5, "S": None, "B": 0.8}, run_id="vX"))
        self.assertEqual(v.verdict, "not_evaluable")
        self.assertIn("component-shift", v.reason)

    def test_v_drift_shadow_no_reject(self):
        """shadow期V暴跌→not_evaluable（记录不拒）"""
        g = _gate(shadow=True, shadow_nights=10)
        # 攒基线（passed历史）
        for i in range(4):
            g.gate(_ev(degradations=0, v=0.8, run_id=f"p{i}"))
        # 喂ΔV样本使ε有限
        g._state["delta_v"] = [0.01, -0.01, 0.02]
        g._save_state()
        v = g.evaluate(_ev(degradations=0, v=0.2, run_id="crash"))  # V暴跌
        self.assertEqual(v.verdict, "not_evaluable")  # shadow不拒
        self.assertIn("shadow", v.reason)

    def test_v_drift_enforce_reject(self):
        """enforce期V暴跌→rejected"""
        g = _gate(shadow=False)  # 直接enforce
        for i in range(4):
            g.gate(_ev(degradations=0, v=0.8, run_id=f"p{i}"))
        g._state["delta_v"] = [0.01, -0.01, 0.02]
        g._save_state()
        v = g.evaluate(_ev(degradations=0, v=0.2, run_id="crash"))
        self.assertEqual(v.verdict, "rejected")
        self.assertIn("b1-v-drift", v.reason)


class TestEpsilon(unittest.TestCase):
    def test_epsilon_cold_start_inf(self):
        self.assertEqual(compute_epsilon([0.1, 0.2], sigma_repeat=0.01), float("inf"))

    def test_epsilon_formula(self):
        """ε=max(3×MAD, 2×σ_repeat)"""
        # ΔV=[0,0,0] → MAD=0 → ε=max(0, 2×0.05)=0.1
        self.assertAlmostEqual(compute_epsilon([0.0, 0.0, 0.0], sigma_repeat=0.05), 0.1)
        # ΔV散布 → MAD主导：median=0.1, 偏差=[0,0.2,0.1,0.3,0.2], MAD=0.2 → ε=0.6
        eps = compute_epsilon([0.1, -0.1, 0.2, -0.2, 0.3], sigma_repeat=0.0)
        self.assertAlmostEqual(eps, 0.6)
        # 注：[0.1,-0.1,0.1,-0.1,0.1]的MAD=0（中位数=0.1恰使过半偏差为0）——
        # MAD抗离群的代价：交替序列中位偏差可为零，此时σ_repeat兜底（max语义）


class TestExecution(unittest.TestCase):
    def test_rollback_on_reject(self):
        g = _gate(d_max=0)
        called = {"n": 0}
        def fake_rollback():
            called["n"] += 1
            return True
        v = g.gate(_ev(degradations=3), rollback_fn=fake_rollback)
        self.assertEqual(v.verdict, "rejected")
        self.assertTrue(v.rolled_back)
        self.assertEqual(called["n"], 1)

    def test_no_rollback_on_pass(self):
        g = _gate()
        called = {"n": 0}
        def fake_rollback():
            called["n"] += 1
            return True
        g.gate(_ev(degradations=0, v=0.7), rollback_fn=fake_rollback)
        self.assertEqual(called["n"], 0)  # passed不回滚

    def test_rollback_fail_escalate(self):
        g = _gate(d_max=0)
        v = g.gate(_ev(degradations=5), rollback_fn=lambda: False)
        self.assertEqual(v.verdict, "rejected")
        self.assertFalse(v.rolled_back)
        self.assertIn("回滚失败", v.reason)
        # 晨报升级
        self.assertIn("⚠️", g.morning_line())


class TestBusAndMorning(unittest.TestCase):
    def test_bus_emit(self):
        events = []
        tmp = Path(tempfile.mkdtemp()) / "g.json"
        g = BurnInGate(state_path=tmp, d_max=0,
                       bus_append=lambda t, p: events.append((t, p)))
        g.gate(_ev(degradations=2, run_id="r1"))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][0], "gate.rejected")
        self.assertEqual(events[0][1]["run_id"], "r1")

    def test_bus_degrade_silent(self):
        """总线故障不抛（降级静默）"""
        tmp = Path(tempfile.mkdtemp()) / "g.json"
        def boom(t, p):
            raise RuntimeError("总线炸了")
        g = BurnInGate(state_path=tmp, bus_append=boom)
        v = g.gate(_ev(degradations=0, v=0.7))  # 不应抛
        self.assertIsNotNone(v.verdict)

    def test_morning_line_no_verdict(self):
        g = _gate()
        self.assertIn("无判定记录", g.morning_line())

    def test_morning_line_consecutive_reject(self):
        g = _gate(d_max=0)
        g.gate(_ev(degradations=1, run_id="a"), rollback_fn=lambda: True)
        g.gate(_ev(degradations=1, run_id="b"), rollback_fn=lambda: True)
        self.assertIn("连续2次拒绝", g.morning_line())


class TestState(unittest.TestCase):
    def test_append_only_history(self):
        g = _gate()
        for i in range(5):
            g.gate(_ev(degradations=0, v=0.7 + i * 0.01, run_id=f"v{i}"))
        self.assertEqual(len(g.state["v_history"]), 5)
        self.assertEqual(len(g.state["delta_v"]), 4)  # 5个V→4个ΔV
        self.assertEqual(g.state["nights_evaluated"], 5)

    def test_persistence_roundtrip(self):
        tmp = Path(tempfile.mkdtemp()) / "g.json"
        g1 = BurnInGate(state_path=tmp)
        g1.gate(_ev(degradations=0, v=0.7, run_id="v1"))
        g2 = BurnInGate(state_path=tmp)
        self.assertEqual(len(g2.state["v_history"]), 1)
        self.assertEqual(g2.state["v_history"][0]["run_id"], "v1")

    def test_shadow_to_enforce_transition(self):
        g = _gate(shadow=True, shadow_nights=3)
        self.assertFalse(g.enforce_ready())  # 初始shadow未就绪
        for i in range(3):
            g.gate(_ev(degradations=0, v=0.7, run_id=f"v{i}"))
        g._state["delta_v"] = [0.01, 0.02, 0.01]  # ≥3样本
        g._save_state()
        self.assertTrue(g.enforce_ready())  # 晚数≥3且ΔV≥3


if __name__ == "__main__":
    unittest.main(verbosity=2)
