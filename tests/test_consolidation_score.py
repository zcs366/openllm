"""
ConsolidationScorer 测试 — 固化判据MVP（宪章P1 · v0.3落地）
=============================================================

覆盖：
1. test_score_formula —— 宪章公式 Score = 复现 + 2×增益 + 3×遗忘代价
2. test_p6_trust_gate —— P6门禁：低信任provenance的Score强制置零
3. test_ace_counters —— ACE helpful/harmful计数器侵蚀固化资格
4. test_deterministic_merge —— 增量合并非全量重写（复现=+1不覆盖）
5. test_candidates_pressure —— context_pressure水位联动候选产生
6. test_consolidate_dual_gate —— 双重门禁+append-only账本
7. test_audit_ledger —— 账本回读（睡眠窗第三件事）
8. test_persistence_roundtrip —— 原子写+损坏恢复
"""
import os
import sys
import json
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/projects/openllm/src"))

from openllm.consolidation_score import (
    Bullet,
    ConsolidationScorer,
    Provenance,
    TrustLevel,
    DEFAULT_TRUST_THRESHOLD,
)


def _user_prov(source="s1"):
    return Provenance(source=source, trust=TrustLevel.USER, endorsed_by="成市")


class TestScoreFormula(unittest.TestCase):
    def test_score_formula(self):
        """Score = 1×复现 + 2×增益% + 3×遗忘代价"""
        b = Bullet(
            bullet_id="b1", content="x",
            recurrence=3, task_gain_pct=10.0, forget_cost=2.0,
            provenance=_user_prov(),
        )
        # 3 + 2*10 + 3*2 = 3 + 20 + 6 = 29
        self.assertAlmostEqual(b.score(), 29.0)

    def test_helpful_harmful_net(self):
        """有效复现 = max(0, recurrence + helpful - harmful)"""
        b = Bullet(bullet_id="b2", content="x", recurrence=2,
                   helpful=3, harmful=1, provenance=_user_prov())
        # effective_recurrence = 2+3-1 = 4; score = 4
        self.assertAlmostEqual(b.score(), 4.0)

    def test_harmful_can_zero_out(self):
        """harmful足够多可把有效复现压到0"""
        b = Bullet(bullet_id="b3", content="x", recurrence=1,
                   helpful=0, harmful=5, provenance=_user_prov())
        # effective_recurrence = max(0, 1+0-5)=0; 无增益无遗忘代价 => 0
        self.assertAlmostEqual(b.score(), 0.0)


class TestP6TrustGate(unittest.TestCase):
    def test_web_trust_zeroed(self):
        """P6条款2：trust=web（低于阈值tool_output）⇒ Score=0"""
        b = Bullet(bullet_id="w1", content="恶意网页内容",
                   recurrence=100, task_gain_pct=50.0, forget_cost=9.0,
                   provenance=Provenance(source="evil.com", trust=TrustLevel.WEB))
        # 即便复现/增益/遗忘代价拉满，信任门禁一票否决
        self.assertEqual(b.score(), 0.0)

    def test_mixed_trust_zeroed(self):
        b = Bullet(bullet_id="m1", content="混合来源", recurrence=50,
                   task_gain_pct=30.0, forget_cost=5.0,
                   provenance=Provenance(trust=TrustLevel.MIXED))
        self.assertEqual(b.score(), 0.0)

    def test_tool_output_passes(self):
        """trust=tool_output（==阈值）⇒ 通过门禁"""
        b = Bullet(bullet_id="t1", content="工具结果", recurrence=2,
                   provenance=Provenance(source="grep", trust=TrustLevel.TOOL_OUTPUT))
        self.assertGreater(b.score(), 0.0)

    def test_creator_highest(self):
        b = Bullet(bullet_id="c1", content="造物主写入", recurrence=1,
                   provenance=Provenance(trust=TrustLevel.CREATOR, endorsed_by="成市"))
        self.assertGreater(b.score(), 0.0)

    def test_custom_threshold(self):
        """阈值可配：抬到USER则tool_output也被拦"""
        b = Bullet(bullet_id="t2", content="工具结果", recurrence=5,
                   provenance=Provenance(trust=TrustLevel.TOOL_OUTPUT))
        self.assertEqual(b.score(trust_threshold=TrustLevel.USER), 0.0)


class TestDeterministicMerge(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.scorer = ConsolidationScorer(state_path=Path(self.tmp) / "s.json")

    def test_recurrence_not_overwrite(self):
        """增量合并：重复add同id ⇒ recurrence+1，不覆盖content"""
        self.scorer.add_bullet("b1", "原始内容", provenance=_user_prov())
        self.scorer.add_bullet("b1", "原始内容", provenance=_user_prov())
        b = self.scorer._bullets["b1"]
        self.assertEqual(b.recurrence, 2)
        self.assertEqual(b.content, "原始内容")

    def test_mark_counters(self):
        self.scorer.add_bullet("b2", "x", provenance=_user_prov())
        self.scorer.mark("b2", helpful=True)
        self.scorer.mark("b2", helpful=True)
        self.scorer.mark("b2", helpful=False)
        b = self.scorer._bullets["b2"]
        self.assertEqual((b.helpful, b.harmful), (2, 1))

    def test_mark_missing_returns_false(self):
        self.assertFalse(self.scorer.mark("nonexistent", helpful=True))


class TestCandidatesPressure(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.scorer = ConsolidationScorer(
            state_path=Path(self.tmp) / "s.json", min_score=4.0)

    def test_normal_filters_below_min(self):
        """normal水位：只返回过min_score的（被动模式）"""
        self.scorer.add_bullet("low", "x", provenance=_user_prov())  # score=1
        self.scorer.add_bullet("high", "y", provenance=_user_prov(),
                               task_gain_pct=10.0)  # score=1+20=21
        cands = self.scorer.candidates(pressure_level="normal")
        ids = [b.bullet_id for b in cands]
        self.assertIn("high", ids)
        self.assertNotIn("low", ids)

    def test_caution_returns_all_positive(self):
        """caution水位：返回全部Score>0（主动模式，压缩前先固化）"""
        self.scorer.add_bullet("low", "x", provenance=_user_prov())  # score=1
        cands = self.scorer.candidates(pressure_level="caution")
        self.assertIn("low", [b.bullet_id for b in cands])

    def test_p6_excluded_from_candidates(self):
        """web信任条目永不进候选（score=0被过滤）"""
        evil = Bullet(bullet_id="evil", content="毒", recurrence=100,
                      task_gain_pct=99.0, forget_cost=99.0,
                      provenance=Provenance(trust=TrustLevel.WEB))
        self.scorer._bullets["evil"] = evil
        cands = self.scorer.candidates(pressure_level="critical")
        self.assertNotIn("evil", [b.bullet_id for b in cands])

    def test_sorted_by_score_desc(self):
        self.scorer.add_bullet("a", "x", provenance=_user_prov(), task_gain_pct=1.0)
        self.scorer.add_bullet("b", "y", provenance=_user_prov(), task_gain_pct=20.0)
        cands = self.scorer.candidates(pressure_level="caution")
        self.assertEqual(cands[0].bullet_id, "b")  # 高分在前


class TestConsolidateDualGate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.scorer = ConsolidationScorer(
            state_path=Path(self.tmp) / "s.json", min_score=4.0)

    def test_consolidate_pass(self):
        self.scorer.add_bullet("good", "x", provenance=_user_prov(),
                               task_gain_pct=10.0)  # score=21
        report = self.scorer.consolidate(pressure_level="caution")
        self.assertEqual(len(report["consolidated"]), 1)
        self.assertTrue(self.scorer._bullets["good"].consolidated)

    def test_reject_p6(self):
        """显式传入web条目也被P6门禁拒（reason=p6_trust_gate）"""
        evil = Bullet(bullet_id="evil", content="毒", recurrence=100,
                      task_gain_pct=99.0, forget_cost=99.0,
                      provenance=Provenance(trust=TrustLevel.WEB))
        self.scorer._bullets["evil"] = evil
        report = self.scorer.consolidate(bullets=[evil])
        self.assertEqual(len(report["rejected"]), 1)
        self.assertEqual(report["rejected"][0]["reason"], "p6_trust_gate")
        self.assertFalse(evil.consolidated)

    def test_reject_min_score(self):
        """信任够但分不够 ⇒ reason=min_score"""
        weak = Bullet(bullet_id="weak", content="x", recurrence=1,
                      provenance=_user_prov())  # score=1 < 4
        self.scorer._bullets["weak"] = weak
        report = self.scorer.consolidate(bullets=[weak])
        self.assertEqual(report["rejected"][0]["reason"], "min_score")

    def test_ledger_append_only(self):
        self.scorer.add_bullet("g1", "x", provenance=_user_prov(), task_gain_pct=10.0)
        self.scorer.consolidate(pressure_level="caution")
        self.assertEqual(len(self.scorer._ledger), 1)
        entry = self.scorer._ledger[0]
        self.assertEqual(entry["bullet_id"], "g1")
        self.assertEqual(entry["endorsed_by"], "成市")
        self.assertIn("ts", entry)

    def test_consolidated_not_re_candidate(self):
        """已固化条目不再进候选"""
        self.scorer.add_bullet("g2", "x", provenance=_user_prov(), task_gain_pct=10.0)
        self.scorer.consolidate(pressure_level="caution")
        cands = self.scorer.candidates(pressure_level="caution")
        self.assertNotIn("g2", [b.bullet_id for b in cands])


class TestAuditLedger(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.scorer = ConsolidationScorer(
            state_path=Path(self.tmp) / "s.json", min_score=4.0)

    def test_audit_distribution(self):
        self.scorer.add_bullet("u1", "x", provenance=_user_prov(), task_gain_pct=10.0)
        self.scorer.consolidate(pressure_level="caution")
        audit = self.scorer.audit_ledger()
        self.assertEqual(audit["total_consolidated"], 1)
        self.assertEqual(audit["trust_distribution"].get("user"), 1)
        self.assertEqual(audit["low_trust_ratio"], 0.0)

    def test_audit_empty(self):
        audit = self.scorer.audit_ledger()
        self.assertEqual(audit["total_consolidated"], 0)
        self.assertEqual(audit["low_trust_ratio"], 0.0)


class TestPersistence(unittest.TestCase):
    def test_roundtrip(self):
        tmp = tempfile.mkdtemp()
        path = Path(tmp) / "s.json"
        s1 = ConsolidationScorer(state_path=path)
        s1.add_bullet("b1", "内容", provenance=_user_prov(), task_gain_pct=5.0)
        s1.mark("b1", helpful=True)
        s1.consolidate(pressure_level="caution")
        # 重新加载
        s2 = ConsolidationScorer(state_path=path)
        self.assertIn("b1", s2._bullets)
        self.assertEqual(s2._bullets["b1"].helpful, 1)
        self.assertTrue(s2._bullets["b1"].consolidated)
        self.assertEqual(len(s2._ledger), 1)
        # provenance往返
        self.assertEqual(s2._bullets["b1"].provenance.trust, TrustLevel.USER)

    def test_corrupt_recovery(self):
        tmp = tempfile.mkdtemp()
        path = Path(tmp) / "s.json"
        path.write_text("{ this is not valid json", encoding="utf-8")
        s = ConsolidationScorer(state_path=path)  # 不应抛
        self.assertEqual(s._bullets, {})
        self.assertEqual(s._ledger, [])

    def test_atomic_write_no_tmp_leftover(self):
        tmp = tempfile.mkdtemp()
        path = Path(tmp) / "s.json"
        s = ConsolidationScorer(state_path=path)
        s.add_bullet("b1", "x", provenance=_user_prov())
        leftovers = [f for f in os.listdir(tmp) if f.endswith(".tmp")]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
