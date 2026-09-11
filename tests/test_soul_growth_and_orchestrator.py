"""
SoulGrowthLedger + ConsolidationOrchestrator 测试
==================================================

覆盖验收判据 SC1-SC4：
  SC1: soul_growth.py import + record/load/stats 接口齐全
  SC2: consolidation_orchestrator.py import + run_cycle 返回dict
  SC3: 四模块真调用（import证据）
  SC4: ≥8条测试

测试矩阵：
  soul_growth:
    1. test_record_and_load —— record后load可读
    2. test_append_only —— 多次record只追加不覆盖
    3. test_load_recent_n —— load(recent_n)只取最近N条
    4. test_stats —— stats返回正确统计
    5. test_load_empty_file —— 文件不存在时返回空
    6. test_rollback_record —— rollback_reason标记不删行

  orchestrator:
    7. test_normal_level_not_acted —— level=normal → acted=False
    8. test_caution_level_acted_with_write —— level=caution → acted=True，growth有写入
    9. test_rehydration_fail_rollback —— validator返回fail → 标记rollback
"""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ── SC3 证据：from import 四模块真调用 ──
from openllm.identity.soul_growth import SoulGrowthLedger, GrowthRecord  # 模块4
from openllm.memory.consolidation_orchestrator import ConsolidationOrchestrator  # 桥梁
from openllm.memory.context_pressure import ContextPressureMonitor  # 模块1
from openllm.consolidation_score import ConsolidationScorer, Provenance, TrustLevel  # 模块2
from openllm.memory.rehydration_validator import RehydrationValidator  # 模块3


# ═══════════════════════════════════════════════════════════════
# SoulGrowthLedger 测试（SC1）
# ═══════════════════════════════════════════════════════════════

class TestSoulGrowthRecord(unittest.TestCase):
    """SC1: record/load/stats 接口齐全。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = Path(self.tmp) / "growth.jsonl"
        self.ledger = SoulGrowthLedger(growth_path=self.path)

    def test_record_and_load(self):
        """record后load可读。"""
        rec = self.ledger.record(
            bullet_id="b1",
            content="WSL中I盘=/mnt/i/",
            origin="consolidation",
            score=29.0,
            provenance_trust="user",
        )
        self.assertEqual(rec["bullet_id"], "b1")
        self.assertEqual(rec["content"], "WSL中I盘=/mnt/i/")
        self.assertEqual(rec["origin"], "consolidation")
        self.assertAlmostEqual(rec["score"], 29.0)
        self.assertEqual(rec["provenance_trust"], "user")
        self.assertIn("ts", rec)

        records = self.ledger.load()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["bullet_id"], "b1")

    def test_append_only_no_overwrite(self):
        """多次record只追加不覆盖。"""
        self.ledger.record(bullet_id="b1", content="内容A", score=10.0)
        self.ledger.record(bullet_id="b2", content="内容B", score=20.0)
        self.ledger.record(bullet_id="b3", content="内容C", score=30.0)

        records = self.ledger.load()
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0]["bullet_id"], "b1")
        self.assertEqual(records[1]["bullet_id"], "b2")
        self.assertEqual(records[2]["bullet_id"], "b3")

    def test_load_recent_n(self):
        """load(recent_n)只取最近N条。"""
        for i in range(10):
            self.ledger.record(bullet_id=f"b{i}", content=f"内容{i}")

        recent = self.ledger.load(recent_n=3)
        self.assertEqual(len(recent), 3)
        self.assertEqual(recent[0]["bullet_id"], "b7")
        self.assertEqual(recent[2]["bullet_id"], "b9")

    def test_stats(self):
        """stats返回正确统计。"""
        self.ledger.record(bullet_id="b1", content="A", origin="consolidation")
        self.ledger.record(bullet_id="b2", content="B", origin="manual")
        self.ledger.record(bullet_id="b3", content="C", origin="consolidation")

        info = self.ledger.stats()
        self.assertEqual(info["total_records"], 3)
        self.assertEqual(info["consolidation_records"], 2)
        self.assertIn("C", info["latest_content_preview"][:100])
        self.assertEqual(str(self.path), info["growth_path"])

    def test_load_empty_file(self):
        """文件不存在时返回空列表。"""
        fake = Path(self.tmp) / "nonexistent.jsonl"
        ledger = SoulGrowthLedger(growth_path=fake)
        self.assertEqual(ledger.load(), [])
        self.assertEqual(ledger.stats()["total_records"], 0)


class TestSoulGrowthRollback(unittest.TestCase):
    """回滚路径：标记rollback_reason，不删行。"""

    def test_rollback_marks_reason(self):
        tmp = tempfile.mkdtemp()
        path = Path(tmp) / "growth.jsonl"
        ledger = SoulGrowthLedger(growth_path=path)

        rec = ledger.record(bullet_id="b1", content="内容")
        # 模拟回滚：直接修改记录中的rollback_reason
        # （实际由orchestrator调用，这里测试容器的append-only特性）
        rec["rollback_reason"] = "rehydration_fail: score=0.100"
        # 追加回滚标记（新行，不删旧行）
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        records = ledger.load()
        self.assertEqual(len(records), 2)
        # 原始记录无rollback
        self.assertEqual(records[0].get("rollback_reason", ""), "")
        # 第二条有rollback
        self.assertIn("rehydration_fail", records[1]["rollback_reason"])


# ═══════════════════════════════════════════════════════════════
# ConsolidationOrchestrator 测试（SC2-SC4）
# ═══════════════════════════════════════════════════════════════

class TestOrchestratorNormalLevel(unittest.TestCase):
    """level=normal → acted=False，不动任何模块。"""

    def test_normal_not_acted(self):
        tmp = tempfile.mkdtemp()
        session_id = "test-normal"

        orch = ConsolidationOrchestrator(
            pressure_state_path=Path(tmp) / "pressure.json",
            scorer_state_path=Path(tmp) / "scorer.json",
            growth_path=Path(tmp) / "growth.jsonl",
        )
        # 未记录任何pressure数据 → level="unknown" → not acted
        report = orch.run_cycle(session_id=session_id)
        self.assertFalse(report["acted"])
        self.assertEqual(report["consolidated_n"], 0)


class TestOrchestratorCautionLevel(unittest.TestCase):
    """level=caution → acted=True，growth有写入。"""

    def test_caution_acted_with_growth_write(self):
        tmp = tempfile.mkdtemp()
        session_id = "test-caution"

        scorer_path = Path(tmp) / "scorer.json"
        growth_path = Path(tmp) / "growth.jsonl"

        # 构造环境：pressure在caution水位
        pressure = ContextPressureMonitor(
            state_path=Path(tmp) / "pressure.json",
        )
        pressure.record_usage(session_id, tokens_used=7500, tokens_budget=10000)
        level = pressure.get_pressure_level(session_id)
        self.assertEqual(level, "caution")

        # scorer里放一个高分bullet —— 先写好文件
        pre_scorer = ConsolidationScorer(
            state_path=scorer_path, min_score=4.0,
        )
        pre_scorer.add_bullet(
            "b1", "WSL中I盘=/mnt/i/",
            provenance=Provenance(source="s1", trust=TrustLevel.USER,
                                  endorsed_by="成市"),
            task_gain_pct=10.0,  # score = 1 + 20 = 21 > min_score
        )

        # orchestrator串联 —— 在bullets写入后初始化，才能加载到
        orch = ConsolidationOrchestrator(
            pressure_state_path=Path(tmp) / "pressure.json",
            scorer_state_path=scorer_path,
            growth_path=growth_path,
        )

        report = orch.run_cycle(session_id=session_id)

        # 验证闭环
        self.assertTrue(report["acted"])
        self.assertEqual(report["level"], "caution")
        self.assertGreaterEqual(report["consolidated_n"], 1)
        self.assertEqual(report["failed_n"], 0)

        # 验证growth有写入
        growth = SoulGrowthLedger(growth_path=growth_path)
        records = growth.load()
        self.assertGreaterEqual(len(records), 1)
        self.assertEqual(records[0]["bullet_id"], "b1")
        self.assertEqual(records[0]["origin"], "consolidation")

        # 验证scorer的bullet已标记consolidated（通过orchestrator的scorer）
        self.assertTrue(orch._scorer._bullets["b1"].consolidated)


class TestOrchestratorRehydrationFailRollback(unittest.TestCase):
    """validator返回fail → 标记rollback_reason。"""

    def test_rollback_on_rehydration_fail(self):
        """当validate返回fail时，growth记录应标记rollback_reason。

        rehydration_validator用content==content做validate（同一字符串），
        cosine相似度=1.0，不会fail。所以我们mock validator让fail发生。
        """
        tmp = tempfile.mkdtemp()
        session_id = "test-rollback"

        scorer_path = Path(tmp) / "scorer.json"
        growth_path = Path(tmp) / "growth.jsonl"

        # 构造环境
        pressure = ContextPressureMonitor(
            state_path=Path(tmp) / "pressure.json",
        )
        pressure.record_usage(session_id, tokens_used=8500, tokens_budget=10000)
        self.assertEqual(pressure.get_pressure_level(session_id), "critical")

        # 先写好scorer状态
        pre_scorer = ConsolidationScorer(
            state_path=scorer_path, min_score=4.0,
        )
        pre_scorer.add_bullet(
            "b1", "critical知识",
            provenance=Provenance(source="s1", trust=TrustLevel.USER),
            task_gain_pct=10.0,
        )

        # 初始化orchestrator（加载已有状态）
        orch = ConsolidationOrchestrator(
            pressure_state_path=Path(tmp) / "pressure.json",
            scorer_state_path=scorer_path,
            growth_path=growth_path,
        )

        # Mock validator返回fail
        def mock_validate(original, rehydrated, threshold=0.6):
            return {
                "score": 0.1,
                "verdict": "fail",
                "threshold": threshold,
                "backend": "mock",
                "original_len": len(original),
                "rehydrated_len": len(rehydrated),
                "length_ratio": len(rehydrated) / max(len(original), 1),
                "ts": time.time(),
            }

        orch._validator.validate = mock_validate

        report = orch.run_cycle(session_id=session_id)

        self.assertTrue(report["acted"])
        self.assertEqual(report["level"], "critical")
        self.assertGreaterEqual(report["consolidated_n"], 1)
        self.assertGreaterEqual(report["failed_n"], 1)

        # growth记录应有rollback_reason（通过load读取最新行）
        growth = SoulGrowthLedger(growth_path=growth_path)
        records = growth.load()
        # load()返回所有行，最后一行应该是带rollback_reason的
        rollback_records = [r for r in records if r.get("rollback_reason")]
        self.assertGreaterEqual(len(rollback_records), 1)
        self.assertIn("rehydration_fail", rollback_records[0]["rollback_reason"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
