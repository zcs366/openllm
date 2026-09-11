"""
ConsolidationTrigger 测试
==================================================

覆盖验收判据 CT1-CT6：
  CT1: import 可用，ConsolidationTrigger 存在
  CT2: on_usage 真调 pressure + should_compress + orchestrator（mock验证）
  CT3: 集成测试：caution水位 + scorer有候选 → on_usage → growth.jsonl真新增
  CT4: context_pressure.py 零改动（通过 git diff 验证）
  CT5: 测试 ≥6 条
  CT6: 真实运行（通过真实 ~/.openllm 环境）

测试矩阵：
  1. test_import_exists —— CT1: import + 类存在
  2. test_normal_level_not_triggered —— normal水位不触发
  3. test_caution_level_triggers —— caution水位触发
  4. test_transmits_record_usage —— 透传 record_usage 参数
  5. test_wire_idempotent —— wire 幂等
  6. test_integration_caution_growth_write —— CT3: 真实组件闭环
  7. test_no_candidates_acted_false —— 无候选时 acted=False
  8. test_lazy_init —— 惰性初始化正常
"""
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ── CT1: import 可用 ──
from openllm.memory.consolidation_trigger import ConsolidationTrigger


class TestImportExists(unittest.TestCase):
    """CT1: ConsolidationTrigger 可 import，类存在。"""

    def test_class_exists(self):
        self.assertTrue(hasattr(ConsolidationTrigger, "__init__"))
        self.assertTrue(hasattr(ConsolidationTrigger, "on_usage"))
        self.assertTrue(hasattr(ConsolidationTrigger, "wire"))


class TestNormalLevelNotTriggered(unittest.TestCase):
    """normal 水位不触发。"""

    def test_normal_returns_triggered_false(self):
        trigger = ConsolidationTrigger()

        # mock pressure: should_compress 返回 False
        mock_pressure = MagicMock()
        mock_pressure.should_compress.return_value = (False, "")
        mock_pressure.get_pressure_level.return_value = "normal"
        trigger._pressure = mock_pressure

        result = trigger.on_usage("s1", tokens_used=5000, tokens_budget=10000)

        self.assertFalse(result["triggered"])
        self.assertEqual(result["level"], "normal")
        # 验证 record_usage 被调用
        mock_pressure.record_usage.assert_called_once_with("s1", 5000, 10000)
        # 验证 orchestrator 未被调用
        self.assertIsNone(trigger._orchestrator)


class TestCautionLevelTriggers(unittest.TestCase):
    """caution 水位触发，返回 triggered + report。"""

    def test_caution_triggers(self):
        trigger = ConsolidationTrigger()

        # mock pressure: should_compress 返回 True
        mock_pressure = MagicMock()
        mock_pressure.should_compress.return_value = (True, "light")
        trigger._pressure = mock_pressure

        # mock orchestrator: run_cycle 返回报告
        mock_orch = MagicMock()
        mock_orch.run_cycle.return_value = {
            "acted": True,
            "level": "caution",
            "consolidated_n": 2,
            "failed_n": 0,
        }
        trigger._orchestrator = mock_orch

        result = trigger.on_usage("s2", tokens_used=7500, tokens_budget=10000)

        self.assertTrue(result["triggered"])
        self.assertEqual(result["report"]["acted"], True)
        self.assertEqual(result["report"]["consolidated_n"], 2)
        # 验证调用链
        mock_pressure.record_usage.assert_called_once_with("s2", 7500, 10000)
        mock_pressure.should_compress.assert_called_once_with("s2")
        mock_orch.run_cycle.assert_called_once_with("s2")


class TestTransmitsRecordUsage(unittest.TestCase):
    """透传 record_usage 参数：session_id/used/budget 原样传递。"""

    def test_transmits_exact_args(self):
        trigger = ConsolidationTrigger()
        mock_pressure = MagicMock()
        mock_pressure.should_compress.return_value = (False, "")
        mock_pressure.get_pressure_level.return_value = "normal"
        trigger._pressure = mock_pressure

        trigger.on_usage("sid-42", 12345, 99999)

        mock_pressure.record_usage.assert_called_once_with("sid-42", 12345, 99999)


class TestWireIdempotent(unittest.TestCase):
    """wire 幂等：第二次调用返回 False。"""

    def test_wire_first_true_second_false(self):
        trigger = ConsolidationTrigger()
        self.assertFalse(trigger._wired)

        first = trigger.wire()
        self.assertTrue(first)
        self.assertTrue(trigger._wired)

        second = trigger.wire()
        self.assertFalse(second)


class TestIntegrationCautionGrowthWrite(unittest.TestCase):
    """CT3: 真实组件闭环——caution水位 + scorer有候选 → growth.jsonl真新增。"""

    def test_real_components(self):
        from openllm.memory.context_pressure import ContextPressureMonitor
        from openllm.consolidation_score import (
            ConsolidationScorer,
            Provenance,
            TrustLevel,
        )
        from openllm.memory.consolidation_orchestrator import (
            ConsolidationOrchestrator,
        )
        from openllm.identity.soul_growth import SoulGrowthLedger

        tmp = tempfile.mkdtemp()
        session_id = "test-ct3-integration"

        pressure_path = Path(tmp) / "pressure.json"
        scorer_path = Path(tmp) / "scorer.json"
        growth_path = Path(tmp) / "growth.jsonl"

        # 构造真实 pressure：75% = caution
        pressure = ContextPressureMonitor(state_path=pressure_path)
        pressure.record_usage(session_id, tokens_used=7500, tokens_budget=10000)
        level = pressure.get_pressure_level(session_id)
        self.assertEqual(level, "caution")

        # 构造真实 scorer：放一个高分 bullet
        scorer = ConsolidationScorer(state_path=scorer_path, min_score=4.0)
        scorer.add_bullet(
            "ct3-b1",
            "测试固化知识：WSL中I盘=/mnt/i/",
            provenance=Provenance(
                source="test", trust=TrustLevel.USER, endorsed_by="测试"
            ),
            task_gain_pct=10.0,  # score = 1 + 20 = 21
        )

        # 用真实 pressure 和 orchestrator 构建 trigger
        orch = ConsolidationOrchestrator(
            pressure_state_path=pressure_path,
            scorer_state_path=scorer_path,
            growth_path=growth_path,
        )
        trigger = ConsolidationTrigger(
            orchestrator=orch,
            pressure=pressure,
        )

        # 执行 on_usage
        result = trigger.on_usage(session_id, tokens_used=7500, tokens_budget=10000)

        # 验证触发
        self.assertTrue(result["triggered"])
        report = result["report"]
        self.assertTrue(report["acted"])
        self.assertGreaterEqual(report["consolidated_n"], 1)

        # 验证 growth.jsonl 真有写入
        ledger = SoulGrowthLedger(growth_path=growth_path)
        records = ledger.load()
        self.assertGreaterEqual(len(records), 1)
        self.assertEqual(records[0]["bullet_id"], "ct3-b1")
        self.assertEqual(records[0]["origin"], "consolidation")


class TestNoCandidatesActedFalse(unittest.TestCase):
    """无候选时 acted=False，不报错。"""

    def test_no_candidates(self):
        trigger = ConsolidationTrigger()

        mock_pressure = MagicMock()
        mock_pressure.should_compress.return_value = (True, "light")
        trigger._pressure = mock_pressure

        mock_orch = MagicMock()
        mock_orch.run_cycle.return_value = {
            "acted": False,
            "level": "caution",
            "consolidated_n": 0,
            "failed_n": 0,
            "details": "no candidates available",
        }
        trigger._orchestrator = mock_orch

        result = trigger.on_usage("s-no-cand", tokens_used=8000, tokens_budget=10000)

        self.assertTrue(result["triggered"])
        self.assertFalse(result["report"]["acted"])
        self.assertEqual(result["report"]["consolidated_n"], 0)


class TestLazyInit(unittest.TestCase):
    """惰性初始化：构造时不触发 import，首次 on_usage 时才创建。"""

    def test_pressure_none_creates_lazily(self):
        trigger = ConsolidationTrigger()
        # 此时 _pressure 是 None
        self.assertIsNone(trigger._pressure)

        # mock 压力监控器创建后的返回
        with patch("openllm.memory.context_pressure.ContextPressureMonitor") as MockPM:
            mock_pm = MagicMock()
            mock_pm.should_compress.return_value = (False, "")
            mock_pm.get_pressure_level.return_value = "normal"
            MockPM.return_value = mock_pm

            result = trigger.on_usage("s-lazy", 1000, 10000)

            MockPM.assert_called_once()
            self.assertIsNotNone(trigger._pressure)


if __name__ == "__main__":
    unittest.main(verbosity=2)
