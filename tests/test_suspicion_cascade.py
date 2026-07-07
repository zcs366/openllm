"""SuspicionCascade 测试 — PAL P0-1
2026-07-07
"""
import json
import os
import time
import tempfile
import unittest

from openllm.governance.engine import SuspicionCascade, SuspicionEntry


class TestNewtonCooling(unittest.TestCase):
    """牛顿冷却衰减函数测试。"""

    def setUp(self):
        self.sc = SuspicionCascade(threshold=0.7, lambda_base=0.1)

    def test_age_zero_no_decay(self):
        """age=0 → λ=λ_base，不衰减。"""
        result = self.sc._newton_cooling(0)
        self.assertAlmostEqual(result, 0.1)

    def test_age_60s(self):
        """age=60s → λ≈λ_base/3。"""
        result = self.sc._newton_cooling(60)
        self.assertAlmostEqual(result, 0.1 / (1 + __import__('math').log(61)), places=4)

    def test_age_3600s(self):
        """age=3600s → λ≈λ_base/4.6。"""
        result = self.sc._newton_cooling(3600)
        expected = 0.1 / (1 + __import__('math').log(3601))
        self.assertAlmostEqual(result, expected, places=4)

    def test_age_large(self):
        """age很大时λ趋近0（不完全消失）。"""
        result = self.sc._newton_cooling(86400 * 365)
        self.assertGreater(result, 0)
        self.assertLess(result, 0.01)


class TestCascadeComputation(unittest.TestCase):
    """级联计算测试。"""

    def setUp(self):
        self.sc = SuspicionCascade(threshold=0.7, lambda_base=0.1)

    def test_no_history(self):
        """无历史→级联=0。"""
        cascade = self.sc._compute_cascade(0.8, [])
        self.assertEqual(cascade, 0.0)

    def test_same_direction_cascade(self):
        """同向累加: 历史高+新高→正级联。"""
        history = [0.8, 0.7, 0.9]  # 最近3条都>0.5
        cascade = self.sc._compute_cascade(0.8, history)
        avg = (0.8 + 0.7 + 0.9) / 3
        self.assertAlmostEqual(cascade, avg * 0.3, places=4)

    def test_opposite_direction_cancel(self):
        """反向抵消: 历史低+新高→负级联。"""
        history = [0.2, 0.3, 0.1]  # 最近3条都<0.5
        cascade = self.sc._compute_cascade(0.8, history)
        self.assertAlmostEqual(cascade, -0.8 * 0.2, places=4)


class TestRecordToolCall(unittest.TestCase):
    """record_tool_call 完整链路测试。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.filepath = os.path.join(self.tmpdir, "test_suspicion.jsonl")
        self.sc = SuspicionCascade(
            threshold=0.7, lambda_base=0.1, filepath=self.filepath
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_normal_sequence_no_alert(self):
        """正常序列10次tool call→不应告警。"""
        for i in range(10):
            result = self.sc.record_tool_call(f"tool_{i}", 0.3, "normal call")
            self.assertFalse(result["alert"], f"第{i+1}次不应告警")

    def test_attack_sequence_triggers_alert(self):
        """攻击序列3次异常→应触发告警。"""
        # 正常序列建立基线
        for i in range(5):
            self.sc.record_tool_call(f"tool_{i}", 0.2, "normal")

        # 攻击序列
        alert_count = 0
        for i in range(3):
            result = self.sc.record_tool_call(f"evil_tool_{i}", 0.9, "suspicious")
            if result["alert"]:
                alert_count += 1

        self.assertGreaterEqual(alert_count, 1, "至少应触发1次告警")

    def test_final_score_bounded(self):
        """最终分数必须在[0, 1]范围内。"""
        for score in [0.0, 0.5, 1.0, 1.5, -0.5]:
            result = self.sc.record_tool_call("tool", max(0.0, min(1.0, score)))
            self.assertGreaterEqual(result["final_score"], 0.0)
            self.assertLessEqual(result["final_score"], 1.0)

    def test_jsonl_written(self):
        """JSONL文件正确写入。"""
        self.sc.record_tool_call("test_tool", 0.5, "test reason")
        self.assertTrue(os.path.exists(self.filepath))

        with open(self.filepath, "r") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)

        obj = json.loads(lines[0])
        self.assertEqual(obj["tool_name"], "test_tool")
        self.assertEqual(obj["suspicion_score"], 0.5)
        self.assertEqual(obj["reason"], "test reason")
        self.assertIn("entry_id", obj)
        self.assertIn("timestamp", obj)

    def test_history_load(self):
        """历史记录正确加载。"""
        # 写入2条记录
        self.sc.record_tool_call("tool_a", 0.4)
        self.sc.record_tool_call("tool_b", 0.6)

        # 新实例加载历史
        sc2 = SuspicionCascade(filepath=self.filepath)
        self.assertEqual(len(sc2._entries), 2)
        self.assertEqual(sc2._entries[0].tool_name, "tool_a")
        self.assertEqual(sc2._entries[1].tool_name, "tool_b")

    def test_recent_entries(self):
        """get_recent返回正确数量。"""
        for i in range(15):
            self.sc.record_tool_call(f"tool_{i}", 0.3)

        recent = self.sc.get_recent(5)
        self.assertEqual(len(recent), 5)
        self.assertEqual(recent[-1]["tool_name"], "tool_14")

    def test_alert_count(self):
        """告警计数正确。"""
        self.sc.record_tool_call("ok_tool", 0.3)
        self.sc.record_tool_call("evil_tool", 0.9)
        self.sc.record_tool_call("ok_tool2", 0.2)

        count = self.sc.get_alert_count(window_seconds=3600)
        self.assertEqual(count, 1)

    def test_performance_under_50ms(self):
        """10次record_tool_call总延迟<500ms（含IO）。"""
        start = time.time()
        for i in range(10):
            self.sc.record_tool_call(f"tool_{i}", 0.5)
        elapsed = (time.time() - start) * 1000
        self.assertLess(elapsed, 500, f"10次调用耗时{elapsed:.0f}ms，应<500ms")


if __name__ == "__main__":
    unittest.main()
