"""
tests/test_feedback_collector.py — feedback_collector 单元测试
==============================================================

覆盖：
1. 信号检测（5种信号）
2. 用户偏好推断
3. 密度调整因子边界
4. 持久化读写
"""

import json
import tempfile
import time
from pathlib import Path

import pytest

from openllm.iko.feedback_collector import (
    FeedbackSignal,
    OutputFeedbackCollector,
)


@pytest.fixture
def collector() -> OutputFeedbackCollector:
    """创建一个内存收集器实例。"""
    return OutputFeedbackCollector()


@pytest.fixture
def tmp_collector(tmp_path: Path) -> OutputFeedbackCollector:
    """创建一个带持久化的收集器实例。"""
    storage = tmp_path / "feedback.json"
    return OutputFeedbackCollector(storage_path=storage)


# ── 信号检测测试 ──


class TestDetectSignal:
    """信号检测测试套件。"""

    def test_detect_accepted(self, collector: OutputFeedbackCollector) -> None:
        """采纳执行 → ACCEPTED。"""
        action = {"user_id": "u1", "action_type": "execute", "time_delta": 2.0}
        signal = collector.detect_signal(action, "out_001")
        assert signal == FeedbackSignal.ACCEPTED

    def test_detect_rejected(self, collector: OutputFeedbackCollector) -> None:
        """明确拒绝 → REJECTED。"""
        action = {"user_id": "u1", "action_type": "reject", "time_delta": 3.0}
        signal = collector.detect_signal(action, "out_002")
        assert signal == FeedbackSignal.REJECTED

    def test_detect_clarified(self, collector: OutputFeedbackCollector) -> None:
        """立即追问 → CLARIFIED。"""
        action = {
            "user_id": "u1",
            "text": "能不能再解释一下？",
            "time_delta": 2.0,
        }
        signal = collector.detect_signal(action, "out_003")
        assert signal == FeedbackSignal.CLARIFIED

    def test_detect_modified(self, collector: OutputFeedbackCollector) -> None:
        """修改后使用 → MODIFIED。"""
        action = {"user_id": "u1", "action_type": "modify", "time_delta": 10.0}
        signal = collector.detect_signal(action, "out_004")
        assert signal == FeedbackSignal.MODIFIED

    def test_detect_ignored(self, collector: OutputFeedbackCollector) -> None:
        """沉默30s换话题 → IGNORED。"""
        action = {
            "user_id": "u1",
            "text": "换个话题吧",
            "time_delta": 35.0,
        }
        signal = collector.detect_signal(action, "out_005")
        assert signal == FeedbackSignal.IGNORED


# ── 偏好推断测试 ──


class TestUserPreferences:
    """用户偏好推断测试套件。"""

    def test_empty_history(self, collector: OutputFeedbackCollector) -> None:
        """无历史时返回默认偏好。"""
        prefs = collector.get_user_preferences("unknown_user")
        assert prefs["total_interactions"] == 0
        assert prefs["preferred_density"] == "medium"
        assert prefs["accepted_ratio"] == 0.0

    def test_high_acceptance(self, collector: OutputFeedbackCollector) -> None:
        """高接受率 → preferred_density 为 high。"""
        for i in range(5):
            action = {"user_id": "u_happy", "action_type": "execute", "time_delta": 1.0}
            collector.detect_signal(action, f"out_h{i}")

        prefs = collector.get_user_preferences("u_happy")
        assert prefs["total_interactions"] == 5
        assert prefs["accepted_ratio"] > 0.8
        assert prefs["preferred_density"] == "high"

    def test_high_rejection(self, collector: OutputFeedbackCollector) -> None:
        """高拒绝率 → preferred_density 为 low。"""
        for i in range(5):
            action = {"user_id": "u_angry", "action_type": "reject", "time_delta": 1.0}
            collector.detect_signal(action, f"out_r{i}")

        prefs = collector.get_user_preferences("u_angry")
        assert prefs["total_interactions"] == 5
        assert prefs["rejection_ratio"] > 0.8
        assert prefs["preferred_density"] == "low"


# ── 密度调整因子测试 ──


class TestDensityAdjustment:
    """密度调整因子测试套件。"""

    def test_default_density(self, collector: OutputFeedbackCollector) -> None:
        """无历史时返回默认密度1.0。"""
        factor = collector.should_adjust_density("new_user")
        assert factor == 1.0

    def test_density_bounded_high(self, collector: OutputFeedbackCollector) -> None:
        """高接受率时密度因子不超过3.0。"""
        for i in range(20):
            action = {"user_id": "u_pure", "action_type": "execute", "time_delta": 0.5}
            collector.detect_signal(action, f"out_b{i}")

        factor = collector.should_adjust_density("u_pure")
        assert 1.0 < factor <= 3.0

    def test_density_bounded_low(self, collector: OutputFeedbackCollector) -> None:
        """高拒绝率时密度因子不低于0.3。"""
        for i in range(20):
            action = {"user_id": "u_hate", "action_type": "reject", "time_delta": 0.5}
            collector.detect_signal(action, f"out_l{i}")

        factor = collector.should_adjust_density("u_hate")
        assert 0.3 <= factor < 1.0

    def test_density_range_clamp(self, collector: OutputFeedbackCollector) -> None:
        """确保因子始终在 [0.3, 3.0] 内。"""
        # 极端接受
        for i in range(100):
            collector.detect_signal(
                {"user_id": "extreme", "action_type": "execute", "time_delta": 0.1},
                f"ext_{i}",
            )
        factor = collector.should_adjust_density("extreme")
        assert 0.3 <= factor <= 3.0


# ── 持久化测试 ──


class TestPersistence:
    """持久化测试套件。"""

    def test_save_and_load(self, tmp_path: Path) -> None:
        """保存后重新加载，历史应完整。"""
        storage = tmp_path / "fb.json"
        c1 = OutputFeedbackCollector(storage_path=storage)
        c1.detect_signal(
            {"user_id": "u_save", "action_type": "execute", "time_delta": 1.0},
            "out_s1",
        )
        c1.save_to_disk()

        c2 = OutputFeedbackCollector(storage_path=storage)
        prefs = c2.get_user_preferences("u_save")
        assert prefs["total_interactions"] == 1

    def test_empty_storage_load(self, tmp_path: Path) -> None:
        """空文件不会崩溃。"""
        storage = tmp_path / "empty.json"
        collector = OutputFeedbackCollector(storage_path=storage)
        assert collector.get_user_preferences("nobody")["total_interactions"] == 0
