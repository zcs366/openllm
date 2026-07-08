"""
tests/test_lambda_calibrator.py — lambda_calibrator 单元测试
=============================================================

覆盖：
1. ACCEPTED高置信度 → λ上升
2. REJECTED高置信度 → λ大幅下降（阿瑞斯约束）
3. CLARIFIED → λ下降 + 触发回滚
4. 连续3次CLARIFIED → should_rollback=True
5. λ范围[0, 1]
6. 持久化 save/load
"""

import json
import tempfile
from pathlib import Path

import pytest

from openllm.iko.feedback_collector import FeedbackSignal
from openllm.iko.lambda_calibrator import (
    LambdaCalibrator,
    LambdaState,
    LAMBDA_DEFAULT,
)


@pytest.fixture
def calibrator() -> LambdaCalibrator:
    """创建一个内存校准器实例。"""
    return LambdaCalibrator()


@pytest.fixture
def tmp_calibrator(tmp_path: Path) -> LambdaCalibrator:
    """创建一个带持久化的校准器实例。"""
    storage = tmp_path / "lambda.json"
    return LambdaCalibrator(storage_path=storage)


# ── 校准规则测试 ──


class TestCalibrationRules:
    """反馈信号校准规则测试套件。"""

    def test_accepted_high_confidence_increases_lambda(
        self, calibrator: LambdaCalibrator
    ) -> None:
        """ACCEPTED + 高置信度(>0.7) → λ += 0.02。"""
        initial = calibrator.get_current_lambda()
        calibrator.update(FeedbackSignal.ACCEPTED, 0.8)
        assert calibrator.get_current_lambda() == pytest.approx(initial + 0.02)

    def test_accepted_low_confidence_increases_lambda(
        self, calibrator: LambdaCalibrator
    ) -> None:
        """ACCEPTED + 低置信度(≤0.7) → λ += 0.01。"""
        initial = calibrator.get_current_lambda()
        calibrator.update(FeedbackSignal.ACCEPTED, 0.5)
        assert calibrator.get_current_lambda() == pytest.approx(initial + 0.01)

    def test_rejected_high_confidence_large_decrease(
        self, calibrator: LambdaCalibrator
    ) -> None:
        """REJECTED + 高置信度(>0.7) → λ -= 0.05（阿瑞斯约束）。"""
        initial = calibrator.get_current_lambda()
        calibrator.update(FeedbackSignal.REJECTED, 0.8)
        assert calibrator.get_current_lambda() == pytest.approx(initial - 0.05)

    def test_rejected_low_confidence_decrease(
        self, calibrator: LambdaCalibrator
    ) -> None:
        """REJECTED + 低置信度(≤0.7) → λ -= 0.02。"""
        initial = calibrator.get_current_lambda()
        calibrator.update(FeedbackSignal.REJECTED, 0.4)
        assert calibrator.get_current_lambda() == pytest.approx(initial - 0.02)

    def test_clarified_decreases_lambda(
        self, calibrator: LambdaCalibrator
    ) -> None:
        """CLARIFIED → λ -= 0.03。"""
        initial = calibrator.get_current_lambda()
        calibrator.update(FeedbackSignal.CLARIFIED, 0.5)
        assert calibrator.get_current_lambda() == pytest.approx(initial - 0.03)

    def test_ignored_no_change(self, calibrator: LambdaCalibrator) -> None:
        """IGNORED → λ不变。"""
        initial = calibrator.get_current_lambda()
        calibrator.update(FeedbackSignal.IGNORED, 0.5)
        assert calibrator.get_current_lambda() == pytest.approx(initial)


# ── 回滚判定测试 ──


class TestRollback:
    """回滚判定测试套件。"""

    def test_three_consecutive_clarified_triggers_rollback(
        self, calibrator: LambdaCalibrator
    ) -> None:
        """连续3次CLARIFIED → should_rollback() = True。"""
        for _ in range(2):
            calibrator.update(FeedbackSignal.CLARIFIED, 0.5)
            assert not calibrator.should_rollback()
        calibrator.update(FeedbackSignal.CLARIFIED, 0.5)
        assert calibrator.should_rollback()

    def test_high_confidence_rejected_triggers_rollback(
        self, calibrator: LambdaCalibrator
    ) -> None:
        """单次高置信度REJECTED → should_rollback() = True。"""
        calibrator.update(FeedbackSignal.REJECTED, 0.8)
        assert calibrator.should_rollback()

    def test_low_lambda_triggers_rollback(
        self, calibrator: LambdaCalibrator
    ) -> None:
        """λ跌至0.3以下 → should_rollback() = True。"""
        # 连续拒绝让λ降到0.3以下
        for _ in range(20):
            calibrator.update(FeedbackSignal.REJECTED, 0.5)
        assert calibrator.get_current_lambda() < 0.3
        assert calibrator.should_rollback()

    def test_clarified_streak_reset_on_accepted(
        self, calibrator: LambdaCalibrator
    ) -> None:
        """ACCEPTED重置连续CLARIFIED计数。"""
        for _ in range(2):
            calibrator.update(FeedbackSignal.CLARIFIED, 0.5)
        calibrator.update(FeedbackSignal.ACCEPTED, 0.8)
        # 再来2次CLARIFIED不应触发（连续计数已重置）
        for _ in range(2):
            calibrator.update(FeedbackSignal.CLARIFIED, 0.5)
        assert not calibrator.should_rollback()

    def test_trigger_transparency_rollback(
        self, calibrator: LambdaCalibrator
    ) -> None:
        """触发透明回滚后λ提升至≥0.6，连续计数归零。"""
        for _ in range(3):
            calibrator.update(FeedbackSignal.CLARIFIED, 0.5)
        assert calibrator.should_rollback()
        calibrator.trigger_transparency_rollback("code")
        assert calibrator.get_current_lambda() >= 0.6
        # 回滚后不再触发
        assert not calibrator.should_rollback()


# ── λ 边界测试 ──


class TestLambdaBounds:
    """λ边界约束测试套件。"""

    def test_lambda_bounded_upper(self, calibrator: LambdaCalibrator) -> None:
        """λ不超过1.0。"""
        for _ in range(200):
            calibrator.update(FeedbackSignal.ACCEPTED, 0.9)
        assert calibrator.get_current_lambda() <= 1.0

    def test_lambda_bounded_lower(self, calibrator: LambdaCalibrator) -> None:
        """λ不低于0.0。"""
        for _ in range(200):
            calibrator.update(FeedbackSignal.REJECTED, 0.8)
        assert calibrator.get_current_lambda() >= 0.0

    def test_mixed_signals_bounded(
        self, calibrator: LambdaCalibrator
    ) -> None:
        """混合信号后λ仍在[0, 1]内。"""
        for i in range(100):
            sig = FeedbackSignal.ACCEPTED if i % 2 == 0 else FeedbackSignal.REJECTED
            calibrator.update(sig, 0.8)
        lam = calibrator.get_current_lambda()
        assert 0.0 <= lam <= 1.0

    def test_clarified_bounded(self, calibrator: LambdaCalibrator) -> None:
        """连续CLARIFIED后λ不低于0.0。"""
        for _ in range(50):
            calibrator.update(FeedbackSignal.CLARIFIED, 0.5)
        assert calibrator.get_current_lambda() >= 0.0


# ── 持久化测试 ──


class TestPersistence:
    """持久化测试套件。"""

    def test_save_and_load(self, tmp_path: Path) -> None:
        """保存后重新加载，状态应完整。"""
        storage = tmp_path / "lambda.json"
        c1 = LambdaCalibrator(storage_path=storage)
        c1.update(FeedbackSignal.ACCEPTED, 0.8)
        c1.update(FeedbackSignal.REJECTED, 0.9)
        c1.save_to_disk()

        c2 = LambdaCalibrator(storage_path=storage)
        assert c2.get_current_lambda() == pytest.approx(c1.get_current_lambda())
        state = c2.get_state()
        assert state.recovery_count == 1
        assert state.error_count == 1

    def test_save_load_preserves_clarified_streak(
        self, tmp_path: Path
    ) -> None:
        """保存/加载保留连续CLARIFIED计数。"""
        storage = tmp_path / "lambda_clarified.json"
        c1 = LambdaCalibrator(storage_path=storage)
        for _ in range(2):
            c1.update(FeedbackSignal.CLARIFIED, 0.5)
        c1.save_to_disk()

        c2 = LambdaCalibrator(storage_path=storage)
        # 还差1次CLARIFIED即可触发回滚
        c2.update(FeedbackSignal.CLARIFIED, 0.5)
        assert c2.should_rollback()

    def test_empty_storage_load(self, tmp_path: Path) -> None:
        """空文件不会崩溃。"""
        storage = tmp_path / "empty_lambda.json"
        cal = LambdaCalibrator(storage_path=storage)
        assert cal.get_current_lambda() == LAMBDA_DEFAULT

    def test_load_corrupt_json(self, tmp_path: Path) -> None:
        """损坏JSON不会崩溃。"""
        storage = tmp_path / "corrupt_lambda.json"
        storage.write_text("not valid json {{{", encoding="utf-8")
        cal = LambdaCalibrator(storage_path=storage)
        assert cal.get_current_lambda() == LAMBDA_DEFAULT


# ── 状态快照测试 ──


class TestStateSnapshot:
    """状态快照测试。"""

    def test_get_state_returns_copy(
        self, calibrator: LambdaCalibrator
    ) -> None:
        """get_state返回独立副本，修改不影响原状态。"""
        calibrator.update(FeedbackSignal.ACCEPTED, 0.8)
        state = calibrator.get_state()
        original_value = state.value
        # 修改快照不影响原状态
        state.value = 0.0
        assert calibrator.get_current_lambda() == pytest.approx(original_value)

    def test_error_and_recovery_counts(
        self, calibrator: LambdaCalibrator
    ) -> None:
        """error_count和recovery_count正确累计。"""
        calibrator.update(FeedbackSignal.ACCEPTED, 0.8)
        calibrator.update(FeedbackSignal.REJECTED, 0.9)
        calibrator.update(FeedbackSignal.ACCEPTED, 0.8)
        state = calibrator.get_state()
        assert state.recovery_count == 2
        assert state.error_count == 1
