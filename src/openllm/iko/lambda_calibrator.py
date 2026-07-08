"""
lambda_calibrator.py — IKO λ 自校准器
======================================

根据用户反馈信号动态校准λ值（输出质量置信度）。
λ ∈ [0.0, 1.0]，初始值0.5，通过反馈回路自校准。

校准规则（七神约束）：
- 赫淮斯托斯：λ必须有自校准回路，不能靠时间自然增长
- 阿瑞斯：高置信度错误（REJECTED+高confidence）触发λ大幅下降
- 德墨忒尔：λ历史必须持久化（JSON文件）

信号响应：
  ACCEPTED + 高置信度(>0.7)  → λ += 0.02
  ACCEPTED + 低置信度(≤0.7)  → λ += 0.01
  REJECTED + 高置信度(>0.7)  → λ -= 0.05（阿瑞斯约束：大幅惩罚）
  REJECTED + 低置信度(≤0.7)  → λ -= 0.02
  CLARIFIED                  → λ -= 0.03 + 触发transparent rollback
  IGNORED                    → λ不变

回滚判定：
  - 连续3次CLARIFIED → True
  - 单次高置信度REJECTED → True
  - λ跌至0.3以下 → True
"""

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from openllm.iko.feedback_collector import FeedbackSignal


# ── λ 边界约束 ──
LAMBDA_MIN: float = 0.0
LAMBDA_MAX: float = 1.0
LAMBDA_DEFAULT: float = 0.5
ROLLBACK_THRESHOLD: float = 0.3
CLARIFIED_STREAK_LIMIT: int = 3


@dataclass
class LambdaState:
    """λ 校准器状态。

    Attributes:
        value: 当前λ值，范围 [0.0, 1.0]
        confidence: 模型对λ值本身的置信度
        last_calibration: 上次校准的Unix时间戳
        error_count: 累计错误（REJECTED）计数
        recovery_count: 累计恢复（ACCEPTED）计数
    """

    value: float = LAMBDA_DEFAULT
    confidence: float = 1.0
    last_calibration: float = 0.0
    error_count: int = 0
    recovery_count: int = 0


class LambdaCalibrator:
    """λ 自校准器。

    根据用户反馈信号动态校准λ值。维护连续CLARIFIED/REJECTED计数，
    支持透明回滚触发和JSON持久化。

    Args:
        storage_path: 持久化JSON文件路径，None则仅内存存储。

    Example:
        >>> cal = LambdaCalibrator()
        >>> cal.update(FeedbackSignal.ACCEPTED, 0.8)
        >>> cal.get_current_lambda()  # > 0.5
    """

    def __init__(self, storage_path: Optional[Path] = None) -> None:
        """初始化校准器。

        Args:
            storage_path: 持久化文件路径（JSON），None则仅内存存储。
        """
        self._storage_path = storage_path
        self._state = LambdaState()
        self._consecutive_clarified: int = 0
        self._last_high_confidence_rejected: bool = False

        if storage_path and storage_path.exists():
            self._load_from_disk()

    def update(self, signal: FeedbackSignal, output_confidence: float) -> None:
        """根据反馈信号校准λ。

        校准后自动clamp到 [0.0, 1.0]，更新状态元数据。

        Args:
            signal: 用户反馈信号
            output_confidence: 模型对该次输出的置信度 [0.0, 1.0]
        """
        current = self._state.value
        high_conf = output_confidence > 0.7

        if signal == FeedbackSignal.ACCEPTED:
            delta = 0.02 if high_conf else 0.01
            self._state.value = current + delta
            self._state.recovery_count += 1
            # 恶化重置连续计数
            self._consecutive_clarified = 0
            self._last_high_confidence_rejected = False

        elif signal == FeedbackSignal.REJECTED:
            if high_conf:
                # 阿瑞斯约束：高置信度错误大幅惩罚
                delta = -0.05
                self._last_high_confidence_rejected = True
            else:
                delta = -0.02
                self._last_high_confidence_rejected = False
            self._state.value = current + delta
            self._state.error_count += 1
            self._consecutive_clarified = 0

        elif signal == FeedbackSignal.CLARIFIED:
            self._state.value = current - 0.03
            self._consecutive_clarified += 1
            self._last_high_confidence_rejected = False

        elif signal == FeedbackSignal.IGNORED:
            # λ不变，但重置恶化标记
            self._consecutive_clarified = 0
            self._last_high_confidence_rejected = False

        # MODIFIED 不在规范中，保守处理：不改变λ
        # Clamp到合法范围
        self._state.value = max(LAMBDA_MIN, min(LAMBDA_MAX, self._state.value))
        self._state.last_calibration = time.time()

    def should_rollback(self) -> bool:
        """判断是否需要透明回滚。

        触发条件（任一满足即True）：
        - 连续3次CLARIFIED
        - 单次高置信度REJECTED
        - λ跌至0.3以下

        Returns:
            True表示应触发透明回滚。
        """
        if self._consecutive_clarified >= CLARIFIED_STREAK_LIMIT:
            return True
        if self._last_high_confidence_rejected:
            return True
        if self._state.value < ROLLBACK_THRESHOLD:
            return True
        return False

    def trigger_transparency_rollback(self, domain: str) -> None:
        """强制提高输出透明度（透明回滚）。

        将λ强制提升至0.6（保守安全区），并记录回滚事件。
        典型用途：用户连续追问或高置信度错误后，系统主动提高输出可解释性。

        Args:
            domain: 触发回滚的领域标识符（如 "math", "code", "general"）
        """
        self._state.value = max(0.6, self._state.value)
        self._state.confidence = max(0.3, self._state.confidence)
        self._consecutive_clarified = 0
        self._last_high_confidence_rejected = False
        self._state.last_calibration = time.time()

    def get_current_lambda(self) -> float:
        """返回当前λ值。

        Returns:
            当前λ值，范围 [0.0, 1.0]。
        """
        return self._state.value

    def get_state(self) -> LambdaState:
        """返回当前状态的快照。

        Returns:
            LambdaState实例的深拷贝。
        """
        return LambdaState(
            value=self._state.value,
            confidence=self._state.confidence,
            last_calibration=self._state.last_calibration,
            error_count=self._state.error_count,
            recovery_count=self._state.recovery_count,
        )

    def save_to_disk(self) -> None:
        """将λ状态持久化到JSON文件（德墨忒尔约束）。"""
        if self._storage_path is None:
            return
        data = {
            "state": asdict(self._state),
            "consecutive_clarified": self._consecutive_clarified,
            "last_high_confidence_rejected": self._last_high_confidence_rejected,
        }
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._storage_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _load_from_disk(self) -> None:
        """从JSON文件加载λ状态。"""
        if self._storage_path is None or not self._storage_path.exists():
            return
        try:
            data = json.loads(self._storage_path.read_text(encoding="utf-8"))
            state_data = data.get("state", {})
            self._state = LambdaState(
                value=state_data.get("value", LAMBDA_DEFAULT),
                confidence=state_data.get("confidence", 1.0),
                last_calibration=state_data.get("last_calibration", 0.0),
                error_count=state_data.get("error_count", 0),
                recovery_count=state_data.get("recovery_count", 0),
            )
            self._consecutive_clarified = data.get("consecutive_clarified", 0)
            self._last_high_confidence_rejected = data.get(
                "last_high_confidence_rejected", False
            )
        except (json.JSONDecodeError, KeyError):
            pass
