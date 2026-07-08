"""
feedback_collector.py — IKO 输出反馈收集器
============================================

收集用户对IKO输出的隐式/显式反馈，推断用户偏好，
动态调整输出密度。

信号检测规则：
- 立即追问 → CLARIFIED
- 采纳执行 → ACCEPTED
- 修改后使用 → MODIFIED
- 沉默30s换话题 → IGNORED
- 明确拒绝 → REJECTED

密度调整因子范围：[0.3, 3.0]（雅典娜约束：防止自我吞噬）
"""

import json
import time
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class FeedbackSignal(Enum):
    """用户反馈信号枚举。"""
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    CLARIFIED = "clarified"
    MODIFIED = "modified"
    IGNORED = "ignored"


class OutputFeedbackCollector:
    """IKO输出反馈收集器。

    收集用户行为反馈，推断用户偏好，计算密度调整因子。

    Attributes:
        _history: 反馈历史字典，按 user_id 分组
        _output_metadata: 输出元数据，按 output_id 分组
        _storage_path: 持久化存储路径（德墨忒尔约束：可持久化）
    """

    # 雅典娜约束：密度调整因子有界[0.3, 3.0]
    DENSITY_MIN: float = 0.3
    DENSITY_MAX: float = 3.0
    DENSITY_DEFAULT: float = 1.0

    # 信号权重映射
    SIGNAL_WEIGHTS: dict[FeedbackSignal, float] = {
        FeedbackSignal.ACCEPTED: 1.0,
        FeedbackSignal.MODIFIED: 0.7,
        FeedbackSignal.CLARIFIED: 0.5,
        FeedbackSignal.IGNORED: -0.3,
        FeedbackSignal.REJECTED: -1.0,
    }

    def __init__(self, storage_path: Optional[Path] = None) -> None:
        """初始化反馈收集器。

        Args:
            storage_path: 持久化文件路径（JSON），None则仅内存存储。
        """
        self._history: dict[str, list[dict[str, Any]]] = {}
        self._output_metadata: dict[str, dict[str, Any]] = {}
        self._storage_path = storage_path
        if storage_path and storage_path.exists():
            self._load_from_disk()

    def detect_signal(self, user_action: dict, output_id: str) -> FeedbackSignal:
        """根据用户行为检测反馈信号。

        检测规则：
        - 用户在输出后立即提问（time_delta < 5s）→ CLARIFIED
        - 用户直接采纳执行（action_type == 'execute'）→ ACCEPTED
        - 用户修改后使用（action_type == 'modify'）→ MODIFIED
        - 用户沉默后换话题（time_delta > 30s 且无引用输出）→ IGNORED
        - 用户明确拒绝（action_type == 'reject'）→ REJECTED

        Args:
            user_action: 用户行为字典，包含以下可选字段：
                - action_type: 行为类型 ('execute'|'modify'|'reject'|'ignore')
                - time_delta: 距上次输出的时间（秒）
                - text: 用户输入文本
                - referenced_output: 引用的输出ID
            output_id: 输出标识符

        Returns:
            检测到的反馈信号。
        """
        action_type = user_action.get("action_type", "")
        time_delta = user_action.get("time_delta", 999.0)
        text = user_action.get("text", "")

        # 显式拒绝
        if action_type == "reject":
            signal = FeedbackSignal.REJECTED
        # 显式采纳
        elif action_type == "execute":
            signal = FeedbackSignal.ACCEPTED
        # 显式修改
        elif action_type == "modify":
            signal = FeedbackSignal.MODIFIED
        # 沉默后换话题
        elif time_delta > 30.0 and not user_action.get("referenced_output"):
            signal = FeedbackSignal.IGNORED
        # 立即追问（text非空且time_delta < 5s）
        elif text and time_delta < 5.0:
            signal = FeedbackSignal.CLARIFIED
        else:
            # 默认为忽略
            signal = FeedbackSignal.IGNORED

        # 记录反馈
        self._record_feedback(output_id, signal, user_action)
        return signal

    def _record_feedback(
        self, output_id: str, signal: FeedbackSignal, user_action: dict
    ) -> None:
        """记录反馈到历史。

        Args:
            output_id: 输出标识符
            signal: 检测到的信号
            user_action: 用户行为字典
        """
        user_id = user_action.get("user_id", "anonymous")
        entry = {
            "output_id": output_id,
            "signal": signal.value,
            "timestamp": time.time(),
            "user_action": user_action,
        }
        self._history.setdefault(user_id, []).append(entry)
        self._output_metadata[output_id] = {
            "signal": signal.value,
            "timestamp": time.time(),
            "user_id": user_id,
        }

    def get_user_preferences(self, user_id: str) -> dict[str, Any]:
        """从历史反馈推断用户偏好。

        分析用户历史中的反馈信号分布，推断偏好维度。

        Args:
            user_id: 用户标识符

        Returns:
            偏好字典，包含：
            - accepted_ratio: 接受率 [0.0, 1.0]
            - rejection_ratio: 拒绝率 [0.0, 1.0]
            - preferred_density: 推测的偏好密度方向 ('high'|'medium'|'low')
            - total_interactions: 总交互次数
            - signal_distribution: 各信号的计数
        """
        history = self._history.get(user_id, [])
        total = len(history)

        if total == 0:
            return {
                "accepted_ratio": 0.0,
                "rejection_ratio": 0.0,
                "preferred_density": "medium",
                "total_interactions": 0,
                "signal_distribution": {},
            }

        # 统计各信号分布
        signal_dist: dict[str, int] = {}
        for entry in history:
            sig = entry.get("signal", "ignored")
            signal_dist[sig] = signal_dist.get(sig, 0) + 1

        accepted_count = signal_dist.get(FeedbackSignal.ACCEPTED.value, 0)
        rejected_count = signal_dist.get(FeedbackSignal.REJECTED.value, 0)
        modified_count = signal_dist.get(FeedbackSignal.MODIFIED.value, 0)
        clarified_count = signal_dist.get(FeedbackSignal.CLARIFIED.value, 0)

        accepted_ratio = accepted_count / total
        rejection_ratio = rejected_count / total

        # 推断偏好密度方向
        positive_signals = accepted_count + modified_count * 0.5
        negative_signals = rejected_count + signal_dist.get(
            FeedbackSignal.IGNORED.value, 0
        ) * 0.5

        if positive_signals > negative_signals * 1.5:
            preferred_density = "high"
        elif negative_signals > positive_signals * 1.5:
            preferred_density = "low"
        else:
            preferred_density = "medium"

        return {
            "accepted_ratio": round(accepted_ratio, 3),
            "rejection_ratio": round(rejection_ratio, 3),
            "preferred_density": preferred_density,
            "total_interactions": total,
            "signal_distribution": signal_dist,
        }

    def should_adjust_density(self, user_id: str) -> float:
        """计算密度调整因子。

        基于用户历史反馈，返回密度调整因子。

        规则：
        - 高接受率 → 增加密度（最大3.0）
        - 高拒绝率 → 降低密度（最小0.3）
        - 无历史 → 返回默认值1.0

        Args:
            user_id: 用户标识符

        Returns:
            密度调整因子，范围 [0.3, 3.0]。
        """
        prefs = self.get_user_preferences(user_id)

        if prefs["total_interactions"] == 0:
            return self.DENSITY_DEFAULT

        # 基于加权信号计算调整因子
        history = self._history.get(user_id, [])
        weighted_sum = 0.0
        for entry in history:
            try:
                sig = FeedbackSignal(entry.get("signal", "ignored"))
                weighted_sum += self.SIGNAL_WEIGHTS.get(sig, 0.0)
            except ValueError:
                continue

        # 归一化到 [-1, 1]
        normalized = weighted_sum / max(len(history), 1)
        # 映射到 [0.3, 3.0]
        factor = self.DENSITY_DEFAULT + normalized * (self.DENSITY_MAX - self.DENSITY_DEFAULT) / 2.0

        # 雅典娜约束：确保在边界内
        return max(self.DENSITY_MIN, min(self.DENSITY_MAX, round(factor, 2)))

    def save_to_disk(self) -> None:
        """将反馈历史持久化到JSON文件（德墨忒尔约束）。"""
        if self._storage_path is None:
            return
        data = {
            "history": self._history,
            "output_metadata": self._output_metadata,
        }
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._storage_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _load_from_disk(self) -> None:
        """从JSON文件加载反馈历史。"""
        if self._storage_path is None or not self._storage_path.exists():
            return
        try:
            data = json.loads(self._storage_path.read_text(encoding="utf-8"))
            self._history = data.get("history", {})
            self._output_metadata = data.get("output_metadata", {})
        except (json.JSONDecodeError, KeyError):
            pass

    def get_output_feedback(self, output_id: str) -> Optional[dict[str, Any]]:
        """获取指定输出的反馈记录。

        Args:
            output_id: 输出标识符

        Returns:
            反馈记录字典，不存在则返回None。
        """
        return self._output_metadata.get(output_id)
