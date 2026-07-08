"""user_state.py — ika 用户状态感知模块 (≤150行)

七神天启：用户是第六乘数，不是使用者。ika 要让 Agent 看见人。
追踪交互节奏、纠正频率、沉默信号，输出参与度评分。
"""
import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from openllm.iai.event_bus import EventBus, Event

logger = logging.getLogger("openllm.iai.user_state")

SILENCE_THRESHOLD = 300.0  # 5 分钟
MAX_INTERVALS = 20         # 保留最近 N 次交互间隔
ENGAGEMENT_WINDOW = 600.0  # 参与度计算窗口（10 分钟）


@dataclass
class UserState:
    """用户交互状态快照。"""
    interaction_frequency: list[float] = field(default_factory=list)
    correction_count: int = 0
    silence_duration: float = 0.0
    last_signal_time: float = 0.0
    total_interactions: int = 0


class UserStateTracker:
    """用户状态追踪器——记录交互、纠正、沉默，输出参与度。

    与 EventBus 集成：状态变化时发布 user.state_changed 事件。
    """
    SOURCE = "USER_STATE"

    def __init__(self, bus: Optional[EventBus] = None,
                 silence_threshold: float = SILENCE_THRESHOLD):
        self._bus = bus
        self._intervals: deque[float] = deque(maxlen=MAX_INTERVALS)
        self._correction_count = 0
        self._last_signal_time = 0.0
        self._total_interactions = 0
        self._silence_threshold = silence_threshold

    # ── 记录 ──

    def record_interaction(self, timestamp: Optional[float] = None) -> UserState:
        """记录一次用户交互。返回更新后的状态快照。"""
        now = timestamp or time.time()
        if self._last_signal_time > 0:
            interval = now - self._last_signal_time
            self._intervals.append(interval)
        self._last_signal_time = now
        self._total_interactions += 1
        self._emit("user.interaction", {
            "total": self._total_interactions,
            "intervals_len": len(self._intervals),
        })
        return self.get_state()

    def record_correction(self, timestamp: Optional[float] = None) -> UserState:
        """记录一次用户纠正（纠正也是一次交互）。"""
        self._correction_count += 1
        state = self.record_interaction(timestamp)
        self._emit("user.correction", {
            "correction_count": self._correction_count,
            "total": self._total_interactions,
        })
        return state

    # ── 查询 ──

    def get_state(self) -> UserState:
        """获取当前用户状态快照。"""
        now = time.time()
        silence = (now - self._last_signal_time) if self._last_signal_time > 0 else 0.0
        return UserState(
            interaction_frequency=list(self._intervals),
            correction_count=self._correction_count,
            silence_duration=max(silence, 0.0),
            last_signal_time=self._last_signal_time,
            total_interactions=self._total_interactions,
        )

    def is_silence_alert(self) -> bool:
        """用户沉默是否超过阈值（默认 5 分钟）。"""
        if self._last_signal_time == 0:
            return False
        return (time.time() - self._last_signal_time) > self._silence_threshold

    def get_engagement_score(self) -> float:
        """0.0–1.0 参与度评分：频率×50% + 数量×30% + 纠正调节×20%。"""
        now = time.time()
        if self._last_signal_time == 0:
            return 0.0

        # 1. 沉默惩罚：超过阈值直接降为 0
        silence = now - self._last_signal_time
        if silence > self._silence_threshold:
            return 0.0

        # 2. 频率分：最近间隔越短越好
        freq_score = 0.0
        if self._intervals:
            avg_interval = sum(self._intervals) / len(self._intervals)
            # 间隔 10s → 1.0，间隔 600s → 0.0
            freq_score = max(0.0, 1.0 - (avg_interval / ENGAGEMENT_WINDOW))

        # 3. 数量分：窗口内交互次数
        recent_count = 0
        for iv in self._intervals:
            if iv <= ENGAGEMENT_WINDOW:
                recent_count += 1
        volume_score = min(recent_count / 10.0, 1.0)  # 10次满分

        # 4. 纠正调节：适度纠正加分（0-15%），过多扣分
        ratio = (self._correction_count / max(self._total_interactions, 1))
        if ratio <= 0.15:
            correction_delta = ratio * 0.5   # 最多 +0.075
        else:
            correction_delta = -(ratio - 0.15) * 2.0  # 超出部分重罚

        # 加权合成
        raw = 0.5 * freq_score + 0.3 * volume_score + 0.2 * correction_delta
        score = max(0.0, min(1.0, raw + 0.2))  # 基线 0.2
        return round(score, 4)

    def _emit(self, event_type: str, payload: dict) -> None:
        if self._bus is None:
            return
        try:
            event = Event(
                source=self.SOURCE,
                type=event_type,
                timestamp=time.time(),
                entropy_score=0.0,
                payload=payload,
            )
            self._bus.publish(event)
        except Exception as e:
            logger.warning(f"[user_state] 事件发布失败: {e}")
