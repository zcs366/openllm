"""遗忘维 ForgettingCurve — 指数衰减主题打分（PAL T-F-9）。

遗忘是智能的另一半骨架。厌倦律IoR=短时抑制已处理主题（秒/分钟级），
ForgettingCurve=长时遗忘衰减（小时/天级）。两者衔接：
  - IoR.get_recent() 提供"最近已处理"→ForgettingCurve当作衰减加速器
  - 已处理+久远的主题得分最低（双重遗忘：IoR短抑制 + ForgettingCurve长衰减）

设计：
  - score(topic, age_seconds) -> float [0,1]：指数衰减打分
  - record_access(topic)：再巩固刷新（访问重置age）
  - apply(topics) -> list：按衰减打分排序（旧的排后/弱化）
  - 半衰期可配：默认1天=86400s

红线：
  - 不重构ISA核心，只加打分层
  - 不替代temperature_engine.py（那是MemoryBus级别的记忆温度）
  - ForgettingCurve作用于topic级，温度引擎作用于entry级
  - 保守接入build_context：只在已有记忆召回后加衰减重排

军规十一/十三：增强不替代。已有temperature_engine覆盖entry级衰减，
本模块补充topic级衰减（build_context用，不影响存储层）。
"""

import math
import time
from typing import Dict, List, Optional, Tuple


# ── 默认参数 ──────────────────────────────────────

DEFAULT_HALF_LIFE_S: float = 86400.0  # 1天 = 86400秒
MIN_SCORE: float = 0.01               # 最低分数（永不归零，保留微弱记忆）
IoR_ACCELERATOR: float = 0.7          # IoR已处理主题的衰减加速因子（0.7=加速30%）


class ForgettingCurve:
    """指数衰减主题打分器。

    公式: score = 2^(-age / half_life) × ior_modifier

    - age: 距上次访问的秒数
    - half_life: 半衰期（可配，默认86400s=1天）
    - ior_modifier: 已处理主题×IoR_ACCELERATOR（更弱）

    再巩固机制:
      record_access(topic) 重置age→0，模拟人类再巩固（reconsolidation）
      每次访问后衰减从头开始，但访问计数累积（未来可做间隔重复）
    """

    def __init__(self, half_life_s: float = DEFAULT_HALF_LIFE_S):
        self._half_life = half_life_s
        self._access_times: Dict[str, float] = {}    # topic -> last_access_time
        self._access_counts: Dict[str, int] = {}     # topic -> access_count
        self._creation_time = time.time()

    @property
    def half_life(self) -> float:
        return self._half_life

    def record_access(self, topic: str):
        """再巩固刷新——访问后衰减重置。

        每次调用将topic的last_access_time更新为now，
        同时递增access_count（为未来间隔重复预留）。
        """
        now = time.time()
        self._access_times[topic] = now
        self._access_counts[topic] = self._access_counts.get(topic, 0) + 1

    def _get_age(self, topic: str, now: Optional[float] = None) -> float:
        """获取topic距上次访问的age秒数。未访问过的topic使用创建时间。"""
        if now is None:
            now = time.time()
        last = self._access_times.get(topic, self._creation_time)
        return max(0.0, now - last)

    def score(self, topic: str, age_seconds: Optional[float] = None,
              ior_handled: bool = False) -> float:
        """指数衰减打分。

        Args:
            topic: 主题标识
            age_seconds: 直接指定age（跳过内部计算，用于测试/外部age）
            ior_handled: 是否已被IoR标记为已处理（True=加速衰减）

        Returns:
            float [MIN_SCORE, 1.0]：衰减分数，越大越"新鲜"
        """
        if age_seconds is None:
            age_seconds = self._get_age(topic)

        # 核心公式: 2^(-age / half_life) → 半衰期处得0.5
        raw = 2.0 ** (-age_seconds / self._half_life)
        score = max(MIN_SCORE, raw)

        # IoR加速：已处理主题衰减更快
        if ior_handled:
            score *= IoR_ACCELERATOR

        return round(score, 6)

    def apply(self, topics: List[str],
              ior_handled_set: Optional[set] = None) -> List[Tuple[str, float]]:
        """按衰减打分排序（高分=新鲜→低分=陈旧）。

        Args:
            topics: 主题列表
            ior_handled_set: IoR已处理主题集合（用于衰减加速）

        Returns:
            [(topic, score), ...] 按score降序排列
        """
        if ior_handled_set is None:
            ior_handled_set = set()

        scored = []
        for t in topics:
            s = self.score(t, ior_handled=(t in ior_handled_set))
            scored.append((t, s))

        # 按score降序（新鲜的在前）
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def get_access_count(self, topic: str) -> int:
        """获取topic的访问次数（间隔重复预留）。"""
        return self._access_counts.get(topic, 0)

    def stats(self) -> dict:
        """统计信息。"""
        return {
            "topics_tracked": len(self._access_times),
            "total_accesses": sum(self._access_counts.values()),
            "half_life_s": self._half_life,
            "half_life_human": _format_duration(self._half_life),
        }


def _format_duration(seconds: float) -> str:
    """将秒数格式化为人类可读的时间。"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}min"
    elif seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    else:
        return f"{seconds / 86400:.1f}d"
