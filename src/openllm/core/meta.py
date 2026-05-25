"""
OpenLLM Meta-Cognition — 元认知层。

Agent自我感知系统。对标全行业零的空白。
认知仪表盘：上下文占用率、注意力衰减、工具成功率、推理步数趋势。
自救行为：过载时自动触发压缩、回退、重启。
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable


class CognitiveState(Enum):
    """认知状态。"""
    HEALTHY = "healthy"       # 正常
    FATIGUED = "fatigued"     # 疲劳（窗口>70%）
    OVERLOADED = "overloaded" # 过载（窗口>85%）
    DEGRADED = "degraded"     # 退化（工具成功率下降）


@dataclass
class MetaSnapshot:
    """认知快照——单次评估的全部指标。"""
    timestamp: float = field(default_factory=time.time)

    # 上下文压力
    context_used_pct: float = 0.0
    context_total: int = 8192

    # 工具调用统计
    tool_calls_total: int = 0
    tool_calls_success: int = 0
    tool_success_rate: float = 1.0

    # 对话统计
    turn_count: int = 0
    avg_response_latency_ms: float = 0.0

    # 综合判断
    cognitive_state: CognitiveState = CognitiveState.HEALTHY
    overloaded: bool = False
    should_compress: bool = False
    should_rest: bool = False

    # 趋势（与前次对比）
    trend: str = "stable"  # "improving" / "stable" / "declining"


class CognitiveDashboard:
    """
    认知仪表盘。

    四维监控信号（议题16）：
      1. 上下文窗口占用率 — >80%触发压缩
      2. 工具调用成功率 — <70%触发策略回退
      3. 推理步数趋势 — 上升=过载
      4. 响应延迟趋势 — 上升=模型压力

    自救行为：
      - 主动触发上下文压缩
      - 建议用户简化问题
      - 标记"需要休息"
    """

    def __init__(
        self,
        max_context: int = 8192,
        overload_threshold: float = 0.80,
        fatigue_threshold: float = 0.70,
        degrade_threshold: float = 0.60,
    ):
        self.max_context = max_context
        self.overload_threshold = overload_threshold
        self.fatigue_threshold = fatigue_threshold
        self.degrade_threshold = degrade_threshold

        # 累积统计
        self.tool_total = 0
        self.tool_success = 0
        self.latency_sum = 0.0
        self.latency_count = 0

        # 历史快照（趋势分析）
        self.history: list[MetaSnapshot] = []
        self.last_snapshot: Optional[MetaSnapshot] = None

        # 自救回调
        self.on_overload: Optional[Callable] = None
        self.on_fatigue: Optional[Callable] = None
        self.on_degrade: Optional[Callable] = None

    def record_tool_call(self, success: bool) -> None:
        """记录一次工具调用。"""
        self.tool_total += 1
        if success:
            self.tool_success += 1

    def record_latency(self, ms: float) -> None:
        """记录一次响应延迟。"""
        self.latency_sum += ms
        self.latency_count += 1

    @property
    def success_rate(self) -> float:
        if self.tool_total == 0:
            return 1.0
        return self.tool_success / self.tool_total

    @property
    def avg_latency(self) -> float:
        if self.latency_count == 0:
            return 0.0
        return self.latency_sum / self.latency_count

    def evaluate(self, context_used: int, turn_count: int) -> MetaSnapshot:
        """
        评估当前认知状态。

        Returns: MetaSnapshot（含自救建议）
        """
        context_pct = context_used / self.max_context if self.max_context > 0 else 0
        rate = self.success_rate
        latency = self.avg_latency

        snap = MetaSnapshot(
            context_used_pct=round(context_pct * 100, 1),
            context_total=self.max_context,
            tool_calls_total=self.tool_total,
            tool_calls_success=self.tool_success,
            tool_success_rate=round(rate, 3),
            turn_count=turn_count,
            avg_response_latency_ms=round(latency, 1),
        )

        # 状态判断
        if context_pct > self.overload_threshold:
            snap.cognitive_state = CognitiveState.OVERLOADED
            snap.overloaded = True
            snap.should_compress = True
        elif context_pct > self.fatigue_threshold:
            snap.cognitive_state = CognitiveState.FATIGUED
            snap.should_compress = True
        elif rate < self.degrade_threshold:
            snap.cognitive_state = CognitiveState.DEGRADED
            snap.should_rest = True
        else:
            snap.cognitive_state = CognitiveState.HEALTHY

        # 趋势分析
        if self.last_snapshot:
            if context_pct > self.last_snapshot.context_used_pct / 100 + 0.1:
                snap.trend = "declining"
            elif rate < self.last_snapshot.tool_success_rate - 0.1:
                snap.trend = "declining"

        # 存储
        self.last_snapshot = snap
        self.history.append(snap)
        if len(self.history) > 100:
            self.history = self.history[-50:]

        # 触发自救
        if snap.overloaded and self.on_overload:
            self.on_overload(snap)
        elif snap.cognitive_state == CognitiveState.FATIGUED and self.on_fatigue:
            self.on_fatigue(snap)

        return snap

    def self_awareness_report(self) -> str:
        """
        生成自我感知报告。

        这是Agent的自述——"我现在的状态"。
        对标议题16：自我状态感知。
        """
        if not self.last_snapshot:
            return "🟢 刚苏醒，尚无认知数据。"

        snap = self.last_snapshot
        state_emoji = {
            CognitiveState.HEALTHY: "🟢",
            CognitiveState.FATIGUED: "🟡",
            CognitiveState.OVERLOADED: "🔴",
            CognitiveState.DEGRADED: "⚠️",
        }

        lines = [
            f"\n{state_emoji.get(snap.cognitive_state, '❓')} 认知状态：{snap.cognitive_state.value}",
            f"  · 上下文占用：{snap.context_used_pct}% ({snap.turn_count}轮对话)",
            f"  · 工具成功率：{snap.tool_success_rate*100:.0f}% ({snap.tool_calls_success}/{snap.tool_calls_total})",
            f"  · 平均延迟：{snap.avg_response_latency_ms:.0f}ms",
        ]

        if snap.should_compress:
            lines.append("  ⚡ 建议：压缩上下文")
        if snap.should_rest:
            lines.append("  ⚡ 建议：简化任务或稍作休息")
        if snap.trend == "declining":
            lines.append("  📉 趋势：认知负载上升中")

        return "\n".join(lines)

    def reset_ephemeral(self):
        """重置瞬时指标（每次会话开始时调用）。"""
        self.latency_sum = 0.0
        self.latency_count = 0
        # 保留工具统计（跨会话趋势有价值）


# ── 自救处理器 ─────────────────────────────────────

class SelfRescue:
    """
    Agent自救系统。

    发现问题 → 自动采取措施，不只报告。
    对标健康报告的"只报警不洒水"问题的修复。
    """

    def __init__(self, dashboard: CognitiveDashboard):
        self.dashboard = dashboard
        self.rescue_count = 0
        self.rescue_log: list[dict] = []

    def on_overload(self, snap: MetaSnapshot) -> str:
        """过载自救：触发上下文压缩。"""
        self.rescue_count += 1
        action = {
            "type": "compress_context",
            "trigger": f"context={snap.context_used_pct}%",
            "time": snap.timestamp,
        }
        self.rescue_log.append(action)
        return (
            f"🔴 上下文过载({snap.context_used_pct}%)。"
            f"建议立即压缩或开启新会话。"
        )

    def on_fatigue(self, snap: MetaSnapshot) -> str:
        """疲劳预警。"""
        self.rescue_count += 1
        action = {
            "type": "fatigue_warning",
            "trigger": f"context={snap.context_used_pct}%",
            "time": snap.timestamp,
        }
        self.rescue_log.append(action)
        return (
            f"🟡 认知疲劳({snap.context_used_pct}%)。"
            f"建议简化当前对话。"
        )

    def on_degrade(self, snap: MetaSnapshot) -> str:
        """退化预警：工具成功率过低。"""
        self.rescue_count += 1
        action = {
            "type": "degradation_alert",
            "trigger": f"success_rate={snap.tool_success_rate}",
            "time": snap.timestamp,
        }
        self.rescue_log.append(action)
        return (
            f"⚠️ 工具成功率下降({snap.tool_success_rate*100:.0f}%)。"
            f"建议检查工具配置。"
        )
