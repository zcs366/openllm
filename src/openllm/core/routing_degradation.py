"""
路由降级链 — 路由失败时Agent还能活

降级链：Router规则 → fallback到全RUN → 默认路由 → 用户干预
降级事件写入JSONL日志（tag=degraded）
降级计数器>3次/小时 → 告警
"""

import time
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from collections import deque


class DegradationTracker:
    """降级追踪器——监控降级频率，触发告警。"""
    
    def __init__(self, window_seconds: int = 3600, threshold: int = 3):
        """
        Args:
            window_seconds: 滑动窗口大小（秒），默认1小时
            threshold: 窗口内降级次数阈值，超过则告警
        """
        self.window_seconds = window_seconds
        self.threshold = threshold
        self._events: deque = deque()  # (timestamp, reason)
    
    def record(self, reason: str):
        """记录一次降级事件。"""
        now = time.time()
        self._events.append((now, reason))
        self._cleanup(now)
    
    def _cleanup(self, now: float):
        """清理过期事件。"""
        cutoff = now - self.window_seconds
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()
    
    @property
    def count(self) -> int:
        """当前窗口内的降级次数。"""
        self._cleanup(time.time())
        return len(self._events)
    
    @property
    def should_alert(self) -> bool:
        """是否需要告警。"""
        return self.count >= self.threshold
    
    def get_status(self) -> dict:
        """获取降级状态摘要。"""
        self._cleanup(time.time())
        return {
            "degradation_count": self.count,
            "threshold": self.threshold,
            "window_seconds": self.window_seconds,
            "should_alert": self.should_alert,
            "recent_reasons": [r for _, r in list(self._events)[-5:]],
        }


def fallback_route(reason: str = "降级fallback"):
    """
    降级路由——所有阶段全RUN。
    当Router规则匹配失败或抛出异常时调用。
    """
    from .router import RouteDecision, PhaseAction
    return [
        RouteDecision("PLAN", PhaseAction.RUN, reason),
        RouteDecision("ACT", PhaseAction.RUN, reason),
        RouteDecision("OBSERVE", PhaseAction.RUN, reason),
        RouteDecision("REFLECT", PhaseAction.RUN, reason),
    ]


def safe_route(router, routing_ctx, tracker: Optional[DegradationTracker] = None):
    """
    安全路由——包裹Router.route()，异常时自动降级。
    
    降级链：
    1. 正常路由 → 返回决策
    2. 路由异常 → fallback到全RUN
    3. 连续降级>3次/小时 → 告警
    
    Returns:
        (decisions, degraded: bool, alert: bool)
    """
    try:
        decisions = router.route(routing_ctx)
        return decisions, False, False
    except Exception as e:
        # 降级：全RUN
        degraded_decisions = fallback_route(f"路由异常降级: {e}")
        
        # 记录降级
        alert = False
        if tracker:
            tracker.record(str(e))
            alert = tracker.should_alert
        
        return degraded_decisions, True, alert
