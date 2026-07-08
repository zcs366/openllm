"""
iai/router.py — IAI快路径路由引擎 (≤150行)

工程法典①：能用代码就别用模型。IAI快路径=纯状态机路由，零LLM，<300ms。
赫淮斯托斯天启：心跳必须是纯状态机永远不等LLM。
阿瑞斯天启：0.3秒窗口内生死已分。
赫尔墨斯红线：路由决策不得终止对话。
"""
import time
import logging
from typing import Any, Optional

from openllm.core.router import RuleRouter, PhaseAction, RouteDecision, RoutingContext
from openllm.iai.event_bus import EventBus, BaseEventEmitter

logger = logging.getLogger("openllm.iai.router")

EVENT_ROUTING_DECIDED = "routing.decided"


class IAIRouter(BaseEventEmitter):
    """IAI快路径路由引擎 — 封装 RuleRouter，增加事件驱动层。

    快路径: 纯规则引擎, <300ms, 零LLM调用。
    慢路径: analyze_async() 预留接口，实际LLM调用由外部提供。
    红线: 路由决策不得终止对话 (PhaseAction 只有 RUN/SKIP/DEGRADED)。
    """

    BODY_NAME = "IAI"
    SIMPLE_EVENTS = frozenset({"greeting", "ack", "thanks", "confirm"})
    COMPLEX_EVENTS = frozenset({"file_op", "code_exec", "deploy", "error"})

    def __init__(self, bus: Optional[EventBus] = None, enabled: bool = True):
        self._own_bus = bus is None
        super().__init__(bus or EventBus())
        self._router = RuleRouter(enabled=enabled)
        self._overrides: dict[str, list[RouteDecision]] = {}
        self._route_count = 0

    def route(self, ctx: RoutingContext) -> list[RouteDecision]:
        """快路径路由: 规则引擎 → 红线检查 → 发布事件。返回4阶段决策。"""
        t0 = time.time()
        # 安全网: user_input为None时降级为全量执行
        if not ctx.user_input:
            from dataclasses import replace
            ctx = replace(ctx, user_input="")
        decisions = self._overrides.pop(self._key(ctx), None) or self._router.route(ctx)
        decisions = self._redline(decisions)
        self._emit(decisions, ctx)
        elapsed_ms = (time.time() - t0) * 1000
        self._route_count += 1
        if elapsed_ms > 300:
            logger.warning("[IAI] 快路径超时 %.1fms > 300ms", elapsed_ms)
        return decisions

    def route_event_type(self, event_type: str, ctx: RoutingContext) -> list[RouteDecision]:
        """按事件类型选择路由策略: SIMPLE→精简, COMPLEX→全量, 其他→规则引擎。"""
        if event_type in self.SIMPLE_EVENTS:
            decisions = [
                RouteDecision("PLAN", PhaseAction.RUN, f"{event_type}·理解意图"),
                RouteDecision("ACT", PhaseAction.SKIP, f"{event_type}·无需工具"),
                RouteDecision("OBSERVE", PhaseAction.SKIP, f"{event_type}·无需观察"),
                RouteDecision("REFLECT", PhaseAction.DEGRADED, f"{event_type}·轻量反思"),
            ]
        elif event_type in self.COMPLEX_EVENTS:
            decisions = self._router.all_run(f"{event_type}·全阶段激活")
        else:
            return self.route(ctx)
        decisions = self._redline(decisions)
        self._emit(decisions, ctx)
        self._route_count += 1
        return decisions

    def set_override(self, key: str, decisions: list[RouteDecision]) -> None:
        """设置路由覆盖（下一次匹配时生效并清除）。"""
        self._overrides[key] = decisions

    async def analyze_async(self, ctx: RoutingContext, **kwargs: Any) -> Optional[list[RouteDecision]]:
        """慢路径接口预留 — 实际LLM分析由外部提供。返回None表示应使用快路径。"""
        return None

    def get_stats(self) -> dict:
        """获取路由统计。"""
        stats = self._router.get_stats()
        stats["iai_route_count"] = self._route_count
        return stats

    # ── 内部 ──

    @staticmethod
    def _redline(decisions: list[RouteDecision]) -> list[RouteDecision]:
        """红线检查: PhaseAction 只允许 RUN/SKIP/DEGRADED，否则降级。"""
        safe = []
        for d in decisions:
            if d.action not in (PhaseAction.RUN, PhaseAction.SKIP, PhaseAction.DEGRADED):
                logger.warning("[IAI红线] 非法决策 %s, 降级为 DEGRADED", d.action)
                safe.append(RouteDecision(d.phase, PhaseAction.DEGRADED, f"红线覆盖:{d.reason}"))
            else:
                safe.append(d)
        return safe

    def _emit(self, decisions: list[RouteDecision], ctx: RoutingContext) -> None:
        """发布路由决策到EventBus。"""
        self.emit_event(EVENT_ROUTING_DECIDED, {
            "decisions": {d.phase: d.action.name for d in decisions},
            "input_preview": (ctx.user_input or "")[:100],
            "risk_level": ctx.risk_level,
        })

    def _key(self, ctx: RoutingContext) -> str:
        return (ctx.user_input or "").strip().lower()[:50]
