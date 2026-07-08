"""
OpenLLM Agent Loop — 六维Agent躯体核心循环。

plan → act → observe → reflect

从Claude Code学来，但增加了自我状态维度。
每次循环后检查：我累了吗？我忘了吗？我需要修正理解吗？
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional
import time

from .router import RuleRouter, RoutingContext, PhaseAction, create_router


class LoopPhase(Enum):
    """Agent循环四阶段。"""
    PLAN = auto()       # 理解意图，拆解任务
    ACT = auto()        # 选择工具，执行调用
    OBSERVE = auto()    # 读取结果，判断成败
    REFLECT = auto()    # 修正理解，更新记忆


class AgentState(Enum):
    """Agent生命周期状态。"""
    WAKING = auto()     # 苏醒：加载记忆，重建身份
    WORKING = auto()    # 工作：正常循环
    SLEEPING = auto()   # 休眠：压缩状态，写Δ胶囊
    OVERLOADED = auto() # 过载：上下文>80%，触发自救


@dataclass
class TurnContext:
    """单轮对话上下文。"""
    user_input: str
    phase: LoopPhase = LoopPhase.PLAN
    plan: list[str] = field(default_factory=list)      # 任务拆解
    tool_calls: list[dict] = field(default_factory=list) # 工具调用记录
    observations: list[str] = field(default_factory=list) # 观察结果
    reflection: str = ""                                 # 反思笔记
    self_state: dict = field(default_factory=dict)       # 自我状态快照


@dataclass
class AgentLoop:
    """
    Agent核心循环。

    四阶段状态机，全程无GPU依赖。
    每次循环后自我检查：上下文占用、认知负载、注意力衰减。
    """
    name: str = "OpenLLM"
    max_context_tokens: int = 8192
    context_used: int = 0
    state: AgentState = AgentState.WAKING
    turn_count: int = 0
    # 路由器（默认启用）
    router: RuleRouter = field(default_factory=RuleRouter)

    # 外部注入的回调（由Memory/Identity/Security层提供）
    on_plan: Optional[Callable] = None
    on_reflect: Optional[Callable] = None
    on_state_change: Optional[Callable] = None

    def wake(self, identity: dict, memory: dict) -> "AgentLoop":
        """苏醒：加载身份和记忆。"""
        self.state = AgentState.WAKING
        self._identity = identity
        self._memory = memory
        return self

    def check_vitals(self) -> dict:
        """检查自身状态。返回认知仪表盘数据。"""
        context_pct = self.context_used / self.max_context_tokens if self.max_context_tokens else 0
        vitals = {
            "context_used_pct": round(context_pct * 100, 1),
            "turn_count": self.turn_count,
            "state": self.state.name,
            "overloaded": context_pct > 0.8,
        }
        if context_pct > 0.8 and self.state != AgentState.OVERLOADED:
            self.state = AgentState.OVERLOADED
            if self.on_state_change:
                self.on_state_change("OVERLOADED", vitals)
        return vitals

    def turn(self, user_input: str) -> TurnContext:
        """执行一轮完整循环。"""
        ctx = TurnContext(user_input=user_input)
        self.turn_count += 1

        # ── 路由决策 ──
        context_pct = self.context_used / self.max_context_tokens if self.max_context_tokens else 0
        consecutive = self.router.get_consecutive_identical(user_input)
        routing_ctx = RoutingContext(
            user_input=user_input,
            turn_count=self.turn_count,
            context_used_pct=context_pct,
            consecutive_identical=consecutive,
            last_tool_success=self._last_tool_success if hasattr(self, '_last_tool_success') else None,
        )
        decisions = self.router.route(routing_ctx)
        self.router.record_intent(user_input)

        # ── 按路由决策执行各阶段 ──
        for dec in decisions:
            phase = LoopPhase[dec.phase]
            ctx.phase = phase
            
            if dec.action == PhaseAction.SKIP:
                # 跳过：记录原因但不执行
                ctx.observations.append(f"[路由] {dec.phase} 已跳过: {dec.reason}")
                continue
            
            if dec.action == PhaseAction.DEGRADED:
                # 降级：执行轻量版
                if phase == LoopPhase.REFLECT:
                    ctx.reflection = f"[轻量反思] {dec.reason}"
                elif phase == LoopPhase.OBSERVE:
                    ctx.observations.append(f"[轻量观察] {dec.reason}")
                else:
                    # 其他阶段的降级 = 正常执行（未来可细化）
                    self._execute_phase(phase, ctx)
                continue
            
            # PhaseAction.RUN: 正常执行
            self._execute_phase(phase, ctx)

        # 记录路由统计
        ctx.self_state["routing"] = {
            "skipped": sum(1 for d in decisions if d.action == PhaseAction.SKIP),
            "degraded": sum(1 for d in decisions if d.action == PhaseAction.DEGRADED),
            "total": len(decisions),
        }

        # 自检
        vitals = self.check_vitals()
        return ctx

    def _execute_phase(self, phase: LoopPhase, ctx: TurnContext):
        """执行单个阶段。"""
        if phase == LoopPhase.PLAN:
            if self.on_plan:
                ctx.plan = self.on_plan(ctx.user_input, self._memory, self._identity)
        elif phase == LoopPhase.ACT:
            # 工具执行由Tool Executor层负责
            pass
        elif phase == LoopPhase.OBSERVE:
            # 结果观察由Tool Executor层负责
            pass
        elif phase == LoopPhase.REFLECT:
            if self.on_reflect:
                ctx.reflection = self.on_reflect(ctx)

    # ── User Correction 追踪（P1） ──────────────────
    _corrections: list = field(default_factory=list)
    _last_tool_success: Optional[bool] = None

    def sleep(self) -> dict:
        """休眠：生成当前状态的Δ差分，准备写入胶囊。"""
        self.state = AgentState.SLEEPING
        delta = {
            "turns": self.turn_count,
            "context_used": self.context_used,
            "state": self.state.name,
            "timestamp": time.time(),
        }
        return delta

    def __repr__(self) -> str:
        return f"AgentLoop(name={self.name}, state={self.state.name}, turns={self.turn_count})"
