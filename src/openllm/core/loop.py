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

        # Phase 1: PLAN — 理解意图，拆解任务
        ctx.phase = LoopPhase.PLAN
        if self.on_plan:
            ctx.plan = self.on_plan(user_input, self._memory, self._identity)

        # Phase 2: ACT — 选择工具并执行
        # （工具执行由Tool Executor层负责，这里只是占位）
        ctx.phase = LoopPhase.ACT

        # Phase 3: OBSERVE — 读取结果
        ctx.phase = LoopPhase.OBSERVE

        # Phase 4: REFLECT — 修正理解，更新记忆
        ctx.phase = LoopPhase.REFLECT
        ctx.self_state = self.check_vitals()
        if self.on_reflect:
            ctx.reflection = self.on_reflect(ctx)

        self.state = AgentState.WORKING
        return ctx

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
