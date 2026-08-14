"""
trace_spec.py — ETAS风格的Trace Spec编译器
=============================================

灵感来源：ETAS TraceSpecAlgebra——编译时trace约束演算。

核心概念：
  spec定义了"什么action序列是被允许的"
  编译为monitor——在运行时验证action trace的合法性

支持三种约束：
  1. allow/deny — 允许/拒绝特定action
  2. temporal ordering — A >> B (A必须先于B)
  3. scope/resource — action参数必须满足条件

设计约束（对齐ETAS）：
  - TraceSpec是编译时抽象，不是运行时回调
  - Monitor是prefix-based的：检查每个trace前缀是否合法
  - 不可证明的约束 → residual runtime check
"""

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from .action_trace import (
    ActionTraceStore, TraceEvent, EventPhase, DenialCause,
)


class SpecKind(str, Enum):
    """spec种类——对齐ETAS trace spec kinds."""
    ALLOW = "allow"       # 允许此action
    DENY = "deny"         # 拒绝此action
    TEMPORAL = "temporal"  # 时序约束: A >> B


@dataclass
class TemporalConstraint:
    """
    时序约束——对齐ETAS temporal obligations。
    
    "approval >> write" = approval必须先于write出现在trace中。
    """
    first: str   # 必须先出现的action
    second: str  # 必须后出现的action
    
    def check_prefix(self, events: list[TraceEvent]) -> bool:
        """
        检查trace前缀是否满足时序约束。
        
        如果trace中出现了second但没有first → 违规。
        """
        has_first = False
        for e in events:
            if e.action_name == self.first and e.phase == EventPhase.COMMIT:
                has_first = True
            if e.action_name == self.second and e.phase == EventPhase.COMMIT:
                if not has_first:
                    return False
        return True


@dataclass
class TraceSpec:
    """
    一个trace规范——定义action序列的合法性约束。
    
    对齐ETAS: TraceSpecAlgebra objects compiled to monitors.
    """
    name: str
    constraints: list[tuple[SpecKind, str]] = field(default_factory=list)
    temporal_constraints: list[TemporalConstraint] = field(default_factory=list)
    # residual checks: 编译时无法证明的约束
    residual_checks: list[str] = field(default_factory=list)
    
    def allow(self, action: str) -> "TraceSpec":
        """允许此action。"""
        self.constraints.append((SpecKind.ALLOW, action))
        return self
    
    def deny(self, action: str) -> "TraceSpec":
        """拒绝此action。"""
        self.constraints.append((SpecKind.DENY, action))
        return self
    
    def require_before(self, first: str, second: str) -> "TraceSpec":
        """时序约束：first必须先于second。"""
        self.temporal_constraints.append(TemporalConstraint(first, second))
        return self
    
    def compile(self) -> "TraceMonitor":
        """编译为运行时monitor。"""
        return TraceMonitor(self)


class TraceMonitor:
    """
    从TraceSpec编译的运行时monitor。
    
    对齐ETAS: prefix-based enforcement。
    每个request/commit事件到来时，monitor检查前缀是否合法。
    """
    
    def __init__(self, spec: TraceSpec):
        self.spec = spec
        self._denied_actions = {
            action for kind, action in spec.constraints
            if kind == SpecKind.DENY
        }
        self._allowed_actions = {
            action for kind, action in spec.constraints
            if kind == SpecKind.ALLOW
        }
    
    def check_event(self, event: TraceEvent, trace: ActionTraceStore) -> tuple[bool, Optional[str]]:
        """
        检查单个事件是否通过monitor。
        
        返回 (passed, denial_reason)。
        对齐ETAS: δ_Π(q_τ, event) ∈ Bad → denied。
        """
        # 只检查commit事件（request事件不触发deny）
        if event.phase != EventPhase.COMMIT:
            return True, None
        
        # deny检查
        if event.action_name in self._denied_actions:
            return False, f"策略拒绝: {event.action_name} 在denied列表中"
        
        # temporal约束检查
        for tc in self.spec.temporal_constraints:
            if event.action_name == tc.second:
                if not tc.check_prefix(trace.events):
                    return False, f"时序违规: {tc.first} 必须先于 {tc.second}"
        
        return True, None
    
    def check_full_trace(self, trace: ActionTraceStore) -> list[tuple[TraceEvent, Optional[str]]]:
        """
        检查完整trace的所有commit事件。
        返回 [(event, denial_reason|None), ...]。
        """
        results = []
        for event in trace.events:
            if event.phase == EventPhase.COMMIT:
                passed, reason = self.check_event(event, trace)
                results.append((event, reason))
        return results


# ── 预定义Spec模板 ─────────────────────────────────

def spec_approval_before(action: str) -> TraceSpec:
    """
    ApprovalBefore模板——对齐ETAS ApprovalBefore<A>。
    
    specApprovalBefore<A> = +Approval.request & +A & (Approval.request >> A)
    """
    return (TraceSpec(f"ApprovalBefore<{action}>")
            .allow("Approval.request")
            .allow(action)
            .require_before("Approval.request", action))


def spec_dry_run_only() -> TraceSpec:
    """只允许dry-run（无commit）。"""
    return (TraceSpec("DryRunOnly")
            .deny("File.write")
            .deny("Email.send")
            .deny("Shell.exec"))


def spec_read_only() -> TraceSpec:
    """只读模式——禁止所有写操作。"""
    return (TraceSpec("ReadOnly")
            .deny("File.write")
            .deny("Shell.exec")
            .deny("Memory.write"))
