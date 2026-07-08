"""
openLLM 规则路由器 v0.1 — 推理步级路由

灵感来源: DMoA (arXiv 2605.15706) 的可微分路由，
但当前版本用规则引擎替代可微分路由（0成本、立即可用）。

核心思想: 不是每个推理步都需要跑全部4个阶段。
路由器根据当前上下文决定哪些阶段激活、哪些跳过。

路由规则 (9条覆盖90%场景):
  - 简单问候/闲聊 → 只跑PLAN+REFLECT（跳过ACT/OBSERVE）
  - 文件操作请求 → 全跑
  - 重复问题 → 跳过PLAN（已知意图）
  - 高风险操作 → 强制全跑+额外验证
  - 上下文过载 → 跳过REFLECT（节省token）
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import re
import time


class PhaseAction(Enum):
    """路由决策：激活/跳过/降级"""
    RUN = auto()       # 正常执行
    SKIP = auto()      # 跳过（不需要）
    DEGRADED = auto()  # 降级执行（轻量版）


@dataclass
class RouteDecision:
    """单个阶段的路由决策"""
    phase: str          # "PLAN" | "ACT" | "OBSERVE" | "REFLECT"
    action: PhaseAction
    reason: str = ""
    confidence: float = 1.0


@dataclass
class RoutingContext:
    """路由决策的输入上下文"""
    user_input: str
    turn_count: int = 0
    context_used_pct: float = 0.0
    last_tool_success: Optional[bool] = None
    consecutive_identical: int = 0  # 连续相同输入次数
    risk_level: str = "low"
    has_pending_tools: bool = False


class RuleRouter:
    """
    规则路由器 — 用if-then规则决定每个阶段的执行策略。
    
    与DMoA的区别:
    - DMoA: Sentence Transformer + GRU + 可微分训练
    - 本路由器: 20条if-then规则，0训练，0额外依赖
    
    与DMoA的共同点:
    - 推理步级路由（不是token级）
    - 根据上下文动态决策
    - 支持稀疏激活（跳过不需要的阶段）
    """
    
    # ── 路由规则表 ──────────────────────────────────
    # 每条规则: (condition_fn, phase_decisions, priority)
    # priority越高越先匹配
    
    # 简单模式: 只需要PLAN+REFLECT
    GREETING_PATTERNS = re.compile(
        r'^(你好|hello|hi|hey|嗨|早|晚安|谢谢|thanks|ok|好的|嗯|哦|知道了)[\s!！.。]*$',
        re.IGNORECASE
    )
    
    # 文件操作: 需要全阶段
    FILE_OPS = re.compile(
        r'(读取|写入|创建|删除|打开|保存|read|write|create|delete|open|save|cat|echo|mkdir)',
        re.IGNORECASE
    )
    
    # 代码相关: 需要全阶段+工具执行
    CODE_OPS = re.compile(
        r'(代码|code|python|script|运行|run|执行|execute|调试|debug|测试|test|import|def |class )',
        re.IGNORECASE
    )
    
    # 搜索/查询: 需要PLAN+ACT(搜索)+OBSERVE(结果)
    QUERY_OPS = re.compile(
        r'(搜索|查找|查询|search|find|look|查一下|帮我找|哪里有)',
        re.IGNORECASE
    )
    
    # 高风险操作: 强制全阶段
    HIGH_RISK = re.compile(
        r'(删除|destroy|sudo|rm |格式化|drop|drop\s+table|不可逆|irreversible)',
        re.IGNORECASE
    )
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.stats = {
            "total_decisions": 0,
            "skipped_phases": 0,
            "degraded_phases": 0,
            "tokens_saved_estimate": 0,
        }
        # 路由历史（用于检测重复模式）
        self._recent_intents: list[str] = []
        self._max_history = 10
    
    def route(self, ctx: RoutingContext) -> list[RouteDecision]:
        """
        根据上下文做出路由决策。
        
        Returns: 4个Phase的决策列表
        """
        if not self.enabled:
            return self._all_run("路由已禁用")
        
        self.stats["total_decisions"] += 1
        decisions = []
        
        # ── 规则匹配（按优先级从高到低）──
        
        # 规则1: 高风险操作 → 强制全跑
        if self.HIGH_RISK.search(ctx.user_input) or ctx.risk_level in ("high", "critical"):
            return self._all_run("高风险操作·全阶段激活")
        
        # 规则2: 上下文过载 → 最大化跳过
        if ctx.context_used_pct > 0.8:
            decisions = [
                RouteDecision("PLAN", PhaseAction.RUN, "上下文过载·只做核心"),
                RouteDecision("ACT", PhaseAction.SKIP, "过载·跳过工具执行"),
                RouteDecision("OBSERVE", PhaseAction.SKIP, "过载·跳过观察"),
                RouteDecision("REFLECT", PhaseAction.DEGRADED, "过载·轻量反思"),
            ]
            self.stats["skipped_phases"] += 2
            self.stats["degraded_phases"] += 1
            return decisions
        
        # 规则3: 简单问候/闲聊 → 最小化
        if self.GREETING_PATTERNS.match(ctx.user_input.strip()):
            decisions = [
                RouteDecision("PLAN", PhaseAction.RUN, "问候·理解意图"),
                RouteDecision("ACT", PhaseAction.SKIP, "问候·无需工具"),
                RouteDecision("OBSERVE", PhaseAction.SKIP, "问候·无需观察"),
                RouteDecision("REFLECT", PhaseAction.DEGRADED, "问候·轻量反思"),
            ]
            self.stats["skipped_phases"] += 2
            self.stats["degraded_phases"] += 1
            return decisions
        
        # 规则4: 重复相同输入 → 跳过PLAN（意图已知）
        if ctx.consecutive_identical >= 2:
            decisions = [
                RouteDecision("PLAN", PhaseAction.SKIP, f"连续{ctx.consecutive_identical}次相同·跳过规划"),
                RouteDecision("ACT", PhaseAction.RUN, "重试·需要执行"),
                RouteDecision("OBSERVE", PhaseAction.RUN, "重试·需要观察结果"),
                RouteDecision("REFLECT", PhaseAction.RUN, "重试·需要反思原因"),
            ]
            self.stats["skipped_phases"] += 1
            return decisions
        
        # 规则5: 文件操作 → 全跑
        if self.FILE_OPS.search(ctx.user_input):
            return self._all_run("文件操作·全阶段激活")
        
        # 规则6: 代码操作 → 全跑
        if self.CODE_OPS.search(ctx.user_input):
            return self._all_run("代码操作·全阶段激活")
        
        # 规则7: 搜索/查询 → 跳过OBSERVE的深度分析
        if self.QUERY_OPS.search(ctx.user_input):
            decisions = [
                RouteDecision("PLAN", PhaseAction.RUN, "查询·理解搜索意图"),
                RouteDecision("ACT", PhaseAction.RUN, "查询·执行搜索"),
                RouteDecision("OBSERVE", PhaseAction.DEGRADED, "查询·轻量结果检查"),
                RouteDecision("REFLECT", PhaseAction.RUN, "查询·记录搜索经验"),
            ]
            self.stats["degraded_phases"] += 1
            return decisions
        
        # 规则8: 上下文>60% → 轻量REFLECT
        if ctx.context_used_pct > 0.6:
            decisions = [
                RouteDecision("PLAN", PhaseAction.RUN, "上下文中等·正常规划"),
                RouteDecision("ACT", PhaseAction.RUN, "上下文中等·正常执行"),
                RouteDecision("OBSERVE", PhaseAction.RUN, "上下文中等·正常观察"),
                RouteDecision("REFLECT", PhaseAction.DEGRADED, "上下文>60%·轻量反思"),
            ]
            self.stats["degraded_phases"] += 1
            return decisions
        
        # 规则9: 上一次工具执行失败 → 强化OBSERVE
        if ctx.last_tool_success is False:
            decisions = [
                RouteDecision("PLAN", PhaseAction.RUN, "工具失败·重新规划"),
                RouteDecision("ACT", PhaseAction.RUN, "工具失败·重试执行"),
                RouteDecision("OBSERVE", PhaseAction.RUN, "工具失败·强化观察"),
                RouteDecision("REFLECT", PhaseAction.RUN, "工具失败·深度反思"),
            ]
            return decisions
        
        # 默认: 全跑
        return self._all_run("默认·全阶段激活")
    
    def _all_run(self, reason: str) -> list[RouteDecision]:
        """全部阶段正常执行。"""
        return [
            RouteDecision("PLAN", PhaseAction.RUN, reason),
            RouteDecision("ACT", PhaseAction.RUN, reason),
            RouteDecision("OBSERVE", PhaseAction.RUN, reason),
            RouteDecision("REFLECT", PhaseAction.RUN, reason),
        ]
    
    def record_intent(self, intent: str):
        """记录用户意图（用于重复检测）。"""
        self._recent_intents.append(intent)
        if len(self._recent_intents) > self._max_history:
            self._recent_intents = self._recent_intents[-self._max_history:]
    
    def get_consecutive_identical(self, current_input: str) -> int:
        """计算连续相同输入次数。"""
        count = 0
        for intent in reversed(self._recent_intents):
            if intent == current_input:
                count += 1
            else:
                break
        return count
    
    def get_stats(self) -> dict:
        """获取路由统计。"""
        total = self.stats["total_decisions"]
        if total == 0:
            return self.stats
        skip_rate = self.stats["skipped_phases"] / (total * 4)  # 4 phases per decision
        degrade_rate = self.stats["degraded_phases"] / (total * 4)
        return {
            **self.stats,
            "skip_rate": f"{skip_rate:.1%}",
            "degrade_rate": f"{degrade_rate:.1%}",
            "efficiency": f"{(skip_rate + degrade_rate * 0.5):.1%}",  # 跳过=100%省, 降级=50%省
        }


# ── 便捷函数 ──────────────────────────────────────

def create_router(enabled: bool = True) -> RuleRouter:
    """创建规则路由器实例。"""
    return RuleRouter(enabled=enabled)
