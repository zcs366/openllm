"""
openLLM 数据模型 — 从main_loop.py提取

所有六体共享的数据类。从main_loop.py line 54-163提取。
"""
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Message:
    """用户输入"""
    text: str
    timestamp: float = field(default_factory=time.time)
    msg_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

@dataclass
class Context:
    """构建好的上下文（喂给推理引擎的）— 七要素"""
    user_message: str
    # ① 身份 — Agent是谁
    identity: dict = field(default_factory=dict)
    # ② 记忆 — 最近决策、因果教训
    memory: dict = field(default_factory=dict)
    # ③ 工具 — 可用的工具列表（动态）
    tools: list = field(default_factory=list)
    # ④ 因果提示 — 上次类似操作的因果历史
    causal_hints: list = field(default_factory=list)
    # ⑤ 认知报告 — D₀感知
    d0_report: dict = field(default_factory=dict)
    # ⑥ 风险上下文 — 当前风险等级
    risk_context: dict = field(default_factory=dict)
    # ⑦ 搜索结果 — 触手脑检索结果
    search_results: list = field(default_factory=list)
    # 厌倦律 IoR：已处理主题抑制提示（抑制重复关注）
    ior_hints: list = field(default_factory=list)
    # 遗忘维 ForgettingCurve：主题衰减打分提示（长时遗忘）
    forgetting_hints: list = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    # DR-20260829-01 P0-C: 身份注入prompt
    identity_block: str = ""

@dataclass
class Prediction:
    """因果预测（Phase 3: 因果预测）"""
    summary: str = ""
    consequences: list[str] = field(default_factory=list)
    confidence: float = 0.5
    risk_signals: list[str] = field(default_factory=list)
    
    def summary_text(self) -> str:
        """返回预测摘要。"""
        return f"[预测] {self.summary} | 置信度={self.confidence:.0%} | 风险信号={len(self.risk_signals)}"

@dataclass
class RiskAssessment:
    """风险评估（Phase 4: 风险评估）"""
    level: str = "low"  # low | medium | high | critical
    blocked: bool = False
    reason: str = ""
    details: list[str] = field(default_factory=list)
    
    def is_blocked(self) -> bool:
        """是否被安全拦截。"""
        return self.blocked or self.level == "critical"

@dataclass
class Proposal:
    """左脑提案"""
    content: str
    confidence: float = 0.5
    evidence: list[str] = field(default_factory=list)
    # 工具调用清单 [{"name": str, "args": dict}]（DR-20260828-01：与Decision.tool_calls同形）
    tool_calls: list[dict] = field(default_factory=list)
    prediction_ref: Optional[Prediction] = None

@dataclass
class Critique:
    """右脑批判"""
    content: str
    verdict: str = ""  # "approve" | "reject" | "revise"
    concerns: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

@dataclass 
class Decision:
    """IO-S仲裁决策"""
    action: str  # "execute" | "deny" | "revise" | "escalate"
    reason: str = ""
    approved: bool = False
    risk_ref: Optional[RiskAssessment] = None
    # 工具调用清单（DR-20260828-01：Decision为tool_calls的携带者，ISN为消费者）
    tool_calls: list[dict] = field(default_factory=list)

@dataclass
class ActionResult:
    """工具执行结果"""
    success: bool
    output: str = ""
    error: str = ""
    duration_ms: float = 0.0

@dataclass
class CausalDelta:
    """因果对照（Phase 8: 因果对照）"""
    prediction_match: bool = False
    delta_summary: str = ""
    learned: list[str] = field(default_factory=list)
    
    def summary_text(self) -> str:
        """返回因果对照摘要。"""
        return f"[因果] 匹配={self.prediction_match} | 学到={len(self.learned)}条"

@dataclass
class TickMetrics:
    """一轮心跳的完整可观测数据"""
    tick_id: str
    phase: str
    status: str
    duration_ms: float = 0.0
    detail: str = ""
