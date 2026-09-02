"""OpenLLM Core — Agent循环、Provider、引擎。

拆分后导出：优先从impl文件导出，保留旧文件兼容。
"""
# 新文件（拆分后）
from .models import (Message, Context, Prediction, RiskAssessment,
                     Proposal, Critique, Decision, ActionResult,
                     CausalDelta, TickMetrics)
from .isa_impl import ISA
# 章鱼I延迟导入（避免循环：iai.octopus→core→core.__init__→iai.octopus）
from .ios_impl import IOS
from .isn_impl import ISN
from .iko_impl import IKO
from .provider_impl import LLMProvider
from .brain_guardian import BrainGuardian, ChangeResult

# 旧文件（兼容期保留）
from .loop import AgentLoop, LoopPhase, AgentState, TurnContext
from .provider import DeepSeekProvider, ModelConfig, ModelResponse, create_provider
from .engine import OpenLLMEngine, AgentConfig
from .meta import CognitiveDashboard, MetaSnapshot, CognitiveState, SelfRescue

# 章鱼I延迟导出——首次访问时从iai.octopus加载
def __getattr__(name):
    if name == "章鱼I":
        from openllm.iai.octopus import 章鱼I
        globals()["章鱼I"] = 章鱼I
        return 章鱼I
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # 新文件
    "Message", "Context", "Prediction", "RiskAssessment",
    "Proposal", "Critique", "Decision", "ActionResult",
    "CausalDelta", "TickMetrics",
    "ISA", "章鱼I", "IOS", "ISN", "IKO", "LLMProvider",
    # 旧文件（兼容）
    "AgentLoop", "LoopPhase", "AgentState", "TurnContext",
    "DeepSeekProvider", "ModelConfig", "ModelResponse", "create_provider",
    "OpenLLMEngine", "AgentConfig",
    "CognitiveDashboard", "MetaSnapshot", "CognitiveState", "SelfRescue",
]

# ═══════════════════════════════════════════════════════════════════
# 待集成模块（沉默代码·不可删除·等待集成）
# ═══════════════════════════════════════════════════════════════════
# 以下模块包含有价值的代码，但尚未集成到公共API中。
# 铁律：不可删除沉默代码。这些模块等待后续集成激活。
#
# ISA 记忆层：
#   - cross_session_chain.py    — 跨session记忆链（ISA子系统）
#   - checkpoint_manager.py     — 检查点管理（ISA持久化）
#   - context_drift_detector.py — 上下文漂移检测（ISA监控）
#
# IOS 决策层：
#   - ios_risk.py               — 风险评估模块
#   - ios_evolve.py             — 进化引擎（IOS自适应）
#   - ios_causal.py             — 因果推理链
#   - ios_arbitrate.py          — 仲裁协议
#   - governance_rule.py        — 治理规则引擎
#
# ISN 信号层：
#   - isn_signal.py             — ISN信号总线
#   - execution_broker.py       — 执行调度器
#
# IKO 输出层：
#   - tool_validator.py         — 工具输入验证
#   - tool_validator_types.py   — 验证器类型定义
#   - tool_executor.py          — 工具执行器
#
# 核心循环增强：
#   - main_loop.py              — 主循环增强版
#   - engine_integrations.py    — 引擎集成适配
#   - engine_utils.py           — 引擎工具函数
#
# 感知与决策：
#   - perception.py             — 感知层（世界模型接口）
#   - decision.py               — 决策框架
#   - execution.py              — 执行框架
#
# 基础设施：
#   - protocol.py               — 协议层（Agent通信）
#   - body_protocol.py          — 肉身协议（外部系统接口）
#   - message_bus.py            — 消息总线
#   - router.py                 — 路由器（请求分发）
#   - gateway.py                — 网关（外部接入）
#
# 可靠性与追踪：
#   - action_trace.py           — 行为追踪链
#   - trace_spec.py             — 追踪规范定义
#   - degradation_trace.py      — 降级追踪
#   - integrity_guardian.py     — 完整性守护者
#   - ecosystem_integrity.py    — 生态完整性检查
#   - failure_tracker.py        — 失败追踪器
#   - validator.py              — 异源验证器（规则+LLM双模式）
#
# 代谢与经济：
#   - token_economy.py          — Token经济（资源预算）
#   - inference_budget.py       — 推理预算管理
#   - heartbeat.py              — 心跳监控
#   - agent_heartbeat.py        — Agent心跳
#   - idle_wander.py            — 空闲游荡（自主探索）
#   - info_metrics.py           — 信息量指标
#   - self_harness.py           — 自体测试桩
#   - oneshot.py                — 一次性执行模式
#
# 集成优先级：IOS层(P0) → ISA记忆(P1) → 感知决策(P2) → 基础设施(P3)
# ═══════════════════════════════════════════════════════════════════
