"""OpenLLM Core — Agent循环、Provider、引擎。

拆分后导出：优先从impl文件导出，保留旧文件兼容。
"""
# 新文件（拆分后）
from .models import (Message, Context, Prediction, RiskAssessment,
                     Proposal, Critique, Decision, ActionResult,
                     CausalDelta, TickMetrics)
from .isa_impl import ISA
from .octopus_impl import 章鱼I
from .ios_impl import IOS
from .isn_impl import ISN
from .iko_impl import IKO
from .provider_impl import LLMProvider

# 旧文件（兼容期保留）
from .loop import AgentLoop, LoopPhase, AgentState, TurnContext
from .provider import DeepSeekProvider, ModelConfig, ModelResponse, create_provider
from .engine import OpenLLMEngine, AgentConfig
from .meta import CognitiveDashboard, MetaSnapshot, CognitiveState, SelfRescue

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
