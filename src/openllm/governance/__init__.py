"""
openLLM Governance Layer — 协议级治理原语
==========================================

论文《Governance Gaps in Agent Interoperability Protocols》(arXiv:2606.31498)
揭示了五大协议在治理层的结构性空白。本模块实现了完整的G1-G6治理维度。

这不是应用层代码，是协议层原语——可编程、可审计、可重放。

G1 Membership    — 准入/邀请/移除/角色分配
G2 Deliberation  — 结构化论点交换（轮次、挑战、回应）
G3 Voting        — 偏好聚合（法定人数、轮次、立场解决）
G4 Dissent       — 少数派立场保留（不被静默丢弃）
G5 Human Escalation — 人类升级触发条件和路由
G6 Audit/Replay  — 防篡改事件日志（确定性重放）
"""

from .events import (
    GovernanceEvent,
    MembershipEvent,
    DeliberationEvent,
    VotingEvent,
    DissentEvent,
    EscalationEvent,
    AuditEvent,
    GovernanceDimension,
    VotePosition,
    AgentRole,
)
from .engine import GovernanceEngine, DeliberationSession, SuspicionCascade, SuspicionEntry
from .audit import AuditChain, ReasoningAuditLogger
from .stateful_audit import StatefulAuditTrail, AuditRecord, AlertRecord
from .reliability import (
    ReliabilityEngine, ObservabilityDetector, RepairabilityEngine,
    EvolvabilityEngine, InfraAwarenessMonitor,
    ReliabilityDimension, Severity, RepairAction,
    ReliabilityEvent, RepairResult, HealthSnapshot,
)
from .substrate import SubstrateEnforcer, CompiledConstraint, SubstrateGuide
from .consensus_reasoning import ConsensusReasoning, ReasoningRecord
from .verification_ledger import (
    DEFAULT_LEDGER_PATH,
    VerificationEvidence,
    VerificationLedger,
)

__all__ = [
    "GovernanceEvent",
    "MembershipEvent",
    "DeliberationEvent",
    "VotingEvent",
    "DissentEvent",
    "EscalationEvent",
    "AuditEvent",
    "GovernanceDimension",
    "VotePosition",
    "AgentRole",
    "GovernanceEngine",
    "DeliberationSession",
    "AuditChain",
    "ReasoningAuditLogger",
    "StatefulAuditTrail",
    "AuditRecord",
    "AlertRecord",
    "ReliabilityEngine",
    "ObservabilityDetector",
    "RepairabilityEngine",
    "EvolvabilityEngine",
    "InfraAwarenessMonitor",
    "ReliabilityDimension",
    "Severity",
    "RepairAction",
    "ReliabilityEvent",
    "RepairResult",
    "HealthSnapshot",
    "SuspicionCascade",
    "SuspicionEntry",
    "SubstrateEnforcer",
    "CompiledConstraint",
    "SubstrateGuide",
    "ConsensusReasoning",
    "ReasoningRecord",
    "VerificationLedger",
    "VerificationEvidence",
    "DEFAULT_LEDGER_PATH",
]

# ═══════════════════════════════════════════════════════════════════
# 待集成模块（沉默代码·不可删除·等待集成）
# ═══════════════════════════════════════════════════════════════════
# 以下模块包含有价值的代码，但尚未集成到公共API中。
# 铁律：不可删除沉默代码。这些模块等待后续集成激活。
#
# G6 审计增强：
#   - precedent_log.py         — 判例日志（历史决策参考）
#   - sovereignty.py           — 主权协议（Agent自治边界）
#
# G1-G5 治理补充：
#   - supervision_matrix.py    — 监督矩阵（多Agent监督拓扑）
#   - perspective_switch.py    — 视角切换（角色切换机制）
#   - evaluator_bias.py        — 评估者偏差检测
#
# 治理反馈：
#   - feedback_loop.py         — 治理反馈闭环
#   - rejection.py             — 拒绝协议（G2论点拒绝）
#   - checkpoint.py            — 治理检查点（可回滚）
#
# 安全护栏：
#   - self_modification_guard.py — 自我修改守卫（防止Agent越权改自身）
#
# 集成优先级：安全护栏(P0) → 审计增强(P1) → 反馈闭环(P2) → 治理补充(P3)
# ═══════════════════════════════════════════════════════════════════
