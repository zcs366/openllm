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
from .engine import GovernanceEngine, DeliberationSession
from .audit import AuditChain, ReasoningAuditLogger
from .stateful_audit import StatefulAuditTrail, AuditRecord, AlertRecord
from .reliability import (
    ReliabilityEngine, ObservabilityDetector, RepairabilityEngine,
    EvolvabilityEngine, InfraAwarenessMonitor,
    ReliabilityDimension, Severity, RepairAction,
    ReliabilityEvent, RepairResult, HealthSnapshot,
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
]
