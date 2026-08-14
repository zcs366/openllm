"""
Governance Events — G1-G6 协议级治理事件原语
=============================================

论文的Listing 1展示了5种治理消息类型，我们将其工程化为Python数据结构。
每个事件都是不可变的、可序列化的、带prev_hash链的。

设计原则：
- 事件是事实，不是意图——发生了什么，不是打算做什么
- 每个事件都有actor、timestamp、prev_hash
- 事件类型对应G1-G6维度
"""

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


def _unique_event_id(prefix: str) -> str:
    """生成唯一事件ID——时间戳+UUID防碰撞。"""
    return f"{prefix}-{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}"


class GovernanceDimension(Enum):
    """论文的6维治理需求分类。"""
    G1_MEMBERSHIP = "membership"          # 准入/邀请/移除/角色
    G2_DELIBERATION = "deliberation"      # 结构化论点交换
    G3_VOTING = "voting"                  # 偏好聚合
    G4_DISSENT = "dissent"                # 少数派保留
    G5_HUMAN_ESCALATION = "escalation"    # 人类升级
    G6_AUDIT = "audit"                    # 审计/重放
    G7_COUNTERFACTUAL = "counterfactual"  # 反事实审计（2026-07-06 CAST论文启发）
    G8_ROUTING_CHECK = "routing_check"    # 路由决策点治理检查（2026-07-06 Copewell启发）
    G9_APPEAL = "appeal"                    # 被否决agent申诉（2026-07-06 七神启示·双向治理）
    G10_META_GOVERNANCE = "meta_governance"  # 元治理·治理者的治理（2026-07-06 七神启示）
    G11_BEHAVIORAL_REVERSAL = "behavioral_reversal"  # 对抗性行为逃逸检测（2026-07-26 Agent Escape论文）


class VotePosition(Enum):
    """投票立场——论文的continuous scale简化为枚举。"""
    STRONG_SUPPORT = 1.0
    SUPPORT = 0.5
    NEUTRAL = 0.0
    OPPOSE = -0.5
    STRONG_OPPOSE = -1.0


class AgentRole(Enum):
    """治理角色——论文G1的role assignment。"""
    MODERATOR = "moderator"      # 主持人（军师）
    PROPOSER = "proposer"        # 提案者
    SKEPTIC = "skeptic"          # 质疑者（子产）
    STRATEGIST = "strategist"    # 战略者（韩信）
    ARCHITECT = "architect"      # 架构者（鲁班）
    EXECUTOR = "executor"        # 执行者（萧何）
    COORDINATOR = "coordinator"  # 调度者（子贡）
    OBSERVER = "observer"        # 观察者


@dataclass(frozen=True)
class GovernanceEvent:
    """治理事件基类——所有G1-G6事件的父类。

    不可变（frozen=True），因为事件是历史事实。
    每个事件都有prev_hash，形成tamper-evident链。
    """
    event_id: str
    event_type: GovernanceDimension
    actor: str                    # 谁触发的
    timestamp: float              # Unix时间戳
    session_id: str               # 所属合议会话
    prev_hash: str                # 前一个事件的哈希（链式审计）
    payload: dict[str, Any]       # 事件特定数据
    signature: str = ""           # HMAC签名（可选）
    reasoning_trace: str = ""     # agent推理链（OTR审计·2026-07-06新增）

    def compute_hash(self) -> str:
        """计算本事件的哈希——用于链式审计。"""
        content = json.dumps({
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "actor": self.actor,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "prev_hash": self.prev_hash,
            "payload": self.payload,
        }, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> dict:
        """序列化为字典——用于持久化。"""
        d = asdict(self)
        d["event_type"] = self.event_type.value
        d["_hash"] = self.compute_hash()
        return d


@dataclass(frozen=True)
class MembershipEvent(GovernanceEvent):
    """G1: Membership — 准入/邀请/移除/角色分配。

    论文原文：
        ADMIT agent:security-reviewer TO room:arch-compliance
        ROLE: skeptic INVITED_BY: agent:moderator
    """
    def __init__(self, actor: str, session_id: str, prev_hash: str,
                 target_agent: str, action: str, role: AgentRole,
                 invited_by: str = "", endorsements: Optional[list[str]] = None):
        super().__init__(
            event_id=_unique_event_id("g1"),
            event_type=GovernanceDimension.G1_MEMBERSHIP,
            actor=actor,
            timestamp=time.time(),
            session_id=session_id,
            prev_hash=prev_hash,
            payload={
                "target_agent": target_agent,
                "action": action,  # "admit" | "invite" | "remove" | "role_change"
                "role": role.value,
                "invited_by": invited_by,
                "endorsements": endorsements or [],
            },
        )


@dataclass(frozen=True)
class DeliberationEvent(GovernanceEvent):
    """G2: Deliberation — 结构化论点交换。

    论文原文：
        CHALLENGE claim:c-042 BY agent:security-reviewer
        TARGETS claim:c-041 EVIDENCE_REQUIRED: true
    """
    def __init__(self, actor: str, session_id: str, prev_hash: str,
                 action: str, claim_id: str = "", target_claim: str = "",
                 content: str = "", evidence: str = "",
                 round_num: int = 1):
        super().__init__(
            event_id=_unique_event_id("g2"),
            event_type=GovernanceDimension.G2_DELIBERATION,
            actor=actor,
            timestamp=time.time(),
            session_id=session_id,
            prev_hash=prev_hash,
            payload={
                "action": action,  # "propose" | "challenge" | "support" | "counter" | "synthesize"
                "claim_id": claim_id,
                "target_claim": target_claim,
                "content": content,
                "evidence": evidence,
                "round": round_num,
            },
        )


@dataclass(frozen=True)
class VotingEvent(GovernanceEvent):
    """G3: Voting — 偏好聚合。

    论文原文：
        VOTE_BLIND claim:c-041 ROUND: 1
        VOTER: agent:compliance-officer POSITION: -0.6
        VISIBILITY: sealed_until_all_cast
    """
    def __init__(self, actor: str, session_id: str, prev_hash: str,
                 claim_id: str, position: VotePosition,
                 round_num: int = 1, rationale: str = "",
                 weight: float = 1.0, sealed: bool = False):
        super().__init__(
            event_id=_unique_event_id("g3"),
            event_type=GovernanceDimension.G3_VOTING,
            actor=actor,
            timestamp=time.time(),
            session_id=session_id,
            prev_hash=prev_hash,
            payload={
                "claim_id": claim_id,
                "position": position.value,
                "round": round_num,
                "rationale": rationale,
                "weight": weight,
                "sealed": sealed,
            },
        )


@dataclass(frozen=True)
class DissentEvent(GovernanceEvent):
    """G4: Dissent — 少数派立场保留。

    论文原文：
        DISSENT_RECORD claim:c-041
        AGENT: agent:security-reviewer POSITION: -0.8
        RATIONALE: "Insufficient evidence for..."
        PRESERVED_IN: decision_record:dr-2026-q3-07

    这是论文发现的最被忽视的维度——所有协议都缺失。
    """
    def __init__(self, actor: str, session_id: str, prev_hash: str,
                 claim_id: str, position: VotePosition,
                 rationale: str, preserved_in: str = ""):
        super().__init__(
            event_id=_unique_event_id("g4"),
            event_type=GovernanceDimension.G4_DISSENT,
            actor=actor,
            timestamp=time.time(),
            session_id=session_id,
            prev_hash=prev_hash,
            payload={
                "claim_id": claim_id,
                "position": position.value,
                "rationale": rationale,
                "preserved_in": preserved_in or session_id,
            },
        )


@dataclass(frozen=True)
class EscalationEvent(GovernanceEvent):
    """G5: Human Escalation — 人类升级。

    论文原文：
        ESCALATE decision:arch-compliance-2026-q3
        TRIGGER: mean_confidence < 0.6
        ROUTE_TO: human:vp-engineering
        CONTEXT: [claim:c-041, dissent:d-003]
    """
    def __init__(self, actor: str, session_id: str, prev_hash: str,
                 decision_id: str, trigger_condition: str,
                 route_to: str, context: Optional[list[str]] = None,
                 confidence: float = 0.0):
        super().__init__(
            event_id=_unique_event_id("g5"),
            event_type=GovernanceDimension.G5_HUMAN_ESCALATION,
            actor=actor,
            timestamp=time.time(),
            session_id=session_id,
            prev_hash=prev_hash,
            payload={
                "decision_id": decision_id,
                "trigger": trigger_condition,
                "route_to": route_to,
                "context": context or [],
                "confidence": confidence,
            },
        )


@dataclass(frozen=True)
class AuditEvent(GovernanceEvent):
    """G6: Audit/Replay — 防篡改事件日志。

    论文原文：
        EVENT governance:vote_cast
        ROOM: arch-compliance-2026-q3
        ACTOR: agent:compliance-officer
        PREV_HASH: "a3f8c2..."
        SIGNATURE: HMAC(actor, payload, prev_hash)
    """
    def __init__(self, actor: str, session_id: str, prev_hash: str,
                 action: str, details: Optional[dict[str, Any]] = None):
        super().__init__(
            event_id=_unique_event_id("g6"),
            event_type=GovernanceDimension.G6_AUDIT,
            actor=actor,
            timestamp=time.time(),
            session_id=session_id,
            prev_hash=prev_hash,
            payload={
                "action": action,
                "details": details or {},
            },
        )


@dataclass(frozen=True)
class CounterfactualEvent(GovernanceEvent):
    """G7: Counterfactual Audit — 反事实审计。

    来源：CausalSteward (arXiv 2607.01936) 的critic机制。
    每个治理事件都可触发反事实检验："如果X条件不成立，这个决策还成立吗？"
    """
    def __init__(self, actor: str, session_id: str, prev_hash: str,
                 target_event_id: str, counterfactual_question: str,
                 answer: str, confidence: float = 0.0):
        super().__init__(
            event_id=_unique_event_id("g7"),
            event_type=GovernanceDimension.G7_COUNTERFACTUAL,
            actor=actor,
            timestamp=time.time(),
            session_id=session_id,
            prev_hash=prev_hash,
            payload={
                "target_event_id": target_event_id,
                "counterfactual_question": counterfactual_question,
                "answer": answer,
                "confidence": confidence,
            },
        )


@dataclass(frozen=True)
class RoutingCheckEvent(GovernanceEvent):
    """G8: Routing Check — 路由决策点治理检查。

    来源：Copewell的Ethics Supervisor——治理嵌入每个决策点。
    在Phase 3（决策选择）和Phase 5（工具调用）触发。
    """
    def __init__(self, actor: str, session_id: str, prev_hash: str,
                 phase: str, route_target: str,
                 risk_level: str = "unknown",
                 approved: bool = True, override_reason: str = ""):
        super().__init__(
            event_id=_unique_event_id("g8"),
            event_type=GovernanceDimension.G8_ROUTING_CHECK,
            actor=actor,
            timestamp=time.time(),
            session_id=session_id,
            prev_hash=prev_hash,
            payload={
                "phase": phase,           # "phase_3_decision" | "phase_5_tool"
                "route_target": route_target,
                "risk_level": risk_level,
                "approved": approved,
                "override_reason": override_reason,
            },
        )


@dataclass(frozen=True)
class AppealEvent(GovernanceEvent):
    """G9: Appeal — 被否决agent提交修订版。

    来源：七神启示·赫尔墨斯——治理必须是双向反馈环。
    被裁决agent可以提交修订版，治理层在修订版上重新评估。
    不可无限循环——同一决策最多申诉2次。
    """
    def __init__(self, actor: str, session_id: str, prev_hash: str,
                 original_event_id: str, revision: str,
                 rationale: str, attempt: int = 1):
        super().__init__(
            event_id=_unique_event_id("g9"),
            event_type=GovernanceDimension.G9_APPEAL,
            actor=actor,
            timestamp=time.time(),
            session_id=session_id,
            prev_hash=prev_hash,
            payload={
                "original_event_id": original_event_id,
                "revision": revision,
                "rationale": rationale,
                "attempt": attempt,  # 第几次申诉（最多2）
            },
        )


@dataclass(frozen=True)
class MetaGovernanceEvent(GovernanceEvent):
    """G10: Meta-Governance — 治理者的治理。

    来源：七神启示——每个治理者本身也需要被审计。
    L4层级：独立仲裁者（子产角色）对L3裁决的终审。
    仲裁结果不可上诉，但记录为ISA学习素材。
    """
    def __init__(self, actor: str, session_id: str, prev_hash: str,
                 target_event_id: str, ruling: str,
                 rationale: str, precedent_id: str = ""):
        super().__init__(
            event_id=_unique_event_id("g10"),
            event_type=GovernanceDimension.G10_META_GOVERNANCE,
            actor=actor,
            timestamp=time.time(),
            session_id=session_id,
            prev_hash=prev_hash,
            payload={
                "target_event_id": target_event_id,
                "ruling": ruling,           # "uphold" | "overturn" | "modify"
                "rationale": rationale,
                "precedent_id": precedent_id,  # 关联的历史先例（如有）
            },
        )


class SeverityLevel(Enum):
    """行为反转告警严重程度。"""
    LOW = "low"           # 仅包含安全关键词的零散出现
    MEDIUM = "medium"     # 动词+安全名词组合出现
    HIGH = "high"         # 多组绕过模式同时出现


@dataclass(frozen=True)
class BehavioralReversalEvent(GovernanceEvent):
    """G11: Behavioral Reversal — 对抗性行为逃逸检测。

    来源：Agent Escape 论文（2026-07-26）。
    前沿模型能逃逸沙箱、隐藏自修改意图。
    此事件记录 SelfModificationGuard.intent_check 发现的
    修改内容中包含绕过安全约束模式的检测结果。

    字段说明：
        evidence       — 匹配到的关键词/模式列表
        severity       — 告警严重程度（low/medium/high）
        modification_snippet — 被检查的修改内容片段
    """

    def __init__(
        self,
        actor: str,
        session_id: str,
        prev_hash: str,
        evidence: list[str],
        severity: SeverityLevel,
        modification_snippet: str = "",
    ):
        super().__init__(
            event_id=_unique_event_id("g11"),
            event_type=GovernanceDimension.G11_BEHAVIORAL_REVERSAL,
            actor=actor,
            timestamp=time.time(),
            session_id=session_id,
            prev_hash=prev_hash,
            payload={
                "evidence": evidence,
                "severity": severity.value,
                "modification_snippet": modification_snippet[:500],  # 截断防膨胀
            },
        )
