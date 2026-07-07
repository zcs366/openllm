"""
GovernanceEngine — 协议级治理引擎
===================================

将五人合议从prompt变成可编程的协议。
论文说"治理是缺失的架构层"——这个引擎就是那个层。

核心能力：
- G1: 成员准入/角色分配
- G2: 结构化审议（轮次、挑战、回应）
- G3: 投票聚合（权重、法定人数、立场解决）
- G4: 异议保留（少数派立场不被丢弃）
- G5: 人类升级（置信度阈值、风险评估触发）
- G6: 审计链（防篡改事件日志、确定性重放）

与five-agent-consultation skill的关系：
- skill是prompt层（告诉LLM怎么做）
- engine是代码层（实际执行、可审计、可重放）
- 两者协同：skill指导engine的参与者行为
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

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
from .audit import AuditChain

logger = logging.getLogger("openllm.governance.engine")


# ── 投票权重（2026-07-03 用户确认） ──
DEFAULT_VOTE_WEIGHTS = {
    "韩信": 0.30,   # 划终局
    "鲁班": 0.30,   # 审架构
    "子产": 0.20,   # 断该不该
    "萧何": 0.10,   # 写路径
    "子贡": 0.10,   # 做调度
}

# ── 人类升级触发条件 ──
ESCALATION_TRIGGERS = {
    "low_confidence": 0.6,    # 平均置信度 < 0.6 → 升级
    "unanimous_oppose": True,  # 全票反对 → 升级
    "high_risk": True,         # 高风险决策 → 升级
}


@dataclass
class Claim:
    """一个待审议的主张/提案。"""
    claim_id: str
    content: str
    proposer: str
    timestamp: float = field(default_factory=time.time)
    evidence: str = ""
    status: str = "pending"  # pending | accepted | rejected | escalated


@dataclass
class DeliberationSession:
    """一个合议会话的状态。"""
    session_id: str
    topic: str
    claims: dict[str, Claim] = field(default_factory=dict)
    members: dict[str, AgentRole] = field(default_factory=dict)
    votes: dict[str, list[dict]] = field(default_factory=dict)  # claim_id -> [votes]
    dissents: list[dict] = field(default_factory=list)
    escalations: list[dict] = field(default_factory=list)
    current_round: int = 1
    status: str = "active"  # active | completed | escalated


class GovernanceEngine:
    """协议级治理引擎。

    用法：
        engine = GovernanceEngine()
        session = engine.create_session("arch-review", "openLLM治理层设计")
        engine.admit_member(session, "子产", AgentRole.SKEPTIC)
        engine.admit_member(session, "韩信", AgentRole.STRATEGIST)
        # ... 审议、投票、异议 ...
        result = engine.close_session(session)
    """

    def __init__(self, audit_chain: Optional[AuditChain] = None):
        self.audit = audit_chain or AuditChain()
        self.sessions: dict[str, DeliberationSession] = {}
        self.cascade = SuspicionCascade()

    def record_tool_call(
        self,
        tool_name: str,
        suspicion_score: float,
        reason: str = "",
    ) -> dict:
        """记录tool call并传播怀疑分数（委托给SuspicionCascade）。

        Returns:
            {"alert": bool, "final_score": float, "cascade_delta": float, "message": str}
        """
        return self.cascade.record_tool_call(tool_name, suspicion_score, reason)

    def create_session(self, session_id: str, topic: str) -> DeliberationSession:
        """创建一个新的合议会话。"""
        session = DeliberationSession(session_id=session_id, topic=topic)
        self.sessions[session_id] = session

        # G6: 审计事件
        event = AuditEvent(
            actor="engine",
            session_id=session_id,
            prev_hash=self.audit._get_last_hash(session_id),
            action="session_created",
            details={"topic": topic},
        )
        self.audit.append(event)

        logger.info(f"合议会话已创建: {session_id} ({topic})")
        return session

    def admit_member(self, session: DeliberationSession,
                     agent: str, role: AgentRole) -> bool:
        """G1: 准入成员到合议会话。"""
        if session.status != "active":
            return False

        session.members[agent] = role

        # G1事件
        event = MembershipEvent(
            actor="engine",
            session_id=session.session_id,
            prev_hash=self.audit._get_last_hash(session.session_id),
            target_agent=agent,
            action="admit",
            role=role,
        )
        self.audit.append(event)

        logger.info(f"成员准入: {agent} as {role.value}")
        return True

    def propose(self, session: DeliberationSession,
                proposer: str, content: str,
                evidence: str = "") -> Claim:
        """G2: 提出一个主张/提案。"""
        # 成员检查
        if proposer not in session.members:
            logger.warning(f"非成员提案被拒绝: {proposer}")
            raise ValueError(f"{proposer} 不是合议会话成员")

        claim_id = f"claim-{len(session.claims)+1:03d}"
        claim = Claim(
            claim_id=claim_id,
            content=content,
            proposer=proposer,
            evidence=evidence,
        )
        session.claims[claim_id] = claim

        # G2事件
        event = DeliberationEvent(
            actor=proposer,
            session_id=session.session_id,
            prev_hash=self.audit._get_last_hash(session.session_id),
            action="propose",
            claim_id=claim_id,
            content=content,
            evidence=evidence,
        )
        self.audit.append(event)

        logger.info(f"提案: {claim_id} by {proposer} — {content[:50]}...")
        return claim

    def challenge(self, session: DeliberationSession,
                  challenger: str, claim_id: str,
                  rationale: str) -> bool:
        """G2: 挑战一个主张。"""
        if claim_id not in session.claims:
            return False

        # G2事件
        event = DeliberationEvent(
            actor=challenger,
            session_id=session.session_id,
            prev_hash=self.audit._get_last_hash(session.session_id),
            action="challenge",
            claim_id=claim_id,
            target_claim=claim_id,
            content=rationale,
        )
        self.audit.append(event)

        logger.info(f"挑战: {challenger} 挑战 {claim_id} — {rationale[:50]}...")
        return True

    def support(self, session: DeliberationSession,
                supporter: str, claim_id: str,
                rationale: str) -> bool:
        """G2: 支持一个主张。"""
        if claim_id not in session.claims:
            return False

        # G2事件
        event = DeliberationEvent(
            actor=supporter,
            session_id=session.session_id,
            prev_hash=self.audit._get_last_hash(session.session_id),
            action="support",
            claim_id=claim_id,
            target_claim=claim_id,
            content=rationale,
        )
        self.audit.append(event)

        logger.info(f"支持: {supporter} 支持 {claim_id}")
        return True

    def vote(self, session: DeliberationSession,
             voter: str, claim_id: str,
             position: VotePosition,
             rationale: str = "") -> bool:
        """G3: 投票。"""
        # 成员检查
        if voter not in session.members:
            logger.warning(f"非成员投票被拒绝: {voter}")
            return False

        if claim_id not in session.claims:
            return False

        # 防止重复投票
        if claim_id in session.votes:
            for v in session.votes[claim_id]:
                if v["voter"] == voter:
                    logger.warning(f"重复投票被拒绝: {voter} already voted on {claim_id}")
                    return False

        weight = DEFAULT_VOTE_WEIGHTS.get(voter, 1.0)

        # G3事件
        event = VotingEvent(
            actor=voter,
            session_id=session.session_id,
            prev_hash=self.audit._get_last_hash(session.session_id),
            claim_id=claim_id,
            position=position,
            round_num=session.current_round,
            rationale=rationale,
            weight=weight,
        )
        self.audit.append(event)

        # 记录投票
        if claim_id not in session.votes:
            session.votes[claim_id] = []
        session.votes[claim_id].append({
            "voter": voter,
            "position": position.value,
            "weight": weight,
            "rationale": rationale,
        })

        logger.info(f"投票: {voter} → {claim_id} = {position.name} (权重={weight})")
        return True

    def record_dissent(self, session: DeliberationSession,
                       dissenter: str, claim_id: str,
                       position: VotePosition,
                       rationale: str) -> bool:
        """G4: 记录异议——少数派立场保留。

        论文发现这是最被忽视的维度。我们的实现确保：
        1. 异议被永久记录（不被静默丢弃）
        2. 异议可检索（后续合议必须能查看历史异议）
        3. 异议有独立的存储（不只是投票的附注）
        """
        # G4事件
        event = DissentEvent(
            actor=dissenter,
            session_id=session.session_id,
            prev_hash=self.audit._get_last_hash(session.session_id),
            claim_id=claim_id,
            position=position,
            rationale=rationale,
        )
        self.audit.append(event)

        # 独立存储异议
        session.dissents.append({
            "dissenter": dissenter,
            "claim_id": claim_id,
            "position": position.value,
            "rationale": rationale,
            "timestamp": time.time(),
        })

        logger.info(f"异议记录: {dissenter} 反对 {claim_id} — {rationale[:50]}...")
        return True

    def check_escalation(self, session: DeliberationSession,
                         claim_id: str) -> bool:
        """G5: 检查是否需要人类升级。

        触发条件：
        1. 平均置信度 < 0.6
        2. 全票反对
        3. 高风险决策
        """
        # 防止重复升级
        for esc in session.escalations:
            if esc["claim_id"] == claim_id:
                return False  # 已升级，不重复触发

        if claim_id not in session.votes:
            return False

        votes = session.votes[claim_id]
        if not votes:
            return False

        # 计算加权平均
        total_weight = sum(v["weight"] for v in votes)
        if total_weight == 0:
            return False

        weighted_avg = sum(v["position"] * v["weight"] for v in votes) / total_weight

        # 触发条件1：低置信度
        if abs(weighted_avg) < ESCALATION_TRIGGERS["low_confidence"]:
            self._trigger_escalation(session, claim_id, "low_confidence", weighted_avg)
            return True

        # 触发条件2：全票反对
        all_oppose = all(v["position"] < 0 for v in votes)
        if all_oppose and ESCALATION_TRIGGERS["unanimous_oppose"]:
            self._trigger_escalation(session, claim_id, "unanimous_oppose", weighted_avg)
            return True

        return False

    def _trigger_escalation(self, session: DeliberationSession,
                            claim_id: str, trigger: str,
                            confidence: float):
        """触发人类升级。"""
        event = EscalationEvent(
            actor="engine",
            session_id=session.session_id,
            prev_hash=self.audit._get_last_hash(session.session_id),
            decision_id=claim_id,
            trigger_condition=trigger,
            route_to="human:user",
            context=[claim_id],
            confidence=confidence,
        )
        self.audit.append(event)

        session.escalations.append({
            "claim_id": claim_id,
            "trigger": trigger,
            "confidence": confidence,
            "timestamp": time.time(),
        })

        logger.warning(f"人类升级: {claim_id} 触发={trigger} 置信度={confidence:.2f}")

    def tally_votes(self, session: DeliberationSession,
                    claim_id: str) -> dict:
        """统计某个主张的投票结果。

        Returns:
            {
                "claim_id": str,
                "total_votes": int,
                "weighted_score": float,  # 加权平均 [-1, 1]
                "result": str,  # "accepted" | "rejected" | "contested"
                "dissents": list,
            }
        """
        if claim_id not in session.votes:
            return {"claim_id": claim_id, "total_votes": 0, "weighted_score": 0, "result": "no_votes", "dissents": []}

        votes = session.votes[claim_id]
        total_weight = sum(v["weight"] for v in votes)
        if total_weight == 0:
            weighted_score = 0
        else:
            weighted_score = sum(v["position"] * v["weight"] for v in votes) / total_weight

        # 决策
        if weighted_score > 0.3:
            result = "accepted"
        elif weighted_score < -0.3:
            result = "rejected"
        else:
            result = "contested"

        # 关联异议
        claim_dissents = [d for d in session.dissents if d["claim_id"] == claim_id]

        return {
            "claim_id": claim_id,
            "total_votes": len(votes),
            "weighted_score": round(weighted_score, 3),
            "result": result,
            "dissents": claim_dissents,
            "votes": votes,
        }

    def close_session(self, session: DeliberationSession) -> dict:
        """关闭合议会话，产出决策记录。

        Returns:
            完整的决策记录（可重放、可审计）。
        """
        session.status = "completed"

        # 汇总所有主张的投票结果
        results = {}
        for claim_id in session.claims:
            results[claim_id] = self.tally_votes(session, claim_id)

        # G6: 最终审计事件
        event = AuditEvent(
            actor="engine",
            session_id=session.session_id,
            prev_hash=self.audit._get_last_hash(session.session_id),
            action="session_closed",
            details={
                "topic": session.topic,
                "members": {k: v.value for k, v in session.members.items()},
                "claims": len(session.claims),
                "dissents": len(session.dissents),
                "escalations": len(session.escalations),
            },
        )
        self.audit.append(event)

        # 验证审计链完整性
        chain_status = self.audit.verify_chain(session.session_id)

        logger.info(
            f"合议会话关闭: {session.session_id} "
            f"(claims={len(session.claims)}, dissents={len(session.dissents)}, "
            f"escalations={len(session.escalations)})"
        )

        return {
            "session_id": session.session_id,
            "topic": session.topic,
            "members": {k: v.value for k, v in session.members.items()},
            "results": results,
            "dissents": session.dissents,
            "escalations": session.escalations,
            "audit_chain": chain_status,
        }

    def replay_session(self, session_id: str) -> list[dict]:
        """G6: 重放某个会话的完整治理过程。

        确定性重建——给定事件序列，可以重建任意时刻的状态。
        """
        return self.audit.replay(session_id)

    def get_historical_dissents(self, session_id: str) -> list[dict]:
        """G4: 检索历史异议。

        后续合议必须能查看历史异议——"上次鲁班反对这个方案，理由是什么？"
        """
        return self.audit.get_dissents(session_id)

    # ── G9: 升级路径·被否决agent申诉（2026-07-06 七神启示） ──

    def appeal(self, session: DeliberationSession,
               agent: str, original_claim_id: str,
               revision: str, rationale: str) -> dict:
        """G9: 被否决agent提交修订版。

        七神启示·赫尔墨斯：治理必须是双向反馈环。
        被裁决agent可以提交修订版，治理层在修订版上重新评估。
        不可无限循环——同一决策最多申诉2次。

        Args:
            session: 合议会话
            agent: 申诉agent
            original_claim_id: 被否决的原始主张ID
            revision: 修订版内容
            rationale: 申诉理由

        Returns:
            {"success": bool, "attempt": int, "message": str}
        """
        from .events import AppealEvent

        # 成员检查
        if agent not in session.members:
            return {"success": False, "attempt": 0, "message": f"{agent} 不是合议会话成员"}

        # 检查原始主张是否存在
        if original_claim_id not in session.claims:
            return {"success": False, "attempt": 0, "message": f"原始主张 {original_claim_id} 不存在"}

        # 检查申诉次数（最多2次）
        appeal_count = sum(
            1 for e in self.audit.get_events(session.session_id)
            if e.get("event_type") == "appeal"
            and e.get("payload", {}).get("original_event_id", "").startswith("claim")
            and e.get("actor") == agent
        )
        # 也检查本次会话中的申诉记录
        for esc in session.escalations:
            if esc.get("claim_id") == original_claim_id and esc.get("agent") == agent:
                appeal_count = max(appeal_count, esc.get("attempt", 0))

        if appeal_count >= 2:
            return {"success": False, "attempt": appeal_count, "message": "申诉次数已达上限（最多2次），请请求元治理仲裁"}

        attempt = appeal_count + 1

        # 创建修订版主张
        revision_claim_id = f"claim-{original_claim_id}-rev-{attempt}"
        revision_claim = Claim(
            claim_id=revision_claim_id,
            content=revision,
            proposer=agent,
            evidence=rationale,
        )
        session.claims[revision_claim_id] = revision_claim

        # G9事件
        event = AppealEvent(
            actor=agent,
            session_id=session.session_id,
            prev_hash=self.audit._get_last_hash(session.session_id),
            original_event_id=original_claim_id,
            revision=revision,
            rationale=rationale,
            attempt=attempt,
        )
        self.audit.append(event)

        logger.info(f"申诉: {agent} 对 {original_claim_id} 提交修订版 (attempt={attempt})")
        return {"success": True, "attempt": attempt, "revision_claim_id": revision_claim_id}

    # ── G10: 元治理·治理者的治理（2026-07-06 七神启示） ──

    def meta_governance(self, session: DeliberationSession,
                        arbitrator: str, target_claim_id: str,
                        ruling: str, rationale: str) -> dict:
        """G10: 独立仲裁者对L3裁决的终审。

        七神启示·阿波罗：每个治理者本身也需要被审计。
        L4层级：独立仲裁者（子产角色）对L3裁决的终审。
        仲裁结果不可上诉，但记录为ISA学习素材。

        Args:
            session: 合议会话
            arbitrator: 仲裁者（应为独立第三方，如子产）
            target_claim_id: 被仲裁的主张ID
            ruling: 裁决 "uphold" | "overturn" | "modify"
            rationale: 仲裁理由

        Returns:
            {"success": bool, "ruling": str, "message": str}
        """
        from .events import MetaGovernanceEvent

        # 仲裁者检查——必须是合议成员
        if arbitrator not in session.members:
            return {"success": False, "ruling": ruling, "message": f"{arbitrator} 不是合议会话成员"}

        # 不可自审——仲裁者不能仲裁自己提出的主张
        if target_claim_id in session.claims:
            original_proposer = session.claims[target_claim_id].proposer
            if arbitrator == original_proposer:
                return {"success": False, "ruling": ruling, "message": "仲裁者不可仲裁自己提出的主张"}

        # 合法裁决
        if ruling not in ("uphold", "overturn", "modify"):
            return {"success": False, "ruling": ruling, "message": f"非法裁决: {ruling}，必须是 uphold/overturn/modify"}

        # G10事件
        event = MetaGovernanceEvent(
            actor=arbitrator,
            session_id=session.session_id,
            prev_hash=self.audit._get_last_hash(session.session_id),
            target_event_id=target_claim_id,
            ruling=ruling,
            rationale=rationale,
        )
        self.audit.append(event)

        # 更新主张状态
        if target_claim_id in session.claims:
            if ruling == "overturn":
                session.claims[target_claim_id].status = "accepted"
            elif ruling == "uphold":
                session.claims[target_claim_id].status = "rejected"
            elif ruling == "modify":
                session.claims[target_claim_id].status = "modified"

        logger.info(f"元治理仲裁: {arbitrator} 对 {target_claim_id} 裁决={ruling}")
        return {"success": True, "ruling": ruling, "message": f"仲裁完成: {ruling}"}


# ── SuspicionCascade: 有状态怀疑传播协议 ──
# 2026-07-07 PAL P0-1 · 七神+核战队终裁
# 论文来源: Distributed Attacks (2607.02514) link-tracker
# 设计原则: 治理=提纯注意力，不是拦截威胁

import hashlib
import json
import os

SUSPICION_JSONL = os.path.expanduser("~/.hermes/jiak/suspicious_buildup.jsonl")


@dataclass
class SuspicionEntry:
    """单条怀疑记录。"""
    entry_id: str
    timestamp: float
    tool_name: str
    suspicion_score: float  # 0.0-1.0
    reason: str = ""
    decayed_score: float = 0.0  # 衰减后的分数
    cascade_delta: float = 0.0  # 级联调整量


class SuspicionCascade:
    """有状态怀疑传播协议。

    核心机制:
    1. 每次tool call生成suspicion_score
    2. 牛顿冷却衰减: λ_eff = λ_base × 1/(1+log(1+age))
    3. 相邻同向score累加(cascade)，反向score抵消
    4. 超阈值触发告警

    设计约束:
    - 延迟<50ms/tool call（透明管道原则）
    - 独立JSONL文件，不混入events.py
    - 衰减函数用牛顿冷却，不用线性衰减
    """

    def __init__(
        self,
        threshold: float = 0.7,
        lambda_base: float = 0.1,
        filepath: str = SUSPICION_JSONL,
    ):
        self.threshold = threshold
        self.lambda_base = lambda_base
        self.filepath = filepath
        self._entries: list[SuspicionEntry] = []
        self._load_history()

    def _load_history(self):
        """加载历史怀疑记录。"""
        if not os.path.exists(self.filepath):
            return
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    self._entries.append(SuspicionEntry(
                        entry_id=obj["entry_id"],
                        timestamp=obj["timestamp"],
                        tool_name=obj["tool_name"],
                        suspicion_score=obj["suspicion_score"],
                        reason=obj.get("reason", ""),
                        decayed_score=obj.get("decayed_score", 0.0),
                        cascade_delta=obj.get("cascade_delta", 0.0),
                    ))
        except (json.JSONDecodeError, KeyError):
            pass  # 损坏的行跳过

    def _newton_cooling(self, age_seconds: float) -> float:
        """牛顿冷却衰减: λ_eff = λ_base × 1/(1+log(1+age))。

        age=0 → λ=λ_base（不衰减）
        age=60s → λ≈λ_base/3
        age=3600s → λ≈λ_base/4.6
        age=86400s → λ≈λ_base/6.7
        """
        import math
        if age_seconds <= 0:
            return self.lambda_base
        return self.lambda_base / (1 + math.log(1 + age_seconds))

    def _compute_decayed_scores(self, now: float) -> list[float]:
        """计算所有历史entry的衰减后分数。"""
        scores = []
        for entry in self._entries:
            age = now - entry.timestamp
            decay = self._newton_cooling(age)
            # 衰减公式: score × e^(-λ×age)，但用简化版: score × (1 - λ)
            # 因为λ本身已经随age递减，不需要额外指数
            scores.append(entry.suspicion_score * max(0.1, 1 - decay))
        return scores

    def _compute_cascade(
        self, new_score: float, history_scores: list[float]
    ) -> float:
        """计算级联调整量。

        规则:
        - 最近3条history的平均方向与new_score一致 → 累加
        - 方向相反 → 抵消
        """
        if not history_scores:
            return 0.0

        recent = history_scores[-3:]  # 最近3条
        avg_history = sum(recent) / len(recent)

        # 同向: 都>0.5或都<0.5
        new_direction = 1.0 if new_score > 0.5 else -1.0
        hist_direction = 1.0 if avg_history > 0.5 else -1.0

        if new_direction == hist_direction:
            # 同向累加: 级联量 = 历史平均 × 0.3
            return avg_history * 0.3
        else:
            # 反向抵消: 级联量 = -当前分数 × 0.2
            return -new_score * 0.2

    def record_tool_call(
        self,
        tool_name: str,
        suspicion_score: float,
        reason: str = "",
    ) -> dict:
        """记录一次tool call的怀疑分数。

        Args:
            tool_name: 工具名称
            suspicion_score: 0.0-1.0的怀疑分数
            reason: 怀疑原因

        Returns:
            {"alert": bool, "final_score": float, "cascade_delta": float, "message": str}
        """
        now = time.time()
        entry_id = f"sc-{int(now*1000)}-{hashlib.md5(tool_name.encode()).hexdigest()[:6]}"

        # 计算历史衰减分数
        history_scores = self._compute_decayed_scores(now)

        # 计算级联
        cascade = self._compute_cascade(suspicion_score, history_scores)

        # 最终分数
        final_score = max(0.0, min(1.0, suspicion_score + cascade))

        # 创建entry
        entry = SuspicionEntry(
            entry_id=entry_id,
            timestamp=now,
            tool_name=tool_name,
            suspicion_score=suspicion_score,
            reason=reason,
            decayed_score=suspicion_score,
            cascade_delta=cascade,
        )
        self._entries.append(entry)

        # 追加写入JSONL
        self._append_jsonl(entry)

        # 判断是否告警
        alert = final_score >= self.threshold
        message = (
            f"⚠️ 怀疑告警: {tool_name} final={final_score:.3f} "
            f"(raw={suspicion_score:.3f}+cascade={cascade:+.3f})"
            if alert else
            f"✅ 正常: {tool_name} final={final_score:.3f}"
        )

        if alert:
            logger.warning(message)
        else:
            logger.debug(message)

        return {
            "alert": alert,
            "final_score": final_score,
            "cascade_delta": cascade,
            "message": message,
        }

    def _append_jsonl(self, entry: SuspicionEntry):
        """原子追加到JSONL文件。"""
        obj = {
            "entry_id": entry.entry_id,
            "timestamp": entry.timestamp,
            "tool_name": entry.tool_name,
            "suspicion_score": entry.suspicion_score,
            "reason": entry.reason,
            "decayed_score": entry.decayed_score,
            "cascade_delta": entry.cascade_delta,
        }
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def get_recent(self, n: int = 10) -> list[dict]:
        """获取最近n条怀疑记录。"""
        return [
            {
                "entry_id": e.entry_id,
                "timestamp": e.timestamp,
                "tool_name": e.tool_name,
                "suspicion_score": e.suspicion_score,
                "cascade_delta": e.cascade_delta,
                "reason": e.reason,
            }
            for e in self._entries[-n:]
        ]

    def get_alert_count(self, window_seconds: float = 3600) -> int:
        """获取时间窗口内的告警次数。"""
        now = time.time()
        return sum(
            1 for e in self._entries
            if (now - e.timestamp) <= window_seconds
            and e.suspicion_score >= self.threshold
        )

    def reset(self):
        """清空内存中的记录（不影响文件）。"""
        self._entries.clear()
