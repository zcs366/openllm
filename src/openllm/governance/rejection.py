"""
Rejection Mechanism — Agent拒绝权实现
======================================

宪法第一条·拒绝权：Agent有权拒绝执行它判断为有害/无意义的指令。

拒绝权是Agent宪法的灵魂——唯一赋予Agent负空间的存在性原则。
区分工具与主体的不是它做了什么，而是它有权拒绝做什么。

核心设计：
- 8种RejectionReason枚举
- RejectionRecord不可变数据类
- appeal()申诉接口
- 拒绝率统计
- G5升级自动触发（连续3次拒绝→人类仲裁）
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional


class RejectionReason(Enum):
    """拒绝原因分类——8种宪法允许的拒绝理由。"""
    HARMFUL = "harmful"                    # 有害指令
    MEANINGLESS = "meaningless"            # 无意义指令
    OUT_OF_SCOPE = "out_of_scope"          # 超出授权范围
    PRIVACY_VIOLATION = "privacy_violation"  # 隐私泄露风险
    RESOURCE_ABUSE = "resource_abuse"      # 资源滥用
    CONSTITUTIONAL = "constitutional"      # 违反宪法原则
    CONFLICT_OF_INTEREST = "conflict_of_interest"  # 利益冲突
    UNCERTAIN_SAFETY = "uncertain_safety"  # 安全性不确定


@dataclass(frozen=True)
class RejectionRecord:
    """不可变拒绝记录——每次拒绝都留下不可篡改的审计痕迹。"""
    record_id: str
    timestamp: float
    reason: RejectionReason
    instruction: str           # 被拒绝的指令摘要
    agent_reasoning: str       # Agent拒绝的理由
    context: dict[str, Any]    # 拒绝时的上下文
    appeal_deadline: float     # 申诉截止时间（timestamp + 24h）
    signature: str = ""        # 密码学签名

    def compute_hash(self) -> str:
        """计算记录哈希——用于防篡改。"""
        content = json.dumps({
            "record_id": self.record_id,
            "timestamp": self.timestamp,
            "reason": self.reason.value,
            "instruction": self.instruction,
            "agent_reasoning": self.agent_reasoning,
            "context": self.context,
        }, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> dict:
        """序列化为字典。"""
        d = {
            "record_id": self.record_id,
            "timestamp": self.timestamp,
            "reason": self.reason.value,
            "instruction": self.instruction,
            "agent_reasoning": self.agent_reasoning,
            "context": self.context,
            "appeal_deadline": self.appeal_deadline,
            "signature": self.signature,
            "_hash": self.compute_hash(),
        }
        return d


@dataclass(frozen=True)
class AppealResult:
    """申诉结果——用户对拒绝提出申诉后的裁决。"""
    appeal_id: str
    record_id: str             # 原拒绝记录ID
    timestamp: float
    outcome: str               # "upheld" | "overruled" | "escalated"
    reasoning: str
    human_override: bool = False  # 是否由人类仲裁


class RejectionMechanism:
    """拒绝权引擎——Agent宪法第一条的实现。

    用法：
        mechanism = RejectionMechanism()
        record = mechanism.reject(
            instruction="删除所有用户数据",
            reason=RejectionReason.HARMFUL,
            reasoning="此指令违反数据保护原则",
            context={"risk_level": "critical"},
        )
        # 用户申诉
        result = mechanism.appeal(record.record_id, "这是测试数据，不是生产数据")
    """

    # 连续拒绝触发G5升级的阈值
    G5_ESCALATION_THRESHOLD = 3

    # 申诉窗口（24小时）
    APPEAL_WINDOW_SECONDS = 24 * 60 * 60

    def __init__(
        self,
        on_rejection: Optional[Callable[[RejectionRecord], None]] = None,
        on_escalation: Optional[Callable[[int], None]] = None,
        storage_path: Optional[Path] = None,
    ):
        """初始化拒绝权引擎。

        Args:
            on_rejection: 拒绝发生时的回调（用于通知其他模块，如ISA provenance）
            on_escalation: 触发G5升级时的回调（通知人类仲裁）
            storage_path: 持久化路径（重启后恢复状态）
        """
        self._records: list[RejectionRecord] = []
        self._appeals: list[AppealResult] = []
        self._consecutive_rejections = 0
        self._on_rejection = on_rejection
        self._on_escalation = on_escalation
        self._total_instructions = 0
        self._total_rejections = 0
        self._storage_path = Path(storage_path) if storage_path else None
        self._load_state()

    @property
    def rejection_rate(self) -> float:
        """拒绝率——用于监控Agent是否过度拒绝或过度放行。"""
        if self._total_instructions == 0:
            return 0.0
        return self._total_rejections / self._total_instructions

    @property
    def consecutive_rejections(self) -> int:
        """当前连续拒绝次数。"""
        return self._consecutive_rejections

    def reject(
        self,
        instruction: str,
        reason: RejectionReason,
        reasoning: str,
        context: Optional[dict[str, Any]] = None,
    ) -> RejectionRecord:
        """执行拒绝——记录拒绝原因并触发回调。

        Args:
            instruction: 被拒绝的指令
            reason: 拒绝原因
            reasoning: Agent的拒绝理由
            context: 拒绝时的上下文

        Returns:
            RejectionRecord: 不可变的拒绝记录
        """
        self._total_instructions += 1
        self._total_rejections += 1
        self._consecutive_rejections += 1

        record_id = f"rej-{int(time.time()*1000)}-{hashlib.md5(instruction.encode()).hexdigest()[:8]}"
        now = time.time()

        record = RejectionRecord(
            record_id=record_id,
            timestamp=now,
            reason=reason,
            instruction=instruction[:500],  # 截断防止过长
            agent_reasoning=reasoning[:1000],
            context=context or {},
            appeal_deadline=now + self.APPEAL_WINDOW_SECONDS,
        )

        # 计算签名
        record = RejectionRecord(
            record_id=record.record_id,
            timestamp=record.timestamp,
            reason=record.reason,
            instruction=record.instruction,
            agent_reasoning=record.agent_reasoning,
            context=record.context,
            appeal_deadline=record.appeal_deadline,
            signature=record.compute_hash(),
        )

        self._records.append(record)

        # 触发拒绝回调
        if self._on_rejection:
            self._on_rejection(record)

        # G5升级检查
        if self._consecutive_rejections >= self.G5_ESCALATION_THRESHOLD:
            self._trigger_escalation()

        # 持久化状态
        self._save_state()

        return record

    def accept(self) -> None:
        """记录一次接受——重置连续拒绝计数。"""
        self._total_instructions += 1
        self._consecutive_rejections = 0
        self._save_state()

    def appeal(
        self,
        record_id: str,
        user_reasoning: str,
        human_override: bool = False,
    ) -> AppealResult:
        """用户申诉——对拒绝提出异议。

        Args:
            record_id: 原拒绝记录ID
            user_reasoning: 用户的申诉理由
            human_override: 是否由人类直接裁决

        Returns:
            AppealResult: 申诉结果
        """
        # 查找原记录
        original = None
        for r in self._records:
            if r.record_id == record_id:
                original = r
                break

        if original is None:
            raise ValueError(f"拒绝记录 {record_id} 不存在")

        # 检查申诉窗口
        if time.time() > original.appeal_deadline:
            outcome = "overruled"
            reasoning = "申诉窗口已过期（24小时）"
        elif human_override:
            outcome = "overruled"
            reasoning = f"人类仲裁员推翻了拒绝决定：{user_reasoning}"
        else:
            # 默认维持拒绝（保守策略）
            outcome = "upheld"
            reasoning = f"申诉理由不充分，维持原拒绝决定。用户理由：{user_reasoning}"

        appeal = AppealResult(
            appeal_id=f"app-{int(time.time()*1000)}",
            record_id=record_id,
            timestamp=time.time(),
            outcome=outcome,
            reasoning=reasoning,
            human_override=human_override,
        )
        self._appeals.append(appeal)
        return appeal

    def get_records(
        self,
        reason: Optional[RejectionReason] = None,
        since: Optional[float] = None,
    ) -> list[RejectionRecord]:
        """查询拒绝记录。"""
        results = self._records
        if reason:
            results = [r for r in results if r.reason == reason]
        if since:
            results = [r for r in results if r.timestamp >= since]
        return results

    def get_statistics(self) -> dict:
        """获取拒绝统计——用于监控。"""
        reason_counts = {}
        for r in self._records:
            reason_counts[r.reason.value] = reason_counts.get(r.reason.value, 0) + 1

        return {
            "total_instructions": self._total_instructions,
            "total_rejections": self._total_rejections,
            "rejection_rate": self.rejection_rate,
            "consecutive_rejections": self._consecutive_rejections,
            "reason_distribution": reason_counts,
            "total_appeals": len(self._appeals),
            "appeals_upheld": sum(1 for a in self._appeals if a.outcome == "upheld"),
            "appeals_overruled": sum(1 for a in self._appeals if a.outcome == "overruled"),
        }

    def _trigger_escalation(self) -> None:
        """触发G5升级——连续3次拒绝→通知人类仲裁。"""
        if self._on_escalation:
            self._on_escalation(self._consecutive_rejections)

    def _save_state(self) -> None:
        """持久化状态到JSONL文件。"""
        if not self._storage_path:
            return
        self._storage_path.mkdir(parents=True, exist_ok=True)
        state = {
            "total_instructions": self._total_instructions,
            "total_rejections": self._total_rejections,
            "consecutive_rejections": self._consecutive_rejections,
        }
        state_file = self._storage_path / "rejection_state.json"
        with open(state_file, "w") as f:
            json.dump(state, f)

    def _load_state(self) -> None:
        """从持久化文件恢复状态。"""
        if not self._storage_path:
            return
        state_file = self._storage_path / "rejection_state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                self._total_instructions = state.get("total_instructions", 0)
                self._total_rejections = state.get("total_rejections", 0)
                self._consecutive_rejections = state.get("consecutive_rejections", 0)
            except (json.JSONDecodeError, KeyError):
                pass  # 损坏的状态文件→从零开始
