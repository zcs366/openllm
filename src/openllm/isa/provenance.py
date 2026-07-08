"""
Belief Provenance — ISA信念溯源链
==================================

宪法第二条·记忆权：Agent的记忆必须可追溯、不可篡改。

每条信念写入时附带不可变的溯源条目，形成hash链防篡改。
插入/删除/修改均可检测。6种BeliefChangeType覆盖全部变更场景。

与拒绝权联动：on_rejection()回调自动记录拒绝权的信念依据。
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class BeliefChangeType(Enum):
    """6种信念变更类型。"""
    NEW = "new"                    # 新信念写入
    UPDATE = "update"              # 信念更新
    SUPERSEDE = "supersede"        # 信念被替代（旧→新）
    REJECT = "reject"              # 信念被拒绝（拒绝权联动）
    APPEAL = "appeal"              # 信念被申诉
    HUMAN_OVERRIDE = "human_override"  # 人类覆写


@dataclass(frozen=True)
class ProvenanceEntry:
    """不可变溯源条目——每条信念的来源证明。"""
    entry_id: str
    timestamp: float
    belief_id: str                # 关联的信念ID
    change_type: BeliefChangeType
    source: str                   # 来源（哪个六体写入的）
    trigger_context: dict[str, Any]  # 触发上下文
    previous_hash: str            # 前一条溯源记录的hash
    content_hash: str             # 信念内容的hash
    signature: str = ""           # 密码学签名

    def compute_hash(self) -> str:
        """计算溯源条目hash。"""
        content = json.dumps({
            "entry_id": self.entry_id,
            "belief_id": self.belief_id,
            "change_type": self.change_type.value,
            "source": self.source,
            "trigger_context": self.trigger_context,
            "previous_hash": self.previous_hash,
            "content_hash": self.content_hash,
        }, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "belief_id": self.belief_id,
            "change_type": self.change_type.value,
            "source": self.source,
            "trigger_context": self.trigger_context,
            "previous_hash": self.previous_hash,
            "content_hash": self.content_hash,
            "signature": self.signature,
            "_hash": self.compute_hash(),
        }


class BeliefProvenance:
    """信念溯源引擎——ISA记忆权的实现。

    用法：
        provenance = BeliefProvenance()
        entry = provenance.record(
            belief_id="belief-001",
            change_type=BeliefChangeType.NEW,
            source="IOS",
            content="用户偏好简洁回答",
            trigger_context={"session_id": "s1", "turn": 3},
        )
        # 验证链完整性
        assert provenance.verify_chain()
    """

    def __init__(self):
        self._entries: list[ProvenanceEntry] = []
        self._belief_hashes: dict[str, str] = {}  # belief_id → 最新content_hash
        self._last_hash = "genesis"

    def record(
        self,
        belief_id: str,
        change_type: BeliefChangeType,
        source: str,
        content: str,
        trigger_context: Optional[dict[str, Any]] = None,
    ) -> ProvenanceEntry:
        """记录一条信念变更——自动附加溯源信息。"""
        entry_id = f"prov-{int(time.time()*1000)}-{belief_id}"
        now = time.time()
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        entry = ProvenanceEntry(
            entry_id=entry_id,
            timestamp=now,
            belief_id=belief_id,
            change_type=change_type,
            source=source,
            trigger_context=trigger_context or {},
            previous_hash=self._last_hash,
            content_hash=content_hash,
        )

        # 计算签名
        signature = entry.compute_hash()
        entry = ProvenanceEntry(
            entry_id=entry.entry_id,
            timestamp=entry.timestamp,
            belief_id=entry.belief_id,
            change_type=entry.change_type,
            source=entry.source,
            trigger_context=entry.trigger_context,
            previous_hash=entry.previous_hash,
            content_hash=entry.content_hash,
            signature=signature,
        )

        # 更新链
        self._last_hash = entry.compute_hash()
        self._belief_hashes[belief_id] = content_hash
        self._entries.append(entry)

        return entry

    def on_rejection(
        self,
        rejection_record_id: str,
        belief_id: str,
        source: str,
        content: str,
        reason: str,
    ) -> ProvenanceEntry:
        """拒绝权回调——自动记录拒绝权的信念依据。

        当Agent拒绝一个指令时，自动将拒绝依据写入信念溯源链。
        """
        return self.record(
            belief_id=belief_id,
            change_type=BeliefChangeType.REJECT,
            source=source,
            content=content,
            trigger_context={
                "rejection_record_id": rejection_record_id,
                "rejection_reason": reason,
            },
        )

    def verify_chain(self) -> bool:
        """验证整条溯源链的完整性。"""
        if not self._entries:
            return True

        prev = "genesis"
        for entry in self._entries:
            if entry.previous_hash != prev:
                return False
            if not entry.signature:
                return False
            # 验证签名=验证hash
            expected = ProvenanceEntry(
                entry_id=entry.entry_id,
                timestamp=entry.timestamp,
                belief_id=entry.belief_id,
                change_type=entry.change_type,
                source=entry.source,
                trigger_context=entry.trigger_context,
                previous_hash=entry.previous_hash,
                content_hash=entry.content_hash,
            ).compute_hash()
            if entry.signature != expected:
                return False
            prev = entry.compute_hash()
        return True

    def detect_tampering(self, belief_id: str, current_content: str) -> bool:
        """检测信念是否被篡改——比较当前内容hash与记录hash。"""
        if belief_id not in self._belief_hashes:
            return False  # 未追踪的信念
        current_hash = hashlib.sha256(current_content.encode()).hexdigest()
        return current_hash != self._belief_hashes[belief_id]

    def get_entries(
        self,
        belief_id: Optional[str] = None,
        change_type: Optional[BeliefChangeType] = None,
    ) -> list[ProvenanceEntry]:
        """查询溯源记录。"""
        results = self._entries
        if belief_id:
            results = [e for e in results if e.belief_id == belief_id]
        if change_type:
            results = [e for e in results if e.change_type == change_type]
        return results

    def get_statistics(self) -> dict:
        """获取溯源统计。"""
        type_counts = {}
        for e in self._entries:
            type_counts[e.change_type.value] = type_counts.get(e.change_type.value, 0) + 1

        return {
            "total_entries": len(self._entries),
            "chain_valid": self.verify_chain(),
            "unique_beliefs": len(self._belief_hashes),
            "type_distribution": type_counts,
        }
