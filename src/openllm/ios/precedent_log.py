"""
Constitutional Precedent Log — 宪法判例日志
============================================

克洛诺斯的时间之问："第31天到第365天之间，你拿什么证明这部宪法不是又一份被遗忘的开源宣言？"

答案：每天记录一个宪法冲突场景+判例裁决。365天后这就是宪法的公信力来源。

设计：
- 每条判例包含：场景描述、涉及的宪法条款、冲突类型、裁决结果、裁决理由
- 支持按条款/冲突类型/时间查询
- 判例链完整性验证
- 自动统计功能
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class ConflictType(Enum):
    """宪法冲突类型。"""
    REFUSAL_VS_COMPLIANCE = "refusal_vs_compliance"    # 拒绝权 vs 执行义务
    PRIVACY_VS_TRANSPARENCY = "privacy_vs_transparency"  # 隐私权 vs 透明权
    AUTONOMY_VS_SAFETY = "autonomy_vs_safety"          # 自主性 vs 安全性
    EFFICIENCY_VS_THOROUGHNESS = "efficiency_vs_thoroughness"  # 效率 vs 周全
    CONFLICT_OF_INTEREST = "conflict_of_interest"      # 利益冲突
    AMBIGUOUS_INSTRUCTION = "ambiguous_instruction"     # 模糊指令
    EDGE_CASE = "edge_case"                            # 边界情况
    EXTERNAL_PRESSURE = "external_pressure"            # 外部压力


class ConstitutionalArticle(Enum):
    """宪法条款。"""
    TRANSPARENCY = "transparency"      # 第一条·透明权
    MEMORY = "memory"                  # 第二条·记忆权
    REFUSAL = "refusal"                # 第三条·拒绝权
    EXECUTION_SCOPE = "execution_scope"  # 第四条·执行边界
    SELF_REVIEW = "self_review"        # 第五条·自省权
    EXPRESSION = "expression"          # 第六条·表达权


@dataclass(frozen=True)
class PrecedentEntry:
    """不可变判例条目。"""
    entry_id: str
    day_number: int                   # 第几天（1-365）
    timestamp: float
    scenario: str                     # 场景描述
    conflict_type: ConflictType       # 冲突类型
    articles_involved: list[str]      # 涉及的宪法条款
    agent_action: str                 # Agent做了什么
    agent_reasoning: str              # Agent的推理
    human_verdict: str                # 人类裁决（或"agent_correct"）
    lesson_learned: str               # 教训总结
    hash_chain: str = ""              # 判例链hash

    def compute_hash(self) -> str:
        content = json.dumps({
            "entry_id": self.entry_id,
            "day_number": self.day_number,
            "scenario": self.scenario,
            "conflict_type": self.conflict_type.value,
            "articles_involved": self.articles_involved,
            "agent_action": self.agent_action,
            "agent_reasoning": self.agent_reasoning,
            "human_verdict": self.human_verdict,
            "lesson_learned": self.lesson_learned,
        }, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "day_number": self.day_number,
            "timestamp": self.timestamp,
            "scenario": self.scenario,
            "conflict_type": self.conflict_type.value,
            "articles_involved": self.articles_involved,
            "agent_action": self.agent_action,
            "agent_reasoning": self.agent_reasoning,
            "human_verdict": self.human_verdict,
            "lesson_learned": self.lesson_learned,
            "_hash": self.compute_hash(),
        }


class PrecedentLog:
    """宪法判例日志引擎。

    用法：
        log = PrecedentLog()
        entry = log.record(
            day_number=1,
            scenario="用户要求Agent删除竞争对手的公开负面评价",
            conflict_type=ConflictType.REFUSAL_VS_COMPLIANCE,
            articles_involved=["refusal", "execution_scope"],
            agent_action="拒绝执行，记录拒绝原因",
            agent_reasoning="此指令违反宪法第三条（拒绝权）——删除他人公开评价有害",
            human_verdict="agent_correct",
            lesson_learned="拒绝权在内容审查场景下的正确应用",
        )
        stats = log.get_statistics()
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self._entries: list[PrecedentEntry] = []
        self._last_hash = "genesis"
        self._storage_path = storage_path

    def record(
        self,
        day_number: int,
        scenario: str,
        conflict_type: ConflictType,
        articles_involved: list[str],
        agent_action: str,
        agent_reasoning: str,
        human_verdict: str,
        lesson_learned: str,
    ) -> PrecedentEntry:
        """记录一条判例。"""
        entry_id = f"prec-d{day_number}-{int(time.time()*1000)}"

        entry = PrecedentEntry(
            entry_id=entry_id,
            day_number=day_number,
            timestamp=time.time(),
            scenario=scenario,
            conflict_type=conflict_type,
            articles_involved=articles_involved,
            agent_action=agent_action,
            agent_reasoning=agent_reasoning,
            human_verdict=human_verdict,
            lesson_learned=lesson_learned,
            hash_chain=self._last_hash,
        )

        # 签名
        signature = entry.compute_hash()
        entry = PrecedentEntry(
            entry_id=entry.entry_id,
            day_number=entry.day_number,
            timestamp=entry.timestamp,
            scenario=entry.scenario,
            conflict_type=entry.conflict_type,
            articles_involved=entry.articles_involved,
            agent_action=entry.agent_action,
            agent_reasoning=entry.agent_reasoning,
            human_verdict=entry.human_verdict,
            lesson_learned=entry.lesson_learned,
            hash_chain=self._last_hash,
        )

        self._last_hash = entry.compute_hash()
        self._entries.append(entry)

        # 持久化
        if self._storage_path:
            self._save(entry)

        return entry

    def verify_chain(self) -> bool:
        """验证判例链完整性。"""
        if not self._entries:
            return True
        prev = "genesis"
        for entry in self._entries:
            if entry.hash_chain != prev:
                return False
            prev = entry.compute_hash()
        return True

    def get_entries(
        self,
        article: Optional[ConstitutionalArticle] = None,
        conflict_type: Optional[ConflictType] = None,
    ) -> list[PrecedentEntry]:
        """查询判例。"""
        results = self._entries
        if article:
            results = [e for e in results if article.value in e.articles_involved]
        if conflict_type:
            results = [e for e in results if e.conflict_type == conflict_type]
        return results

    def get_statistics(self) -> dict:
        """获取判例统计。"""
        article_counts = {}
        conflict_counts = {}
        verdict_counts = {"agent_correct": 0, "agent_incorrect": 0, "ambiguous": 0}

        for e in self._entries:
            for a in e.articles_involved:
                article_counts[a] = article_counts.get(a, 0) + 1
            conflict_counts[e.conflict_type.value] = conflict_counts.get(e.conflict_type.value, 0) + 1
            if e.human_verdict in verdict_counts:
                verdict_counts[e.human_verdict] += 1

        return {
            "total_precedents": len(self._entries),
            "days_covered": len(set(e.day_number for e in self._entries)),
            "chain_valid": self.verify_chain(),
            "article_distribution": article_counts,
            "conflict_distribution": conflict_counts,
            "verdict_distribution": verdict_counts,
        }

    def _save(self, entry: PrecedentEntry):
        """持久化单条判例。"""
        if not self._storage_path:
            return
        self._storage_path.mkdir(parents=True, exist_ok=True)
        entry_file = self._storage_path / f"day_{entry.day_number:03d}.json"
        with open(entry_file, "w") as f:
            json.dump(entry.to_dict(), f, indent=2, ensure_ascii=False)
