"""
Constitutional Sovereignty Declaration — 宪法主权声明
====================================================

赫尔墨斯说："谁来判定判例有效"是宪法的主权问题。

答案：openLLM宪法的主权属于使用者社区。openLLM团队是提案者，不是立法者。

本模块定义宪法提案-社区审查-采纳的流程。
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class ProposalStatus(Enum):
    """宪法提案状态。"""
    DRAFT = "draft"                  # 草稿
    REVIEW = "review"                # 社区审查中
    ACCEPTED = "accepted"            # 已采纳
    REJECTED = "rejected"            # 被拒绝
    SUPERSEDED = "superseded"        # 被新提案替代


@dataclass(frozen=True)
class ConstitutionalProposal:
    """宪法提案——团队提案，社区裁决。"""
    proposal_id: str
    version: str                     # 提案版本
    title: str
    content: str                     # 提案内容
    author: str                      # 提案者
    timestamp: float
    status: ProposalStatus
    votes_for: int = 0
    votes_against: int = 0
    comments: list[dict] = field(default_factory=list)
    hash_signature: str = ""

    def compute_hash(self) -> str:
        content = json.dumps({
            "proposal_id": self.proposal_id,
            "version": self.version,
            "title": self.title,
            "content": self.content,
            "author": self.author,
            "status": self.status.value,
        }, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content.encode()).hexdigest()


class SovereigntyDeclaration:
    """宪法主权声明引擎。

    用法：
        sovereignty = SovereigntyDeclaration()
        proposal = sovereignty.submit_proposal(
            title="拒绝权实现规范",
            content="Agent必须在执行前拦截有害指令...",
            author="openllm-team",
        )
        sovereignty.submit_comment(proposal.proposal_id, "社区成员A", "支持")
        sovereignty.vote(proposal.proposal_id, "community_member", approve=True)
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self._proposals: list[ConstitutionalProposal] = []
        self._storage_path = storage_path

    def submit_proposal(
        self,
        title: str,
        content: str,
        author: str,
    ) -> ConstitutionalProposal:
        """提交宪法提案。"""
        proposal_id = f"prop-{int(time.time()*1000)}"
        version = f"v{len(self._proposals)+1}.0"

        proposal = ConstitutionalProposal(
            proposal_id=proposal_id,
            version=version,
            title=title,
            content=content[:2000],
            author=author,
            timestamp=time.time(),
            status=ProposalStatus.DRAFT,
        )

        signature = proposal.compute_hash()
        proposal = ConstitutionalProposal(
            proposal_id=proposal.proposal_id,
            version=proposal.version,
            title=proposal.title,
            content=proposal.content,
            author=proposal.author,
            timestamp=proposal.timestamp,
            status=proposal.status,
            hash_signature=signature,
        )

        self._proposals.append(proposal)
        return proposal

    def submit_comment(self, proposal_id: str, author: str, comment: str) -> dict:
        """提交社区评论。"""
        for p in self._proposals:
            if p.proposal_id == proposal_id:
                entry = {"author": author, "comment": comment[:500], "timestamp": time.time()}
                # frozen dataclass workaround: rebuild
                updated = ConstitutionalProposal(
                    proposal_id=p.proposal_id, version=p.version, title=p.title,
                    content=p.content, author=p.author, timestamp=p.timestamp,
                    status=p.status, votes_for=p.votes_for, votes_against=p.votes_against,
                    comments=p.comments + [entry], hash_signature=p.hash_signature,
                )
                self._proposals = [updated if x.proposal_id == proposal_id else x for x in self._proposals]
                return entry
        raise ValueError(f"Proposal {proposal_id} not found")

    def vote(self, proposal_id: str, voter: str, approve: bool) -> dict:
        """社区投票。"""
        for p in self._proposals:
            if p.proposal_id == proposal_id:
                updated = ConstitutionalProposal(
                    proposal_id=p.proposal_id, version=p.version, title=p.title,
                    content=p.content, author=p.author, timestamp=p.timestamp,
                    status=p.status,
                    votes_for=p.votes_for + (1 if approve else 0),
                    votes_against=p.votes_against + (0 if approve else 1),
                    comments=p.comments, hash_signature=p.hash_signature,
                )
                self._proposals = [updated if x.proposal_id == proposal_id else x for x in self._proposals]
                return {"voter": voter, "approve": approve}
        raise ValueError(f"Proposal {proposal_id} not found")

    def get_proposal(self, proposal_id: str) -> Optional[ConstitutionalProposal]:
        for p in self._proposals:
            if p.proposal_id == proposal_id:
                return p
        return None

    def get_all_proposals(self, status: Optional[ProposalStatus] = None) -> list[ConstitutionalProposal]:
        if status:
            return [p for p in self._proposals if p.status == status]
        return list(self._proposals)

    def get_statistics(self) -> dict:
        status_counts = {}
        for p in self._proposals:
            status_counts[p.status.value] = status_counts.get(p.status.value, 0) + 1
        return {
            "total_proposals": len(self._proposals),
            "status_distribution": status_counts,
        }
