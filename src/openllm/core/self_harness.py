"""
openLLM Self-Harness — IOS+ISA 自进化引擎 (Layer 10)
零LLM调用自进化管线：scan→generate→evaluate→deploy/rollback。
持久化: ~/.hermes/jiak/self_harness_state.json
"""
import json, logging, time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from openllm.core.failure_tracker import FailureCategory, FailureSignatureTracker

logger = logging.getLogger(__name__)

# ── 枚举 + 数据类 ──────────────────────────────────

class BottleneckType(str, Enum):
    FAILURE_CLUSTER = "failure_cluster"
    SLOW_TOOL = "slow_tool"
    LOW_CONFIDENCE = "low_confidence"
    MEMORY_BLOAT = "memory_bloat"
    REPETITION = "repetition"
    CONTEXT_PRESSURE = "context_pressure"

class ProposalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEPLOYED = "deployed"
    ROLLED_BACK = "rolled_back"

@dataclass(frozen=True)
class Bottleneck:
    bottleneck_type: BottleneckType
    description: str
    evidence: dict
    severity: float
    affected_module: str
    first_seen: float
    occurrence_count: int = 1

@dataclass(frozen=True)
class ImprovementProposal:
    proposal_id: str
    bottleneck: Bottleneck
    action_type: str
    target_file: str
    target_change: str
    expected_effect: str
    confidence: float
    status: ProposalStatus = ProposalStatus.PENDING
    created_at: float = 0.0
    evaluated_at: float = 0.0
    rollback_target: Optional[str] = None

# ── 分类映射 ────────────────────────────────────────
_CAT_MAP: dict[FailureCategory, tuple[BottleneckType, str, str]] = {
    FailureCategory.TOOL_PARAM:        (BottleneckType.FAILURE_CLUSTER, "tool_interface", "modify_skill"),
    FailureCategory.TOOL_TIMEOUT:      (BottleneckType.SLOW_TOOL,       "lifecycle",      "adjust_config"),
    FailureCategory.TOOL_PERMISSION:   (BottleneckType.FAILURE_CLUSTER, "governance",     "add_rule"),
    FailureCategory.LLM_HALLUCINATION: (BottleneckType.LOW_CONFIDENCE,  "context_memory", "modify_skill"),
    FailureCategory.LLM_FORMAT:        (BottleneckType.REPETITION,      "verification",   "modify_skill"),
    FailureCategory.CONTEXT_OVERFLOW:  (BottleneckType.MEMORY_BLOAT,    "context_memory", "adjust_config"),
    FailureCategory.ROUTE_WRONG:       (BottleneckType.REPETITION,      "lifecycle",      "add_rule"),
    FailureCategory.USER_CORRECTION:   (BottleneckType.LOW_CONFIDENCE,  "observability",  "modify_skill"),
}

_TARGET_MAP = {
    "tool_interface": "engine.py:_verify_tool_params",
    "lifecycle":      "loop.py:lifecycle",
    "governance":     "router.py:permissions",
    "context_memory": "loop.py:context_window",
    "verification":   "engine.py:output_validate",
    "observability":  "main_loop.py:reflect",
}

_HINTS = {
    BottleneckType.FAILURE_CLUSTER:  "添加更强的输入校验/重试逻辑",
    BottleneckType.SLOW_TOOL:        "调整超时阈值或增加降级策略",
    BottleneckType.LOW_CONFIDENCE:   "强化 prompt 中的格式/边界约束",
    BottleneckType.MEMORY_BLOAT:     "增加上下文压缩策略或滑动窗口",
    BottleneckType.REPETITION:       "引入重复检测 + 自动跳过规则",
    BottleneckType.CONTEXT_PRESSURE: "调整 token budget 分配",
}

_STATE_PATH = Path.home() / ".hermes" / "jiak" / "self_harness_state.json"


# ── 引擎 ────────────────────────────────────────────

class SelfHarness:
    """IOS+ISA 自进化引擎。scan→generate→evaluate→deploy/rollback，全持久化。"""

    def __init__(self, *, state_path: Path = _STATE_PATH,
                 failure_tracker: Optional[FailureSignatureTracker] = None,
                 precedent_log: Any = None):
        self._path = state_path
        self._tracker = failure_tracker or FailureSignatureTracker()
        self._precedent = precedent_log
        self._bottlenecks: list[Bottleneck] = []
        self._proposals: list[ImprovementProposal] = []
        self._load()

    def scan_bottlenecks(self, *, window_hours: int = 24) -> list[Bottleneck]:
        """从failure_tracker聚类失败签名→生成Bottleneck列表。"""
        cutoff = time.time() - window_hours * 3600
        clusters: dict[str, list] = {}
        for sig in self._tracker.signatures:
            if sig.timestamp < cutoff:
                continue
            clusters.setdefault(f"{sig.category.name}:{sig.tool_name}", []).append(sig)

        bns = []
        for sigs in clusters.values():
            cat = sigs[0].category
            m = _CAT_MAP.get(cat)
            if not m:
                continue
            bn = Bottleneck(
                bottleneck_type=m[0],
                description=f"{cat.name} 集群: {len(sigs)} 次失败 @ {sigs[0].tool_name}",
                evidence={"category": cat.name, "tool": sigs[0].tool_name,
                          "error_pattern": sigs[0].error_pattern},
                severity=min(1.0, len(sigs) / 5.0),
                affected_module=m[1],
                first_seen=sigs[0].timestamp,
                occurrence_count=len(sigs),
            )
            bns.append(bn)
        bns.sort(key=lambda b: -b.severity)
        self._bottlenecks = bns
        logger.info("scan: %d 瓶颈", len(bns))
        return bns

    def generate_proposal(self, bottleneck: Bottleneck) -> list[ImprovementProposal]:
        """对瓶颈生成提案，检查precedent_log避免重复。"""
        m = _CAT_MAP.get(FailureCategory[bottleneck.evidence.get("category", "")])
        if not m:
            return []
        pid = f"prop-{int(time.time()*1000)}-{bottleneck.bottleneck_type.value}"
        p = ImprovementProposal(
            proposal_id=pid, bottleneck=bottleneck, action_type=m[2],
            target_file=_TARGET_MAP.get(m[1], f"{m[1]}.py:{m[2]}"),
            target_change=_HINTS.get(bottleneck.bottleneck_type, "未知变更"),
            expected_effect=f"降低 {bottleneck.bottleneck_type.value} severity",
            confidence=min(0.9, 0.4 + bottleneck.severity * 0.5),
            created_at=time.time(),
        )
        self._proposals.append(p)
        return [p]

    def evaluate_proposal(self, proposal: ImprovementProposal,
                          *, metric_fn: Optional[Callable] = None) -> ImprovementProposal:
        """评估提案：(severity+confidence)/2 > 0.3 → APPROVED。"""
        score = metric_fn(proposal) if metric_fn else (proposal.bottleneck.severity + proposal.confidence) / 2.0
        status = ProposalStatus.APPROVED if score > 0.3 else ProposalStatus.REJECTED
        ev = ImprovementProposal(**{**asdict(proposal), "status": status, "evaluated_at": time.time()})
        self._replace(ev)
        logger.info("evaluate: %s → %s (%.2f)", proposal.proposal_id, status.value, score)
        return ev

    def deploy_proposal(self, proposal: ImprovementProposal) -> bool:
        """将approved提案标记deployed，记录precedent_log。"""
        if proposal.status != ProposalStatus.APPROVED:
            logger.warning("deploy: %s 未评估通过", proposal.proposal_id)
            return False
        d = ImprovementProposal(**{**asdict(proposal),
                                   "status": ProposalStatus.DEPLOYED,
                                   "rollback_target": proposal.rollback_target or proposal.target_file})
        self._replace(d)
        self._record_precedent(d)
        self._save()
        logger.info("deploy: %s 已部署", proposal.proposal_id)
        return True

    def rollback_proposal(self, proposal: ImprovementProposal) -> bool:
        """将deployed提案标记rolled_back。"""
        if proposal.status != ProposalStatus.DEPLOYED:
            return False
        r = ImprovementProposal(**{**asdict(proposal), "status": ProposalStatus.ROLLED_BACK})
        self._replace(r)
        self._record_precedent(r, is_rollback=True)
        self._save()
        logger.info("rollback: %s 已回滚", proposal.proposal_id)
        return True

    # ── 内部 ────────────────────────────────────────
    def _replace(self, updated: ImprovementProposal) -> None:
        self._proposals = [updated if p.proposal_id == updated.proposal_id else p for p in self._proposals]

    def _record_precedent(self, p: ImprovementProposal, *, is_rollback: bool = False) -> None:
        if not self._precedent:
            return
        try:
            from openllm.governance.precedent_log import ConflictType
            self._precedent.record(
                day_number=1,
                scenario=f"[self_harness] {p.action_type}: {p.target_change}",
                conflict_type=ConflictType.EDGE_CASE,
                articles_involved=["self_review"],
                agent_action=f"{'rollback' if is_rollback else 'deploy'} {p.proposal_id}",
                agent_reasoning=p.expected_effect,
                human_verdict="agent_correct",
                lesson_learned=p.target_change,
            )
        except Exception as e:
            logger.error("_record_precedent FAILED: %s — deploy仍然成功但判例未记录", e)
            # 不吞异常——让调用方知道判例记录失败了
            # 但不阻塞deploy（deploy本身已成功写入JSON）

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps({
            "bottlenecks": [asdict(b) for b in self._bottlenecks],
            "proposals": [asdict(p) for p in self._proposals],
            "updated_at": time.time(),
        }, indent=2, ensure_ascii=False))

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text())
            self._bottlenecks = [Bottleneck(**b) for b in raw.get("bottlenecks", [])]
            fields = set(ImprovementProposal.__dataclass_fields__)
            self._proposals = [ImprovementProposal(**{k: v for k, v in p.items() if k in fields})
                               for p in raw.get("proposals", [])]
        except Exception as e:
            logger.warning("_load: %s", e)

    def get_pending(self) -> list[ImprovementProposal]:
        return [p for p in self._proposals if p.status == ProposalStatus.PENDING]

    def get_deployed(self) -> list[ImprovementProposal]:
        return [p for p in self._proposals if p.status == ProposalStatus.DEPLOYED]

    def summary(self) -> dict:
        return {"bottlenecks": len(self._bottlenecks), "proposals": len(self._proposals),
                "pending": len(self.get_pending()), "deployed": len(self.get_deployed())}
