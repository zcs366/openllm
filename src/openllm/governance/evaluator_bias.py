"""
EvaluatorBiasTracker — Layer 5 Contagion Networks
==================================================

评估者偏差检测模块：追踪每个评估者（agent）的判决与实际结果之间的
系统性偏差，分析偏差在网络中的传播（contagion）效应。

核心能力：
- 记录评估判决与最终结果
- 计算每个评估者的偏差分（-1.0 ~ 1.0）
- 检测偏差是否在网络中扩散（contagion analysis）
- 判断是否需要隔离某个评估者

纯规则+统计，零LLM调用。
数据持久化到 ~/.hermes/governance/evaluator_bias_state.json
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("openllm.governance.evaluator_bias")

# ── 持久化路径 ──
_STATE_DIR = Path.home() / ".hermes" / "governance"
_STATE_PATH = _STATE_DIR / "evaluator_bias_state.json"

# ── 偏差阈值 ──
DEFAULT_BIAS_THRESHOLD = 0.3
# contagion传播判定：被错误评估影响的下游决策占比
CONTAGION_RATIO_THRESHOLD = 0.5


@dataclass
class EvaluationRecord:
    """单次评估记录。"""
    agent_id: str
    claim_id: str
    verdict: float       # -1.0（强烈反对）~ 1.0（强烈支持）
    confidence: float    # 0.0 ~ 1.0
    timestamp: float
    outcome: Optional[float] = None   # 实际结果，None=尚未记录
    outcome_recorded_at: Optional[float] = None


@dataclass
class BiasState:
    """持久化状态。"""
    evaluations: list[dict] = field(default_factory=list)
    # claim_id -> list of agent_ids that evaluated it（传播图）
    contagion_graph: dict[str, list[str]] = field(default_factory=dict)
    # claim_id -> actual_outcome
    outcomes: dict[str, float] = field(default_factory=dict)


class EvaluatorBiasTracker:
    """评估者偏差检测器。

    追踪每个agent的评估判决与实际结果的偏差，
    检测偏差在网络中的传播效应。

    用法::

        tracker = EvaluatorBiasTracker()
        tracker.record_evaluation("韩信", "c001", 0.8, 0.9)
        tracker.record_outcome("c001", 0.2)
        print(tracker.compute_bias("韩信"))  # 偏差分
    """

    def __init__(self, state_path: Optional[Path] = None):
        self._state_path = state_path or _STATE_PATH
        self._state = self._load_state()
        self._records: list[EvaluationRecord] = [
            EvaluationRecord(**r) for r in self._state.evaluations
        ]

    # ── 公共 API ──

    def record_evaluation(
        self, agent_id: str, claim_id: str, verdict: float, confidence: float
    ) -> None:
        """记录一次评估。

        Args:
            agent_id: 评估者标识（如 "韩信"）
            claim_id: 被评估声明的ID
            verdict: 评估判决，-1.0~1.0
            confidence: 评估置信度，0.0~1.0
        """
        rec = EvaluationRecord(
            agent_id=agent_id,
            claim_id=claim_id,
            verdict=max(-1.0, min(1.0, verdict)),
            confidence=max(0.0, min(1.0, confidence)),
            timestamp=time.time(),
        )
        self._records.append(rec)
        self._sync_to_state()
        self._persist()

    def record_outcome(self, claim_id: str, actual_outcome: float) -> None:
        """记录实际结果，与之前的评估比对。"""
        actual_outcome = max(-1.0, min(1.0, actual_outcome))
        for rec in self._records:
            if rec.claim_id == claim_id and rec.outcome is None:
                rec.outcome = actual_outcome
                rec.outcome_recorded_at = time.time()
        self._sync_to_state()
        self._persist()

    def compute_bias(self, agent_id: str) -> float:
        """计算评估者的系统性偏差。

        偏差 = mean(verdict - actual_outcome)，范围 -1.0~1.0。
        正值=系统性高估，负值=系统性低估。
        无已结算评估时返回 0.0。
        """
        diffs = [
            rec.verdict - rec.outcome
            for rec in self._records
            if rec.agent_id == agent_id and rec.outcome is not None
        ]
        if not diffs:
            return 0.0
        return round(sum(diffs) / len(diffs), 4)

    def is_biased(self, agent_id: str, threshold: float = DEFAULT_BIAS_THRESHOLD) -> bool:
        """判断评估者偏差是否超过阈值。"""
        return abs(self.compute_bias(agent_id)) > threshold

    def get_contagion_analysis(self) -> dict[str, Any]:
        """分析偏差传播（contagion networks）。

        检测模式：当一个评估者的错误评估影响了下游决策，
        即另一个评估者在同一claim上做出了受前者影响的判决。

        返回::

            {
                "biased_agents": [...],
                "affected_claims": [...],
                "contagion_edges": [
                    {"from": "A", "to": "B", "claim": "c001",
                     "a_verdict": 0.8, "b_verdict": 0.7,
                     "actual": 0.1, "spread": true}
                ],
                "contagion_ratio": 0.0  # 被传播的claim占比
            }
        """
        biased_agents = [
            aid for aid in self._agent_ids() if self.is_biased(aid)
        ]

        # 找出所有有实际结果的claim
        settled_claims = {
            rec.claim_id
            for rec in self._records
            if rec.outcome is not None
        }

        # 对每个settled claim，检查多个evaluator之间的一致性
        affected_claims: list[str] = []
        contagion_edges: list[dict] = []

        for claim_id in settled_claims:
            claim_recs = [
                rec for rec in self._records
                if rec.claim_id == claim_id and rec.outcome is not None
            ]
            if len(claim_recs) < 2:
                continue

            actual: float = claim_recs[0].outcome  # type: ignore[assignment]

            # 如果claim上有多个evaluator，检查是否存在传播
            # 传播 = A先评估且有偏差，B后评估且偏差方向与A一致
            by_time = sorted(claim_recs, key=lambda r: r.timestamp)
            for i in range(len(by_time)):
                for j in range(i + 1, len(by_time)):
                    a, b = by_time[i], by_time[j]
                    a_wrong = (a.verdict - actual) * (b.verdict - actual) > 0
                    a_first = a.timestamp < b.timestamp
                    same_bias_dir = (a.verdict - actual) > 0 == (b.verdict - actual) > 0
                    if a_first and a_wrong and same_bias_dir and a.agent_id != b.agent_id:
                        spread = abs(b.verdict - actual) > abs(a.verdict - actual) * 0.5
                        contagion_edges.append({
                            "from": a.agent_id,
                            "to": b.agent_id,
                            "claim": claim_id,
                            "a_verdict": a.verdict,
                            "b_verdict": b.verdict,
                            "actual": actual,
                            "spread": spread,
                        })
            if any(e["claim"] == claim_id and e["spread"] for e in contagion_edges):
                affected_claims.append(claim_id)

        # 传播比率
        total = len(settled_claims)
        contagion_ratio = round(len(affected_claims) / total, 4) if total else 0.0

        return {
            "biased_agents": biased_agents,
            "affected_claims": affected_claims,
            "contagion_edges": contagion_edges,
            "contagion_ratio": contagion_ratio,
        }

    def get_agent_stats(self, agent_id: str) -> dict[str, Any]:
        """获取单个评估者的统计摘要。"""
        recs = [r for r in self._records if r.agent_id == agent_id]
        settled = [r for r in recs if r.outcome is not None]
        return {
            "agent_id": agent_id,
            "total_evaluations": len(recs),
            "settled_evaluations": len(settled),
            "bias": self.compute_bias(agent_id),
            "is_biased": self.is_biased(agent_id),
            "avg_confidence": round(
                sum(r.confidence for r in recs) / len(recs), 4
            ) if recs else 0.0,
        }

    # ── 内部方法 ──

    def _agent_ids(self) -> list[str]:
        return list({r.agent_id for r in self._records})

    def _sync_to_state(self) -> None:
        self._state.evaluations = [asdict(r) for r in self._records]
        # 构建contagion graph
        graph: dict[str, list[str]] = defaultdict(list)
        for rec in self._records:
            graph[rec.claim_id].append(rec.agent_id)
        self._state.contagion_graph = {k: list(set(v)) for k, v in graph.items()}
        for rec in self._records:
            if rec.outcome is not None:
                self._state.outcomes[rec.claim_id] = rec.outcome

    def _load_state(self) -> BiasState:
        if self._state_path.exists():
            try:
                raw = json.loads(self._state_path.read_text(encoding="utf-8"))
                return BiasState(
                    evaluations=raw.get("evaluations", []),
                    contagion_graph=raw.get("contagion_graph", {}),
                    outcomes=raw.get("outcomes", {}),
                )
            except (json.JSONDecodeError, KeyError):
                logger.warning("偏差状态文件损坏，使用空状态: %s", self._state_path)
        return BiasState()

    def _persist(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(
            json.dumps(asdict(self._state), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
