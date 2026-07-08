"""
ConsensusReasoning — IOS Layer 2 MACR（多Agent共识推理）
=======================================================
多Agent独立推理→共识度计算→重试决策。纯规则，零LLM。
持久化: ~/.hermes/governance/consensus_reasoning_state.json
"""
from __future__ import annotations

import json, logging, re, time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger("openllm.governance.consensus_reasoning")
_STATE_PATH = Path.home() / ".hermes" / "governance" / "consensus_reasoning_state.json"
DEFAULT_CONSENSUS_THRESHOLD = 0.6
_ROLE_PROFILES: dict[str, tuple[str, float]] = {
    "韩信": ("strategy", 0.85), "鲁班": ("engineering", 0.80),
    "子产": ("judgment", 0.75), "萧何": ("planning", 0.70),
    "子贡": ("execution", 0.65),
}
_STOPS = {"the", "a", "an", "is", "of", "to", "in", "on", "for"}


@dataclass(frozen=True)
class ReasoningRecord:
    """单次推理记录（不可变）。"""
    claim_id: str
    agent_id: str
    conclusion: str
    keywords: list[str]
    confidence: float
    reasoning_text: str
    timestamp: float


class ConsensusReasoning:
    """多Agent共识推理引擎。规则模板推理→关键词重叠共识→重试决策。"""

    def __init__(self, state_path: str | Path | None = None):
        self._path = Path(state_path) if state_path else _STATE_PATH
        self._history: list[dict] = self._load()

    def independent_reasoning(
        self, claim_id: str, agents: list[str]
    ) -> dict[str, ReasoningRecord]:
        """各Agent对claim独立推理（规则模板模拟）。"""
        records = {aid: self._simulate(claim_id, aid) for aid in agents}
        self._persist(records.values())
        return records

    def compute_consensus(
        self, reasonings: dict[str, ReasoningRecord]
    ) -> tuple[float, str]:
        """计算共识度（≥半数Agent的关键词重叠率）和摘要。"""
        if len(reasonings) < 2:
            return 0.0, "Agent数不足，无法计算共识"
        freq: Counter[str] = Counter()
        for rec in reasonings.values():
            freq.update(rec.keywords)
        half = len(reasonings) / 2.0
        consensus_kws = [kw for kw, c in freq.items() if c >= half]
        score = round(len(consensus_kws) / len(freq), 4) if freq else 0.0
        dominant = sorted(freq, key=freq.get, reverse=True)[:5]  # type: ignore
        conclusions = Counter(r.conclusion for r in reasonings.values())
        summary = (
            f"{len(reasonings)}个Agent，共识度={score:.2%}，"
            f"关键词：{'、'.join(dominant)}，结论：{conclusions.most_common(3)}"
        )
        return score, summary

    def should_retry(
        self, consensus_score: float, threshold: float = DEFAULT_CONSENSUS_THRESHOLD
    ) -> bool:
        """共识度低于阈值时返回True。"""
        return consensus_score < threshold

    def get_retry_strategy(self, reasonings: dict[str, ReasoningRecord]) -> str:
        """选策略：add_more_agents / focus_alignment / confrontational_analysis。"""
        if not reasonings:
            return "fresh_start"
        avg_conf = sum(r.confidence for r in reasonings.values()) / len(reasonings)
        if avg_conf < 0.5:
            return "add_more_agents"
        conclusions = [r.conclusion for r in reasonings.values()]
        if len(set(conclusions)) > len(conclusions) * 0.6:
            return "focus_alignment"
        return "confrontational_analysis"

    def get_history(self) -> list[dict]:
        """获取完整推理历史。"""
        return list(self._history)

    def _simulate(self, claim_id: str, agent_id: str) -> ReasoningRecord:
        """基于角色的规则模板推理（无LLM）。"""
        bias, base_conf = _ROLE_PROFILES.get(agent_id, ("general", 0.7))
        seed = hash((claim_id, agent_id)) % 100
        kws = [t.lower() for t in re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+", claim_id)
               if t.lower() not in _STOPS and len(t) > 1]
        kws = list(dict.fromkeys(kws + [bias, f"ev{seed % 3}", f"f{seed % 5}"]))[:8]
        conf = min(1.0, base_conf + (seed % 10) / 100.0)
        return ReasoningRecord(
            claim_id=claim_id, agent_id=agent_id,
            conclusion=f"{'支持' if seed % 3 != 0 else '反对'}_{claim_id}",
            keywords=kws, confidence=round(conf, 4),
            reasoning_text=f"{agent_id}({bias})对{claim_id}的推理",
            timestamp=time.time(),
        )

    def _persist(self, records) -> None:
        for rec in records:
            self._history.append(asdict(rec))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._history, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _load(self) -> list[dict]:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, KeyError):
                logger.warning("共识推理状态损坏: %s", self._path)
        return []
