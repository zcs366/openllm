"""
ConsolidationScorer — 固化判据 MVP（宪章P1第一议程 · v0.3落地）
================================================================

回答的问题：什么上下文值得持久化（从W_ICL/MEMORY快层固化到SOUL/Iam慢层）。

骨架来源：ACE（ICLR 2026, arXiv 2510.04618）的bullet结构——
  id + helpful/harmful计数器 + 增量delta更新（禁全量重写）。
我们的补维：**provenance**（ACE缺、MemOS有字段无授权语义）——
  宪章P6：来源信任分级，不可信内容Score强制置零，不得触发固化。

宪章公式（v0.3·五、度量）：
  Score = 复现次数 + 2×任务增益% + 3×遗忘代价
  且 provenance.trust < τ ⇒ Score = 0（不可固化）

触发位（宪章v0.2·赫淮斯托斯裁）：
  context_pressure caution水位触发候选，嵌入compaction aggressive策略。

设计原则（与compaction_control/context_pressure同族）：
  - 纯规则引擎，零LLM调用
  - 确定性合并：LLM只可提案（add_bullet/update_counters的调用方），
    打分/门禁/合并逻辑全部在本模块内确定性执行（ACE确定性merge的移植）
  - append-only账本：固化历史只增不改（授权链H的投影）

上下文格言：
  门认来源，不认辞令。
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("openllm.consolidation_score")


# ── 信任分级（宪章P6条款1）──────────────────────────────
class TrustLevel(Enum):
    """provenance信任五级，从高到低。"""
    CREATOR = "creator"        # 造物主写入（Iam/SOUL级内容）
    USER = "user"              # 用户当次输入
    TOOL_OUTPUT = "tool_output"  # 工具执行结果（本机可控环境）
    WEB = "web"                # 外部网页/第三方内容
    MIXED = "mixed"            # 混合来源（无法拆分时按最低级处理）


# 信任级 → 数值权重（用于阈值比较）
_TRUST_ORDER = {
    TrustLevel.CREATOR: 5,
    TrustLevel.USER: 4,
    TrustLevel.TOOL_OUTPUT: 3,
    TrustLevel.WEB: 2,
    TrustLevel.MIXED: 1,
}

# 默认固化阈值：低于此信任级不可固化（P6条款2）
DEFAULT_TRUST_THRESHOLD = TrustLevel.TOOL_OUTPUT

# Score公式权重（宪章v0.3）
W_RECURRENCE = 1.0   # 复现次数
W_GAIN = 2.0         # 任务增益%
W_FORGET = 3.0       # 遗忘代价


@dataclass
class Provenance:
    """条目来源标签（宪章P6条款1 · ACE缺维的补位）。"""
    source: str = "unknown"          # 来源标识（session_id/URL/工具名）
    trust: TrustLevel = TrustLevel.MIXED
    ingested_at: float = field(default_factory=time.time)
    endorsed_by: str = ""            # 背书者（门禁P的签名位，可空=未背书）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "trust": self.trust.value,
            "ingested_at": self.ingested_at,
            "endorsed_by": self.endorsed_by,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Provenance":
        return cls(
            source=d.get("source", "unknown"),
            trust=TrustLevel(d.get("trust", "mixed")),
            ingested_at=d.get("ingested_at", time.time()),
            endorsed_by=d.get("endorsed_by", ""),
        )


@dataclass
class Bullet:
    """固化候选条目——ACE bullet骨架 + provenance补维。

    Attributes:
        bullet_id: 唯一标识（ACE: unique identifier）
        content: 知识单元（策略/领域概念/失败模式）
        helpful: 被标记有帮助的次数（ACE计数器）
        harmful: 被标记有害的次数（ACE计数器）
        recurrence: 复现次数（跨session重复出现的次数）
        task_gain_pct: 最近一次测得的任务增益百分比（0-100）
        forget_cost: 遗忘代价（0-10，丢掉这条的损失评估）
        provenance: 来源标签（P6）
        created_at / updated_at: 时间戳
        consolidated: 是否已固化（固化后置True，条目升级为持久层）
    """
    bullet_id: str
    content: str
    helpful: int = 0
    harmful: int = 0
    recurrence: int = 1
    task_gain_pct: float = 0.0
    forget_cost: float = 0.0
    provenance: Provenance = field(default_factory=Provenance)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    consolidated: bool = False

    def score(self, trust_threshold: TrustLevel = DEFAULT_TRUST_THRESHOLD) -> float:
        """宪章公式：Score = 复现次数 + 2×任务增益% + 3×遗忘代价。

        P6门禁：provenance.trust低于阈值 ⇒ Score强制置零。
        harmful净扣：ACE的harmful计数器在此生效——
          有效复现 = max(0, recurrence + helpful - harmful)，
          有害标记直接侵蚀固化资格（毒条目攒不出分）。
        """
        # ── P6门禁（先于一切计算）──
        if _TRUST_ORDER[self.provenance.trust] < _TRUST_ORDER[trust_threshold]:
            return 0.0
        effective_recurrence = max(0, self.recurrence + self.helpful - self.harmful)
        return (
            W_RECURRENCE * effective_recurrence
            + W_GAIN * self.task_gain_pct
            + W_FORGET * self.forget_cost
        )

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "bullet_id": self.bullet_id,
            "content": self.content,
            "helpful": self.helpful,
            "harmful": self.harmful,
            "recurrence": self.recurrence,
            "task_gain_pct": self.task_gain_pct,
            "forget_cost": self.forget_cost,
            "provenance": self.provenance.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "consolidated": self.consolidated,
        }
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Bullet":
        return cls(
            bullet_id=d["bullet_id"],
            content=d["content"],
            helpful=d.get("helpful", 0),
            harmful=d.get("harmful", 0),
            recurrence=d.get("recurrence", 1),
            task_gain_pct=d.get("task_gain_pct", 0.0),
            forget_cost=d.get("forget_cost", 0.0),
            provenance=Provenance.from_dict(d.get("provenance", {})),
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
            consolidated=d.get("consolidated", False),
        )


class ConsolidationScorer:
    """固化判据引擎——纯规则，零LLM。

    职责：
      1. add_bullet / update_counters —— 接收提案（调用方可是LLM，
         但合并逻辑在这里确定性执行：ACE"LLM只提案"纪律）
      2. candidates() —— 从context_pressure信号产生固化候选
         （caution水位触发，宪章v0.2度量表）
      3. consolidate() —— 过门禁的条目升级为固化，写入append-only账本
      4. ledger —— 固化历史只增不改（授权链H的投影）

    用法：
        scorer = ConsolidationScorer(state_path=Path("~/.openllm/consolidation.json"))
        scorer.add_bullet("b1", "WSL中I盘=/mnt/i/", provenance=Provenance(
            source="session-42", trust=TrustLevel.USER))
        scorer.mark("b1", helpful=True)
        cands = scorer.candidates(pressure_level="caution")
        report = scorer.consolidate(cands)
    """

    def __init__(
        self,
        state_path: Optional[Path] = None,
        trust_threshold: TrustLevel = DEFAULT_TRUST_THRESHOLD,
        min_score: float = 4.0,
    ):
        if state_path is None:
            state_path = Path.home() / ".openllm" / "consolidation_state.json"
        self._state_path = Path(state_path)
        self.trust_threshold = trust_threshold
        self.min_score = min_score  # 固化门槛：Score低于此值不固化
        self._bullets: Dict[str, Bullet] = {}
        self._ledger: List[Dict[str, Any]] = []  # append-only固化账本
        self._load()

    # ── 提案接口（调用方可是LLM，合并是确定性的）──────────

    def add_bullet(
        self,
        bullet_id: str,
        content: str,
        provenance: Optional[Provenance] = None,
        task_gain_pct: float = 0.0,
        forget_cost: float = 0.0,
    ) -> Bullet:
        """新增固化候选。已存在则视为一次复现（recurrence+1）。"""
        if bullet_id in self._bullets:
            b = self._bullets[bullet_id]
            b.recurrence += 1
            b.updated_at = time.time()
            # 增益/遗忘代价取最新观测
            if task_gain_pct:
                b.task_gain_pct = task_gain_pct
            if forget_cost:
                b.forget_cost = forget_cost
            self._save()
            return b
        b = Bullet(
            bullet_id=bullet_id,
            content=content,
            provenance=provenance or Provenance(),
            task_gain_pct=task_gain_pct,
            forget_cost=forget_cost,
        )
        self._bullets[bullet_id] = b
        self._save()
        return b

    def mark(self, bullet_id: str, helpful: bool) -> bool:
        """ACE计数器：Generator标记bullet有帮助/有害。"""
        b = self._bullets.get(bullet_id)
        if b is None:
            return False
        if helpful:
            b.helpful += 1
        else:
            b.harmful += 1
        b.updated_at = time.time()
        self._save()
        return True

    # ── 候选产生（context_pressure联动位）──────────────────

    def candidates(
        self,
        pressure_level: str = "normal",
        top_n: int = 10,
    ) -> List[Bullet]:
        """产生固化候选，按Score降序。

        pressure_level: context_pressure的分级（normal/caution/critical）。
        宪章v0.2：caution水位触发候选。normal时只返回已过min_score的
        （被动模式）；caution/critical时返回全部未固化且Score>0的
        （主动模式——压力越高越该把值钱的搬出去，compaction前先固化）。
        """
        scored = [
            (b.score(self.trust_threshold), b)
            for b in self._bullets.values()
            if not b.consolidated
        ]
        scored = [(s, b) for s, b in scored if s > 0]  # P6门禁+零分过滤
        scored.sort(key=lambda x: (-x[0], x[1].created_at))
        if pressure_level == "normal":
            scored = [(s, b) for s, b in scored if s >= self.min_score]
        return [b for _, b in scored[:top_n]]

    # ── 固化执行（过门禁 → 升级 + 记账）───────────────────

    def consolidate(
        self,
        bullets: Optional[List[Bullet]] = None,
        pressure_level: str = "caution",
    ) -> Dict[str, Any]:
        """固化一批条目。返回报告（固化了谁/拒了谁/为什么）。

        双重门禁：①P6信任门禁（score()内置）②min_score质量门禁。
        账本append-only：每次固化记录(bullet_id, score, trust, endorsed_by, ts)。
        """
        if bullets is None:
            bullets = self.candidates(pressure_level=pressure_level)
        report = {"consolidated": [], "rejected": []}
        for b in bullets:
            s = b.score(self.trust_threshold)
            if s <= 0:
                report["rejected"].append(
                    {"bullet_id": b.bullet_id, "reason": "p6_trust_gate",
                     "trust": b.provenance.trust.value}
                )
                continue
            if s < self.min_score:
                report["rejected"].append(
                    {"bullet_id": b.bullet_id, "reason": "min_score",
                     "score": round(s, 2)}
                )
                continue
            b.consolidated = True
            b.updated_at = time.time()
            self._ledger.append({
                "bullet_id": b.bullet_id,
                "score": round(s, 2),
                "trust": b.provenance.trust.value,
                "endorsed_by": b.provenance.endorsed_by,
                "ts": time.time(),
            })
            report["consolidated"].append(
                {"bullet_id": b.bullet_id, "score": round(s, 2)}
            )
        self._save()
        return report

    # ── 账本读取（宪章附录3：认领即审计）──────────────────

    def audit_ledger(self, recent_n: int = 20) -> Dict[str, Any]:
        """回读固化账本——睡眠窗第三件事（翻账本审计）的MVP位。

        检测项：①近期固化的信任分布（单向漂移探测：全是低信任=异常）
        ②固化速率（突增=可能在攒毒）
        """
        recent = self._ledger[-recent_n:]
        trust_counts: Dict[str, int] = {}
        for e in recent:
            trust_counts[e["trust"]] = trust_counts.get(e["trust"], 0) + 1
        low_trust = sum(
            v for k, v in trust_counts.items()
            if _TRUST_ORDER[TrustLevel(k)] <= _TRUST_ORDER[TrustLevel.WEB]
        )
        return {
            "total_consolidated": len(self._ledger),
            "recent_n": len(recent),
            "trust_distribution": trust_counts,
            "low_trust_ratio": (low_trust / len(recent)) if recent else 0.0,
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "bullets": len(self._bullets),
            "consolidated": sum(1 for b in self._bullets.values() if b.consolidated),
            "pending": sum(1 for b in self._bullets.values() if not b.consolidated),
            "ledger_entries": len(self._ledger),
        }

    # ── 持久化（DiskPersistence同款原子写）─────────────────

    def _save(self) -> None:
        data = {
            "bullets": {k: v.to_dict() for k, v in self._bullets.items()},
            "ledger": self._ledger,  # append-only：只整体重写文件，永不删条目
        }
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self._state_path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.write("\n")
            os.replace(tmp, str(self._state_path))
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _load(self) -> None:
        if not self._state_path.exists():
            return
        try:
            with open(self._state_path, encoding="utf-8") as f:
                data = json.load(f)
            self._bullets = {
                k: Bullet.from_dict(v) for k, v in data.get("bullets", {}).items()
            }
            self._ledger = data.get("ledger", [])
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("固化状态损坏，从空开始: %s", e)
            self._bullets = {}
            self._ledger = []
