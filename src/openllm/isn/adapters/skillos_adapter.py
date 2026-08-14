"""
SkillOS Adapter — RL-Based Skill Curation → UnifiedSkillConfig
==============================================================

SkillOS 是 Google 提出的 RL-based skill curation 框架，核心思想：
  - 技能（skill）在文本空间中竞争生存
  - 策展器（curator）用 RL 信号学习 keep/delete/merge 策略
  - 同一任务组（task_group）内的技能竞争有限的"存活槽位"
  - 压缩奖励（compression_reward）奖励用更少 token 达到同等质量

本模块将 SkillOS 的 RL curation 决策转译为 UnifiedSkillConfig.optimization_hints，
供 ISN Skill Lifecycle Manager 消费。

关键概念：
  - SkillUsageHistory: 技能使用历史记录
  - SkillScore: 多维度 RL 风格评分
  - CurationDecision: 策展决策（keep/merge/retire）
  - CurationPolicy: 基于 RL 信号的策展策略

用法：
    adapter = SkillOSAdapter()
    decisions = adapter.curate([skill1, skill2, skill3])
    for skill, decision in decisions:
        adapter.apply_decision(skill, decision)
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("openllm.isn.skillos_adapter")


# ═══════════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════════

class CurationAction(str, Enum):
    """策展行动。"""
    KEEP = "keep"
    MERGE = "merge"
    RETIRE = "retire"


# ═══════════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SkillUsageRecord:
    """单次技能使用记录。"""
    timestamp: str                      # ISO timestamp
    success: bool
    latency_ms: float = 0.0
    token_cost: float = 0.0
    quality_score: Optional[float] = None  # 执行后质量评估 [0, 1]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillUsageHistory:
    """技能使用历史 — SkillOS 的核心输入。

    RL curation 基于这些信号学习 keep/merge/retire 策略。
    """
    skill_name: str
    task_group_id: Optional[str] = None  # RL 分组训练 ID
    records: List[SkillUsageRecord] = field(default_factory=list)
    curator_score: Optional[float] = None   # 策展器质量分 [0, 1]
    compression_reward: Optional[float] = None  # 压缩奖励信号
    total_token_size: int = 0               # 技能文本总 token 数
    content_hash: str = ""                  # 技能内容指纹（用于去重）

    @property
    def total_calls(self) -> int:
        return len(self.records)

    @property
    def success_rate(self) -> float:
        if not self.records:
            return 0.0
        return sum(1 for r in self.records if r.success) / len(self.records)

    @property
    def avg_latency_ms(self) -> float:
        if not self.records:
            return 0.0
        return sum(r.latency_ms for r in self.records) / len(self.records)

    @property
    def avg_token_cost(self) -> float:
        if not self.records:
            return 0.0
        return sum(r.token_cost for r in self.records) / len(self.records)

    @property
    def avg_quality_score(self) -> float:
        scored = [r.quality_score for r in self.records if r.quality_score is not None]
        if not scored:
            return 0.0
        return sum(scored) / len(scored)

    @property
    def last_used_at(self) -> Optional[str]:
        if not self.records:
            return None
        return self.records[-1].timestamp


@dataclass
class SkillScore:
    """RL 风格多维度评分。

    每个维度归一化到 [0, 1]，通过加权组合得到 composite_score。
    """
    success_signal: float = 0.0       # 成功率信号
    efficiency_signal: float = 0.0    # 效率信号 (quality / token_cost)
    recency_signal: float = 0.0       # 时效信号（衰减）
    uniqueness_signal: float = 0.0    # 独特性信号
    compression_signal: float = 0.0   # 压缩奖励信号
    curator_signal: float = 0.0       # 策展器信号
    composite_score: float = 0.0      # 加权综合分
    confidence: float = 0.0           # 评分置信度（基于数据量）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": round(self.success_signal, 4),
            "efficiency": round(self.efficiency_signal, 4),
            "recency": round(self.recency_signal, 4),
            "uniqueness": round(self.uniqueness_signal, 4),
            "compression": round(self.compression_signal, 4),
            "curator": round(self.curator_signal, 4),
            "composite": round(self.composite_score, 4),
            "confidence": round(self.confidence, 4),
        }


@dataclass
class CurationDecision:
    """策展决策输出。"""
    skill_name: str
    action: CurationAction
    score: SkillScore
    merge_target: Optional[str] = None   # merge 时的目标技能名
    reason: str = ""                     # 人类可读的决策原因
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "skill_name": self.skill_name,
            "action": self.action.value,
            "score": self.score.to_dict(),
            "reason": self.reason,
        }
        if self.merge_target:
            d["merge_target"] = self.merge_target
        return d


# ═══════════════════════════════════════════════════════════════════════════════
# Curation Policy
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CurationPolicy:
    """RL-based 策展策略 — 控制 keep/merge/retire 阈值。

    阈值通过 RL 训练迭代更新。此处提供初始值和手动调优接口。
    """
    # 动作阈值（composite_score [0, 1]）
    keep_threshold: float = 0.6        # ≥ 此分 → keep
    merge_threshold: float = 0.3       # [merge_threshold, keep_threshold) → merge 候选
    retire_threshold: float = 0.1      # < 此分 → retire

    # 权重（RL policy network 的近似）
    weight_success: float = 0.25
    weight_efficiency: float = 0.20
    weight_recency: float = 0.15
    weight_uniqueness: float = 0.15
    weight_compression: float = 0.10
    weight_curator: float = 0.15

    # 衰减参数
    recency_half_life_days: float = 30.0  # 时效半衰期（天）

    # 数据充足性阈值
    min_calls_for_confidence: int = 3  # 至少 N 次调用才给高置信度

    # 任务组容量（每个任务组最多保留 N 个技能）
    max_skills_per_group: int = 10

    def validate(self) -> List[str]:
        """验证策略参数。"""
        errors = []
        if self.retire_threshold >= self.merge_threshold:
            errors.append("retire_threshold must be < merge_threshold")
        if self.merge_threshold >= self.keep_threshold:
            errors.append("merge_threshold must be < keep_threshold")
        total_w = (
            self.weight_success + self.weight_efficiency + self.weight_recency
            + self.weight_uniqueness + self.weight_compression + self.weight_curator
        )
        if abs(total_w - 1.0) > 0.01:
            errors.append(f"weights must sum to 1.0, got {total_w:.4f}")
        return errors


# ═══════════════════════════════════════════════════════════════════════════════
# SkillOSAdapter
# ═══════════════════════════════════════════════════════════════════════════════

class SkillOSAdapter:
    """SkillOS → UnifiedSkillConfig 适配器。

    职责：
      1. 接收技能使用历史（SkillUsageHistory 列表）
      2. 计算 RL 风格多维度评分（SkillScore）
      3. 生成策展决策（keep / merge / retire）
      4. 识别合并候选（内容相似 + 低独特性）
      5. 将决策转译为 UnifiedSkillConfig.optimization_hints

    用法：
        adapter = SkillOSAdapter(policy=CurationPolicy(...))
        histories = [history1, history2, ...]
        decisions = adapter.curate(histories)

        # 应用决策
        for decision in decisions:
            adapter.apply_decision(config, decision)
    """

    def __init__(
        self,
        policy: Optional[CurationPolicy] = None,
        reference_time: Optional[datetime] = None,
    ):
        """
        Args:
            policy: 策展策略参数（None = 使用默认值）
            reference_time: 参考时间点（用于 recency 计算，None = now）
        """
        self.policy = policy or CurationPolicy()
        self._reference_time = reference_time or datetime.now(timezone.utc)
        self._curation_history: List[CurationDecision] = []

        # 验证策略
        errors = self.policy.validate()
        if errors:
            raise ValueError(f"Invalid curation policy: {errors}")

    # ═══════════════════════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════════════════════

    def score_skill(self, history: SkillUsageHistory) -> SkillScore:
        """评估单个技能的 RL 风格多维度评分。

        Args:
            history: 技能使用历史

        Returns:
            SkillScore 包含各维度信号和综合分
        """
        # 1. 成功信号
        success_signal = self._compute_success_signal(history)

        # 2. 效率信号
        efficiency_signal = self._compute_efficiency_signal(history)

        # 3. 时效信号
        recency_signal = self._compute_recency_signal(history)

        # 4. 独特性信号（需要全局信息，此处给默认值）
        uniqueness_signal = 0.0  # 由 curate() 统一计算

        # 5. 压缩奖励信号
        compression_signal = self._compute_compression_signal(history)

        # 6. 策展器信号
        curator_signal = history.curator_score if history.curator_score is not None else 0.0

        # 7. 加权综合分
        composite = self._weighted_composite(
            success_signal, efficiency_signal, recency_signal,
            uniqueness_signal, compression_signal, curator_signal,
        )

        # 8. 置信度（基于数据量）
        confidence = self._compute_confidence(history)

        return SkillScore(
            success_signal=success_signal,
            efficiency_signal=efficiency_signal,
            recency_signal=recency_signal,
            uniqueness_signal=uniqueness_signal,
            compression_signal=compression_signal,
            curator_signal=curator_signal,
            composite_score=composite,
            confidence=confidence,
        )

    def recommend_action(self, history: SkillUsageHistory) -> CurationDecision:
        """为单个技能生成策展决策（不需要全局信息）。

        Args:
            history: 技能使用历史

        Returns:
            CurationDecision 包含行动和原因
        """
        score = self.score_skill(history)
        action, reason = self._decide_action(score, history.skill_name)
        return CurationDecision(
            skill_name=history.skill_name,
            action=action,
            score=score,
            reason=reason,
        )

    def curate(
        self,
        histories: List[SkillUsageHistory],
    ) -> List[CurationDecision]:
        """批量策展 — 考虑全局信息（任务组容量、合并候选、独特性）。

        这是核心入口，模拟 SkillOS 的 RL curation policy。

        Args:
            histories: 所有技能的使用历史列表

        Returns:
            策展决策列表（每个技能一个决策）
        """
        if not histories:
            return []

        # 1. 按任务组分组
        groups: Dict[Optional[str], List[SkillUsageHistory]] = {}
        for h in histories:
            gid = h.task_group_id
            groups.setdefault(gid, []).append(h)

        # 2. 每组内计算独特性 + 策展
        all_decisions: List[CurationDecision] = []
        for gid, group_histories in groups.items():
            # 计算组内独特性信号
            uniqueness_map = self._compute_group_uniqueness(group_histories)

            # 每个技能评分
            scored: List[Tuple[SkillUsageHistory, SkillScore]] = []
            for h in group_histories:
                s = self.score_skill(h)
                s.uniqueness_signal = uniqueness_map.get(h.skill_name, 0.5)
                # 重新计算 composite（包含 uniqueness）
                s.composite_score = self._weighted_composite(
                    s.success_signal, s.efficiency_signal, s.recency_signal,
                    s.uniqueness_signal, s.compression_signal, s.curator_signal,
                )
                scored.append((h, s))

            # 按 composite_score 降序排序
            scored.sort(key=lambda x: x[1].composite_score, reverse=True)

            # 容量限制：超过 max_skills_per_group 的低分技能标记为 merge/retire
            decisions = self._decide_group(scored, gid)
            all_decisions.extend(decisions)

        # 3. 全局合并候选检测（跨任务组）
        all_decisions = self._detect_merge_candidates(all_decisions, histories)

        self._curation_history.extend(all_decisions)
        return all_decisions

    def apply_decision(
        self,
        config: Any,  # UnifiedSkillConfig
        decision: CurationDecision,
    ) -> Any:
        """将策展决策应用到 UnifiedSkillConfig。

        更新 optimization_hints 和 lifecycle_state。

        Args:
            config: UnifiedSkillConfig 实例
            decision: 策展决策

        Returns:
            修改后的 config
        """
        from openllm.isn.unified_skill_config import (
            OptimizationHints,
            SkillLifecycleState,
        )

        # 确保 optimization_hints 存在
        if config.optimization_hints is None:
            config.optimization_hints = OptimizationHints()

        hints = config.optimization_hints
        hints.should_optimize = decision.action != CurationAction.KEEP
        hints.confidence = decision.score.confidence
        hints.last_analyzed_at = datetime.now(timezone.utc).isoformat()

        if decision.action == CurationAction.KEEP:
            hints.priority = 0
            hints.reason = f"Skill scored {decision.score.composite_score:.3f}, kept"
            hints.suggested_focus = "maintain"
            # 如果是 dormant 且分数高，恢复到 active
            if hasattr(config, 'lifecycle_state'):
                if config.lifecycle_state == SkillLifecycleState.DORMANT:
                    config.transition(SkillLifecycleState.ACTIVE)

        elif decision.action == CurationAction.MERGE:
            hints.priority = 2
            hints.reason = (
                f"Merge candidate: {decision.reason}. "
                f"Target: {decision.merge_target or 'TBD'}"
            )
            hints.suggested_focus = "merge"
            hints.metadata["merge_target"] = decision.merge_target or ""
            hints.metadata["curation_score"] = decision.score.composite_score
            if hasattr(config, 'lifecycle_state'):
                config.transition(SkillLifecycleState.DEPRECATED)

        elif decision.action == CurationAction.RETIRE:
            hints.priority = 3
            hints.reason = f"Retire: {decision.reason}"
            hints.suggested_focus = "retire"
            hints.metadata["retirement_reason"] = decision.reason
            hints.metadata["curation_score"] = decision.score.composite_score
            if hasattr(config, 'lifecycle_state'):
                if config.lifecycle_state == SkillLifecycleState.ACTIVE:
                    config.transition(SkillLifecycleState.DEPRECATED)
                elif config.lifecycle_state == SkillLifecycleState.DORMANT:
                    config.transition(SkillLifecycleState.DEPRECATED)

        # 写入完整评分到 metadata
        hints.metadata["skillos_score"] = decision.score.to_dict()
        hints.metadata["skillos_action"] = decision.action.value

        return config

    @property
    def curation_history(self) -> List[CurationDecision]:
        """历史策展决策。"""
        return list(self._curation_history)

    def reset(self):
        """重置 adapter 状态。"""
        self._curation_history.clear()

    # ═══════════════════════════════════════════════════════════════════════
    # Signal Computation (RL-style)
    # ═══════════════════════════════════════════════════════════════════════

    def _compute_success_signal(self, history: SkillUsageHistory) -> float:
        """成功信号：基于成功率。"""
        if history.total_calls == 0:
            return 0.0
        return min(1.0, max(0.0, history.success_rate))

    def _compute_efficiency_signal(self, history: SkillUsageHistory) -> float:
        """效率信号：quality / token_cost，归一化到 [0, 1]。

        高效率 = 高质量 + 低 token 消耗。
        """
        if history.total_calls == 0 or history.avg_token_cost <= 0:
            return 0.0

        avg_quality = history.avg_quality_score
        if avg_quality <= 0:
            # 没有 quality_score 时，用 success_rate 近似
            avg_quality = history.success_rate

        # 效率 = quality / (1 + token_cost_per_call)
        # 归一化：除以典型值 100 tokens/call 得到相对效率
        raw_efficiency = avg_quality / (1.0 + history.avg_token_cost / 100.0)
        return min(1.0, max(0.0, raw_efficiency))

    def _compute_recency_signal(self, history: SkillUsageHistory) -> float:
        """时效信号：指数衰减，基于最后使用时间。"""
        last_used = history.last_used_at
        if last_used is None:
            return 0.0

        try:
            last_dt = datetime.fromisoformat(last_used.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return 0.0

        delta_days = (self._reference_time - last_dt).total_seconds() / 86400.0
        if delta_days < 0:
            delta_days = 0.0

        half_life = self.policy.recency_half_life_days
        # 指数衰减：e^(-ln2 * delta / half_life) = 0.5^(delta/half_life)
        return math.pow(0.5, delta_days / half_life)

    def _compute_compression_signal(self, history: SkillUsageHistory) -> float:
        """压缩奖励信号：奖励小而精的技能。

        compression_reward 如果有值则直接使用，
        否则根据 total_token_size 近似。
        """
        if history.compression_reward is not None:
            return min(1.0, max(0.0, history.compression_reward))

        if history.total_token_size <= 0:
            return 0.0

        # Token 越少得分越高（假设 500 token 为"完美压缩"基准）
        # sigmoid-like: 1 / (1 + token_size / 500)
        return min(1.0, 1.0 / (1.0 + history.total_token_size / 500.0))

    def _compute_group_uniqueness(
        self, histories: List[SkillUsageHistory]
    ) -> Dict[str, float]:
        """计算组内技能独特性：基于 content_hash 重复度。

        独特性 = 1 - (同 hash 技能数 - 1) / 总技能数
        高独特性 = 该技能是组内唯一的
        """
        if len(histories) <= 1:
            return {h.skill_name: 1.0 for h in histories}

        # 统计每个 hash 出现的次数
        hash_counts: Dict[str, int] = {}
        for h in histories:
            hsh = h.content_hash or h.skill_name  # fallback to name
            hash_counts[hsh] = hash_counts.get(hsh, 0) + 1

        total = len(histories)
        result: Dict[str, float] = {}
        for h in histories:
            hsh = h.content_hash or h.skill_name
            dup_count = hash_counts[hsh]
            if dup_count <= 1:
                result[h.skill_name] = 1.0  # 唯一
            else:
                # 重复越多，独特性越低
                result[h.skill_name] = 1.0 - (dup_count - 1) / total

        return result

    def _compute_confidence(self, history: SkillUsageHistory) -> float:
        """评分置信度：基于数据量的 sigmoid 函数。"""
        if history.total_calls == 0:
            return 0.0

        min_calls = self.policy.min_calls_for_confidence
        if min_calls <= 0:
            return 1.0

        # sigmoid: 1 / (1 + e^(-(calls - min_calls) / 3))
        x = (history.total_calls - min_calls) / 3.0
        return 1.0 / (1.0 + math.exp(-x))

    def _weighted_composite(
        self,
        success: float,
        efficiency: float,
        recency: float,
        uniqueness: float,
        compression: float,
        curator: float,
    ) -> float:
        """加权综合评分。"""
        p = self.policy
        return (
            p.weight_success * success
            + p.weight_efficiency * efficiency
            + p.weight_recency * recency
            + p.weight_uniqueness * uniqueness
            + p.weight_compression * compression
            + p.weight_curator * curator
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Decision Logic
    # ═══════════════════════════════════════════════════════════════════════

    def _decide_action(
        self, score: SkillScore, skill_name: str
    ) -> Tuple[CurationAction, str]:
        """根据综合评分决定行动。"""
        s = score.composite_score
        p = self.policy

        if s >= p.keep_threshold:
            reason = (
                f"Score {s:.3f} >= keep_threshold {p.keep_threshold:.2f}. "
                f"Signals: success={score.success_signal:.2f}, "
                f"efficiency={score.efficiency_signal:.2f}, "
                f"recency={score.recency_signal:.2f}"
            )
            return CurationAction.KEEP, reason

        elif s >= p.retire_threshold:
            reason = (
                f"Score {s:.3f} in [retire, keep) range "
                f"[{p.retire_threshold:.2f}, {p.keep_threshold:.2f}). "
                f"Candidate for merge or optimization"
            )
            return CurationAction.MERGE, reason

        else:
            reason = (
                f"Score {s:.3f} < retire_threshold {p.retire_threshold:.2f}. "
                f"Low quality and/or unused — recommended for retirement"
            )
            return CurationAction.RETIRE, reason

    def _decide_group(
        self,
        scored: List[Tuple[SkillUsageHistory, SkillScore]],
        group_id: Optional[str],
    ) -> List[CurationDecision]:
        """按任务组策展：考虑容量限制。"""
        max_skills = self.policy.max_skills_per_group
        decisions: List[CurationDecision] = []

        for rank, (history, score) in enumerate(scored):
            # 超出容量的技能强制 retire（硬约束，不受分数影响）
            if rank >= max_skills:
                reason = (
                    f"Exceeded group capacity ({rank + 1} > {max_skills}). "
                    f"Score {score.composite_score:.3f} — forced retire"
                )
                decisions.append(CurationDecision(
                    skill_name=history.skill_name,
                    action=CurationAction.RETIRE,
                    score=score,
                    reason=reason,
                    metadata={"group_id": group_id, "rank": rank},
                ))
            else:
                action, reason = self._decide_action(score, history.skill_name)
                decisions.append(CurationDecision(
                    skill_name=history.skill_name,
                    action=action,
                    score=score,
                    reason=reason,
                    metadata={"group_id": group_id, "rank": rank},
                ))

        return decisions

    def _detect_merge_candidates(
        self,
        decisions: List[CurationDecision],
        histories: List[SkillUsageHistory],
    ) -> List[CurationDecision]:
        """跨技能合并候选检测：内容相似的技能配对。"""
        history_map = {h.skill_name: h for h in histories}

        # 收集所有 merge 候选
        merge_candidates = [
            d for d in decisions if d.action == CurationAction.MERGE
        ]

        if len(merge_candidates) < 2:
            return decisions

        # 计算候选之间的相似度（基于 content_hash）
        paired: Dict[str, str] = {}  # skill_name → merge_target
        for i, d1 in enumerate(merge_candidates):
            if d1.skill_name in paired:
                continue
            h1 = history_map.get(d1.skill_name)
            if not h1:
                continue

            best_match: Optional[str] = None
            best_score = 0.0

            for j, d2 in enumerate(merge_candidates):
                if i == j or d2.skill_name in paired:
                    continue
                h2 = history_map.get(d2.skill_name)
                if not h2:
                    continue

                # 相同 content_hash = 完全重复 → 强合并
                if (h1.content_hash and h2.content_hash
                        and h1.content_hash == h2.content_hash):
                    # 选质量更高的保留
                    if h1.avg_quality_score >= h2.avg_quality_score:
                        paired[d2.skill_name] = d1.skill_name
                        best_match = d2.skill_name
                        best_score = 1.0
                    else:
                        paired[d1.skill_name] = d2.skill_name
                        break
                else:
                    # 同任务组的 merge 候选自动配对（取最高分的作为目标）
                    if (h1.task_group_id == h2.task_group_id
                            and h1.task_group_id is not None):
                        if d1.score.composite_score >= d2.score.composite_score:
                            paired[d2.skill_name] = d1.skill_name
                        else:
                            paired[d1.skill_name] = d2.skill_name
                        break

        # 更新决策
        updated: List[CurationDecision] = []
        for d in decisions:
            if d.skill_name in paired:
                d.merge_target = paired[d.skill_name]
                d.reason += f" → merge into {paired[d.skill_name]}"
                d.metadata["merge_target"] = paired[d.skill_name]
            updated.append(d)

        return updated
