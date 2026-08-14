"""
SkillOpt Adapter — 文本空间优化信号 → UnifiedSkillConfig
=========================================================

SkillOpt 核心思想：技能（skill）是一段可编辑的文本（prompt/instructions），
优化过程是在文本空间中做 add/delete/replace 操作，通过验证分数筛选。

本模块将 SkillOpt 的 scored rollouts 转译为 UnifiedSkillConfig.optimization_hints，
供 ISN Skill Lifecycle Manager 消费。

关键概念：
  - Rollout：一次完整的优化尝试（edit proposal + validation score）
  - Edit：对技能文本的 bounded 修改（add/delete/replace）
  - Learning-rate budget：每轮可接受的最大 token 变动量
  - Rejected-edit buffer：被拒绝的 edit 用于反面学习
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("openllm.isn.skillopt_adapter")


# ── 枚举与数据结构 ──

class EditOp(str, Enum):
    """编辑操作类型。"""
    ADD = "add"
    DELETE = "delete"
    REPLACE = "replace"


@dataclass
class TextEdit:
    """一条文本编辑提案。"""
    op: EditOp
    target_section: str          # 目标技能中的 section/段落名称
    content: str                 # 要添加/替换的内容
    old_content: str = ""        # replace 时的原始内容；delete 时为被删内容
    token_delta: int = 0         # 预估 token 变动量（正=增加，负=删除）

    def __post_init__(self):
        if self.op == EditOp.DELETE and not self.old_content:
            logger.warning("DELETE edit 未指定 old_content")
        if self.op == EditOp.REPLACE and not self.old_content:
            logger.warning("REPLACE edit 未指定 old_content")


@dataclass
class ScoredEdit:
    """带分数的编辑提案 — SkillOpt rollout 的核心单元。"""
    edit: TextEdit
    validation_score: float      # 0.0 ~ 1.0
    baseline_score: float        # 应用前的基线分
    rollout_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def score_delta(self) -> float:
        return self.validation_score - self.baseline_score

    @property
    def accepted(self) -> bool:
        """是否被接受（严格改进）。"""
        return self.validation_score > self.baseline_score


@dataclass
class OptimizationHint:
    """写入 UnifiedSkillConfig.optimization_hints 的单条提示。"""
    hint_type: str               # add / delete / replace / rejected_pattern / budget_usage
    section: str
    content: str
    score_delta: float = 0.0
    rejection_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """一次 optimize() 调用的完整结果。"""
    accepted_edits: List[ScoredEdit]
    rejected_edits: List[ScoredEdit]
    optimization_hints: List[OptimizationHint]
    budget_used: int             # 已用 token budget
    budget_limit: int            # token budget 上限
    iteration: int               # 第几轮优化

    def to_config_dict(self) -> Dict[str, Any]:
        """转换为 UnifiedSkillConfig.optimization_hints 字段格式。"""
        return {
            "accepted_edits": [
                {
                    "op": se.edit.op.value,
                    "section": se.edit.target_section,
                    "content": se.edit.content,
                    "token_delta": se.edit.token_delta,
                    "score_delta": round(se.score_delta, 4),
                }
                for se in self.accepted_edits
            ],
            "rejected_edits": [
                {
                    "op": se.edit.op.value,
                    "section": se.edit.target_section,
                    "reason": se.metadata.get("rejection_reason", "no_improvement"),
                    "score_delta": round(se.score_delta, 4),
                }
                for se in self.rejected_edits
            ],
            "hints": [
                {
                    "type": h.hint_type,
                    "section": h.section,
                    "content": h.content,
                    "score_delta": round(h.score_delta, 4) if h.score_delta else 0,
                }
                for h in self.optimization_hints
            ],
            "budget": {
                "used": self.budget_used,
                "limit": self.budget_limit,
                "utilization": round(self.budget_used / max(self.budget_limit, 1), 4),
            },
            "iteration": self.iteration,
        }


# ── UnifiedSkillConfig 占位 ──

@dataclass
class UnifiedSkillConfig:
    """ISN Skill Lifecycle Manager 的统一技能配置。

    这是 SkillOpt adapter 的输出目标。optimization_hints 字段
    由 adapter 的 to_config_dict() 生成并填充。
    """
    skill_id: str
    name: str
    description: str = ""
    version: str = "0.1.0"
    status: str = "active"       # active / dormant / deprecated / retired
    tags: List[str] = field(default_factory=list)
    content: str = ""            # 技能的文本内容（prompt/instructions）
    sections: Dict[str, str] = field(default_factory=dict)  # 分段内容
    optimization_hints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── 核心 Adapter ──

class SkillOptAdapter:
    """
    SkillOpt → UnifiedSkillConfig 适配器。

    职责：
    1. 接收 scored rollouts（排序后的编辑提案列表）
    2. 生成 bounded add/delete/replace edits
    3. 严格接受改进（validation_score > baseline_score）
    4. 管理 textual learning-rate budget（每轮最大 token 变动）
    5. 维护 rejected-edit buffer 用于反面学习

    用法：
        adapter = SkillOptAdapter(token_budget=500)
        result = adapter.optimize(scored_rollouts, iteration=1)
        config.optimization_hints = result.to_config_dict()
    """

    def __init__(
        self,
        token_budget: int = 500,
        max_edits_per_iteration: int = 5,
        rejected_buffer_size: int = 20,
        min_score_delta: float = 0.0,
        priority_weight: float = 0.6,
    ):
        """
        Args:
            token_budget: 每轮可接受的最大 token 变动量（learning-rate budget）
            max_edits_per_iteration: 每轮最多接受的编辑数
            rejected_buffer_size: 被拒绝 edit buffer 的最大容量
            min_score_delta: 接受 edit 的最小分数改进阈值（>0 才接受）
            priority_weight: score_delta 优先级权重（0~1），
                             用于在 budget 内选择最优 edit 组合
        """
        self.token_budget = token_budget
        self.max_edits_per_iteration = max_edits_per_iteration
        self.rejected_buffer_size = rejected_buffer_size
        self.min_score_delta = max(min_score_delta, 0.0)
        self.priority_weight = max(0.0, min(1.0, priority_weight))

        # 内部状态
        self._rejected_buffer: List[ScoredEdit] = []
        self._acceptance_history: List[Dict[str, Any]] = []
        self._iteration = 0
        self._total_tokens_used = 0

    @property
    def rejected_buffer(self) -> List[ScoredEdit]:
        """只读访问被拒绝编辑 buffer。"""
        return list(self._rejected_buffer)

    @property
    def iteration(self) -> int:
        return self._iteration

    @property
    def total_tokens_used(self) -> int:
        return self._total_tokens_used

    def optimize(
        self,
        scored_rollouts: List[ScoredEdit],
        iteration: int = 0,
    ) -> OptimizationResult:
        """
        核心优化流程。

        1. 过滤：只保留严格改进的 edits
        2. 排序：按 score_delta 降序
        3. 预算约束：在 token_budget 内贪心选择最优组合
        4. 约束：不超过 max_edits_per_iteration
        5. 记录被拒绝的 edits 到 buffer

        Args:
            scored_rollouts: 带分数的编辑提案列表
            iteration: 当前迭代轮次

        Returns:
            OptimizationResult
        """
        self._iteration = iteration
        budget_remaining = self.token_budget
        accepted: List[ScoredEdit] = []
        rejected: List[ScoredEdit] = []

        # 1. 分离：严格改进 vs 被拒绝
        candidates = []  # 严格改进的
        for sr in scored_rollouts:
            if sr.validation_score > sr.baseline_score and sr.score_delta >= self.min_score_delta:
                candidates.append(sr)
            else:
                rejected.append(sr)

        # 2. 按 score_delta 降序排序
        candidates.sort(key=lambda x: x.score_delta, reverse=True)

        # 3. 贪心选择：在 budget 内选最优组合
        for candidate in candidates:
            if len(accepted) >= self.max_edits_per_iteration:
                rejected.append(candidate)
                candidate.metadata["rejection_reason"] = "max_edits_exceeded"
                continue

            edit_token_delta = abs(candidate.edit.token_delta)

            if edit_token_delta <= budget_remaining:
                accepted.append(candidate)
                budget_remaining -= edit_token_delta
            else:
                rejected.append(candidate)
                candidate.metadata["rejection_reason"] = "token_budget_exceeded"

        # 4. 更新内部状态
        budget_used = self.token_budget - budget_remaining
        self._total_tokens_used += budget_used

        # 5. 维护 rejected buffer（FIFO + 容量限制）
        for r in rejected:
            self._add_to_rejected_buffer(r)

        # 6. 生成 optimization hints
        hints = self._generate_hints(accepted, rejected, budget_used)

        logger.info(
            f"SkillOpt iteration={iteration}: "
            f"accepted={len(accepted)}, rejected={len(rejected)}, "
            f"budget_used={budget_used}/{self.token_budget}"
        )

        return OptimizationResult(
            accepted_edits=accepted,
            rejected_edits=rejected,
            optimization_hints=hints,
            budget_used=budget_used,
            budget_limit=self.token_budget,
            iteration=iteration,
        )

    def propose_edits(
        self,
        skill_config: UnifiedSkillConfig,
        rollout_score: float = 0.0,
    ) -> List[ScoredEdit]:
        """
        基于当前技能配置生成候选 edits。

        这是 SkillOpt 的 edit proposal 阶段。在实际使用中，
        这个方法可以被替换为更复杂的 proposal 生成器。

        Args:
            skill_config: 当前技能配置
            rollout_score: 基线分数

        Returns:
            候选 ScoredEdit 列表
        """
        candidates = []

        # 策略1：对每个 section 尝试 add
        for section_name, section_content in skill_config.sections.items():
            # 检查是否有重复行（可以优化）
            lines = section_content.split("\n")
            if len(lines) < 3:
                candidates.append(ScoredEdit(
                    edit=TextEdit(
                        op=EditOp.ADD,
                        target_section=section_name,
                        content="# 可优化：该段落过短",
                        token_delta=8,
                    ),
                    validation_score=0.0,
                    baseline_score=rollout_score,
                    rollout_id=f"propose-{section_name}-short",
                ))

            # 检查冗余行
            seen_lines = set()
            duplicates = []
            for line in lines:
                stripped = line.strip()
                if stripped and stripped in seen_lines:
                    duplicates.append(stripped)
                seen_lines.add(stripped)

            if duplicates:
                candidates.append(ScoredEdit(
                    edit=TextEdit(
                        op=EditOp.DELETE,
                        target_section=section_name,
                        content="",  # DELETE: new content is empty
                        old_content="\n".join(duplicates),
                        token_delta=-len(" ".join(duplicates).split()),
                    ),
                    validation_score=0.0,
                    baseline_score=rollout_score,
                    rollout_id=f"propose-{section_name}-dedup",
                ))

        return candidates

    def get_rejected_patterns(self) -> List[Dict[str, Any]]:
        """
        从 rejected buffer 中提取失败模式。

        用于反面学习：哪些编辑策略被反复拒绝？

        Returns:
            失败模式列表
        """
        if not self._rejected_buffer:
            return []

        # 按 op 类型和 section 聚合
        pattern_counts: Dict[str, List[ScoredEdit]] = {}
        for sr in self._rejected_buffer:
            key = f"{sr.edit.op.value}:{sr.edit.target_section}"
            pattern_counts.setdefault(key, []).append(sr)

        patterns = []
        for key, edits in pattern_counts.items():
            op_type, section = key.split(":", 1)
            avg_delta = sum(e.score_delta for e in edits) / len(edits)
            reasons = [e.metadata.get("rejection_reason", "unknown") for e in edits]
            primary_reason = max(set(reasons), key=reasons.count) if reasons else "unknown"

            patterns.append({
                "op": op_type,
                "section": section,
                "count": len(edits),
                "avg_score_delta": round(avg_delta, 4),
                "primary_rejection_reason": primary_reason,
                "suggestion": self._suggest_from_pattern(op_type, avg_delta, primary_reason),
            })

        patterns.sort(key=lambda p: p["count"], reverse=True)
        return patterns

    def to_unified_config(
        self,
        skill_config: UnifiedSkillConfig,
        result: OptimizationResult,
    ) -> UnifiedSkillConfig:
        """
        将优化结果写入 UnifiedSkillConfig.optimization_hints。

        Args:
            skill_config: 目标技能配置（会被原地修改）
            result: optimize() 的返回值

        Returns:
            修改后的 skill_config
        """
        skill_config.optimization_hints = result.to_config_dict()

        # 附加 rejected patterns 作为元信息
        patterns = self.get_rejected_patterns()
        if patterns:
            skill_config.optimization_hints["rejected_patterns"] = patterns

        # 附加 budget 历史
        skill_config.optimization_hints["total_tokens_used"] = self._total_tokens_used
        skill_config.optimization_hints["total_iterations"] = self._iteration

        return skill_config

    def reset(self):
        """重置 adapter 状态。"""
        self._rejected_buffer.clear()
        self._acceptance_history.clear()
        self._iteration = 0
        self._total_tokens_used = 0

    # ── 内部方法 ──

    def _add_to_rejected_buffer(self, edit: ScoredEdit):
        """FIFO 模式维护 rejected buffer。"""
        self._rejected_buffer.append(edit)
        if len(self._rejected_buffer) > self.rejected_buffer_size:
            self._rejected_buffer.pop(0)  # 弹出最旧的

    def _generate_hints(
        self,
        accepted: List[ScoredEdit],
        rejected: List[ScoredEdit],
        budget_used: int,
    ) -> List[OptimizationHint]:
        """从 accepted/rejected edits 生成 hints。"""
        hints = []

        # 已接受的 edits → hints
        for sr in accepted:
            hints.append(OptimizationHint(
                hint_type=sr.edit.op.value,
                section=sr.edit.target_section,
                content=sr.edit.content or sr.edit.old_content,
                score_delta=sr.score_delta,
                metadata={"rollout_id": sr.rollout_id, "status": "accepted"},
            ))

        # 被拒绝的 edits → rejected_pattern hints
        for sr in rejected:
            hints.append(OptimizationHint(
                hint_type=f"rejected_{sr.edit.op.value}",
                section=sr.edit.target_section,
                content=sr.metadata.get("rejection_reason", ""),
                score_delta=sr.score_delta,
                rejection_reason=sr.metadata.get("rejection_reason", ""),
                metadata={"rollout_id": sr.rollout_id, "status": "rejected"},
            ))

        # Budget 使用 hint
        hints.append(OptimizationHint(
            hint_type="budget_usage",
            section="__budget__",
            content=f"{budget_used}/{self.token_budget} tokens used",
            metadata={
                "used": budget_used,
                "limit": self.token_budget,
                "utilization": round(budget_used / max(self.token_budget, 1), 4),
            },
        ))

        return hints

    def _suggest_from_pattern(
        self, op_type: str, avg_delta: float, reason: str
    ) -> str:
        """从失败模式生成建议。"""
        if op_type == "add" and avg_delta < 0:
            return "反复尝试添加但降低分数，建议停止该 section 的添加操作"
        if op_type == "delete" and avg_delta < 0:
            return "删除操作降低分数，该内容可能有保护作用"
        if reason == "token_budget_exceeded":
            return "编辑过大，建议拆分为更小的增量修改"
        if reason == "max_edits_per_iteration":
            return "候选 edit 过多，建议增加 max_edits_per_iteration 或更严格的预过滤"
        return "无特殊建议"
