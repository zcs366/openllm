"""
ISN UnifiedSkillConfig — 五源融合共享数据模型
=============================================

所有 5 个技能框架（SkillOS / OpenSkill / SkCC / SkillOpt / meta-skills）
通过此统一配置模型进行互操作。

设计原则（来自 PAL v1 2026-07-24）：
  1. SkillOpt 是心跳 — optimization_hints 必须是运行时热数据
  2. 退出工程化是盲区 — lifecycle_state 覆盖 deprecated→retired 全路径
  3. SKIR 必须是认知基质不是序列化格式 — 这是 dataclass 而非 JSON schema
  4. 所有框架字段均可选 — 只有 core 字段必填，框架特有字段按需填充

用法：
    from openllm.isn.unified_skill_config import UnifiedSkillConfig, SkillLifecycleState

    config = UnifiedSkillConfig(
        name="search-fallback",
        description="搜索降级级联",
        version="1.0.0",
        lifecycle_state=SkillLifecycleState.ACTIVE,
    )

    # 从特定框架迁移
    config = UnifiedSkillConfig.from_skillos(repo_data)
    config = UnifiedSkillConfig.from_openskill(community_data)
    config = UnifiedSkillConfig.from_skcc(contract_data)
    config = UnifiedSkillConfig.from_skillopt(metrics_data)
    config = UnifiedSkillConfig.from_metaskills(meta_data)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════════

class SkillLifecycleState(Enum):
    """技能生命周期状态 — 统一五源状态机。

    状态转移：
        active → dormant     （低频使用/暂存）
        active → deprecated  （被替代/标记退役）
        dormant → active     （重新激活）
        dormant → deprecated （长期休眠后退役）
        deprecated → retired  （彻底删除/归档）
    """
    ACTIVE = "active"
    DORMANT = "dormant"
    DEPRECATED = "deprecated"
    RETIRED = "retired"


class SkillFrameworkSource(Enum):
    """技能来源框架标识。"""
    SKILLOS = "skillos"
    OPENSKILL = "openskill"
    SKCC = "skcc"
    SKILLOPT = "skillopt"
    META_SKILLS = "meta_skills"
    UNIFIED = "unified"  # 原生创建，非迁移


# ═══════════════════════════════════════════════════════════════════════════════
# Framework-specific sub-models
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OptimizationHints:
    """SkillOpt 优化信号 — 心跳数据。

    SkillOpt 是心跳：这些信号必须是运行时热数据，而非静态配置。
    """
    should_optimize: bool = False
    priority: int = 0             # 0=无, 1=低, 2=中, 3=高, 4=P0
    reason: str = ""              # 人类可读的优化原因
    suggested_focus: str = ""     # 建议优化方向（如 "prompt_quality", "token_cost"）
    last_analyzed_at: Optional[str] = None  # ISO timestamp
    confidence: float = 0.0       # 优化建议置信度 [0.0, 1.0]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceTraceability:
    """OpenSkill 来源溯源 — 社区共享的 provenance 信息。"""
    upstream_repo: str = ""       # 上游仓库 URL
    upstream_version: str = ""    # 上游版本
    forked_at: Optional[str] = None  # ISO timestamp
    license: str = ""             # 如 "MIT", "Apache-2.0"
    original_author: str = ""     # 原始作者
    attribution: str = ""         # 归属声明
    modifications: list[str] = field(default_factory=list)  # 本地改动列表


@dataclass
class SkillUsageMetrics:
    """SkillOpt 运行时使用指标。"""
    total_calls: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_latency_ms: float = 0.0
    avg_token_cost: float = 0.0
    last_used_at: Optional[str] = None  # ISO timestamp
    success_rate: float = 0.0           # computed convenience field

    def __post_init__(self) -> None:
        if self.total_calls > 0 and self.success_rate == 0.0:
            self.success_rate = self.success_count / self.total_calls


@dataclass
class SkillCompositionContract:
    """SkCC 技能组合合约 — 输入/输出接口声明。"""
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    requires_approval: bool = False   # 是否需要执行前确认
    timeout_seconds: Optional[int] = None
    fallback_config: Optional[str] = None  # 降级策略 skill name


@dataclass
class MetaSkillConfig:
    """Meta-skill 配置 — 编排其他技能的高阶技能。"""
    orchestration_pattern: str = ""  # 如 "sequential", "parallel", "pipeline"
    managed_skills: list[str] = field(default_factory=list)
    composition_rules: list[str] = field(default_factory=list)
    max_concurrency: int = 1
    abort_on_failure: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# UnifiedSkillConfig
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class UnifiedSkillConfig:
    """ISN 五源融合统一技能配置。

    所有 5 个技能框架共享此数据模型。
    Core 字段（name, description, version, lifecycle_state）必填。
    Framework-specific 字段按需填充（默认值合理，零配置可用）。
    """

    # ── Core fields (all frameworks) ──────────────────────────────────────
    name: str = ""
    description: str = ""
    version: str = "0.0.1"
    author: str = ""
    lifecycle_state: SkillLifecycleState = SkillLifecycleState.ACTIVE
    source_framework: SkillFrameworkSource = SkillFrameworkSource.UNIFIED
    domain_tags: list[str] = field(default_factory=list)

    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # ── SkillOS-specific ──────────────────────────────────────────────────
    task_group_id: Optional[str] = None     # RL 分组训练 ID
    curator_score: Optional[float] = None   # 策展器质量分
    compression_reward: Optional[float] = None  # 压缩奖励信号

    # ── OpenSkill-specific ────────────────────────────────────────────────
    source_traceability: Optional[SourceTraceability] = None

    # ── SkCC-specific ─────────────────────────────────────────────────────
    composition_contract: Optional[SkillCompositionContract] = None
    dependencies: list[str] = field(default_factory=list)
    composes_with: list[str] = field(default_factory=list)

    # ── SkillOpt-specific ─────────────────────────────────────────────────
    optimization_hints: Optional[OptimizationHints] = None
    usage_metrics: Optional[SkillUsageMetrics] = None

    # ── Meta-skill-specific ───────────────────────────────────────────────
    meta_config: Optional[MetaSkillConfig] = None

    # ── Risk level (ISN v2 classifier) ───────────────────────────────────
    risk_level: str = "medium"  # critical / high / medium / low

    # ═══════════════════════════════════════════════════════════════════════
    # Factory methods — from specific frameworks
    # ═══════════════════════════════════════════════════════════════════════

    @classmethod
    def from_skillos(cls, data: dict[str, Any]) -> UnifiedSkillConfig:
        """从 SkillOS SKILL.md / repo 格式迁移。

        SkillOS 关键字段：task_group_id, curator_score, compression_reward。
        """
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            version=data.get("version", "0.0.1"),
            author=data.get("author", ""),
            lifecycle_state=SkillLifecycleState(
                data.get("lifecycle_state", "active")
            ),
            source_framework=SkillFrameworkSource.SKILLOS,
            domain_tags=data.get("domain_tags", []),
            task_group_id=data.get("task_group_id"),
            curator_score=data.get("curator_score"),
            compression_reward=data.get("compression_reward"),
        )

    @classmethod
    def from_openskill(cls, data: dict[str, Any]) -> UnifiedSkillConfig:
        """从 OpenSkill 社区共享格式迁移。

        OpenSkill 关键字段：source_traceability (provenance)。
        """
        trace_data = data.get("source_traceability")
        trace = None
        if trace_data:
            trace = SourceTraceability(
                upstream_repo=trace_data.get("upstream_repo", ""),
                upstream_version=trace_data.get("upstream_version", ""),
                forked_at=trace_data.get("forked_at"),
                license=trace_data.get("license", ""),
                original_author=trace_data.get("original_author", ""),
                attribution=trace_data.get("attribution", ""),
                modifications=trace_data.get("modifications", []),
            )
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            version=data.get("version", "0.0.1"),
            author=data.get("author", ""),
            lifecycle_state=SkillLifecycleState(
                data.get("lifecycle_state", "active")
            ),
            source_framework=SkillFrameworkSource.OPENSKILL,
            domain_tags=data.get("domain_tags", []),
            source_traceability=trace,
        )

    @classmethod
    def from_skcc(cls, data: dict[str, Any]) -> UnifiedSkillConfig:
        """从 SkCC (Skill Composition Contracts) 格式迁移。

        SkCC 关键字段：input_schema, output_schema, dependencies。
        """
        contract_data = data.get("composition_contract")
        contract = None
        if contract_data:
            contract = SkillCompositionContract(
                input_schema=contract_data.get("input_schema", {}),
                output_schema=contract_data.get("output_schema", {}),
                requires_approval=contract_data.get("requires_approval", False),
                timeout_seconds=contract_data.get("timeout_seconds"),
                fallback_config=contract_data.get("fallback_config"),
            )
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            version=data.get("version", "0.0.1"),
            author=data.get("author", ""),
            lifecycle_state=SkillLifecycleState(
                data.get("lifecycle_state", "active")
            ),
            source_framework=SkillFrameworkSource.SKCC,
            domain_tags=data.get("domain_tags", []),
            composition_contract=contract,
            dependencies=data.get("dependencies", []),
            composes_with=data.get("composes_with", []),
        )

    @classmethod
    def from_skillopt(cls, data: dict[str, Any]) -> UnifiedSkillConfig:
        """从 SkillOpt 优化框架格式迁移。

        SkillOpt 关键字段：optimization_hints, usage_metrics。
        这些是心跳数据——SkillOpt 是心跳。
        """
        hints_data = data.get("optimization_hints")
        hints = None
        if hints_data:
            hints = OptimizationHints(
                should_optimize=hints_data.get("should_optimize", False),
                priority=hints_data.get("priority", 0),
                reason=hints_data.get("reason", ""),
                suggested_focus=hints_data.get("suggested_focus", ""),
                last_analyzed_at=hints_data.get("last_analyzed_at"),
                confidence=hints_data.get("confidence", 0.0),
                metadata=hints_data.get("metadata", {}),
            )

        metrics_data = data.get("usage_metrics")
        metrics = None
        if metrics_data:
            total = metrics_data.get("total_calls", 0)
            success = metrics_data.get("success_count", 0)
            metrics = SkillUsageMetrics(
                total_calls=total,
                success_count=success,
                failure_count=metrics_data.get("failure_count", 0),
                avg_latency_ms=metrics_data.get("avg_latency_ms", 0.0),
                avg_token_cost=metrics_data.get("avg_token_cost", 0.0),
                last_used_at=metrics_data.get("last_used_at"),
                success_rate=(success / total) if total > 0 else 0.0,
            )

        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            version=data.get("version", "0.0.1"),
            author=data.get("author", ""),
            lifecycle_state=SkillLifecycleState(
                data.get("lifecycle_state", "active")
            ),
            source_framework=SkillFrameworkSource.SKILLOPT,
            domain_tags=data.get("domain_tags", []),
            optimization_hints=hints,
            usage_metrics=metrics,
        )

    @classmethod
    def from_metaskills(cls, data: dict[str, Any]) -> UnifiedSkillConfig:
        """从 meta-skill 编排格式迁移。

        Meta-skill 关键字段：orchestration_pattern, managed_skills。
        """
        meta_data = data.get("meta_config")
        meta = None
        if meta_data:
            meta = MetaSkillConfig(
                orchestration_pattern=meta_data.get("orchestration_pattern", ""),
                managed_skills=meta_data.get("managed_skills", []),
                composition_rules=meta_data.get("composition_rules", []),
                max_concurrency=meta_data.get("max_concurrency", 1),
                abort_on_failure=meta_data.get("abort_on_failure", True),
            )
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            version=data.get("version", "0.0.1"),
            author=data.get("author", ""),
            lifecycle_state=SkillLifecycleState(
                data.get("lifecycle_state", "active")
            ),
            source_framework=SkillFrameworkSource.META_SKILLS,
            domain_tags=data.get("domain_tags", []),
            meta_config=meta,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Lifecycle management
    # ═══════════════════════════════════════════════════════════════════════

    # Valid state transitions (PAL: 退出工程化盲区 → 明确状态机)
    _VALID_TRANSITIONS: ClassVar[dict[SkillLifecycleState, list[SkillLifecycleState]]] = {
        SkillLifecycleState.ACTIVE:    [SkillLifecycleState.DORMANT, SkillLifecycleState.DEPRECATED],
        SkillLifecycleState.DORMANT:   [SkillLifecycleState.ACTIVE, SkillLifecycleState.DEPRECATED],
        SkillLifecycleState.DEPRECATED: [SkillLifecycleState.RETIRED],
        SkillLifecycleState.RETIRED:   [],  # 终态
    }

    def transition(self, new_state: SkillLifecycleState) -> bool:
        """执行生命周期状态转移。返回是否成功。"""
        allowed = self._VALID_TRANSITIONS.get(self.lifecycle_state, [])
        if new_state in allowed:
            self.lifecycle_state = new_state
            self.updated_at = datetime.now(timezone.utc).isoformat()
            return True
        return False

    def can_transition_to(self, new_state: SkillLifecycleState) -> bool:
        """检查是否可以转移到目标状态（不实际执行）。"""
        allowed = self._VALID_TRANSITIONS.get(self.lifecycle_state, [])
        return new_state in allowed

    # ═══════════════════════════════════════════════════════════════════════
    # Serialization
    # ═══════════════════════════════════════════════════════════════════════

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict（JSON-safe）。"""
        def _convert(obj: Any) -> Any:
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert(v) for v in obj]
            if hasattr(obj, "__dataclass_fields__"):
                d: dict[str, Any] = {}
                for fname, fval in obj.__dataclass_fields__.items():
                    if fname.startswith("_"):
                        continue
                    val = getattr(obj, fname)
                    d[fname] = _convert(val)
                return d
            return obj

        return _convert(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UnifiedSkillConfig:
        """从 dict 反序列化。"""
        # Convert lifecycle_state string back to enum
        if "lifecycle_state" in data and isinstance(data["lifecycle_state"], str):
            data["lifecycle_state"] = SkillLifecycleState(data["lifecycle_state"])
        # Convert source_framework string back to enum
        if "source_framework" in data and isinstance(data["source_framework"], str):
            data["source_framework"] = SkillFrameworkSource(data["source_framework"])
        # Reconstruct nested dataclasses
        if "source_traceability" in data and isinstance(data["source_traceability"], dict):
            data["source_traceability"] = SourceTraceability(**data["source_traceability"])
        if "optimization_hints" in data and isinstance(data["optimization_hints"], dict):
            data["optimization_hints"] = OptimizationHints(**data["optimization_hints"])
        if "usage_metrics" in data and isinstance(data["usage_metrics"], dict):
            data["usage_metrics"] = SkillUsageMetrics(**data["usage_metrics"])
        if "composition_contract" in data and isinstance(data["composition_contract"], dict):
            data["composition_contract"] = SkillCompositionContract(**data["composition_contract"])
        if "meta_config" in data and isinstance(data["meta_config"], dict):
            data["meta_config"] = MetaSkillConfig(**data["meta_config"])
        # Remove unknown keys to avoid TypeError
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known and not k.startswith("_")}
        return cls(**filtered)

    # ═══════════════════════════════════════════════════════════════════════
    # Validation
    # ═══════════════════════════════════════════════════════════════════════

    def validate(self) -> list[str]:
        """验证配置，返回错误列表（空 = 通过）。"""
        errors: list[str] = []
        if not self.name:
            errors.append("name is required")
        if not self.version:
            errors.append("version is required")
        if not isinstance(self.lifecycle_state, SkillLifecycleState):
            errors.append(f"invalid lifecycle_state: {self.lifecycle_state}")
        if self.curator_score is not None and not (0.0 <= self.curator_score <= 1.0):
            errors.append(f"curator_score must be in [0.0, 1.0], got {self.curator_score}")
        if self.optimization_hints is not None:
            h = self.optimization_hints
            if not (0.0 <= h.confidence <= 1.0):
                errors.append(f"optimization_hints.confidence must be in [0.0, 1.0], got {h.confidence}")
            if not (0 <= h.priority <= 4):
                errors.append(f"optimization_hints.priority must be in [0, 4], got {h.priority}")
        if self.risk_level not in ("critical", "high", "medium", "low"):
            errors.append(f"invalid risk_level: {self.risk_level}")
        return errors

    def is_valid(self) -> bool:
        """是否通过验证。"""
        return len(self.validate()) == 0
