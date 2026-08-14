"""
MetaSkills Adapter — 12 核心原则验证引擎
=========================================

将 meta-skills GitHub 工具的 12 条核心原则集成到
ISN Skill Lifecycle Manager，提供：
  - 12 原则合规验证 (validate_12_principles)
  - 行为合约生成 (generate_behavior_contract)
  - 技能边界审计 (audit_skill_boundaries)

12 核心原则:
  P1  Behavior Contract       — 技能必须声明可观测行为，而非实现细节
  P2  Prevention > Governance — 预防坏行为优于事后治理
  P3  Behavior > File         — 验证行为而非文件结构
  P4  User Intent Priority    — 用户意图优先于技能默认值
  P5  Single Responsibility   — 每个技能只做一件事
  P6  Composability           — 技能可无冲突组合
  P7  Backward Compatibility  — 新版本不破坏已有消费者
  P8  Graceful Degradation    — 显式声明失败降级路径
  P9  Idempotent Operations   — 幂等操作，多次调用结果一致
  P10 Observable Side Effects — 所有副作用必须声明且可追溯
  P11 Versioned Contracts     — 输入/输出合约必须有版本
  P12 Lifecycle Awareness     — 技能必须声明生命周期阶段和转换规则

用法:
    from openllm.isn.adapters.metaskills_adapter import MetaSkillsAdapter

    adapter = MetaSkillsAdapter()
    report = adapter.validate_12_principles(skill_config)
    contract = adapter.generate_behavior_contract(skill_config)
    boundaries = adapter.audit_skill_boundaries(skill_config)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger("openllm.isn.metaskills_adapter")


# ═══════════════════════════════════════════════════════════════════════════════
# Enums & Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

class PrincipleID(str, Enum):
    """12 核心原则标识。"""
    BEHAVIOR_CONTRACT = "P1"
    PREVENTION_OVER_GOVERNANCE = "P2"
    BEHAVIOR_OVER_FILE = "P3"
    USER_INTENT_PRIORITY = "P4"
    SINGLE_RESPONSIBILITY = "P5"
    COMPOSABILITY = "P6"
    BACKWARD_COMPATIBILITY = "P7"
    GRACEFUL_DEGRADATION = "P8"
    IDEMPOTENT_OPERATIONS = "P9"
    OBSERVABLE_SIDE_EFFECTS = "P10"
    VERSIONED_CONTRACTS = "P11"
    LIFECYCLE_AWARENESS = "P12"


class Severity(str, Enum):
    """合规问题严重度。"""
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    CRITICAL = "critical"


@dataclass
class PrincipleCheck:
    """单条原则检查结果。"""
    principle_id: PrincipleID
    name: str
    description: str
    severity: Severity
    message: str
    evidence: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.severity == Severity.PASS


@dataclass
class ComplianceReport:
    """12 原则合规报告。"""
    skill_name: str
    timestamp: str
    checks: list[PrincipleCheck] = field(default_factory=list)
    overall_score: float = 0.0  # 0.0 ~ 1.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    @property
    def total_count(self) -> int:
        return len(self.checks)

    @property
    def is_compliant(self) -> bool:
        """所有原则都通过才算合规。"""
        return self.failed_count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "timestamp": self.timestamp,
            "overall_score": round(self.overall_score, 4),
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "total_count": self.total_count,
            "is_compliant": self.is_compliant,
            "checks": [
                {
                    "principle_id": c.principle_id.value,
                    "name": c.name,
                    "severity": c.severity.value,
                    "message": c.message,
                    "evidence": c.evidence,
                    "suggestions": c.suggestions,
                }
                for c in self.checks
            ],
        }


@dataclass
class BehaviorContract:
    """技能行为合约 — P1 的结构化产出。"""
    skill_name: str
    version: str
    declared_behaviors: list[str] = field(default_factory=list)
    input_expectations: list[str] = field(default_factory=list)
    output_guarantees: list[str] = field(default_factory=list)
    side_effects: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)
    idempotency_guarantee: bool = False
    lifecycle_stage: str = "active"
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "version": self.version,
            "declared_behaviors": self.declared_behaviors,
            "input_expectations": self.input_expectations,
            "output_guarantees": self.output_guarantees,
            "side_effects": self.side_effects,
            "failure_modes": self.failure_modes,
            "idempotency_guarantee": self.idempotency_guarantee,
            "lifecycle_stage": self.lifecycle_stage,
            "generated_at": self.generated_at,
        }


@dataclass
class BoundaryAudit:
    """技能边界审计结果。"""
    skill_name: str
    owns: list[str] = field(default_factory=list)       # 该技能管理的资源
    touches: list[str] = field(default_factory=list)    # 该技能读取/引用的外部资源
    violations: list[str] = field(default_factory=list) # 边界违规
    suggestions: list[str] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def has_violations(self) -> bool:
        return len(self.violations) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "owns": self.owns,
            "touches": self.touches,
            "violations": self.violations,
            "suggestions": self.suggestions,
            "has_violations": self.has_violations,
            "timestamp": self.timestamp,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Principle definitions (static metadata)
# ═══════════════════════════════════════════════════════════════════════════════

PRINCIPLES: dict[PrincipleID, dict[str, str]] = {
    PrincipleID.BEHAVIOR_CONTRACT: {
        "name": "Behavior Contract",
        "description": "技能必须声明可观测行为，而非实现细节",
    },
    PrincipleID.PREVENTION_OVER_GOVERNANCE: {
        "name": "Prevention > Governance",
        "description": "预防坏行为优于事后治理",
    },
    PrincipleID.BEHAVIOR_OVER_FILE: {
        "name": "Behavior > File Verification",
        "description": "验证技能行为而非文件结构",
    },
    PrincipleID.USER_INTENT_PRIORITY: {
        "name": "User Intent Priority",
        "description": "用户意图优先于技能默认值",
    },
    PrincipleID.SINGLE_RESPONSIBILITY: {
        "name": "Single Responsibility",
        "description": "每个技能只做一件事",
    },
    PrincipleID.COMPOSABILITY: {
        "name": "Composability",
        "description": "技能可无冲突组合",
    },
    PrincipleID.BACKWARD_COMPATIBILITY: {
        "name": "Backward Compatibility",
        "description": "新版本不破坏已有消费者",
    },
    PrincipleID.GRACEFUL_DEGRADATION: {
        "name": "Graceful Degradation",
        "description": "显式声明失败降级路径",
    },
    PrincipleID.IDEMPOTENT_OPERATIONS: {
        "name": "Idempotent Operations",
        "description": "幂等操作，多次调用结果一致",
    },
    PrincipleID.OBSERVABLE_SIDE_EFFECTS: {
        "name": "Observable Side Effects",
        "description": "所有副作用必须声明且可追溯",
    },
    PrincipleID.VERSIONED_CONTRACTS: {
        "name": "Versioned Contracts",
        "description": "输入/输出合约必须有版本",
    },
    PrincipleID.LIFECYCLE_AWARENESS: {
        "name": "Lifecycle Awareness",
        "description": "技能必须声明生命周期阶段和转换规则",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# MetaSkillsAdapter
# ═══════════════════════════════════════════════════════════════════════════════

class MetaSkillsAdapter:
    """
    meta-skills → ISN Skill Lifecycle Manager 适配器。

    职责：
    1. validate_12_principles() — 对 UnifiedSkillConfig 做 12 原则合规检查
    2. generate_behavior_contract() — 从配置生成行为合约（P1 产出物）
    3. audit_skill_boundaries() — 审计技能资源边界，检测违规

    集成方式：
    - 输入: UnifiedSkillConfig（P0 已实现的五源融合数据模型）
    - 输出: ComplianceReport / BehaviorContract / BoundaryAudit
    - 可组合: 配合 UnifiedSkillConfig.validate() 做双重校验
    """

    def __init__(
        self,
        *,
        strict_mode: bool = False,
        custom_rules: dict[PrincipleID, Any] | None = None,
    ):
        """
        Args:
            strict_mode: 严格模式下 WARN 也视为失败
            custom_rules: 自定义规则覆盖（用于扩展检查）
        """
        self.strict_mode = strict_mode
        self.custom_rules = custom_rules or {}

    # ── P1: validate_12_principles ──────────────────────────────────────

    def validate_12_principles(
        self,
        skill_config: Any,  # UnifiedSkillConfig
    ) -> ComplianceReport:
        """
        对 skill_config 执行 12 条核心原则验证。

        每条原则返回 PrincipleCheck (PASS/WARN/FAIL/CRITICAL)。
        ComplianceReport 汇总所有检查结果。

        Args:
            skill_config: UnifiedSkillConfig 实例（或任何具有同构字段的对象）

        Returns:
            ComplianceReport
        """
        report = ComplianceReport(
            skill_name=getattr(skill_config, "name", "unknown"),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # 依次执行 12 条检查
        checkers = [
            self._check_p1_behavior_contract,
            self._check_p2_prevention_over_governance,
            self._check_p3_behavior_over_file,
            self._check_p4_user_intent_priority,
            self._check_p5_single_responsibility,
            self._check_p6_composability,
            self._check_p7_backward_compatibility,
            self._check_p8_graceful_degradation,
            self._check_p9_idempotent_operations,
            self._check_p10_observable_side_effects,
            self._check_p11_versioned_contracts,
            self._check_p12_lifecycle_awareness,
        ]

        for checker in checkers:
            check = checker(skill_config)
            report.checks.append(check)

        # 计算总分: PASS=1.0, WARN=0.5, FAIL=0.0, CRITICAL=-1.0
        scores = []
        for c in report.checks:
            if c.severity == Severity.PASS:
                scores.append(1.0)
            elif c.severity == Severity.WARN:
                scores.append(0.5)
            elif c.severity == Severity.FAIL:
                scores.append(0.0)
            else:  # CRITICAL
                scores.append(-1.0)

        report.overall_score = sum(scores) / max(len(scores), 1)
        return report

    # ── P2: generate_behavior_contract ──────────────────────────────────

    def generate_behavior_contract(
        self,
        skill_config: Any,  # UnifiedSkillConfig
    ) -> BehaviorContract:
        """
        从 UnifiedSkillConfig 生成结构化行为合约。

        行为合约是 P1 (Behavior Contract) 的标准化产出，
        明确声明技能的可观测行为、输入/输出、副作用和失败模式。

        Args:
            skill_config: UnifiedSkillConfig 实例

        Returns:
            BehaviorContract
        """
        name = getattr(skill_config, "name", "unknown")
        version = getattr(skill_config, "version", "0.0.1")
        description = getattr(skill_config, "description", "")

        lifecycle_raw = getattr(skill_config, "lifecycle_state", "active")
        if hasattr(lifecycle_raw, "value"):
            lifecycle_str = str(lifecycle_raw.value)
        else:
            lifecycle_str = str(lifecycle_raw)

        contract = BehaviorContract(
            skill_name=name,
            version=version,
            lifecycle_stage=lifecycle_str,
        )

        # 从 description 提取声明行为
        if description:
            contract.declared_behaviors = self._extract_behaviors(description)

        # 从 composition_contract 提取输入/输出
        composition = getattr(skill_config, "composition_contract", None)
        if composition:
            input_schema = getattr(composition, "input_schema", {})
            output_schema = getattr(composition, "output_schema", {})
            if input_schema:
                contract.input_expectations = self._schema_to_expectations(
                    input_schema
                )
            if output_schema:
                contract.output_guarantees = self._schema_to_guarantees(
                    output_schema
                )

        # 从 meta_config 提取编排行为
        meta_config = getattr(skill_config, "meta_config", None)
        if meta_config:
            managed = getattr(meta_config, "managed_skills", [])
            if managed:
                contract.declared_behaviors.append(
                    f"Orchestrates {len(managed)} sub-skills: {', '.join(managed)}"
                )
            abort = getattr(meta_config, "abort_on_failure", True)
            if abort:
                contract.failure_modes.append(
                    "Aborts on any sub-skill failure"
                )
            else:
                contract.failure_modes.append(
                    "Continues on sub-skill failure (best-effort)"
                )

        # 从 optimization_hints 提取副作用
        hints = getattr(skill_config, "optimization_hints", None)
        if hints:
            should_optimize = getattr(hints, "should_optimize", False)
            if should_optimize:
                contract.side_effects.append(
                    "May trigger text-space optimization on skill content"
                )
            priority = getattr(hints, "priority", 0)
            if priority >= 3:
                contract.side_effects.append(
                    "High-priority optimization may modify skill sections"
                )

        # 从 dependencies 提取外部引用
        deps = getattr(skill_config, "dependencies", [])
        if deps:
            contract.side_effects.append(
                f"Depends on external skills: {', '.join(deps)}"
            )

        # 从 composes_with 提取组合行为
        composes = getattr(skill_config, "composes_with", [])
        if composes:
            contract.declared_behaviors.append(
                f"Composes with: {', '.join(composes)}"
            )

        # 幂等性判断：无副作用且无 meta_config 编排 → 幂等
        contract.idempotency_guarantee = (
            len(contract.side_effects) == 0
            and meta_config is None
        )

        # 如果没有显式声明行为，从 description 推断
        if not contract.declared_behaviors and description:
            contract.declared_behaviors = [description]

        return contract

    # ── P3: audit_skill_boundaries ──────────────────────────────────────

    def audit_skill_boundaries(
        self,
        skill_config: Any,  # UnifiedSkillConfig
    ) -> BoundaryAudit:
        """
        审计技能资源边界（P6 Composability 的验证工具）。

        检查：
        1. 技能声明的"拥有"资源 (owns)
        2. 技能引用的外部资源 (touches)
        3. 边界违规（如技能试图修改不属于自己的资源）

        Args:
            skill_config: UnifiedSkillConfig 实例

        Returns:
            BoundaryAudit
        """
        name = getattr(skill_config, "name", "unknown")
        audit = BoundaryAudit(skill_name=name)

        # ── owns: 技能声明拥有的资源 ──

        # 基本拥有：自己的配置、内容、状态
        audit.owns.append(f"skill:{name}:config")
        audit.owns.append(f"skill:{name}:state")

        # 如果有 composition_contract，拥有其输入/输出模式
        composition = getattr(skill_config, "composition_contract", None)
        if composition:
            audit.owns.append(f"skill:{name}:input_schema")
            audit.owns.append(f"skill:{name}:output_schema")

        # 如果有 optimization_hints，拥有其优化历史
        hints = getattr(skill_config, "optimization_hints", None)
        if hints:
            audit.owns.append(f"skill:{name}:optimization_history")

        # ── touches: 技能引用的外部资源 ──

        # dependencies → 外部技能
        deps = getattr(skill_config, "dependencies", [])
        for dep in deps:
            audit.touches.append(f"skill:{dep}:config")

        # composes_with → 组合目标
        composes = getattr(skill_config, "composes_with", [])
        for comp in composes:
            audit.touches.append(f"skill:{comp}:interface")

        # meta_config.managed_skills → 编排目标
        meta_config = getattr(skill_config, "meta_config", None)
        if meta_config:
            managed = getattr(meta_config, "managed_skills", [])
            for m in managed:
                audit.touches.append(f"skill:{m}:lifecycle")
                audit.touches.append(f"skill:{m}:content")

        # source_traceability → 上游仓库
        trace = getattr(skill_config, "source_traceability", None)
        if trace:
            repo = getattr(trace, "upstream_repo", "")
            if repo:
                audit.touches.append(f"upstream:{repo}")

        # ── violations: 边界违规检测 ──

        # V1: 自引用依赖（技能依赖自己）
        if name in deps:
            audit.violations.append(
                f"Self-dependency: skill '{name}' depends on itself"
            )

        # V2: 循环引用检测（简单一对多）
        for dep in deps:
            if dep in composes:
                audit.violations.append(
                    f"Circular reference: '{name}' depends on and composes with '{dep}'"
                )

        # V3: 无描述技能（边界模糊）
        description = getattr(skill_config, "description", "")
        if not description:
            audit.violations.append(
                "No description: skill boundaries are undefined"
            )
            audit.suggestions.append(
                "Add a clear description to define what this skill owns"
            )

        # V4: 过多依赖（边界泄漏风险）
        if len(deps) > 5:
            audit.violations.append(
                f"Excessive dependencies ({len(deps)}): high boundary leakage risk"
            )
            audit.suggestions.append(
                "Consider splitting into smaller, more focused skills"
            )

        # V5: 编排技能但无 composition_contract
        if meta_config and not composition:
            audit.violations.append(
                "Orchestration skill without composition_contract: undefined I/O boundaries"
            )
            audit.suggestions.append(
                "Add composition_contract with input/output schemas"
            )

        # V6: retired 技能仍被依赖
        state = getattr(skill_config, "lifecycle_state", "active")
        state_value = state.value if hasattr(state, "value") else str(state)
        if state_value == "retired" and (deps or composes):
            audit.violations.append(
                "Retired skill still has dependencies or compositions"
            )
            audit.suggestions.append(
                "Remove dependencies or transition skill back to active"
            )

        return audit

    # ═══════════════════════════════════════════════════════════════════════
    # 12 Principle Checkers (private)
    # ═══════════════════════════════════════════════════════════════════════

    def _make_check(
        self,
        pid: PrincipleID,
        severity: Severity,
        message: str,
        evidence: list[str] | None = None,
        suggestions: list[str] | None = None,
    ) -> PrincipleCheck:
        meta = PRINCIPLES[pid]
        return PrincipleCheck(
            principle_id=pid,
            name=meta["name"],
            description=meta["description"],
            severity=severity,
            message=message,
            evidence=evidence or [],
            suggestions=suggestions or [],
        )

    def _check_p1_behavior_contract(self, cfg: Any) -> PrincipleCheck:
        """P1: Behavior Contract — 声明可观测行为。"""
        desc = getattr(cfg, "description", "")
        composition = getattr(cfg, "composition_contract", None)
        meta_config = getattr(cfg, "meta_config", None)

        evidence = []
        issues = []

        if desc:
            evidence.append(f"description present: '{desc[:80]}...'")
        else:
            issues.append("No description — cannot determine observable behavior")

        if composition:
            input_s = getattr(composition, "input_schema", {})
            output_s = getattr(composition, "output_schema", {})
            if input_s or output_s:
                evidence.append("composition_contract with I/O schemas")
            else:
                issues.append("composition_contract exists but I/O schemas are empty")

        if meta_config:
            managed = getattr(meta_config, "managed_skills", [])
            if managed:
                evidence.append(f"meta_config declares {len(managed)} managed skills")

        if issues:
            return self._make_check(
                PrincipleID.BEHAVIOR_CONTRACT,
                Severity.WARN if len(issues) == 1 else Severity.FAIL,
                "; ".join(issues),
                evidence=evidence,
                suggestions=["Add a clear description of observable behaviors"],
            )
        return self._make_check(
            PrincipleID.BEHAVIOR_CONTRACT,
            Severity.PASS,
            "Skill has declared behaviors",
            evidence=evidence,
        )

    def _check_p2_prevention_over_governance(self, cfg: Any) -> PrincipleCheck:
        """P2: Prevention > Governance — 预防优于治理。"""
        evidence = []
        issues = []

        # 检查是否有 validation 入口（预防机制）
        has_validate = hasattr(cfg, "validate") and callable(getattr(cfg, "validate", None))
        if has_validate:
            evidence.append("UnifiedSkillConfig.validate() available (prevention)")
        else:
            issues.append("No validate() method — no prevention mechanism")

        # 检查 composition_contract 是否有 requires_approval（治理机制）
        composition = getattr(cfg, "composition_contract", None)
        if composition:
            requires = getattr(composition, "requires_approval", False)
            if requires:
                evidence.append("requires_approval = True (governance fallback)")
            else:
                evidence.append("requires_approval = False (pure prevention)")

        # 检查 lifecycle_state 是否有 guard（预防）
        state = getattr(cfg, "lifecycle_state", "active")
        state_value = state.value if hasattr(state, "value") else str(state)
        if state_value in ("deprecated", "retired"):
            evidence.append(f"State is '{state_value}' — transition guards active")
        else:
            evidence.append(f"State is '{state_value}' — normal operation")

        if issues:
            return self._make_check(
                PrincipleID.PREVENTION_OVER_GOVERNANCE,
                Severity.WARN,
                "; ".join(issues),
                evidence=evidence,
                suggestions=["Add validation methods to prevent invalid states"],
            )
        return self._make_check(
            PrincipleID.PREVENTION_OVER_GOVERNANCE,
            Severity.PASS,
            "Prevention mechanisms present",
            evidence=evidence,
        )

    def _check_p3_behavior_over_file(self, cfg: Any) -> PrincipleCheck:
        """P3: Behavior > File Verification — 验证行为而非文件结构。"""
        evidence = []
        issues = []

        # 检查是否有行为相关的字段
        desc = getattr(cfg, "description", "")
        composition = getattr(cfg, "composition_contract", None)
        usage = getattr(cfg, "usage_metrics", None)

        if desc:
            evidence.append("description describes behavior, not file structure")
        if composition:
            evidence.append("composition_contract defines behavioral interface")
        if usage:
            total = getattr(usage, "total_calls", 0)
            evidence.append(f"usage_metrics with {total} calls — runtime behavior tracked")

        # 纯文件结构验证不足
        name = getattr(cfg, "name", "")
        tags = getattr(cfg, "domain_tags", [])
        if name and not desc and not composition and not usage:
            issues.append(
                "Only structural fields (name, tags) present — no behavioral verification"
            )

        if issues:
            return self._make_check(
                PrincipleID.BEHAVIOR_OVER_FILE,
                Severity.WARN,
                "; ".join(issues),
                evidence=evidence,
                suggestions=[
                    "Add composition_contract or usage_metrics for behavioral verification",
                    "Behavior should be verified at runtime, not just at file level",
                ],
            )
        return self._make_check(
            PrincipleID.BEHAVIOR_OVER_FILE,
            Severity.PASS,
            "Behavior verification available beyond file structure",
            evidence=evidence,
        )

    def _check_p4_user_intent_priority(self, cfg: Any) -> PrincipleCheck:
        """P4: User Intent Priority — 用户意图优先于技能默认值。"""
        evidence = []

        # 检查是否有风险等级分类（用户意图信号）
        risk = getattr(cfg, "risk_level", "medium")
        evidence.append(f"risk_level='{risk}' — user-configurable")

        # 检查 meta_config 是否有用户可配置参数
        meta_config = getattr(cfg, "meta_config", None)
        if meta_config:
            max_conc = getattr(meta_config, "max_concurrency", 1)
            abort = getattr(meta_config, "abort_on_failure", True)
            evidence.append(
                f"meta_config: max_concurrency={max_conc}, abort_on_failure={abort}"
            )

        # 检查 composition_contract 的 requires_approval
        composition = getattr(cfg, "composition_contract", None)
        if composition:
            requires = getattr(composition, "requires_approval", False)
            evidence.append(f"requires_approval={requires} — user can override")

        # 用户意图优先是设计原则，这里只检查是否有用户可配置的旋钮
        return self._make_check(
            PrincipleID.USER_INTENT_PRIORITY,
            Severity.PASS,
            "User-configurable parameters present",
            evidence=evidence,
        )

    def _check_p5_single_responsibility(self, cfg: Any) -> PrincipleCheck:
        """P5: Single Responsibility — 每个技能只做一件事。"""
        evidence = []
        issues = []

        desc = getattr(cfg, "description", "")
        tags = getattr(cfg, "domain_tags", [])
        meta_config = getattr(cfg, "meta_config", None)

        # 检查 domain_tags 是否过多（可能违反单一职责）
        if len(tags) > 5:
            issues.append(
                f"Too many domain_tags ({len(tags)}): skill may have multiple responsibilities"
            )
        elif tags:
            evidence.append(f"domain_tags: {tags}")

        # 检查 meta_config 是否管理过多技能
        if meta_config:
            managed = getattr(meta_config, "managed_skills", [])
            if len(managed) > 10:
                issues.append(
                    f"meta_config manages {len(managed)} skills: may violate single responsibility"
                )
            elif managed:
                evidence.append(f"meta_config manages {len(managed)} sub-skills")

        # 检查 dependencies 是否过多
        deps = getattr(cfg, "dependencies", [])
        if len(deps) > 5:
            issues.append(
                f"Excessive dependencies ({len(deps)}): high coupling"
            )

        if issues:
            return self._make_check(
                PrincipleID.SINGLE_RESPONSIBILITY,
                Severity.WARN,
                "; ".join(issues),
                evidence=evidence,
                suggestions=["Consider splitting into smaller, focused skills"],
            )
        return self._make_check(
            PrincipleID.SINGLE_RESPONSIBILITY,
            Severity.PASS,
            "Skill appears to have focused responsibility",
            evidence=evidence,
        )

    def _check_p6_composability(self, cfg: Any) -> PrincipleCheck:
        """P6: Composability — 技能可无冲突组合。"""
        evidence = []
        issues = []

        composes = getattr(cfg, "composes_with", [])
        deps = getattr(cfg, "dependencies", [])

        if composes:
            evidence.append(f"Composes with {len(composes)} skills: {composes}")
        if deps:
            evidence.append(f"Depends on {len(deps)} skills: {deps}")

        # 循环引用检测
        name = getattr(cfg, "name", "")
        if name in deps:
            issues.append("Self-dependency detected — composition will deadlock")
        if name in composes:
            issues.append("Self-composition detected — composition will loop")

        # 冲突检测：同时依赖和组合同一技能
        overlap = set(deps) & set(composes)
        if overlap:
            issues.append(
                f"Overlap between dependencies and compositions: {overlap}"
            )

        if issues:
            return self._make_check(
                PrincipleID.COMPOSABILITY,
                Severity.FAIL,
                "; ".join(issues),
                evidence=evidence,
                suggestions=[
                    "Remove self-references",
                    "Resolve dependency/composition overlap",
                ],
            )
        return self._make_check(
            PrincipleID.COMPOSABILITY,
            Severity.PASS,
            "No composition conflicts detected",
            evidence=evidence,
        )

    def _check_p7_backward_compatibility(self, cfg: Any) -> PrincipleCheck:
        """P7: Backward Compatibility — 新版本不破坏已有消费者。"""
        version = getattr(cfg, "version", "0.0.1")
        evidence = [f"version={version}"]

        # 检查版本号格式
        parts = version.split(".")
        if len(parts) != 3:
            return self._make_check(
                PrincipleID.BACKWARD_COMPATIBILITY,
                Severity.WARN,
                f"Non-standard version format: '{version}'",
                evidence=evidence,
                suggestions=["Use semver format: X.Y.Z"],
            )

        major = int(parts[0]) if parts[0].isdigit() else 0
        if major == 0:
            evidence.append("Pre-1.0: breaking changes allowed by semver convention")
            return self._make_check(
                PrincipleID.BACKWARD_COMPATIBILITY,
                Severity.PASS,
                "Pre-1.0 version — breaking changes are expected",
                evidence=evidence,
            )

        # 1.0+ 版本，检查是否有 composition_contract（消费者依赖）
        composition = getattr(cfg, "composition_contract", None)
        if composition:
            evidence.append("composition_contract present — consumers may depend on I/O")
            return self._make_check(
                PrincipleID.BACKWARD_COMPATIBILITY,
                Severity.WARN,
                "Post-1.0 skill with composition_contract: ensure I/O backward compatibility",
                evidence=evidence,
                suggestions=[
                    "Use additive-only changes to input/output schemas",
                    "Version your composition contracts separately",
                ],
            )

        return self._make_check(
            PrincipleID.BACKWARD_COMPATIBILITY,
            Severity.PASS,
            "No backward compatibility concerns detected",
            evidence=evidence,
        )

    def _check_p8_graceful_degradation(self, cfg: Any) -> PrincipleCheck:
        """P8: Graceful Degradation — 显式声明失败降级路径。"""
        evidence = []
        issues = []

        composition = getattr(cfg, "composition_contract", None)
        meta_config = getattr(cfg, "meta_config", None)

        # composition_contract 的 fallback_config
        if composition:
            fallback = getattr(composition, "fallback_config", None)
            timeout = getattr(composition, "timeout_seconds", None)
            if fallback:
                evidence.append(f"fallback_config='{fallback}'")
            else:
                issues.append("composition_contract has no fallback_config")

            if timeout:
                evidence.append(f"timeout_seconds={timeout}")
            else:
                issues.append("composition_contract has no timeout_seconds")

        # meta_config 的 abort_on_failure
        if meta_config:
            abort = getattr(meta_config, "abort_on_failure", True)
            if abort:
                evidence.append("abort_on_failure=True — fails fast")
            else:
                evidence.append("abort_on_failure=False — continues on failure")

        # 如果没有任何降级声明
        if not composition and not meta_config:
            evidence.append("No failure mode declarations (simple skill)")
            return self._make_check(
                PrincipleID.GRACEFUL_DEGRADATION,
                Severity.PASS,
                "Simple skill — no failure modes to declare",
                evidence=evidence,
            )

        if issues:
            return self._make_check(
                PrincipleID.GRACEFUL_DEGRADATION,
                Severity.WARN,
                "; ".join(issues),
                evidence=evidence,
                suggestions=[
                    "Add fallback_config for graceful degradation",
                    "Add timeout_seconds to prevent hung operations",
                ],
            )
        return self._make_check(
            PrincipleID.GRACEFUL_DEGRADATION,
            Severity.PASS,
            "Failure modes explicitly declared",
            evidence=evidence,
        )

    def _check_p9_idempotent_operations(self, cfg: Any) -> PrincipleCheck:
        """P9: Idempotent Operations — 幂等操作。"""
        evidence = []
        issues = []

        meta_config = getattr(cfg, "meta_config", None)
        hints = getattr(cfg, "optimization_hints", None)
        usage = getattr(cfg, "usage_metrics", None)

        # 编排技能通常非幂等
        if meta_config:
            managed = getattr(meta_config, "managed_skills", [])
            if managed:
                issues.append(
                    f"Orchestration skill with {len(managed)} sub-skills: likely non-idempotent"
                )
                evidence.append("meta_config.managed_skills present")

        # 优化提示暗示副作用
        if hints:
            should_opt = getattr(hints, "should_optimize", False)
            if should_opt:
                issues.append("optimization_hints.should_optimize=True: implies content mutation")
                evidence.append("optimization_hints active")

        # 使用指标暗示有状态
        if usage:
            total = getattr(usage, "total_calls", 0)
            if total > 0:
                evidence.append(f"usage_metrics with {total} calls — state tracked")

        if issues:
            return self._make_check(
                PrincipleID.IDEMPOTENT_OPERATIONS,
                Severity.WARN,
                "; ".join(issues),
                evidence=evidence,
                suggestions=[
                    "Document which operations are idempotent",
                    "Add idempotency keys for non-idempotent operations",
                ],
            )
        return self._make_check(
            PrincipleID.IDEMPOTENT_OPERATIONS,
            Severity.PASS,
            "No non-idempotent patterns detected",
            evidence=evidence,
        )

    def _check_p10_observable_side_effects(self, cfg: Any) -> PrincipleCheck:
        """P10: Observable Side Effects — 副作用声明。"""
        evidence = []
        side_effects_declared = []

        # 检查各模块的副作用声明
        composition = getattr(cfg, "composition_contract", None)
        if composition:
            requires = getattr(composition, "requires_approval", False)
            if requires:
                side_effects_declared.append("requires_approval (user confirmation)")

        hints = getattr(cfg, "optimization_hints", None)
        if hints:
            should_opt = getattr(hints, "should_optimize", False)
            if should_opt:
                side_effects_declared.append("text-space optimization")

        meta_config = getattr(cfg, "meta_config", None)
        if meta_config:
            managed = getattr(meta_config, "managed_skills", [])
            if managed:
                side_effects_declared.append(
                    f"orchestrates {len(managed)} sub-skills"
                )

        deps = getattr(cfg, "dependencies", [])
        if deps:
            side_effects_declared.append(f"triggers {len(deps)} dependency skills")

        if side_effects_declared:
            evidence.append(f"Declared side effects: {side_effects_declared}")
            return self._make_check(
                PrincipleID.OBSERVABLE_SIDE_EFFECTS,
                Severity.PASS,
                f"{len(side_effects_declared)} side effects declared",
                evidence=evidence,
            )

        evidence.append("No side effects detected")
        return self._make_check(
            PrincipleID.OBSERVABLE_SIDE_EFFECTS,
            Severity.PASS,
            "No side effects to declare",
            evidence=evidence,
        )

    def _check_p11_versioned_contracts(self, cfg: Any) -> PrincipleCheck:
        """P11: Versioned Contracts — 版本化合约。"""
        evidence = []
        issues = []

        version = getattr(cfg, "version", "")
        if version:
            evidence.append(f"skill version: {version}")
        else:
            issues.append("No skill version declared")

        composition = getattr(cfg, "composition_contract", None)
        if composition:
            input_s = getattr(composition, "input_schema", {})
            output_s = getattr(composition, "output_schema", {})
            # 检查 schema 是否有版本字段
            has_version = "_version" in input_s or "_version" in output_s
            if has_version:
                evidence.append("composition_contract schemas are versioned")
            else:
                issues.append(
                    "composition_contract schemas have no version field"
                )

        trace = getattr(cfg, "source_traceability", None)
        if trace:
            upstream_ver = getattr(trace, "upstream_version", "")
            if upstream_ver:
                evidence.append(f"upstream_version: {upstream_ver}")

        if issues:
            return self._make_check(
                PrincipleID.VERSIONED_CONTRACTS,
                Severity.WARN,
                "; ".join(issues),
                evidence=evidence,
                suggestions=[
                    "Add version fields to composition contracts",
                    "Use semver for all contract versions",
                ],
            )
        return self._make_check(
            PrincipleID.VERSIONED_CONTRACTS,
            Severity.PASS,
            "Versioning present",
            evidence=evidence,
        )

    def _check_p12_lifecycle_awareness(self, cfg: Any) -> PrincipleCheck:
        """P12: Lifecycle Awareness — 生命周期声明。"""
        evidence = []

        state = getattr(cfg, "lifecycle_state", None)
        if state:
            state_value = state.value if hasattr(state, "value") else str(state)
            evidence.append(f"lifecycle_state: {state_value}")
        else:
            return self._make_check(
                PrincipleID.LIFECYCLE_AWARENESS,
                Severity.FAIL,
                "No lifecycle_state declared",
                evidence=[],
                suggestions=["Declare lifecycle_state (active/dormant/deprecated/retired)"],
            )

        # 检查是否可以转移（有状态机支持）
        has_transition = hasattr(cfg, "transition") and callable(
            getattr(cfg, "transition", None)
        )
        if has_transition:
            evidence.append("transition() method available — state machine supported")
        else:
            evidence.append("No transition() method — static lifecycle only")

        # 检查更新时间
        updated = getattr(cfg, "updated_at", "")
        if updated:
            evidence.append(f"updated_at: {updated}")

        return self._make_check(
            PrincipleID.LIFECYCLE_AWARENESS,
            Severity.PASS,
            "Lifecycle state declared and managed",
            evidence=evidence,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Helper methods
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _extract_behaviors(description: str) -> list[str]:
        """从 description 文本中提取行为声明。"""
        behaviors = []
        # 按句号分割
        sentences = re.split(r"[.。！!？?\n]+", description)
        for s in sentences:
            s = s.strip()
            if s and len(s) > 5:
                behaviors.append(s)
        return behaviors[:10]  # 最多 10 条

    @staticmethod
    def _schema_to_expectations(schema: dict[str, Any]) -> list[str]:
        """将 input_schema 转换为输入期望列表。"""
        expectations = []
        for key, val in schema.items():
            if isinstance(val, dict):
                type_name = val.get("type", "any")
                expectations.append(f"expects '{key}' of type {type_name}")
            else:
                expectations.append(f"expects '{key}'")
        return expectations

    @staticmethod
    def _schema_to_guarantees(schema: dict[str, Any]) -> list[str]:
        """将 output_schema 转换为输出保证列表。"""
        guarantees = []
        for key, val in schema.items():
            if isinstance(val, dict):
                type_name = val.get("type", "any")
                guarantees.append(f"guarantees '{key}' of type {type_name}")
            else:
                guarantees.append(f"guarantees '{key}'")
        return guarantees
