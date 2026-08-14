"""
SkCC Adapter — UnifiedSkillConfig → SKIR (Skill Intermediate Representation)
============================================================================

SkCC (Skill Composition Contracts) 是 Microsoft 的技能编译框架，
本模块将其核心思想落地为 ISN Skill Lifecycle Manager 的跨框架适配器。

SKIR（Skill Intermediate Representation）是跨框架可移植性的认知基质，
不是序列化格式。它捕获技能的本质结构——身份、合约、组合图、安全剖面、
生命周期——使任何目标框架都能消费。

设计原则（PAL v1 2026-07-24）：
  1. SKIR 必须是认知基质不是序列化格式
  2. 编译是单向的：UnifiedSkillConfig → SKIR → 目标框架
  3. 安全约束是编译期门控，不是运行时检查
  4. 组合合约必须显式声明输入/输出/超时/降级

用法：
    from openllm.isn.unified_skill_config import UnifiedSkillConfig, SkillCompositionContract
    from openllm.isn.adapters.skcc_adapter import SkCCAdapter

    config = UnifiedSkillConfig(
        name="search-fallback",
        description="搜索降级级联",
        version="1.0.0",
        composition_contract=SkillCompositionContract(
            input_schema={"query": {"type": "string"}},
            output_schema={"results": {"type": "array"}},
            requires_approval=False,
            timeout_seconds=30,
        ),
        dependencies=["bing-search", "web-search"],
    )

    adapter = SkCCAdapter()
    skir = adapter.compile(config)            # → SKIR dict
    safe = adapter.validate_security(config)   # → SecurityReport
    export = adapter.export(skir, target="skillos")  # → framework-specific dict
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from openllm.isn.unified_skill_config import (
    SkillCompositionContract,
    SkillFrameworkSource,
    SkillLifecycleState,
    UnifiedSkillConfig,
)

logger = logging.getLogger("openllm.isn.skcc_adapter")


# ═══════════════════════════════════════════════════════════════════════════════
# Enums & Constants
# ═══════════════════════════════════════════════════════════════════════════════

class SKIRVersion(str, Enum):
    """SKIR schema version."""
    V1 = "1.0"


class ExportTarget(str, Enum):
    """Supported export target frameworks."""
    SKILLOS = "skillos"
    OPENSKILL = "openskill"
    SKILLOPT = "skillopt"
    META_SKILLS = "meta_skills"
    UNIFIED = "unified"


class SecurityRiskLevel(str, Enum):
    """Security risk levels for constraint validation."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityViolationType(str, Enum):
    """Types of security violations detected during validation."""
    CIRCULAR_DEPENDENCY = "circular_dependency"
    UNTRUSTED_DEPENDENCY = "untrusted_dependency"
    EVAL_IN_SCHEMA = "eval_in_schema"
    EXEC_IN_SCHEMA = "exec_in_schema"
    DEPTH_EXCEEDED = "depth_exceeded"
    MISSING_TIMEOUT = "missing_timeout"
    MISSING_FALLBACK = "missing_fallback"
    APPROVAL_REQUIRED_BUT_MISSING = "approval_required_but_missing"
    EMPTY_NAME = "empty_name"
    INVALID_VERSION = "invalid_version"
    UNSAFE_CONTENT_PATTERN = "unsafe_content_pattern"
    RISK_LEVEL_ESCALATION = "risk_level_escalation"


# Maximum allowed composition depth (root → leaf) to prevent infinite recursion
MAX_COMPOSITION_DEPTH = 8

# Patterns that should never appear in input/output schemas
_UNSAFE_SCHEMA_PATTERNS = [
    "eval(",
    "exec(",
    "__import__(",
    "subprocess.",
    "os.system(",
    "os.popen(",
    "shutil.",
]

# Trusted dependency prefixes (empty = no trust list = deny by default)
# In production this would be a configurable allowlist
_TRUSTED_DEPENDENCY_PREFIXES: list[str] = []


# ═══════════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SecurityViolation:
    """A single security constraint violation."""
    violation_type: SecurityViolationType
    severity: SecurityRiskLevel
    message: str
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "violation_type": self.violation_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "context": self.context,
        }


@dataclass
class SecurityReport:
    """Security validation report — pass/fail + violations."""
    passed: bool
    violations: list[SecurityViolation] = field(default_factory=list)
    checked_constraints: int = 0
    risk_level: SecurityRiskLevel = SecurityRiskLevel.LOW

    @property
    def violation_count(self) -> int:
        return len(self.violations)

    @property
    def critical_violations(self) -> list[SecurityViolation]:
        return [v for v in self.violations if v.severity == SecurityRiskLevel.CRITICAL]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "violation_count": self.violation_count,
            "critical_count": len(self.critical_violations),
            "checked_constraints": self.checked_constraints,
            "risk_level": self.risk_level.value,
            "violations": [v.to_dict() for v in self.violations],
        }


@dataclass
class CompositionNode:
    """A node in the skill composition graph."""
    skill_name: str
    version: str
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: Optional[int] = None
    fallback: Optional[str] = None
    requires_approval: bool = False
    depth: int = 0


@dataclass
class SKIR:
    """Skill Intermediate Representation — 认知基质。

    SKIR 不是序列化格式，而是技能在跨框架空间中的结构化表达。
    它捕获：身份、合约、组合图、安全剖面、生命周期。
    """
    version: str = SKIRVersion.V1.value
    # ── Identity ──
    skill_name: str = ""
    description: str = ""
    skill_version: str = "0.0.1"
    author: str = ""
    source_framework: str = "unified"
    domain_tags: list[str] = field(default_factory=list)
    # ── Composition Contract ──
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    requires_approval: bool = False
    timeout_seconds: Optional[int] = None
    fallback_skill: Optional[str] = None
    # ── Composition Graph ──
    dependencies: list[str] = field(default_factory=list)
    composes_with: list[str] = field(default_factory=list)
    composition_depth: int = 0
    # ── Security Profile ──
    risk_level: str = "medium"
    security_passed: bool = False
    security_violations: int = 0
    # ── Lifecycle ──
    lifecycle_state: str = "active"
    # ── Portability ──
    export_hints: dict[str, Any] = field(default_factory=dict)
    # ── Integrity ──
    content_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        """序列化为 JSON-safe dict。"""
        return {
            "version": self.version,
            "identity": {
                "name": self.skill_name,
                "description": self.description,
                "version": self.skill_version,
                "author": self.author,
                "source_framework": self.source_framework,
                "domain_tags": self.domain_tags,
            },
            "contract": {
                "input_schema": self.input_schema,
                "output_schema": self.output_schema,
                "requires_approval": self.requires_approval,
                "timeout_seconds": self.timeout_seconds,
                "fallback_skill": self.fallback_skill,
            },
            "composition_graph": {
                "dependencies": self.dependencies,
                "composes_with": self.composes_with,
                "depth": self.composition_depth,
            },
            "security": {
                "risk_level": self.risk_level,
                "passed": self.security_passed,
                "violations": self.security_violations,
            },
            "lifecycle": {
                "state": self.lifecycle_state,
            },
            "portability": {
                "export_hints": self.export_hints,
            },
            "integrity": {
                "content_hash": self.content_hash,
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SkCCAdapter — Core
# ═══════════════════════════════════════════════════════════════════════════════

class SkCCAdapter:
    """
    SkCC → SKIR 编译器。

    职责：
      1. compile(): UnifiedSkillConfig → SKIR (认知基质)
      2. validate_security(): 安全约束门控
      3. export(): SKIR → 目标框架格式
      4. build_composition_graph(): 组合图构建 + 深度检查

    用法：
        adapter = SkCCAdapter(trusted_deps={"web-search", "bing-search"})
        skir = adapter.compile(config)
        report = adapter.validate_security(config)
        if report.passed:
            skillos_export = adapter.export(skir, target="skillos")
    """

    def __init__(
        self,
        trusted_deps: Optional[set[str]] = None,
        max_depth: int = MAX_COMPOSITION_DEPTH,
        allow_untrusted: bool = False,
    ):
        """
        Args:
            trusted_deps: 可信依赖名称集合。None = 空集（deny-by-default）。
            max_depth: 最大组合深度。
            allow_untrusted: 是否允许未信任依赖（默认 False）。
        """
        self.trusted_deps: set[str] = trusted_deps if trusted_deps is not None else set()
        self.max_depth = max_depth
        self.allow_untrusted = allow_untrusted

    # ── Public API ────────────────────────────────────────────────────────

    def compile(self, config: UnifiedSkillConfig) -> SKIR:
        """
        将 UnifiedSkillConfig 编译为 SKIR 认知基质。

        这是单向编译：UnifiedSkillConfig → SKIR。
        编译过程中自动执行安全验证。

        Args:
            config: 统一技能配置

        Returns:
            SKIR 认知基质

        Raises:
            ValueError: config 不合法（空 name 等）
        """
        if not config.name:
            raise ValueError("Cannot compile skill with empty name")

        # Build composition contract
        contract = config.composition_contract
        input_schema = contract.input_schema if contract else {}
        output_schema = contract.output_schema if contract else {}
        requires_approval = contract.requires_approval if contract else False
        timeout_seconds = contract.timeout_seconds if contract else None
        fallback_skill = contract.fallback_config if contract else None

        # Build composition graph
        composition_depth = self._compute_composition_depth(config.dependencies)

        # Compute content hash for integrity
        content_hash = self._compute_content_hash(config)

        # Generate export hints
        export_hints = self._generate_export_hints(config)

        # Security validation (auto-run during compile)
        security_report = self.validate_security(config)

        skir = SKIR(
            skill_name=config.name,
            description=config.description,
            skill_version=config.version,
            author=config.author,
            source_framework=config.source_framework.value,
            domain_tags=list(config.domain_tags),
            input_schema=input_schema,
            output_schema=output_schema,
            requires_approval=requires_approval,
            timeout_seconds=timeout_seconds,
            fallback_skill=fallback_skill,
            dependencies=list(config.dependencies),
            composes_with=list(config.composes_with),
            composition_depth=composition_depth,
            risk_level=config.risk_level,
            security_passed=security_report.passed,
            security_violations=security_report.violation_count,
            lifecycle_state=config.lifecycle_state.value,
            export_hints=export_hints,
            content_hash=content_hash,
        )

        logger.info(
            f"SKIR compiled: {config.name} v{config.version} "
            f"| security={'PASS' if security_report.passed else 'FAIL'} "
            f"| violations={security_report.violation_count} "
            f"| depth={composition_depth}"
        )

        return skir

    def validate_security(self, config: UnifiedSkillConfig) -> SecurityReport:
        """
        安全约束验证 — 编译期门控。

        检查清单：
        1. 基础字段验证（name, version）
        2. Schema 安全性（无 eval/exec 等危险模式）
        3. 依赖信任验证
        4. 组合深度检查
        5. 高风险技能的审批/超时/降级要求
        6. 风险等级一致性

        Args:
            config: 统一技能配置

        Returns:
            SecurityReport
        """
        violations: list[SecurityViolation] = []
        checked = 0

        # 1. Basic field validation
        checked += 1
        if not config.name:
            violations.append(SecurityViolation(
                violation_type=SecurityViolationType.EMPTY_NAME,
                severity=SecurityRiskLevel.CRITICAL,
                message="Skill name is empty",
            ))

        checked += 1
        if not config.version:
            violations.append(SecurityViolation(
                violation_type=SecurityViolationType.INVALID_VERSION,
                severity=SecurityRiskLevel.HIGH,
                message="Skill version is empty",
            ))

        # 2. Schema safety — check for dangerous patterns
        contract = config.composition_contract
        if contract:
            for schema_name, schema in [
                ("input_schema", contract.input_schema),
                ("output_schema", contract.output_schema),
            ]:
                checked += 1
                violations.extend(
                    self._check_schema_safety(schema, schema_name)
                )

        # 3. Dependency trust
        checked += 1
        violations.extend(self._check_dependency_trust(config.dependencies))

        # 4. Composition depth
        checked += 1
        depth = self._compute_composition_depth(config.dependencies)
        if depth > self.max_depth:
            violations.append(SecurityViolation(
                violation_type=SecurityViolationType.DEPTH_EXCEEDED,
                severity=SecurityRiskLevel.HIGH,
                message=f"Composition depth {depth} exceeds maximum {self.max_depth}",
                context={"depth": depth, "max_depth": self.max_depth},
            ))

        # 5. High-risk requirements
        risk = config.risk_level
        if risk in ("critical", "high"):
            checked += 1
            if contract is None or not contract.requires_approval:
                violations.append(SecurityViolation(
                    violation_type=SecurityViolationType.APPROVAL_REQUIRED_BUT_MISSING,
                    severity=SecurityRiskLevel.HIGH,
                    message=f"Risk level '{risk}' requires requires_approval=True",
                    context={"risk_level": risk},
                ))

            checked += 1
            if contract is None or contract.timeout_seconds is None:
                violations.append(SecurityViolation(
                    violation_type=SecurityViolationType.MISSING_TIMEOUT,
                    severity=SecurityRiskLevel.MEDIUM,
                    message=f"Risk level '{risk}' requires timeout_seconds",
                    context={"risk_level": risk},
                ))

            checked += 1
            if contract is None or not contract.fallback_config:
                violations.append(SecurityViolation(
                    violation_type=SecurityViolationType.MISSING_FALLBACK,
                    severity=SecurityRiskLevel.MEDIUM,
                    message=f"Risk level '{risk}' requires fallback_config",
                    context={"risk_level": risk},
                ))

        # 6. Risk level consistency
        checked += 1
        violations.extend(self._check_risk_consistency(config))

        # Determine overall result
        critical_count = sum(
            1 for v in violations if v.severity == SecurityRiskLevel.CRITICAL
        )
        high_count = sum(
            1 for v in violations if v.severity == SecurityRiskLevel.HIGH
        )

        # Compute aggregate risk level
        if critical_count > 0:
            agg_risk = SecurityRiskLevel.CRITICAL
        elif high_count > 0:
            agg_risk = SecurityRiskLevel.HIGH
        elif violations:
            agg_risk = SecurityRiskLevel.MEDIUM
        else:
            agg_risk = SecurityRiskLevel(config.risk_level) if config.risk_level in [e.value for e in SecurityRiskLevel] else SecurityRiskLevel.LOW

        passed = critical_count == 0 and high_count == 0

        report = SecurityReport(
            passed=passed,
            violations=violations,
            checked_constraints=checked,
            risk_level=agg_risk,
        )

        if not passed:
            logger.warning(
                f"Security validation FAILED for {config.name}: "
                f"{len(violations)} violations "
                f"({critical_count} critical, {high_count} high)"
            )

        return report

    def export(
        self,
        skir: SKIR,
        target: str = "unified",
    ) -> dict[str, Any]:
        """
        将 SKIR 导出为目标框架格式。

        Args:
            skir: SKIR 认知基质
            target: 目标框架 (skillos / openskill / skillopt / meta_skills / unified)

        Returns:
            目标框架格式的 dict

        Raises:
            ValueError: 不支持的目标框架
        """
        try:
            export_target = ExportTarget(target)
        except ValueError:
            raise ValueError(
                f"Unsupported export target: '{target}'. "
                f"Supported: {[e.value for e in ExportTarget]}"
            )

        exporters = {
            ExportTarget.SKILLOS: self._export_skillos,
            ExportTarget.OPENSKILL: self._export_openskill,
            ExportTarget.SKILLOPT: self._export_skillopt,
            ExportTarget.META_SKILLS: self._export_meta_skills,
            ExportTarget.UNIFIED: self._export_unified,
        }

        return exporters[export_target](skir)

    def build_composition_graph(
        self,
        config: UnifiedSkillConfig,
        known_skills: Optional[dict[str, UnifiedSkillConfig]] = None,
    ) -> dict[str, Any]:
        """
        构建组合图 — 分析依赖链并返回图结构。

        Args:
            config: 技能配置
            known_skills: 已知技能注册表（用于解析依赖）

        Returns:
            组合图 dict: {nodes: [...], edges: [...], depth: int, cycles: bool}
        """
        known = known_skills or {}
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        visited: set[str] = set()
        has_cycle = False

        def _visit(skill_name: str, depth: int) -> None:
            nonlocal has_cycle
            if skill_name in visited:
                has_cycle = True
                return
            visited.add(skill_name)

            # Determine the skill's config (use known or the root config)
            skill_config = known.get(skill_name)
            if skill_config is None and skill_name == config.name:
                skill_config = config
            deps = skill_config.dependencies if skill_config else []
            contract = skill_config.composition_contract if skill_config else None

            nodes.append({
                "name": skill_name,
                "depth": depth,
                "has_contract": contract is not None,
                "timeout": contract.timeout_seconds if contract else None,
            })

            for dep in deps:
                edges.append({"from": skill_name, "to": dep})
                _visit(dep, depth + 1)

        _visit(config.name, 0)

        return {
            "nodes": nodes,
            "edges": edges,
            "depth": self._compute_composition_depth(
                config.dependencies, known_skills=known
            ),
            "cycles": has_cycle,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
        }

    # ── Internal: Composition Depth ───────────────────────────────────────

    def _compute_composition_depth(
        self,
        dependencies: list[str],
        known_skills: Optional[dict[str, UnifiedSkillConfig]] = None,
        _visited: Optional[set[str]] = None,
    ) -> int:
        """递归计算组合深度（最长依赖链）。"""
        if not dependencies:
            return 0

        visited = _visited or set()
        known = known_skills or {}
        max_child_depth = 0

        for dep in dependencies:
            if dep in visited:
                continue  # Cycle protection
            visited.add(dep)
            dep_config = known.get(dep)
            if dep_config and dep_config.dependencies:
                child_depth = self._compute_composition_depth(
                    dep_config.dependencies, known, visited
                )
                max_child_depth = max(max_child_depth, child_depth)

        return 1 + max_child_depth

    # ── Internal: Content Hash ────────────────────────────────────────────

    def _compute_content_hash(self, config: UnifiedSkillConfig) -> str:
        """计算技能内容的 SHA-224 哈希（完整性校验）。"""
        # Hash key fields (not timestamps — those change)
        hash_input = json.dumps({
            "name": config.name,
            "description": config.description,
            "version": config.version,
            "dependencies": sorted(config.dependencies),
            "composes_with": sorted(config.composes_with),
        }, sort_keys=True, ensure_ascii=False)
        return hashlib.sha224(hash_input.encode("utf-8")).hexdigest()[:16]

    # ── Internal: Export Hints ────────────────────────────────────────────

    def _generate_export_hints(self, config: UnifiedSkillConfig) -> dict[str, Any]:
        """为目标框架生成导出提示。"""
        hints: dict[str, Any] = {}

        # SkillOS hints
        hints["skillos"] = {
            "needs_task_group": config.task_group_id is None,
            "has_curator_score": config.curator_score is not None,
        }

        # OpenSkill hints
        hints["openskill"] = {
            "has_provenance": config.source_traceability is not None,
            "needs_attribution": config.author != "",
        }

        # SkillOpt hints
        hints["skillopt"] = {
            "has_optimization_data": config.optimization_hints is not None,
            "has_metrics": config.usage_metrics is not None,
        }

        # Meta-skills hints
        hints["meta_skills"] = {
            "is_meta_skill": config.meta_config is not None,
            "orchestration_pattern": (
                config.meta_config.orchestration_pattern
                if config.meta_config
                else None
            ),
        }

        return hints

    # ── Internal: Schema Safety ───────────────────────────────────────────

    def _check_schema_safety(
        self,
        schema: dict[str, Any],
        schema_name: str,
    ) -> list[SecurityViolation]:
        """检查 schema 中是否有危险模式。"""
        violations: list[SecurityViolation] = []
        schema_str = json.dumps(schema, ensure_ascii=False)

        for pattern in _UNSAFE_SCHEMA_PATTERNS:
            if pattern in schema_str:
                if pattern.startswith(("eval", "exec")):
                    vtype = (
                        SecurityViolationType.EVAL_IN_SCHEMA
                        if pattern.startswith("eval")
                        else SecurityViolationType.EXEC_IN_SCHEMA
                    )
                else:
                    vtype = SecurityViolationType.UNSAFE_CONTENT_PATTERN

                violations.append(SecurityViolation(
                    violation_type=vtype,
                    severity=SecurityRiskLevel.CRITICAL,
                    message=f"Dangerous pattern '{pattern}' found in {schema_name}",
                    context={"schema_name": schema_name, "pattern": pattern},
                ))

        return violations

    # ── Internal: Dependency Trust ────────────────────────────────────────

    def _check_dependency_trust(
        self,
        dependencies: list[str],
    ) -> list[SecurityViolation]:
        """检查依赖是否在信任列表中。"""
        violations: list[SecurityViolation] = []

        if self.allow_untrusted:
            return violations

        for dep in dependencies:
            # Empty trust list + not allowing untrusted = deny all
            if self.trusted_deps and dep not in self.trusted_deps:
                violations.append(SecurityViolation(
                    violation_type=SecurityViolationType.UNTRUSTED_DEPENDENCY,
                    severity=SecurityRiskLevel.HIGH,
                    message=f"Dependency '{dep}' is not in trusted list",
                    context={"dependency": dep, "trusted": sorted(self.trusted_deps)},
                ))
            elif not self.trusted_deps and dep:
                # No trust list configured = deny by default
                violations.append(SecurityViolation(
                    violation_type=SecurityViolationType.UNTRUSTED_DEPENDENCY,
                    severity=SecurityRiskLevel.MEDIUM,
                    message=f"No trust list configured; dependency '{dep}' unverified",
                    context={"dependency": dep},
                ))

        return violations

    # ── Internal: Risk Consistency ────────────────────────────────────────

    def _check_risk_consistency(
        self,
        config: UnifiedSkillConfig,
    ) -> list[SecurityViolation]:
        """检查风险等级与其他字段的一致性。"""
        violations: list[SecurityViolation] = []

        # High-risk skills with many dependencies should be critical
        if (
            config.risk_level == "high"
            and len(config.dependencies) > 5
        ):
            violations.append(SecurityViolation(
                violation_type=SecurityViolationType.RISK_LEVEL_ESCALATION,
                severity=SecurityRiskLevel.HIGH,
                message=(
                    f"Skill has {len(config.dependencies)} dependencies "
                    f"but risk_level='high'; consider 'critical'"
                ),
                context={
                    "risk_level": config.risk_level,
                    "dependency_count": len(config.dependencies),
                },
            ))

        return violations

    # ── Internal: Exporters ───────────────────────────────────────────────

    def _export_skillos(self, skir: SKIR) -> dict[str, Any]:
        """导出为 SkillOS 格式。"""
        return {
            "name": skir.skill_name,
            "description": skir.description,
            "version": skir.skill_version,
            "author": skir.author,
            "lifecycle_state": skir.lifecycle_state,
            "domain_tags": skir.domain_tags,
            "skcc_source": {
                "input_schema": skir.input_schema,
                "output_schema": skir.output_schema,
                "dependencies": skir.dependencies,
                "risk_level": skir.risk_level,
                "content_hash": skir.content_hash,
            },
            "task_group_id": skir.export_hints.get("skillos", {}).get(
                "needs_task_group", True
            ),
        }

    def _export_openskill(self, skir: SKIR) -> dict[str, Any]:
        """导出为 OpenSkill 格式。"""
        return {
            "name": skir.skill_name,
            "description": skir.description,
            "version": skir.skill_version,
            "author": skir.author,
            "license": "MIT",
            "source_traceability": {
                "upstream_framework": "skcc",
                "content_hash": skir.content_hash,
                "compilation_version": skir.version,
            },
            "composition_contract": {
                "input_schema": skir.input_schema,
                "output_schema": skir.output_schema,
            },
            "dependencies": skir.dependencies,
            "tags": skir.domain_tags,
        }

    def _export_skillopt(self, skir: SKIR) -> dict[str, Any]:
        """导出为 SkillOpt 格式。"""
        return {
            "name": skir.skill_name,
            "version": skir.skill_version,
            "content_hash": skir.content_hash,
            "optimization_target": {
                "schema_compatible": bool(skir.input_schema or skir.output_schema),
                "has_timeout": skir.timeout_seconds is not None,
                "has_fallback": skir.fallback_skill is not None,
                "dependency_count": len(skir.dependencies),
            },
            "risk_level": skir.risk_level,
            "lifecycle_state": skir.lifecycle_state,
        }

    def _export_meta_skills(self, skir: SKIR) -> dict[str, Any]:
        """导出为 meta-skill 编排格式。"""
        return {
            "name": skir.skill_name,
            "description": skir.description,
            "version": skir.skill_version,
            "orchestration": {
                "managed_skills": skir.dependencies,
                "composition_rules": skir.composes_with,
                "max_concurrency": 1,
                "abort_on_failure": True,
            },
            "contract": {
                "input_schema": skir.input_schema,
                "output_schema": skir.output_schema,
                "timeout_seconds": skir.timeout_seconds,
            },
            "depth": skir.composition_depth,
        }

    def _export_unified(self, skir: SKIR) -> dict[str, Any]:
        """导出为统一格式（完整 SKIR dict）。"""
        return skir.to_dict()
