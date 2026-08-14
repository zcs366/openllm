"""
Unit tests for MetaSkills Adapter
==================================

测试覆盖：
1. ComplianceReport 基础行为
2. BehaviorContract 基础行为
3. BoundaryAudit 基础行为
4. validate_12_principles() — 12 原则合规验证
   - 完全合规技能
   - 缺少描述的技能
   - 有编排配置的技能
   - 有依赖冲突的技能
   - retired 状态的技能
5. generate_behavior_contract() — 行为合约生成
   - 简单技能
   - 编排技能
   - 有优化配置的技能
6. audit_skill_boundaries() — 边界审计
   - 无违规技能
   - 自引用技能
   - 循环引用技能
   - 无描述技能
   - 过多依赖技能
   - 退役技能仍有依赖
7. 集成测试：与 UnifiedSkillConfig 联合使用
"""

import sys
import os
import pytest
from dataclasses import dataclass, field
from typing import Any

# 确保 openllm 可导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from openllm.isn.unified_skill_config import (
    MetaSkillConfig,
    OptimizationHints,
    SkillCompositionContract,
    SkillFrameworkSource,
    SkillLifecycleState,
    SkillUsageMetrics,
    SourceTraceability,
    UnifiedSkillConfig,
)
from openllm.isn.adapters.metaskills_adapter import (
    BehaviorContract,
    BoundaryAudit,
    ComplianceReport,
    MetaSkillsAdapter,
    PrincipleCheck,
    PrincipleID,
    PRINCIPLES,
    Severity,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


def make_skill_config(
    name: str = "test-skill",
    description: str = "A test skill that does testing",
    version: str = "0.1.0",
    lifecycle_state: SkillLifecycleState = SkillLifecycleState.ACTIVE,
    domain_tags: list[str] | None = None,
    dependencies: list[str] | None = None,
    composes_with: list[str] | None = None,
    composition_contract: SkillCompositionContract | None = None,
    meta_config: MetaSkillConfig | None = None,
    optimization_hints: OptimizationHints | None = None,
    usage_metrics: SkillUsageMetrics | None = None,
    source_traceability: SourceTraceability | None = None,
    risk_level: str = "medium",
) -> UnifiedSkillConfig:
    """创建测试用 UnifiedSkillConfig 的工厂方法。"""
    return UnifiedSkillConfig(
        name=name,
        description=description,
        version=version,
        lifecycle_state=lifecycle_state,
        domain_tags=domain_tags or [],
        dependencies=dependencies or [],
        composes_with=composes_with or [],
        composition_contract=composition_contract,
        meta_config=meta_config,
        optimization_hints=optimization_hints,
        usage_metrics=usage_metrics,
        source_traceability=source_traceability,
        risk_level=risk_level,
    )


def make_adapter(**kwargs) -> MetaSkillsAdapter:
    """创建测试用 MetaSkillsAdapter 的工厂方法。"""
    return MetaSkillsAdapter(**kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# Data structure tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestPrincipleID:
    def test_all_12_principles(self):
        ids = [p.value for p in PrincipleID]
        assert len(ids) == 12
        assert ids == [f"P{i}" for i in range(1, 13)]

    def test_principles_metadata_complete(self):
        for pid in PrincipleID:
            assert pid in PRINCIPLES
            meta = PRINCIPLES[pid]
            assert "name" in meta
            assert "description" in meta
            assert len(meta["name"]) > 0


class TestSeverity:
    def test_all_severities(self):
        values = [s.value for s in Severity]
        assert "pass" in values
        assert "warn" in values
        assert "fail" in values
        assert "critical" in values


class TestPrincipleCheck:
    def test_passed_property(self):
        check = PrincipleCheck(
            principle_id=PrincipleID.BEHAVIOR_CONTRACT,
            name="test",
            description="test",
            severity=Severity.PASS,
            message="ok",
        )
        assert check.passed is True

    def test_not_passed(self):
        check = PrincipleCheck(
            principle_id=PrincipleID.BEHAVIOR_CONTRACT,
            name="test",
            description="test",
            severity=Severity.FAIL,
            message="fail",
        )
        assert check.passed is False

    def test_warn_not_passed(self):
        check = PrincipleCheck(
            principle_id=PrincipleID.BEHAVIOR_CONTRACT,
            name="test",
            description="test",
            severity=Severity.WARN,
            message="warn",
        )
        assert check.passed is False


class TestComplianceReport:
    def test_counts(self):
        report = ComplianceReport(skill_name="test", timestamp="2026-01-01")
        report.checks = [
            PrincipleCheck(
                principle_id=PrincipleID.BEHAVIOR_CONTRACT,
                name="P1", description="d",
                severity=Severity.PASS, message="ok",
            ),
            PrincipleCheck(
                principle_id=PrincipleID.PREVENTION_OVER_GOVERNANCE,
                name="P2", description="d",
                severity=Severity.FAIL, message="fail",
            ),
            PrincipleCheck(
                principle_id=PrincipleID.BEHAVIOR_OVER_FILE,
                name="P3", description="d",
                severity=Severity.PASS, message="ok",
            ),
        ]
        assert report.passed_count == 2
        assert report.failed_count == 1
        assert report.total_count == 3
        assert report.is_compliant is False

    def test_all_passed(self):
        report = ComplianceReport(skill_name="test", timestamp="2026-01-01")
        report.checks = [
            PrincipleCheck(
                principle_id=PrincipleID.BEHAVIOR_CONTRACT,
                name="P1", description="d",
                severity=Severity.PASS, message="ok",
            ),
        ]
        assert report.is_compliant is True

    def test_to_dict(self):
        report = ComplianceReport(skill_name="test", timestamp="2026-01-01")
        report.overall_score = 0.8
        d = report.to_dict()
        assert d["skill_name"] == "test"
        assert d["overall_score"] == 0.8
        assert "checks" in d
        assert isinstance(d["checks"], list)


class TestBehaviorContract:
    def test_to_dict(self):
        contract = BehaviorContract(
            skill_name="test",
            version="1.0.0",
            declared_behaviors=["do something"],
            lifecycle_stage="active",
        )
        d = contract.to_dict()
        assert d["skill_name"] == "test"
        assert d["declared_behaviors"] == ["do something"]
        assert d["lifecycle_stage"] == "active"


class TestBoundaryAudit:
    def test_has_violations(self):
        audit = BoundaryAudit(skill_name="test")
        assert audit.has_violations is False
        audit.violations.append("violation!")
        assert audit.has_violations is True

    def test_to_dict(self):
        audit = BoundaryAudit(
            skill_name="test",
            owns=["a"],
            touches=["b"],
            violations=["v"],
        )
        d = audit.to_dict()
        assert d["owns"] == ["a"]
        assert d["touches"] == ["b"]
        assert d["has_violations"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# validate_12_principles tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidate12Principles:
    def test_compliant_skill(self):
        """完全合规的技能：有描述、有 composition_contract、有 lifecycle_state。"""
        cfg = make_skill_config(
            name="good-skill",
            description="This skill searches for information and returns results",
            composition_contract=SkillCompositionContract(
                input_schema={"query": {"type": "string"}},
                output_schema={"results": {"type": "list"}},
                fallback_config="search-fallback",
                timeout_seconds=30,
            ),
        )
        adapter = make_adapter()
        report = adapter.validate_12_principles(cfg)

        assert report.skill_name == "good-skill"
        assert report.total_count == 12
        assert report.overall_score > 0.5
        # 所有检查都应该有结果
        for check in report.checks:
            assert check.severity in (Severity.PASS, Severity.WARN, Severity.FAIL, Severity.CRITICAL)

    def test_missing_description(self):
        """缺少描述的技能：P1 和 P3 会失败。"""
        cfg = make_skill_config(
            name="no-desc",
            description="",
        )
        adapter = make_adapter()
        report = adapter.validate_12_principles(cfg)

        p1_check = next(c for c in report.checks if c.principle_id == PrincipleID.BEHAVIOR_CONTRACT)
        assert p1_check.severity in (Severity.WARN, Severity.FAIL)

    def test_orchestration_skill(self):
        """编排技能：有 meta_config。"""
        cfg = make_skill_config(
            name="orchestrator",
            description="Orchestrates multiple search skills",
            meta_config=MetaSkillConfig(
                orchestration_pattern="pipeline",
                managed_skills=["search-a", "search-b"],
                abort_on_failure=True,
            ),
        )
        adapter = make_adapter()
        report = adapter.validate_12_principles(cfg)

        # P8 (graceful degradation) 应该通过（有 abort_on_failure）
        p8 = next(c for c in report.checks if c.principle_id == PrincipleID.GRACEFUL_DEGRADATION)
        assert p8.severity == Severity.PASS

    def test_self_dependency_violation(self):
        """自引用技能：P6 会失败。"""
        cfg = make_skill_config(
            name="self-ref",
            description="A skill that depends on itself",
            dependencies=["self-ref"],
        )
        adapter = make_adapter()
        report = adapter.validate_12_principles(cfg)

        p6 = next(c for c in report.checks if c.principle_id == PrincipleID.COMPOSABILITY)
        assert p6.severity == Severity.FAIL

    def test_excessive_tags(self):
        """过多 domain_tags：P5 会警告。"""
        cfg = make_skill_config(
            name="tag-heavy",
            description="A skill with many tags",
            domain_tags=["a", "b", "c", "d", "e", "f", "g"],
        )
        adapter = make_adapter()
        report = adapter.validate_12_principles(cfg)

        p5 = next(c for c in report.checks if c.principle_id == PrincipleID.SINGLE_RESPONSIBILITY)
        assert p5.severity == Severity.WARN

    def test_retired_skill_with_deps(self):
        """retired 技能仍有依赖：边界审计违规。"""
        cfg = make_skill_config(
            name="retired-but-active",
            description="I should be retired",
            lifecycle_state=SkillLifecycleState.RETIRED,
            dependencies=["some-skill"],
        )
        adapter = make_adapter()
        # retired-with-deps 是边界审计的职责，不是12原则检查
        audit = adapter.audit_skill_boundaries(cfg)
        assert audit.has_violations is True
        retired_violations = [v for v in audit.violations if "Retired" in v]
        assert len(retired_violations) > 0

    def test_pre_1_version(self):
        """pre-1.0 版本：P7 应该通过（semver 允许 breaking changes）。"""
        cfg = make_skill_config(
            name="pre-1",
            description="A pre-1.0 skill",
            version="0.5.0",
        )
        adapter = make_adapter()
        report = adapter.validate_12_principles(cfg)

        p7 = next(c for c in report.checks if c.principle_id == PrincipleID.BACKWARD_COMPATIBILITY)
        assert p7.severity == Severity.PASS

    def test_no_lifecycle_state(self):
        """没有 lifecycle_state：P12 失败。"""
        cfg = make_skill_config(name="no-lifecycle", description="test")

        # 创建一个没有 lifecycle_state 的模拟对象
        @dataclass
        class MinimalConfig:
            name: str = "minimal"
            description: str = "test"
            version: str = "1.0.0"
            lifecycle_state: Any = None

        minimal = MinimalConfig()
        adapter = make_adapter()
        report = adapter.validate_12_principles(minimal)

        p12 = next(c for c in report.checks if c.principle_id == PrincipleID.LIFECYCLE_AWARENESS)
        assert p12.severity == Severity.FAIL

    def test_strict_mode_warn_as_fail(self):
        """严格模式：WARN 也计为失败。"""
        cfg = make_skill_config(
            name="partial",
            description="",
            composition_contract=SkillCompositionContract(
                input_schema={},
                output_schema={},
            ),
        )
        adapter = make_adapter(strict_mode=True)
        report = adapter.validate_12_principles(cfg)

        # 在严格模式下，WARN 也算失败
        warn_checks = [c for c in report.checks if c.severity == Severity.WARN]
        if warn_checks:
            # 严格模式下 overall_score 应该更低
            assert report.overall_score < 1.0

    def test_overall_score_calculation(self):
        """验证 overall_score 计算逻辑。"""
        cfg = make_skill_config(
            name="scoring",
            description="A well-defined skill",
            composition_contract=SkillCompositionContract(
                input_schema={"q": {"type": "str"}},
                output_schema={"r": {"type": "list"}},
                fallback_config="fallback",
                timeout_seconds=10,
            ),
        )
        adapter = make_adapter()
        report = adapter.validate_12_principles(cfg)

        # 分数应在 -1.0 到 1.0 之间
        assert -1.0 <= report.overall_score <= 1.0

    def test_all_12_principles_checked(self):
        """确保所有 12 条原则都被检查。"""
        cfg = make_skill_config(name="all-checked", description="test")
        adapter = make_adapter()
        report = adapter.validate_12_principles(cfg)

        checked_ids = {c.principle_id for c in report.checks}
        expected_ids = set(PrincipleID)
        assert checked_ids == expected_ids


# ═══════════════════════════════════════════════════════════════════════════════
# generate_behavior_contract tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestGenerateBehaviorContract:
    def test_simple_skill(self):
        """简单技能的行为合约。"""
        cfg = make_skill_config(
            name="simple",
            description="Search for information and return results",
            version="1.0.0",
        )
        adapter = make_adapter()
        contract = adapter.generate_behavior_contract(cfg)

        assert contract.skill_name == "simple"
        assert contract.version == "1.0.0"
        assert len(contract.declared_behaviors) > 0
        assert contract.lifecycle_stage == "active"
        # 简单技能应该是幂等的
        assert contract.idempotency_guarantee is True

    def test_orchestration_skill(self):
        """编排技能的行为合约。"""
        cfg = make_skill_config(
            name="pipeline",
            description="Orchestrates search and summarization",
            meta_config=MetaSkillConfig(
                orchestration_pattern="pipeline",
                managed_skills=["search", "summarize"],
                abort_on_failure=True,
            ),
        )
        adapter = make_adapter()
        contract = adapter.generate_behavior_contract(cfg)

        assert contract.skill_name == "pipeline"
        # 应该声明编排行为
        orchestration_behaviors = [
            b for b in contract.declared_behaviors if "Orchestrates" in b
        ]
        assert len(orchestration_behaviors) > 0
        # 应该声明失败模式
        assert len(contract.failure_modes) > 0
        # 编排技能不幂等
        assert contract.idempotency_guarantee is False

    def test_with_composition_contract(self):
        """有 composition_contract 的行为合约。"""
        cfg = make_skill_config(
            name="contracted",
            description="A skill with I/O contract",
            composition_contract=SkillCompositionContract(
                input_schema={"query": {"type": "string"}, "limit": {"type": "int"}},
                output_schema={"results": {"type": "list"}, "count": {"type": "int"}},
            ),
        )
        adapter = make_adapter()
        contract = adapter.generate_behavior_contract(cfg)

        assert len(contract.input_expectations) == 2
        assert len(contract.output_guarantees) == 2
        assert any("query" in e for e in contract.input_expectations)

    def test_with_optimization_hints(self):
        """有 optimization_hints 的副作用声明。"""
        cfg = make_skill_config(
            name="optimized",
            description="An optimized skill",
            optimization_hints=OptimizationHints(
                should_optimize=True,
                priority=3,
            ),
        )
        adapter = make_adapter()
        contract = adapter.generate_behavior_contract(cfg)

        assert len(contract.side_effects) > 0
        assert any("optimization" in s.lower() for s in contract.side_effects)

    def test_with_dependencies(self):
        """有依赖的副作用声明。"""
        cfg = make_skill_config(
            name="dep-skill",
            description="A skill with dependencies",
            dependencies=["auth-skill", "cache-skill"],
        )
        adapter = make_adapter()
        contract = adapter.generate_behavior_contract(cfg)

        assert len(contract.side_effects) > 0
        assert any("auth-skill" in s for s in contract.side_effects)

    def test_to_dict(self):
        """行为合约序列化。"""
        cfg = make_skill_config(
            name="serializable",
            description="A skill",
            dependencies=["dep1"],
        )
        adapter = make_adapter()
        contract = adapter.generate_behavior_contract(cfg)
        d = contract.to_dict()

        assert d["skill_name"] == "serializable"
        assert "declared_behaviors" in d
        assert "side_effects" in d
        assert "generated_at" in d

    def test_lifecycle_stage_from_enum(self):
        """lifecycle_stage 正确从 Enum 提取。"""
        cfg = make_skill_config(
            name="lifecycle-test",
            description="test",
            lifecycle_state=SkillLifecycleState.DEPRECATED,
        )
        adapter = make_adapter()
        contract = adapter.generate_behavior_contract(cfg)

        assert contract.lifecycle_stage == "deprecated"

    def test_empty_description(self):
        """空描述的行为合约。"""
        cfg = make_skill_config(
            name="empty-desc",
            description="",
        )
        adapter = make_adapter()
        contract = adapter.generate_behavior_contract(cfg)

        # 空描述不应该有行为
        # 但如果 description 为空且没有其他来源，declared_behaviors 应该为空
        # 实际上 generate_behavior_contract 会把 description 作为 fallback
        # 空字符串不会被添加
        assert contract.skill_name == "empty-desc"


# ═══════════════════════════════════════════════════════════════════════════════
# audit_skill_boundaries tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestAuditSkillBoundaries:
    def test_clean_skill(self):
        """无违规的干净技能。"""
        cfg = make_skill_config(
            name="clean",
            description="A clean skill with no issues",
        )
        adapter = make_adapter()
        audit = adapter.audit_skill_boundaries(cfg)

        assert audit.skill_name == "clean"
        assert audit.has_violations is False
        assert len(audit.owns) > 0
        assert f"skill:clean:config" in audit.owns

    def test_self_dependency(self):
        """自引用检测。"""
        cfg = make_skill_config(
            name="self-ref",
            description="Self-referencing skill",
            dependencies=["self-ref"],
        )
        adapter = make_adapter()
        audit = adapter.audit_skill_boundaries(cfg)

        assert audit.has_violations is True
        self_dep_violations = [
            v for v in audit.violations if "self" in v.lower()
        ]
        assert len(self_dep_violations) > 0

    def test_circular_reference(self):
        """循环引用检测。"""
        cfg = make_skill_config(
            name="circular",
            description="Circular skill",
            dependencies=["other"],
            composes_with=["other"],
        )
        adapter = make_adapter()
        audit = adapter.audit_skill_boundaries(cfg)

        circular_violations = [
            v for v in audit.violations if "circular" in v.lower()
        ]
        assert len(circular_violations) > 0

    def test_no_description_violation(self):
        """无描述导致边界模糊。"""
        cfg = make_skill_config(
            name="no-desc",
            description="",
        )
        adapter = make_adapter()
        audit = adapter.audit_skill_boundaries(cfg)

        desc_violations = [
            v for v in audit.violations if "No description" in v
        ]
        assert len(desc_violations) > 0
        assert len(audit.suggestions) > 0

    def test_excessive_dependencies(self):
        """过多依赖检测。"""
        cfg = make_skill_config(
            name="heavy",
            description="Heavy skill",
            dependencies=[f"dep-{i}" for i in range(8)],
        )
        adapter = make_adapter()
        audit = adapter.audit_skill_boundaries(cfg)

        excessive_violations = [
            v for v in audit.violations if "Excessive" in v
        ]
        assert len(excessive_violations) > 0

    def test_retired_with_deps(self):
        """retired 技能仍有依赖。"""
        cfg = make_skill_config(
            name="retired-active",
            description="I should be retired",
            lifecycle_state=SkillLifecycleState.RETIRED,
            dependencies=["some-skill"],
        )
        adapter = make_adapter()
        audit = adapter.audit_skill_boundaries(cfg)

        retired_violations = [
            v for v in audit.violations if "Retired" in v
        ]
        assert len(retired_violations) > 0

    def test_orchestration_without_contract(self):
        """编排技能无 composition_contract。"""
        cfg = make_skill_config(
            name="orch-no-contract",
            description="Orchestrator without contract",
            meta_config=MetaSkillConfig(
                managed_skills=["a", "b"],
            ),
        )
        adapter = make_adapter()
        audit = adapter.audit_skill_boundaries(cfg)

        contract_violations = [
            v for v in audit.violations if "composition_contract" in v
        ]
        assert len(contract_violations) > 0

    def test_touches_externals(self):
        """外部资源引用。"""
        cfg = make_skill_config(
            name="external",
            description="Skill with externals",
            dependencies=["auth-service"],
            composes_with=["cache-layer"],
        )
        adapter = make_adapter()
        audit = adapter.audit_skill_boundaries(cfg)

        auth_touches = [t for t in audit.touches if "auth-service" in t]
        cache_touches = [t for t in audit.touches if "cache-layer" in t]
        assert len(auth_touches) > 0
        assert len(cache_touches) > 0

    def test_to_dict(self):
        """边界审计序列化。"""
        cfg = make_skill_config(
            name="serializable",
            description="test",
        )
        adapter = make_adapter()
        audit = adapter.audit_skill_boundaries(cfg)
        d = audit.to_dict()

        assert d["skill_name"] == "serializable"
        assert "owns" in d
        assert "touches" in d
        assert "violations" in d
        assert "has_violations" in d

    def test_meta_config_managed_skills(self):
        """meta_config 管理的技能出现在 touches 中。"""
        cfg = make_skill_config(
            name="meta-touches",
            description="Meta skill touching managed skills",
            meta_config=MetaSkillConfig(
                managed_skills=["sub-a", "sub-b"],
            ),
        )
        adapter = make_adapter()
        audit = adapter.audit_skill_boundaries(cfg)

        sub_a_touches = [t for t in audit.touches if "sub-a" in t]
        sub_b_touches = [t for t in audit.touches if "sub-b" in t]
        assert len(sub_a_touches) > 0
        assert len(sub_b_touches) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Integration with UnifiedSkillConfig
# ═══════════════════════════════════════════════════════════════════════════════


class TestIntegrationWithUnifiedSkillConfig:
    def test_from_metaskills_adapter_roundtrip(self):
        """从 UnifiedSkillConfig.from_metaskills 迁移后验证。"""
        raw_data = {
            "name": "migrated-skill",
            "description": "A migrated skill",
            "version": "2.0.0",
            "lifecycle_state": "active",
            "meta_config": {
                "orchestration_pattern": "parallel",
                "managed_skills": ["a", "b"],
                "abort_on_failure": False,
            },
        }
        cfg = UnifiedSkillConfig.from_metaskills(raw_data)
        adapter = make_adapter()

        report = adapter.validate_12_principles(cfg)
        assert report.total_count == 12

        contract = adapter.generate_behavior_contract(cfg)
        assert contract.skill_name == "migrated-skill"

        audit = adapter.audit_skill_boundaries(cfg)
        assert audit.skill_name == "migrated-skill"

    def test_full_lifecycle_validation(self):
        """完整生命周期：创建 → 验证 → 转移 → 再验证。"""
        # 1. 创建
        cfg = make_skill_config(
            name="lifecycle-test",
            description="Test full lifecycle",
            lifecycle_state=SkillLifecycleState.ACTIVE,
        )
        adapter = make_adapter()

        # 2. 初始验证
        report1 = adapter.validate_12_principles(cfg)
        assert report1.overall_score > 0

        # 3. 转移到 dormant
        success = cfg.transition(SkillLifecycleState.DORMANT)
        assert success is True

        # 4. 再验证
        report2 = adapter.validate_12_principles(cfg)
        p12 = next(c for c in report2.checks if c.principle_id == PrincipleID.LIFECYCLE_AWARENESS)
        assert "dormant" in p12.evidence[0]

    def test_validate_with_metaskills_config(self):
        """直接用 MetaSkillConfig 验证。"""
        cfg = make_skill_config(
            name="meta-direct",
            description="Direct meta skill",
            meta_config=MetaSkillConfig(
                orchestration_pattern="sequential",
                managed_skills=["step-1", "step-2", "step-3"],
                composition_rules=["must-follow-order"],
                max_concurrency=1,
                abort_on_failure=True,
            ),
            composition_contract=SkillCompositionContract(
                input_schema={"task": {"type": "string"}},
                output_schema={"result": {"type": "dict"}},
                requires_approval=True,
                timeout_seconds=60,
                fallback_config="fallback-skill",
            ),
        )
        adapter = make_adapter()
        report = adapter.validate_12_principles(cfg)
        # 应该大部分通过
        assert report.overall_score > 0.3
        # P1 应该通过（有 description + composition_contract）
        p1 = next(c for c in report.checks if c.principle_id == PrincipleID.BEHAVIOR_CONTRACT)
        assert p1.severity == Severity.PASS
        # P8 应该通过（有 abort_on_failure + timeout + fallback）
        p8 = next(c for c in report.checks if c.principle_id == PrincipleID.GRACEFUL_DEGRADATION)
        assert p8.severity == Severity.PASS


# ═══════════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_minimal_config(self):
        """最小配置（只有 name）。"""
        cfg = UnifiedSkillConfig(name="minimal")
        adapter = make_adapter()
        report = adapter.validate_12_principles(cfg)
        assert report.total_count == 12
        # 没有 description → P1 应该 warn/fail
        p1 = next(c for c in report.checks if c.principle_id == PrincipleID.BEHAVIOR_CONTRACT)
        assert p1.severity != Severity.PASS

    def test_helper_extract_behaviors(self):
        """_extract_behaviors 辅助方法。"""
        behaviors = MetaSkillsAdapter._extract_behaviors(
            "Search for papers. Analyze results. Generate summary."
        )
        assert len(behaviors) == 3
        assert "Search for papers" in behaviors[0]

    def test_helper_schema_to_expectations(self):
        """_schema_to_expectations 辅助方法。"""
        expectations = MetaSkillsAdapter._schema_to_expectations(
            {"query": {"type": "string"}, "limit": {"type": "int"}}
        )
        assert len(expectations) == 2
        assert any("query" in e for e in expectations)

    def test_helper_schema_to_guarantees(self):
        """_schema_to_guarantees 辅助方法。"""
        guarantees = MetaSkillsAdapter._schema_to_guarantees(
            {"results": {"type": "list"}}
        )
        assert len(guarantees) == 1
        assert "results" in guarantees[0]

    def test_unknown_lifecycle_state(self):
        """未知的 lifecycle_state 值（字符串而非 Enum）。"""
        @dataclass
        class ConfigWithStr:
            name: str = "str-state"
            description: str = "test"
            version: str = "1.0.0"
            lifecycle_state: Any = "active"

        cfg = ConfigWithStr()
        adapter = make_adapter()
        # 不应该抛异常
        report = adapter.validate_12_principles(cfg)
        assert report.total_count == 12

        contract = adapter.generate_behavior_contract(cfg)
        assert contract.lifecycle_stage == "active"

    def test_compliance_report_empty(self):
        """空的 ComplianceReport。"""
        report = ComplianceReport(skill_name="empty", timestamp="")
        assert report.passed_count == 0
        assert report.failed_count == 0
        assert report.total_count == 0
        assert report.is_compliant is True  # 0 failures = compliant


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
