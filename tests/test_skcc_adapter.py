"""
Unit tests for SkCC Adapter
============================

测试覆盖：
1. SKIR 数据结构（to_dict 格式、字段完整性）
2. SkCCAdapter.compile() — 核心编译流程
   - 正常编译 + 字段映射
   - 空 name 异常
   - 组合合约字段提取
   - 依赖 + composes_with 映射
   - content_hash 一致性
3. SkCCAdapter.validate_security() — 安全约束验证
   - 基础字段（name/version）
   - Schema 安全（eval/exec/危险模式）
   - 依赖信任验证
   - 组合深度超限
   - 高风险技能要求（approval/timeout/fallback）
   - 风险等级一致性
4. SkCCAdapter.export() — 框架导出
   - 5 个目标框架格式
   - 不支持的目标框架异常
5. SkCCAdapter.build_composition_graph() — 组合图
   - 正常图构建
   - 循环检测
6. Integration: compile → validate → export 全流程
"""

import sys
import os
import pytest

# Ensure openllm is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from openllm.isn.unified_skill_config import (
    SkillCompositionContract,
    SkillFrameworkSource,
    SkillLifecycleState,
    UnifiedSkillConfig,
)
from openllm.isn.adapters.skcc_adapter import (
    SKIR,
    SKIRVersion,
    ExportTarget,
    SecurityReport,
    SecurityRiskLevel,
    SecurityViolation,
    SecurityViolationType,
    SkCCAdapter,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

def make_config(
    name: str = "test-skill",
    description: str = "A test skill",
    version: str = "1.0.0",
    risk_level: str = "medium",
    dependencies: list[str] | None = None,
    composes_with: list[str] | None = None,
    with_contract: bool = True,
    contract_overrides: dict | None = None,
) -> UnifiedSkillConfig:
    """创建测试用 UnifiedSkillConfig。"""
    contract = None
    if with_contract:
        contract_data = {
            "input_schema": {"query": {"type": "string", "required": True}},
            "output_schema": {"results": {"type": "array"}},
            "requires_approval": False,
            "timeout_seconds": 30,
            "fallback_config": "web-search",
        }
        if contract_overrides:
            contract_data.update(contract_overrides)
        contract = SkillCompositionContract(**contract_data)

    return UnifiedSkillConfig(
        name=name,
        description=description,
        version=version,
        lifecycle_state=SkillLifecycleState.ACTIVE,
        source_framework=SkillFrameworkSource.SKCC,
        composition_contract=contract,
        dependencies=dependencies or [],
        composes_with=composes_with or [],
        risk_level=risk_level,
    )


def make_high_risk_config() -> UnifiedSkillConfig:
    """创建高风险技能配置。"""
    return make_config(
        name="critical-deploy",
        risk_level="critical",
        dependencies=["deploy-engine", "infra-manager", "config-store"],
        contract_overrides={
            "requires_approval": True,
            "timeout_seconds": 60,
            "fallback_config": "rollback-skill",
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SKIR Data Structure Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSKIR:
    def test_version(self):
        skir = SKIR()
        assert skir.version == SKIRVersion.V1.value

    def test_to_dict_structure(self):
        skir = SKIR(skill_name="my-skill", skill_version="2.0.0")
        d = skir.to_dict()

        assert "version" in d
        assert "identity" in d
        assert "contract" in d
        assert "composition_graph" in d
        assert "security" in d
        assert "lifecycle" in d
        assert "portability" in d
        assert "integrity" in d

    def test_to_dict_identity(self):
        skir = SKIR(
            skill_name="test",
            description="desc",
            skill_version="1.0",
            author="zcs",
            source_framework="skcc",
            domain_tags=["search", "web"],
        )
        d = skir.to_dict()
        identity = d["identity"]
        assert identity["name"] == "test"
        assert identity["description"] == "desc"
        assert identity["version"] == "1.0"
        assert identity["author"] == "zcs"
        assert identity["source_framework"] == "skcc"
        assert identity["domain_tags"] == ["search", "web"]

    def test_to_dict_contract(self):
        skir = SKIR(
            input_schema={"q": {"type": "string"}},
            output_schema={"r": {"type": "array"}},
            requires_approval=True,
            timeout_seconds=30,
            fallback_skill="backup",
        )
        d = skir.to_dict()
        contract = d["contract"]
        assert contract["input_schema"] == {"q": {"type": "string"}}
        assert contract["output_schema"] == {"r": {"type": "array"}}
        assert contract["requires_approval"] is True
        assert contract["timeout_seconds"] == 30
        assert contract["fallback_skill"] == "backup"

    def test_to_dict_security(self):
        skir = SKIR(
            risk_level="high",
            security_passed=True,
            security_violations=0,
        )
        d = skir.to_dict()
        sec = d["security"]
        assert sec["risk_level"] == "high"
        assert sec["passed"] is True
        assert sec["violations"] == 0

    def test_to_dict_lifecycle(self):
        skir = SKIR(lifecycle_state="dormant")
        d = skir.to_dict()
        assert d["lifecycle"]["state"] == "dormant"


# ═══════════════════════════════════════════════════════════════════════════════
# Compile Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSkCCAdapterCompile:
    def test_basic_compile(self):
        """正常编译：字段正确映射到 SKIR。"""
        adapter = SkCCAdapter(allow_untrusted=True)
        config = make_config()
        skir = adapter.compile(config)

        assert skir.skill_name == "test-skill"
        assert skir.description == "A test skill"
        assert skir.skill_version == "1.0.0"
        assert skir.source_framework == "skcc"
        assert skir.lifecycle_state == "active"

    def test_compile_empty_name_raises(self):
        """空 name 应抛 ValueError。"""
        adapter = SkCCAdapter()
        config = make_config(name="")
        with pytest.raises(ValueError, match="empty name"):
            adapter.compile(config)

    def test_compile_contract_fields(self):
        """组合合约字段正确提取。"""
        adapter = SkCCAdapter(allow_untrusted=True)
        config = make_config(with_contract=True)
        skir = adapter.compile(config)

        assert skir.input_schema == {"query": {"type": "string", "required": True}}
        assert skir.output_schema == {"results": {"type": "array"}}
        assert skir.requires_approval is False
        assert skir.timeout_seconds == 30
        assert skir.fallback_skill == "web-search"

    def test_compile_no_contract(self):
        """没有合约时，字段为默认值。"""
        adapter = SkCCAdapter(allow_untrusted=True)
        config = make_config(with_contract=False)
        skir = adapter.compile(config)

        assert skir.input_schema == {}
        assert skir.output_schema == {}
        assert skir.requires_approval is False
        assert skir.timeout_seconds is None
        assert skir.fallback_skill is None

    def test_compile_dependencies(self):
        """依赖和 composes_with 正确映射。"""
        adapter = SkCCAdapter(allow_untrusted=True)
        config = make_config(
            dependencies=["dep-a", "dep-b"],
            composes_with=["comp-x"],
        )
        skir = adapter.compile(config)

        assert skir.dependencies == ["dep-a", "dep-b"]
        assert skir.composes_with == ["comp-x"]

    def test_compile_content_hash_deterministic(self):
        """同一 config 产生相同的 content_hash。"""
        adapter = SkCCAdapter(allow_untrusted=True)
        config = make_config()
        skir1 = adapter.compile(config)
        skir2 = adapter.compile(config)
        assert skir1.content_hash == skir2.content_hash

    def test_compile_content_hash_varies_with_name(self):
        """不同 name 产生不同的 content_hash。"""
        adapter = SkCCAdapter(allow_untrusted=True)
        skir1 = adapter.compile(make_config(name="skill-a"))
        skir2 = adapter.compile(make_config(name="skill-b"))
        assert skir1.content_hash != skir2.content_hash

    def test_compile_risk_level(self):
        """风险等级正确传递。"""
        adapter = SkCCAdapter(allow_untrusted=True)
        skir = adapter.compile(make_config(risk_level="critical"))
        assert skir.risk_level == "critical"

    def test_compile_domain_tags(self):
        """domain_tags 正确映射。"""
        adapter = SkCCAdapter(allow_untrusted=True)
        config = make_config()
        config.domain_tags = ["search", "fallback"]
        skir = adapter.compile(config)
        assert skir.domain_tags == ["search", "fallback"]

    def test_compile_security_auto_runs(self):
        """编译时自动运行安全验证。"""
        adapter = SkCCAdapter(allow_untrusted=True)
        skir = adapter.compile(make_config())
        # Medium-risk skill with valid contract should pass
        assert skir.security_passed is True
        assert skir.security_violations == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Security Validation Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSkCCAdapterSecurity:
    def test_pass_valid_config(self):
        """合法配置通过安全验证。"""
        adapter = SkCCAdapter(allow_untrusted=True)
        config = make_config(
            contract_overrides={
                "requires_approval": False,
                "timeout_seconds": 30,
                "fallback_config": "backup",
            }
        )
        report = adapter.validate_security(config)
        assert report.passed is True
        assert report.violation_count == 0
        assert report.checked_constraints > 0

    def test_fail_empty_name(self):
        """空 name 触发 CRITICAL 违规。"""
        adapter = SkCCAdapter()
        config = make_config(name="")
        report = adapter.validate_security(config)
        assert report.passed is False
        assert any(
            v.violation_type == SecurityViolationType.EMPTY_NAME
            for v in report.violations
        )

    def test_fail_empty_version(self):
        """空 version 触发 HIGH 违规。"""
        adapter = SkCCAdapter()
        config = make_config(version="")
        report = adapter.validate_security(config)
        assert report.passed is False
        assert any(
            v.violation_type == SecurityViolationType.INVALID_VERSION
            for v in report.violations
        )

    def test_fail_eval_in_schema(self):
        """input_schema 中的 eval() 触发 CRITICAL 违规。"""
        adapter = SkCCAdapter(allow_untrusted=True)
        config = make_config(
            contract_overrides={
                "input_schema": {"code": {"type": "string", "example": "eval('1+1')"}},
            }
        )
        report = adapter.validate_security(config)
        assert report.passed is False
        assert any(
            v.violation_type == SecurityViolationType.EVAL_IN_SCHEMA
            for v in report.violations
        )

    def test_fail_exec_in_schema(self):
        """output_schema 中的 exec() 触发 CRITICAL 违规。"""
        adapter = SkCCAdapter(allow_untrusted=True)
        config = make_config(
            contract_overrides={
                "output_schema": {"result": {"type": "string", "note": "exec(os.system('ls'))"}},
            }
        )
        report = adapter.validate_security(config)
        assert report.passed is False
        assert any(
            v.violation_type == SecurityViolationType.EXEC_IN_SCHEMA
            for v in report.violations
        )

    def test_fail_subprocess_in_schema(self):
        """schema 中的 subprocess. 触发 CRITICAL 违规。"""
        adapter = SkCCAdapter(allow_untrusted=True)
        config = make_config(
            contract_overrides={
                "input_schema": {"cmd": {"type": "string", "default": "subprocess.run"}},
            }
        )
        report = adapter.validate_security(config)
        assert report.passed is False
        assert any(
            v.violation_type == SecurityViolationType.UNSAFE_CONTENT_PATTERN
            for v in report.violations
        )

    def test_fail_untrusted_dependency(self):
        """未信任依赖触发 HIGH 违规。"""
        adapter = SkCCAdapter(trusted_deps={"web-search"})
        config = make_config(dependencies=["web-search", "unknown-dep"])
        report = adapter.validate_security(config)
        assert report.passed is False
        untrusted = [
            v for v in report.violations
            if v.violation_type == SecurityViolationType.UNTRUSTED_DEPENDENCY
        ]
        assert len(untrusted) == 1
        assert untrusted[0].context["dependency"] == "unknown-dep"

    def test_fail_no_trust_list(self):
        """无信任列表时所有依赖触发 MEDIUM 违规。"""
        adapter = SkCCAdapter()  # no trusted_deps, not allow_untrusted
        config = make_config(dependencies=["dep-a", "dep-b"])
        report = adapter.validate_security(config)
        # Both deps should be flagged
        untrusted = [
            v for v in report.violations
            if v.violation_type == SecurityViolationType.UNTRUSTED_DEPENDENCY
        ]
        assert len(untrusted) == 2

    def test_pass_trusted_dependency(self):
        """信任依赖不触发违规。"""
        adapter = SkCCAdapter(trusted_deps={"dep-a", "dep-b"})
        config = make_config(dependencies=["dep-a", "dep-b"])
        report = adapter.validate_security(config)
        untrusted = [
            v for v in report.violations
            if v.violation_type == SecurityViolationType.UNTRUSTED_DEPENDENCY
        ]
        assert len(untrusted) == 0

    def test_fail_depth_exceeded(self):
        """组合深度超限触发 HIGH 违规。"""
        # validate_security uses _compute_composition_depth(config.dependencies)
        # which only computes depth from direct deps (no known_skills).
        # depth=1 for config with deps=["a", "b", "c"].
        # So max_depth=0 should trigger the violation.
        adapter = SkCCAdapter(max_depth=0)
        config = make_config(dependencies=["a", "b", "c"])
        report = adapter.validate_security(config)
        depth_violations = [
            v for v in report.violations
            if v.violation_type == SecurityViolationType.DEPTH_EXCEEDED
        ]
        assert len(depth_violations) == 1

    def test_fail_critical_missing_approval(self):
        """critical 技能缺少 requires_approval 触发 HIGH 违规。"""
        adapter = SkCCAdapter(allow_untrusted=True)
        config = make_config(
            risk_level="critical",
            contract_overrides={"requires_approval": False},
        )
        report = adapter.validate_security(config)
        assert report.passed is False
        approval_v = [
            v for v in report.violations
            if v.violation_type == SecurityViolationType.APPROVAL_REQUIRED_BUT_MISSING
        ]
        assert len(approval_v) == 1

    def test_fail_critical_missing_timeout(self):
        """critical 技能缺少 timeout 触发 MEDIUM 违规。"""
        adapter = SkCCAdapter(allow_untrusted=True)
        config = make_config(
            risk_level="critical",
            contract_overrides={"timeout_seconds": None},
        )
        report = adapter.validate_security(config)
        timeout_v = [
            v for v in report.violations
            if v.violation_type == SecurityViolationType.MISSING_TIMEOUT
        ]
        assert len(timeout_v) == 1

    def test_fail_critical_missing_fallback(self):
        """critical 技能缺少 fallback 触发 MEDIUM 违规。"""
        adapter = SkCCAdapter(allow_untrusted=True)
        config = make_config(
            risk_level="critical",
            contract_overrides={"fallback_config": None},
        )
        report = adapter.validate_security(config)
        fb_v = [
            v for v in report.violations
            if v.violation_type == SecurityViolationType.MISSING_FALLBACK
        ]
        assert len(fb_v) == 1

    def test_fail_critical_no_contract(self):
        """critical 技能无合约触发 3 个违规（approval + timeout + fallback）。"""
        adapter = SkCCAdapter(allow_untrusted=True)
        config = make_config(risk_level="critical", with_contract=False)
        report = adapter.validate_security(config)
        assert report.passed is False
        assert report.violation_count >= 3

    def test_pass_critical_with_full_contract(self):
        """critical 技能有完整合约则通过。"""
        adapter = SkCCAdapter(allow_untrusted=True)
        config = make_high_risk_config()
        report = adapter.validate_security(config)
        # Should pass: approval=True, timeout=60, fallback=rollback-skill
        assert report.passed is True

    def test_fail_risk_level_escalation(self):
        """高风险+多依赖应建议升级到 critical。"""
        adapter = SkCCAdapter(allow_untrusted=True)
        config = make_config(
            risk_level="high",
            dependencies=["a", "b", "c", "d", "e", "f"],  # > 5 deps
            contract_overrides={
                "requires_approval": True,
                "timeout_seconds": 30,
                "fallback_config": "backup",
            },
        )
        report = adapter.validate_security(config)
        escalation = [
            v for v in report.violations
            if v.violation_type == SecurityViolationType.RISK_LEVEL_ESCALATION
        ]
        assert len(escalation) == 1

    def test_security_report_to_dict(self):
        """SecurityReport.to_dict() 格式正确。"""
        adapter = SkCCAdapter()
        config = make_config(name="")
        report = adapter.validate_security(config)
        d = report.to_dict()

        assert "passed" in d
        assert "violation_count" in d
        assert "critical_count" in d
        assert "checked_constraints" in d
        assert "risk_level" in d
        assert "violations" in d
        assert isinstance(d["violations"], list)
        assert d["passed"] is False

    def test_security_violation_to_dict(self):
        """SecurityViolation.to_dict() 格式正确。"""
        v = SecurityViolation(
            violation_type=SecurityViolationType.EMPTY_NAME,
            severity=SecurityRiskLevel.CRITICAL,
            message="test",
            context={"key": "value"},
        )
        d = v.to_dict()
        assert d["violation_type"] == "empty_name"
        assert d["severity"] == "critical"
        assert d["message"] == "test"
        assert d["context"]["key"] == "value"


# ═══════════════════════════════════════════════════════════════════════════════
# Export Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSkCCAdapterExport:
    def _compile(self, **kwargs) -> SKIR:
        adapter = SkCCAdapter(allow_untrusted=True)
        config = make_config(**kwargs)
        return adapter.compile(config)

    def test_export_skillos(self):
        skir = self._compile()
        adapter = SkCCAdapter()
        result = adapter.export(skir, target="skillos")

        assert "name" in result
        assert "skcc_source" in result
        assert result["skcc_source"]["input_schema"] == skir.input_schema
        assert result["skcc_source"]["risk_level"] == "medium"

    def test_export_openskill(self):
        skir = self._compile()
        adapter = SkCCAdapter()
        result = adapter.export(skir, target="openskill")

        assert "source_traceability" in result
        assert result["source_traceability"]["upstream_framework"] == "skcc"
        assert "composition_contract" in result
        assert result["license"] == "MIT"

    def test_export_skillopt(self):
        skir = self._compile()
        adapter = SkCCAdapter()
        result = adapter.export(skir, target="skillopt")

        assert "optimization_target" in result
        assert result["optimization_target"]["has_timeout"] is True
        assert result["optimization_target"]["has_fallback"] is True

    def test_export_meta_skills(self):
        skir = self._compile(dependencies=["dep-a"])
        adapter = SkCCAdapter()
        result = adapter.export(skir, target="meta_skills")

        assert "orchestration" in result
        assert result["orchestration"]["managed_skills"] == ["dep-a"]
        assert "contract" in result

    def test_export_unified(self):
        skir = self._compile()
        adapter = SkCCAdapter()
        result = adapter.export(skir, target="unified")

        # Unified should be the full SKIR dict
        assert "version" in result
        assert "identity" in result
        assert "contract" in result
        assert "security" in result

    def test_export_unsupported_target_raises(self):
        skir = self._compile()
        adapter = SkCCAdapter()
        with pytest.raises(ValueError, match="Unsupported export target"):
            adapter.export(skir, target="nonexistent-framework")

    def test_export_all_targets_produce_dict(self):
        """所有支持的导出目标产生有效 dict。"""
        skir = self._compile()
        adapter = SkCCAdapter()
        for target in ExportTarget:
            result = adapter.export(skir, target=target.value)
            assert isinstance(result, dict)
            # Unified format nests under 'identity'; others use top-level 'name'
            if target == ExportTarget.UNIFIED:
                assert "identity" in result
            else:
                assert "name" in result


# ═══════════════════════════════════════════════════════════════════════════════
# Composition Graph Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCompositionGraph:
    def test_simple_graph(self):
        """简单依赖图。"""
        adapter = SkCCAdapter(allow_untrusted=True)
        config = make_config(dependencies=["dep-a", "dep-b"])
        graph = adapter.build_composition_graph(config)

        assert graph["total_nodes"] >= 1  # at least root
        assert graph["cycles"] is False
        assert graph["depth"] >= 1

    def test_nested_graph(self):
        """嵌套依赖图。"""
        adapter = SkCCAdapter(allow_untrusted=True)
        config = make_config(name="root", dependencies=["a"])
        known = {
            "a": make_config(name="a", dependencies=["a1"]),
            "a1": make_config(name="a1", dependencies=[]),
        }
        graph = adapter.build_composition_graph(config, known_skills=known)

        assert graph["depth"] == 2  # root → a → a1 (depth counts edges)
        node_names = [n["name"] for n in graph["nodes"]]
        assert "root" in node_names
        assert "a" in node_names
        assert "a1" in node_names

    def test_cycle_detection(self):
        """循环依赖检测。"""
        adapter = SkCCAdapter(allow_untrusted=True)
        config = make_config(name="a", dependencies=["b"])
        known = {
            "a": make_config(name="a", dependencies=["b"]),
            "b": make_config(name="b", dependencies=["a"]),
        }
        graph = adapter.build_composition_graph(config, known_skills=known)
        assert graph["cycles"] is True

    def test_empty_dependencies(self):
        """无依赖时图只有根节点。"""
        adapter = SkCCAdapter(allow_untrusted=True)
        config = make_config(dependencies=[])
        graph = adapter.build_composition_graph(config)
        assert graph["total_nodes"] == 1
        assert graph["total_edges"] == 0
        assert graph["depth"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    def test_full_pipeline_compile_validate_export(self):
        """完整管线：compile → validate → export。"""
        adapter = SkCCAdapter(trusted_deps={"web-search", "bing-search"})
        config = make_config(
            dependencies=["web-search"],
            contract_overrides={
                "requires_approval": False,
                "timeout_seconds": 15,
                "fallback_config": "local-cache",
            },
        )

        # 1. Compile
        skir = adapter.compile(config)
        assert skir.skill_name == "test-skill"

        # 2. Validate (already done in compile, but can re-check)
        report = adapter.validate_security(config)
        assert report.passed is True

        # 3. Export to all targets
        for target in ExportTarget:
            exported = adapter.export(skir, target=target.value)
            assert isinstance(exported, dict)

    def test_security_blocks_dangerous_compile(self):
        """安全违规不阻止编译（但标记失败）。"""
        adapter = SkCCAdapter(allow_untrusted=True)
        config = make_config(
            name="dangerous",
            contract_overrides={
                "input_schema": {"code": "eval('malicious')"},
            },
        )
        skir = adapter.compile(config)
        assert skir.security_passed is False
        assert skir.security_violations >= 1

    def test_multi_iteration_compile(self):
        """多次编译同一 config 产生一致结果。"""
        adapter = SkCCAdapter(allow_untrusted=True)
        config = make_config()

        results = [adapter.compile(config) for _ in range(5)]
        hashes = [r.content_hash for r in results]
        assert len(set(hashes)) == 1  # all same hash

    def test_export_preserves_hash(self):
        """导出保留 content_hash。"""
        adapter = SkCCAdapter(allow_untrusted=True)
        config = make_config()
        skir = adapter.compile(config)

        unified = adapter.export(skir, target="unified")
        assert unified["integrity"]["content_hash"] == skir.content_hash

    def test_adapter_configurable_trust(self):
        """adapter 的信任配置影响验证结果。"""
        config = make_config(dependencies=["dep-x"])

        # Strict: dep-x not trusted
        strict = SkCCAdapter(trusted_deps={"other-dep"})
        report1 = strict.validate_security(config)
        assert report1.passed is False

        # Permissive: allow_untrusted
        permissive = SkCCAdapter(allow_untrusted=True)
        report2 = permissive.validate_security(config)
        assert report2.passed is True


# ═══════════════════════════════════════════════════════════════════════════════
# Enum & Constants Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestEnums:
    def test_skir_version(self):
        assert SKIRVersion.V1.value == "1.0"

    def test_export_targets(self):
        targets = [e.value for e in ExportTarget]
        assert "skillos" in targets
        assert "openskill" in targets
        assert "skillopt" in targets
        assert "meta_skills" in targets
        assert "unified" in targets

    def test_security_risk_levels(self):
        levels = [e.value for e in SecurityRiskLevel]
        assert "low" in levels
        assert "medium" in levels
        assert "high" in levels
        assert "critical" in levels

    def test_violation_types(self):
        types = [e.value for e in SecurityViolationType]
        assert "circular_dependency" in types
        assert "eval_in_schema" in types
        assert "depth_exceeded" in types


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
