"""ISN UnifiedSkillConfig 单元测试。"""
import json
from typing import ClassVar
import pytest

from openllm.isn.unified_skill_config import (
    OptimizationHints,
    SkillCompositionContract,
    SkillFrameworkSource,
    SkillLifecycleState,
    SkillUsageMetrics,
    MetaSkillConfig,
    SourceTraceability,
    UnifiedSkillConfig,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Enum tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSkillLifecycleState:
    def test_all_states(self):
        states = [s.value for s in SkillLifecycleState]
        assert states == ["active", "dormant", "deprecated", "retired"]

    def test_active_is_default(self):
        config = UnifiedSkillConfig(name="test")
        assert config.lifecycle_state == SkillLifecycleState.ACTIVE

    def test_from_string(self):
        state = SkillLifecycleState("deprecated")
        assert state == SkillLifecycleState.DEPRECATED


class TestSkillFrameworkSource:
    def test_all_sources(self):
        sources = [s.value for s in SkillFrameworkSource]
        assert "skillos" in sources
        assert "openskill" in sources
        assert "skcc" in sources
        assert "skillopt" in sources
        assert "meta_skills" in sources
        assert "unified" in sources


# ═══════════════════════════════════════════════════════════════════════════════
# Core UnifiedSkillConfig tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestUnifiedSkillConfigCore:
    def test_default_construction(self):
        config = UnifiedSkillConfig(name="test-skill")
        assert config.name == "test-skill"
        assert config.lifecycle_state == SkillLifecycleState.ACTIVE
        assert config.source_framework == SkillFrameworkSource.UNIFIED
        assert config.version == "0.0.1"
        assert config.domain_tags == []
        assert config.created_at  # not empty

    def test_full_construction(self):
        config = UnifiedSkillConfig(
            name="search-cascade",
            description="搜索降级级联",
            version="2.1.0",
            author="zcs",
            lifecycle_state=SkillLifecycleState.ACTIVE,
            source_framework=SkillFrameworkSource.UNIFIED,
            domain_tags=["search", "fallback"],
            risk_level="medium",
        )
        assert config.name == "search-cascade"
        assert config.description == "搜索降级级联"
        assert config.version == "2.1.0"
        assert config.author == "zcs"
        assert "search" in config.domain_tags
        assert config.risk_level == "medium"

    def test_framework_fields_default_none(self):
        config = UnifiedSkillConfig(name="test")
        assert config.task_group_id is None
        assert config.curator_score is None
        assert config.compression_reward is None
        assert config.source_traceability is None
        assert config.composition_contract is None
        assert config.optimization_hints is None
        assert config.usage_metrics is None
        assert config.meta_config is None
        assert config.dependencies == []
        assert config.composes_with == []


# ═══════════════════════════════════════════════════════════════════════════════
# Factory method tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFromSkillos:
    def test_basic_migration(self):
        data = {
            "name": "skill-a",
            "description": "SkillOS test skill",
            "version": "1.0.0",
            "author": "skillos-team",
            "lifecycle_state": "active",
            "domain_tags": ["nlp", "classification"],
            "task_group_id": "group-42",
            "curator_score": 0.85,
            "compression_reward": 0.92,
        }
        config = UnifiedSkillConfig.from_skillos(data)
        assert config.name == "skill-a"
        assert config.source_framework == SkillFrameworkSource.SKILLOS
        assert config.task_group_id == "group-42"
        assert config.curator_score == 0.85
        assert config.compression_reward == 0.92
        assert "nlp" in config.domain_tags

    def test_minimal_data(self):
        config = UnifiedSkillConfig.from_skillos({"name": "minimal"})
        assert config.name == "minimal"
        assert config.source_framework == SkillFrameworkSource.SKILLOS
        assert config.task_group_id is None
        assert config.lifecycle_state == SkillLifecycleState.ACTIVE

    def test_lifecycle_dormant(self):
        config = UnifiedSkillConfig.from_skillos({
            "name": "old-skill",
            "lifecycle_state": "dormant",
        })
        assert config.lifecycle_state == SkillLifecycleState.DORMANT


class TestFromOpenskill:
    def test_basic_migration_with_trace(self):
        data = {
            "name": "community-search",
            "description": "Community search skill",
            "version": "0.5.0",
            "author": "community-user",
            "source_traceability": {
                "upstream_repo": "https://github.com/example/skills",
                "upstream_version": "0.4.0",
                "forked_at": "2026-07-01T10:00:00Z",
                "license": "MIT",
                "original_author": "upstream-dev",
                "attribution": "Forked from upstream/skills",
                "modifications": ["added Chinese support", "fixed edge case"],
            },
        }
        config = UnifiedSkillConfig.from_openskill(data)
        assert config.name == "community-search"
        assert config.source_framework == SkillFrameworkSource.OPENSKILL
        assert config.source_traceability is not None
        assert config.source_traceability.upstream_repo == "https://github.com/example/skills"
        assert config.source_traceability.license == "MIT"
        assert len(config.source_traceability.modifications) == 2

    def test_minimal_no_trace(self):
        config = UnifiedSkillConfig.from_openskill({"name": "no-trace"})
        assert config.source_traceability is None
        assert config.source_framework == SkillFrameworkSource.OPENSKILL


class TestFromSkcc:
    def test_basic_migration_with_contract(self):
        data = {
            "name": "data-processor",
            "description": "Processes data",
            "composition_contract": {
                "input_schema": {"type": "object", "properties": {"data": {"type": "array"}}},
                "output_schema": {"type": "object", "properties": {"result": {"type": "string"}}},
                "requires_approval": True,
                "timeout_seconds": 30,
                "fallback_config": "default-processor",
            },
            "dependencies": ["data-fetcher", "data-validator"],
            "composes_with": ["report-generator", "data-visualizer"],
        }
        config = UnifiedSkillConfig.from_skcc(data)
        assert config.name == "data-processor"
        assert config.source_framework == SkillFrameworkSource.SKCC
        assert config.composition_contract is not None
        assert config.composition_contract.requires_approval is True
        assert config.composition_contract.timeout_seconds == 30
        assert "data-fetcher" in config.dependencies
        assert "report-generator" in config.composes_with

    def test_minimal_no_contract(self):
        config = UnifiedSkillConfig.from_skcc({"name": "simple"})
        assert config.composition_contract is None
        assert config.dependencies == []


class TestFromSkillopt:
    def test_basic_migration_with_hints_and_metrics(self):
        data = {
            "name": "search-engine",
            "description": "Search engine",
            "optimization_hints": {
                "should_optimize": True,
                "priority": 3,
                "reason": "success_rate dropped below 90%",
                "suggested_focus": "prompt_quality",
                "last_analyzed_at": "2026-07-24T00:00:00Z",
                "confidence": 0.87,
                "metadata": {"regression_score": 0.15},
            },
            "usage_metrics": {
                "total_calls": 1000,
                "success_count": 850,
                "failure_count": 150,
                "avg_latency_ms": 120.5,
                "avg_token_cost": 350.0,
                "last_used_at": "2026-07-23T23:59:00Z",
            },
        }
        config = UnifiedSkillConfig.from_skillopt(data)
        assert config.name == "search-engine"
        assert config.source_framework == SkillFrameworkSource.SKILLOPT
        assert config.optimization_hints is not None
        assert config.optimization_hints.should_optimize is True
        assert config.optimization_hints.priority == 3
        assert config.optimization_hints.confidence == 0.87
        assert config.optimization_hints.metadata["regression_score"] == 0.15
        assert config.usage_metrics is not None
        assert config.usage_metrics.total_calls == 1000
        assert config.usage_metrics.success_rate == pytest.approx(0.85)

    def test_zero_calls_no_division_by_zero(self):
        data = {
            "name": "new-skill",
            "usage_metrics": {"total_calls": 0, "success_count": 0},
        }
        config = UnifiedSkillConfig.from_skillopt(data)
        assert config.usage_metrics.success_rate == 0.0

    def test_minimal_no_opt(self):
        config = UnifiedSkillConfig.from_skillopt({"name": "plain"})
        assert config.optimization_hints is None
        assert config.usage_metrics is None


class TestFromMetaskills:
    def test_basic_migration_with_meta(self):
        data = {
            "name": "pipeline-orchestrator",
            "description": "Orchestrates data pipeline",
            "meta_config": {
                "orchestration_pattern": "pipeline",
                "managed_skills": ["fetcher", "transformer", "loader"],
                "composition_rules": ["order: fetch→transform→load", "parallel_ok: none"],
                "max_concurrency": 3,
                "abort_on_failure": False,
            },
        }
        config = UnifiedSkillConfig.from_metaskills(data)
        assert config.name == "pipeline-orchestrator"
        assert config.source_framework == SkillFrameworkSource.META_SKILLS
        assert config.meta_config is not None
        assert config.meta_config.orchestration_pattern == "pipeline"
        assert len(config.meta_config.managed_skills) == 3
        assert config.meta_config.max_concurrency == 3
        assert config.meta_config.abort_on_failure is False

    def test_minimal_no_meta(self):
        config = UnifiedSkillConfig.from_metaskills({"name": "bare"})
        assert config.meta_config is None


# ═══════════════════════════════════════════════════════════════════════════════
# Lifecycle state machine tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLifecycleStateMachine:
    def test_active_to_dormant(self):
        config = UnifiedSkillConfig(name="s", lifecycle_state=SkillLifecycleState.ACTIVE)
        assert config.transition(SkillLifecycleState.DORMANT) is True
        assert config.lifecycle_state == SkillLifecycleState.DORMANT

    def test_active_to_deprecated(self):
        config = UnifiedSkillConfig(name="s", lifecycle_state=SkillLifecycleState.ACTIVE)
        assert config.transition(SkillLifecycleState.DEPRECATED) is True
        assert config.lifecycle_state == SkillLifecycleState.DEPRECATED

    def test_active_to_retired_invalid(self):
        config = UnifiedSkillConfig(name="s", lifecycle_state=SkillLifecycleState.ACTIVE)
        assert config.transition(SkillLifecycleState.RETIRED) is False
        assert config.lifecycle_state == SkillLifecycleState.ACTIVE  # unchanged

    def test_dormant_to_active(self):
        config = UnifiedSkillConfig(name="s", lifecycle_state=SkillLifecycleState.DORMANT)
        assert config.transition(SkillLifecycleState.ACTIVE) is True

    def test_dormant_to_deprecated(self):
        config = UnifiedSkillConfig(name="s", lifecycle_state=SkillLifecycleState.DORMANT)
        assert config.transition(SkillLifecycleState.DEPRECATED) is True

    def test_deprecated_to_retired(self):
        config = UnifiedSkillConfig(name="s", lifecycle_state=SkillLifecycleState.DEPRECATED)
        assert config.transition(SkillLifecycleState.RETIRED) is True
        assert config.lifecycle_state == SkillLifecycleState.RETIRED

    def test_retired_is_terminal(self):
        config = UnifiedSkillConfig(name="s", lifecycle_state=SkillLifecycleState.RETIRED)
        assert config.transition(SkillLifecycleState.ACTIVE) is False
        assert config.transition(SkillLifecycleState.DORMANT) is False
        assert config.transition(SkillLifecycleState.DEPRECATED) is False

    def test_can_transition_to(self):
        config = UnifiedSkillConfig(name="s", lifecycle_state=SkillLifecycleState.ACTIVE)
        assert config.can_transition_to(SkillLifecycleState.DORMANT) is True
        assert config.can_transition_to(SkillLifecycleState.RETIRED) is False

    def test_transition_updates_timestamp(self):
        config = UnifiedSkillConfig(name="s", lifecycle_state=SkillLifecycleState.ACTIVE)
        old_ts = config.updated_at
        import time; time.sleep(0.01)
        config.transition(SkillLifecycleState.DORMANT)
        assert config.updated_at >= old_ts


# ═══════════════════════════════════════════════════════════════════════════════
# Validation tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidation:
    def test_valid_config(self):
        config = UnifiedSkillConfig(name="valid-skill", version="1.0.0")
        assert config.is_valid()
        assert config.validate() == []

    def test_missing_name(self):
        config = UnifiedSkillConfig(name="")
        errors = config.validate()
        assert "name is required" in errors

    def test_missing_version(self):
        config = UnifiedSkillConfig(name="x", version="")
        errors = config.validate()
        assert "version is required" in errors

    def test_invalid_curator_score_negative(self):
        config = UnifiedSkillConfig(name="x", curator_score=-0.1)
        errors = config.validate()
        assert any("curator_score" in e for e in errors)

    def test_invalid_curator_score_over_one(self):
        config = UnifiedSkillConfig(name="x", curator_score=1.5)
        errors = config.validate()
        assert any("curator_score" in e for e in errors)

    def test_valid_curator_score(self):
        config = UnifiedSkillConfig(name="x", curator_score=0.85)
        errors = config.validate()
        assert not any("curator_score" in e for e in errors)

    def test_invalid_optimization_hints_confidence(self):
        config = UnifiedSkillConfig(
            name="x",
            optimization_hints=OptimizationHints(confidence=1.5),
        )
        errors = config.validate()
        assert any("confidence" in e for e in errors)

    def test_invalid_optimization_hints_priority(self):
        config = UnifiedSkillConfig(
            name="x",
            optimization_hints=OptimizationHints(priority=5),
        )
        errors = config.validate()
        assert any("priority" in e for e in errors)

    def test_invalid_risk_level(self):
        config = UnifiedSkillConfig(name="x", risk_level="unknown")
        errors = config.validate()
        assert any("risk_level" in e for e in errors)

    def test_valid_risk_levels(self):
        for level in ("critical", "high", "medium", "low"):
            config = UnifiedSkillConfig(name="x", risk_level=level)
            assert not any("risk_level" in e for e in config.validate())


# ═══════════════════════════════════════════════════════════════════════════════
# Serialization round-trip tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSerialization:
    def test_to_dict_basic(self):
        config = UnifiedSkillConfig(name="test", lifecycle_state=SkillLifecycleState.ACTIVE)
        d = config.to_dict()
        assert d["name"] == "test"
        assert d["lifecycle_state"] == "active"  # enum → string

    def test_to_dict_with_nested(self):
        config = UnifiedSkillConfig(
            name="opt-skill",
            optimization_hints=OptimizationHints(
                should_optimize=True,
                priority=3,
                confidence=0.8,
            ),
            source_traceability=SourceTraceability(
                upstream_repo="https://example.com",
                license="MIT",
            ),
        )
        d = config.to_dict()
        assert d["optimization_hints"]["should_optimize"] is True
        assert d["optimization_hints"]["priority"] == 3
        assert d["source_traceability"]["license"] == "MIT"

    def test_to_dict_json_serializable(self):
        config = UnifiedSkillConfig(
            name="json-test",
            optimization_hints=OptimizationHints(confidence=0.9),
            source_traceability=SourceTraceability(license="Apache-2.0"),
        )
        d = config.to_dict()
        # Must not raise
        json_str = json.dumps(d)
        assert "json-test" in json_str

    def test_round_trip(self):
        original = UnifiedSkillConfig(
            name="round-trip",
            description="测试往返序列化",
            version="2.0.0",
            lifecycle_state=SkillLifecycleState.DORMANT,
            source_framework=SkillFrameworkSource.SKILLOPT,
            domain_tags=["test", "serialization"],
            optimization_hints=OptimizationHints(
                should_optimize=True,
                priority=2,
                confidence=0.75,
            ),
            source_traceability=SourceTraceability(
                upstream_repo="https://github.com/test",
                license="MIT",
                modifications=["change1"],
            ),
            dependencies=["dep-a", "dep-b"],
        )
        d = original.to_dict()
        restored = UnifiedSkillConfig.from_dict(d)
        assert restored.name == original.name
        assert restored.lifecycle_state == original.lifecycle_state
        assert restored.source_framework == original.source_framework
        assert restored.optimization_hints.should_optimize is True
        assert restored.source_traceability.upstream_repo == "https://github.com/test"
        assert restored.dependencies == ["dep-a", "dep-b"]

    def test_from_dict_unknown_keys_ignored(self):
        d = {
            "name": "resilient",
            "unknown_field": "should_be_ignored",
            "another_unknown": 42,
        }
        config = UnifiedSkillConfig.from_dict(d)
        assert config.name == "resilient"

    def test_round_trip_all_frameworks(self):
        """Verify serialization round-trips for data produced by each factory."""
        factories = {
            "skillos": UnifiedSkillConfig.from_skillos({
                "name": "s", "task_group_id": "g1", "curator_score": 0.5,
            }),
            "openskill": UnifiedSkillConfig.from_openskill({
                "name": "o", "source_traceability": {"upstream_repo": "url"},
            }),
            "skcc": UnifiedSkillConfig.from_skcc({
                "name": "c", "dependencies": ["d1"],
            }),
            "skillopt": UnifiedSkillConfig.from_skillopt({
                "name": "p", "optimization_hints": {"should_optimize": True},
            }),
            "meta_skills": UnifiedSkillConfig.from_metaskills({
                "name": "m", "meta_config": {"managed_skills": ["a", "b"]},
            }),
        }
        for fw_name, config in factories.items():
            d = config.to_dict()
            restored = UnifiedSkillConfig.from_dict(d)
            assert restored.name == config.name, f"round-trip failed for {fw_name}"
            assert restored.source_framework == config.source_framework, f"framework mismatch for {fw_name}"


# ═══════════════════════════════════════════════════════════════════════════════
# Sub-model tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestOptimizationHints:
    def test_defaults(self):
        h = OptimizationHints()
        assert h.should_optimize is False
        assert h.priority == 0
        assert h.confidence == 0.0
        assert h.metadata == {}

    def test_full(self):
        h = OptimizationHints(
            should_optimize=True,
            priority=4,
            reason="critical regression",
            suggested_focus="token_cost",
            confidence=0.95,
            metadata={"score_delta": -0.3},
        )
        assert h.priority == 4
        assert h.metadata["score_delta"] == -0.3


class TestSourceTraceability:
    def test_defaults(self):
        t = SourceTraceability()
        assert t.upstream_repo == ""
        assert t.modifications == []

    def test_full(self):
        t = SourceTraceability(
            upstream_repo="https://github.com/test",
            upstream_version="1.0.0",
            license="MIT",
            original_author="dev",
            modifications=["fix1", "fix2"],
        )
        assert len(t.modifications) == 2


class TestSkillUsageMetrics:
    def test_defaults(self):
        m = SkillUsageMetrics()
        assert m.total_calls == 0
        assert m.success_rate == 0.0

    def test_success_rate(self):
        m = SkillUsageMetrics(total_calls=200, success_count=180, failure_count=20)
        assert m.success_rate == pytest.approx(0.9)


class TestSkillCompositionContract:
    def test_defaults(self):
        c = SkillCompositionContract()
        assert c.input_schema == {}
        assert c.requires_approval is False
        assert c.timeout_seconds is None


class TestMetaSkillConfig:
    def test_defaults(self):
        m = MetaSkillConfig()
        assert m.orchestration_pattern == ""
        assert m.managed_skills == []
        assert m.max_concurrency == 1
        assert m.abort_on_failure is True

    def test_full(self):
        m = MetaSkillConfig(
            orchestration_pattern="parallel",
            managed_skills=["a", "b", "c"],
            max_concurrency=5,
            abort_on_failure=False,
        )
        assert len(m.managed_skills) == 3
        assert m.max_concurrency == 5


# ═══════════════════════════════════════════════════════════════════════════════
# Integration / edge case tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_empty_domain_tags(self):
        config = UnifiedSkillConfig(name="x", domain_tags=[])
        assert config.domain_tags == []

    def test_risk_level_default(self):
        config = UnifiedSkillConfig(name="x")
        assert config.risk_level == "medium"

    def test_multiple_factory_data_combined(self):
        """A config can carry data from multiple frameworks after manual merge."""
        config = UnifiedSkillConfig(
            name="hybrid-skill",
            source_framework=SkillFrameworkSource.UNIFIED,
            task_group_id="g1",  # SkillOS
            source_traceability=SourceTraceability(license="MIT"),  # OpenSkill
            optimization_hints=OptimizationHints(should_optimize=True),  # SkillOpt
        )
        assert config.task_group_id == "g1"
        assert config.source_traceability.license == "MIT"
        assert config.optimization_hints.should_optimize is True

    def test_is_valid_comprehensive(self):
        config = UnifiedSkillConfig(
            name="complete",
            version="1.0.0",
            curator_score=0.9,
            optimization_hints=OptimizationHints(confidence=0.8, priority=2),
            risk_level="low",
        )
        assert config.is_valid()
