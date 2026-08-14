"""
ISN Cross-Profile Consistency Checker — 单元测试
=================================================

测试覆盖：
  1. VersionConflict — 跨 profile 版本差异检测与解决建议
  2. ConstraintConflict — 约束差异检测（lifecycle_state / risk_level）
  3. ProfileConsistencyChecker — 完整检查流水线
  4. ConflictReport — 报告生成与摘要
  5. 边界情况：空 profile / 单 profile / 全一致 / 混合冲突
  6. 退出工程化路径一致性
"""

import pytest
from openllm.isn.unified_skill_config import (
    SkillCompositionContract,
    SkillLifecycleState,
    UnifiedSkillConfig,
)
from openllm.isn.profile_consistency import (
    ConflictReport,
    ConflictSeverity,
    ConflictType,
    ConstraintConflict,
    ProfileConsistencyChecker,
    SkillConflict,
    VersionConflict,
    _suggest_constraint_resolution,
    _suggest_version_resolution,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def skill_a_v1():
    """Skill A version 1.0.0, active, medium risk."""
    return UnifiedSkillConfig(
        name="search-fallback",
        description="搜索降级级联",
        version="1.0.0",
        lifecycle_state=SkillLifecycleState.ACTIVE,
        risk_level="medium",
        dependencies=["web-fetch"],
    )


@pytest.fixture
def skill_a_v2():
    """Skill A version 2.0.0, active, medium risk."""
    return UnifiedSkillConfig(
        name="search-fallback",
        description="搜索降级级联 v2",
        version="2.0.0",
        lifecycle_state=SkillLifecycleState.ACTIVE,
        risk_level="medium",
        dependencies=["web-fetch"],
    )


@pytest.fixture
def skill_a_deprecated():
    """Skill A version 1.0.0, deprecated."""
    return UnifiedSkillConfig(
        name="search-fallback",
        description="搜索降级级联",
        version="1.0.0",
        lifecycle_state=SkillLifecycleState.DEPRECATED,
        risk_level="medium",
        dependencies=["web-fetch"],
    )


@pytest.fixture
def skill_a_retired():
    """Skill A version 1.0.0, retired."""
    return UnifiedSkillConfig(
        name="search-fallback",
        description="搜索降级级联",
        version="1.0.0",
        lifecycle_state=SkillLifecycleState.RETIRED,
        risk_level="medium",
        dependencies=["web-fetch"],
    )


@pytest.fixture
def skill_b():
    """Skill B — different skill entirely."""
    return UnifiedSkillConfig(
        name="paper-analysis",
        description="论文分析",
        version="1.2.0",
        lifecycle_state=SkillLifecycleState.ACTIVE,
        risk_level="low",
    )


@pytest.fixture
def skill_b_high_risk():
    """Skill B with high risk level."""
    return UnifiedSkillConfig(
        name="paper-analysis",
        description="论文分析",
        version="1.2.0",
        lifecycle_state=SkillLifecycleState.ACTIVE,
        risk_level="high",
    )


@pytest.fixture
def empty_checker():
    """Empty checker with no profiles."""
    return ProfileConsistencyChecker()


@pytest.fixture
def two_profile_checker(skill_a_v1, skill_a_v2, skill_b):
    """Checker with two profiles that have a version conflict on skill_a."""
    checker = ProfileConsistencyChecker()
    checker.register_profile("sa_main", {"search-fallback": skill_a_v1, "paper-analysis": skill_b})
    checker.register_profile("sa_worker", {"search-fallback": skill_a_v2, "paper-analysis": skill_b})
    return checker


@pytest.fixture
def consistent_checker(skill_a_v1, skill_b):
    """Checker with two profiles that are fully consistent."""
    checker = ProfileConsistencyChecker()
    checker.register_profile("sa_main", {"search-fallback": skill_a_v1, "paper-analysis": skill_b})
    checker.register_profile("sa_worker", {"search-fallback": skill_a_v1, "paper-analysis": skill_b})
    return checker


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Profile registration
# ═══════════════════════════════════════════════════════════════════════════════

class TestProfileRegistration:
    def test_register_single_profile(self, empty_checker, skill_a_v1):
        empty_checker.register_profile("sa_main", {"skill_a": skill_a_v1})
        assert "sa_main" in empty_checker.profiles
        assert len(empty_checker.profiles["sa_main"]) == 1

    def test_register_multiple_profiles(self, empty_checker, skill_a_v1, skill_b):
        empty_checker.register_profile("sa_main", {"skill_a": skill_a_v1})
        empty_checker.register_profile("sa_worker", {"skill_b": skill_b})
        assert len(empty_checker.profiles) == 2

    def test_unregister_profile(self, empty_checker, skill_a_v1):
        empty_checker.register_profile("sa_main", {"skill_a": skill_a_v1})
        assert empty_checker.unregister_profile("sa_main")
        assert "sa_main" not in empty_checker.profiles

    def test_unregister_nonexistent(self, empty_checker):
        assert not empty_checker.unregister_profile("ghost")

    def test_get_all_skill_names(self, two_profile_checker):
        names = two_profile_checker.get_all_skill_names()
        assert "search-fallback" in names
        assert "paper-analysis" in names
        assert len(names) == 2

    def test_get_skill_across_profiles(self, two_profile_checker):
        across = two_profile_checker.get_skill_across_profiles("search-fallback")
        assert "sa_main" in across
        assert "sa_worker" in across
        assert across["sa_main"] is not None
        assert across["sa_worker"] is not None

    def test_get_missing_skill_across_profiles(self, two_profile_checker):
        across = two_profile_checker.get_skill_across_profiles("nonexistent")
        assert across["sa_main"] is None
        assert across["sa_worker"] is None


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Version conflict detection
# ═══════════════════════════════════════════════════════════════════════════════

class TestVersionConflicts:
    def test_no_version_conflict_when_consistent(self, consistent_checker):
        conflicts = consistent_checker.check_version_conflicts()
        assert len(conflicts) == 0

    def test_detects_major_version_conflict(self, two_profile_checker):
        conflicts = two_profile_checker.check_version_conflicts()
        assert len(conflicts) == 1
        assert conflicts[0].skill_name == "search-fallback"
        assert conflicts[0].conflict_type == ConflictType.VERSION_MISMATCH
        # 1.0.0 → 2.0.0 is major, should be ERROR
        assert conflicts[0].severity == ConflictSeverity.ERROR

    def test_detects_minor_version_conflict(self, skill_a_v1, skill_b):
        skill_a_1_1 = UnifiedSkillConfig(
            name="search-fallback", version="1.1.0",
            lifecycle_state=SkillLifecycleState.ACTIVE,
        )
        checker = ProfileConsistencyChecker()
        checker.register_profile("sa_main", {"search-fallback": skill_a_v1})
        checker.register_profile("sa_worker", {"search-fallback": skill_a_1_1})

        conflicts = checker.check_version_conflicts()
        assert len(conflicts) == 1
        # 1.0.0 → 1.1.0 is minor, should be WARNING
        assert conflicts[0].severity == ConflictSeverity.WARNING

    def test_detects_patch_version_conflict(self, skill_a_v1):
        skill_a_1_0_1 = UnifiedSkillConfig(
            name="search-fallback", version="1.0.1",
            lifecycle_state=SkillLifecycleState.ACTIVE,
        )
        checker = ProfileConsistencyChecker()
        checker.register_profile("sa_main", {"search-fallback": skill_a_v1})
        checker.register_profile("sa_worker", {"search-fallback": skill_a_1_0_1})

        conflicts = checker.check_version_conflicts()
        assert len(conflicts) == 1
        # 1.0.0 → 1.0.1 is patch, should be INFO
        assert conflicts[0].severity == ConflictSeverity.INFO

    def test_version_conflict_versions_dict(self, two_profile_checker):
        conflicts = two_profile_checker.check_version_conflicts()
        assert conflicts[0].versions["sa_main"] == "1.0.0"
        assert conflicts[0].versions["sa_worker"] == "2.0.0"

    def test_no_conflict_with_single_profile(self, empty_checker, skill_a_v1):
        empty_checker.register_profile("sa_main", {"skill_a": skill_a_v1})
        conflicts = empty_checker.check_version_conflicts()
        assert len(conflicts) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Constraint conflict detection
# ═══════════════════════════════════════════════════════════════════════════════

class TestConstraintConflicts:
    def test_no_constraint_conflict_when_consistent(self, consistent_checker):
        conflicts = consistent_checker.check_constraint_conflicts()
        assert len(conflicts) == 0

    def test_detects_lifecycle_state_conflict(
        self, skill_a_v1, skill_a_deprecated
    ):
        checker = ProfileConsistencyChecker()
        checker.register_profile("sa_main", {"search-fallback": skill_a_v1})
        checker.register_profile("sa_worker", {"search-fallback": skill_a_deprecated})

        conflicts = checker.check_constraint_conflicts()
        assert len(conflicts) == 1
        assert conflicts[0].field_name == "lifecycle_state"
        assert conflicts[0].conflict_type == ConflictType.CONSTRAINT_DIVERGENCE

    def test_active_vs_retired_is_critical(self, skill_a_v1, skill_a_retired):
        checker = ProfileConsistencyChecker()
        checker.register_profile("sa_main", {"search-fallback": skill_a_v1})
        checker.register_profile("sa_worker", {"search-fallback": skill_a_retired})

        conflicts = checker.check_constraint_conflicts()
        assert len(conflicts) == 1
        assert conflicts[0].severity == ConflictSeverity.CRITICAL

    def test_active_vs_deprecated_is_error(self, skill_a_v1, skill_a_deprecated):
        checker = ProfileConsistencyChecker()
        checker.register_profile("sa_main", {"search-fallback": skill_a_v1})
        checker.register_profile("sa_worker", {"search-fallback": skill_a_deprecated})

        conflicts = checker.check_constraint_conflicts()
        assert len(conflicts) == 1
        assert conflicts[0].severity == ConflictSeverity.ERROR

    def test_detects_risk_level_conflict(self, skill_b, skill_b_high_risk):
        checker = ProfileConsistencyChecker()
        checker.register_profile("sa_main", {"paper-analysis": skill_b})
        checker.register_profile("sa_worker", {"paper-analysis": skill_b_high_risk})

        conflicts = checker.check_constraint_conflicts()
        assert len(conflicts) == 1
        assert conflicts[0].field_name == "risk_level"

    def test_critical_vs_low_risk_is_critical(self):
        skill_critical = UnifiedSkillConfig(
            name="x", version="1.0.0", risk_level="critical",
            lifecycle_state=SkillLifecycleState.ACTIVE,
        )
        skill_low = UnifiedSkillConfig(
            name="x", version="1.0.0", risk_level="low",
            lifecycle_state=SkillLifecycleState.ACTIVE,
        )
        checker = ProfileConsistencyChecker()
        checker.register_profile("sa_main", {"x": skill_critical})
        checker.register_profile("sa_worker", {"x": skill_low})

        conflicts = checker.check_constraint_conflicts()
        assert len(conflicts) == 1
        assert conflicts[0].severity == ConflictSeverity.CRITICAL

    def test_no_conflict_single_profile(self, empty_checker, skill_a_v1):
        empty_checker.register_profile("sa_main", {"skill_a": skill_a_v1})
        conflicts = empty_checker.check_constraint_conflicts()
        assert len(conflicts) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Resolution suggestions
# ═══════════════════════════════════════════════════════════════════════════════

class TestResolutionSuggestions:
    def test_version_suggestion_mentions_profiles(self, two_profile_checker):
        conflicts = two_profile_checker.check_version_conflicts()
        suggestion = conflicts[0].suggestion
        assert "sa_main" in suggestion
        assert "sa_worker" in suggestion

    def test_version_suggestion_mentions_versions(self, two_profile_checker):
        conflicts = two_profile_checker.check_version_conflicts()
        suggestion = conflicts[0].suggestion
        assert "1.0.0" in suggestion
        assert "2.0.0" in suggestion

    def test_version_suggestion_mentions_major(self, two_profile_checker):
        conflicts = two_profile_checker.check_version_conflicts()
        suggestion = conflicts[0].suggestion
        assert "MAJOR" in suggestion or "breaking" in suggestion.lower()

    def test_constraint_suggestion_for_lifecycle(self, skill_a_v1, skill_a_deprecated):
        checker = ProfileConsistencyChecker()
        checker.register_profile("sa_main", {"search-fallback": skill_a_v1})
        checker.register_profile("sa_worker", {"search-fallback": skill_a_deprecated})

        conflicts = checker.check_constraint_conflicts()
        suggestion = conflicts[0].suggestion
        assert "active" in suggestion.lower() or "deprecated" in suggestion.lower()

    def test_constraint_suggestion_for_risk(self, skill_b, skill_b_high_risk):
        checker = ProfileConsistencyChecker()
        checker.register_profile("sa_main", {"paper-analysis": skill_b})
        checker.register_profile("sa_worker", {"paper-analysis": skill_b_high_risk})

        conflicts = checker.check_constraint_conflicts()
        suggestion = conflicts[0].suggestion
        assert "high" in suggestion.lower() or "strictest" in suggestion.lower()

    def test_suggest_resolution_method(self, two_profile_checker):
        conflicts = two_profile_checker.check_version_conflicts()
        checker = two_profile_checker
        suggestion = checker.suggest_resolution(conflicts[0])
        assert len(suggestion) > 0

    def test_suggest_resolution_for_missing_skill(self):
        checker = ProfileConsistencyChecker()
        checker.register_profile("sa_main", {"skill_a": UnifiedSkillConfig(
            name="skill_a", version="1.0.0",
            lifecycle_state=SkillLifecycleState.ACTIVE,
        )})

        conflicts = checker.run_full_check().conflicts
        # No conflict if only 1 profile
        assert len([c for c in conflicts if c.conflict_type == ConflictType.MISSING_SKILL]) == 0

    def test_suggest_resolution_for_missing_in_multi_profile(self, skill_a_v1):
        checker = ProfileConsistencyChecker()
        checker.register_profile("sa_main", {"skill_a": skill_a_v1, "skill_b": skill_a_v1})
        checker.register_profile("sa_worker", {"skill_a": skill_a_v1})

        report = checker.run_full_check()
        missing = [c for c in report.conflicts if c.conflict_type == ConflictType.MISSING_SKILL]
        assert len(missing) == 1
        suggestion = checker.suggest_resolution(missing[0])
        assert "skill_b" in suggestion


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Full check pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class TestFullCheck:
    def test_clean_report_when_consistent(self, consistent_checker):
        report = consistent_checker.run_full_check()
        assert report.is_clean
        assert len(report.conflicts) == 0
        assert len(report.validation_errors) == 0

    def test_report_has_version_conflict(self, two_profile_checker):
        report = two_profile_checker.run_full_check()
        assert not report.is_clean
        version_conflicts = [
            c for c in report.conflicts
            if c.conflict_type == ConflictType.VERSION_MISMATCH
        ]
        assert len(version_conflicts) == 1

    def test_report_profiles_checked(self, two_profile_checker):
        report = two_profile_checker.run_full_check()
        assert "sa_main" in report.profiles_checked
        assert "sa_worker" in report.profiles_checked

    def test_report_skills_checked(self, two_profile_checker):
        report = two_profile_checker.run_full_check()
        assert "search-fallback" in report.skills_checked
        assert "paper-analysis" in report.skills_checked

    def test_report_summary_clean(self, consistent_checker):
        report = consistent_checker.run_full_check()
        summary = report.summary()
        assert "✅" in summary
        assert "no conflicts" in summary.lower() or "consistent" in summary.lower()

    def test_report_summary_with_conflicts(self, two_profile_checker):
        report = two_profile_checker.run_full_check()
        summary = report.summary()
        assert "Conflicts" in summary
        assert "1" in summary  # at least one conflict

    def test_full_check_catches_multiple_conflict_types(self, skill_a_v1, skill_a_retired):
        """Version + lifecycle + risk all different."""
        skill_a_v2_retired_high = UnifiedSkillConfig(
            name="search-fallback",
            description="搜索降级级联 v2",
            version="2.0.0",
            lifecycle_state=SkillLifecycleState.RETIRED,
            risk_level="high",
        )
        checker = ProfileConsistencyChecker()
        checker.register_profile("sa_main", {"search-fallback": skill_a_v1})
        checker.register_profile("sa_worker", {"search-fallback": skill_a_v2_retired_high})

        report = checker.run_full_check()
        assert not report.is_clean
        conflict_types = {c.conflict_type for c in report.conflicts}
        assert ConflictType.VERSION_MISMATCH in conflict_types
        assert ConflictType.CONSTRAINT_DIVERGENCE in conflict_types

    def test_validation_errors_detected(self):
        """Skill with empty name fails validation."""
        bad_config = UnifiedSkillConfig(name="", version="1.0.0")
        good_config = UnifiedSkillConfig(
            name="good-skill", version="1.0.0",
            lifecycle_state=SkillLifecycleState.ACTIVE,
        )
        checker = ProfileConsistencyChecker()
        checker.register_profile("sa_main", {"bad": bad_config, "good": good_config})

        report = checker.run_full_check()
        assert "sa_main" in report.validation_errors
        assert any("name is required" in e for e in report.validation_errors["sa_main"])

    def test_empty_profiles_check(self, empty_checker):
        report = empty_checker.run_full_check()
        assert report.is_clean
        assert len(report.profiles_checked) == 0

    def test_single_profile_check(self, empty_checker, skill_a_v1):
        empty_checker.register_profile("sa_main", {"skill_a": skill_a_v1})
        report = empty_checker.run_full_check()
        assert report.is_clean
        assert len(report.conflicts) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Missing skill detection
# ═══════════════════════════════════════════════════════════════════════════════

class TestMissingSkills:
    def test_no_missing_when_symmetric(self, consistent_checker):
        report = consistent_checker.run_full_check()
        missing = [c for c in report.conflicts if c.conflict_type == ConflictType.MISSING_SKILL]
        assert len(missing) == 0

    def test_detects_missing_skill(self, skill_a_v1):
        checker = ProfileConsistencyChecker()
        checker.register_profile("sa_main", {"skill_a": skill_a_v1, "skill_b": skill_a_v1})
        checker.register_profile("sa_worker", {"skill_a": skill_a_v1})

        report = checker.run_full_check()
        missing = [c for c in report.conflicts if c.conflict_type == ConflictType.MISSING_SKILL]
        assert len(missing) == 1
        assert missing[0].skill_name == "skill_b"
        assert "sa_worker" in missing[0].details["missing_in"]

    def test_missing_skill_severity_majority_present(self, skill_a_v1):
        """Skill in 2/3 profiles → WARNING."""
        checker = ProfileConsistencyChecker()
        checker.register_profile("sa_main", {"skill_a": skill_a_v1, "shared": skill_a_v1})
        checker.register_profile("sa_worker", {"skill_a": skill_a_v1, "shared": skill_a_v1})
        checker.register_profile("sa_solo", {"skill_a": skill_a_v1})

        report = checker.run_full_check()
        missing = [c for c in report.conflicts if c.conflict_type == ConflictType.MISSING_SKILL]
        assert len(missing) == 1
        assert missing[0].severity == ConflictSeverity.WARNING

    def test_missing_skill_severity_minority_present(self, skill_a_v1):
        """Skill in 1/3 profiles → INFO."""
        checker = ProfileConsistencyChecker()
        checker.register_profile("sa_main", {"skill_a": skill_a_v1, "rare": skill_a_v1})
        checker.register_profile("sa_worker", {"skill_a": skill_a_v1})
        checker.register_profile("sa_solo", {"skill_a": skill_a_v1})

        report = checker.run_full_check()
        missing = [c for c in report.conflicts if c.conflict_type == ConflictType.MISSING_SKILL]
        assert len(missing) == 1
        assert missing[0].severity == ConflictSeverity.INFO


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Constraint resolution helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestConstraintResolutionHelpers:
    def test_version_resolution_major(self):
        versions = {"sa_main": "1.0.0", "sa_worker": "2.0.0"}
        suggestion = _suggest_version_resolution(versions)
        assert "MAJOR" in suggestion or "breaking" in suggestion.lower()

    def test_version_resolution_minor(self):
        versions = {"sa_main": "1.0.0", "sa_worker": "1.1.0"}
        suggestion = _suggest_version_resolution(versions)
        assert "Minor" in suggestion or "minor" in suggestion.lower()

    def test_version_resolution_same(self):
        versions = {"sa_main": "1.0.0", "sa_worker": "1.0.0"}
        suggestion = _suggest_version_resolution(versions)
        assert suggestion == ""  # no conflict

    def test_constraint_resolution_lifecycle_active_vs_deprecated(self):
        values = {
            "sa_main": SkillLifecycleState.ACTIVE,
            "sa_worker": SkillLifecycleState.DEPRECATED,
        }
        suggestion = _suggest_constraint_resolution("lifecycle_state", values)
        assert "active" in suggestion.lower()
        assert "deprecated" in suggestion.lower()

    def test_constraint_resolution_lifecycle_retired_vs_active(self):
        values = {
            "sa_main": SkillLifecycleState.RETIRED,
            "sa_worker": SkillLifecycleState.ACTIVE,
        }
        suggestion = _suggest_constraint_resolution("lifecycle_state", values)
        assert "retire" in suggestion.lower()

    def test_constraint_resolution_risk_mismatch(self):
        values = {"sa_main": "low", "sa_worker": "critical"}
        suggestion = _suggest_constraint_resolution("risk_level", values)
        assert "strictest" in suggestion.lower() or "critical" in suggestion.lower()

    def test_constraint_resolution_generic_field(self):
        values = {"sa_main": "foo", "sa_worker": "bar"}
        suggestion = _suggest_constraint_resolution("custom_field", values)
        assert "foo" in suggestion
        assert "bar" in suggestion


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Report data structure
# ═══════════════════════════════════════════════════════════════════════════════

class TestConflictReport:
    def test_report_is_clean_by_default(self):
        report = ConflictReport()
        assert report.is_clean

    def test_report_not_clean_with_conflicts(self):
        report = ConflictReport(
            conflicts=[
                SkillConflict(
                    skill_name="x",
                    conflict_type=ConflictType.MISSING_SKILL,
                    severity=ConflictSeverity.WARNING,
                )
            ]
        )
        assert not report.is_clean

    def test_report_not_clean_with_validation_errors(self):
        report = ConflictReport(
            validation_errors={"sa_main": ["name is required"]}
        )
        assert not report.is_clean

    def test_error_count(self):
        report = ConflictReport(
            conflicts=[
                SkillConflict(skill_name="a", conflict_type=ConflictType.VERSION_MISMATCH,
                              severity=ConflictSeverity.ERROR),
                SkillConflict(skill_name="b", conflict_type=ConflictType.VERSION_MISMATCH,
                              severity=ConflictSeverity.CRITICAL),
                SkillConflict(skill_name="c", conflict_type=ConflictType.VERSION_MISMATCH,
                              severity=ConflictSeverity.WARNING),
            ]
        )
        assert report.error_count == 2

    def test_warning_count(self):
        report = ConflictReport(
            conflicts=[
                SkillConflict(skill_name="a", conflict_type=ConflictType.VERSION_MISMATCH,
                              severity=ConflictSeverity.WARNING),
                SkillConflict(skill_name="b", conflict_type=ConflictType.VERSION_MISMATCH,
                              severity=ConflictSeverity.INFO),
            ]
        )
        assert report.warning_count == 1

    def test_suggestions_list(self):
        report = ConflictReport(
            conflicts=[
                SkillConflict(skill_name="a", conflict_type=ConflictType.VERSION_MISMATCH,
                              severity=ConflictSeverity.WARNING, suggestion="do X"),
                SkillConflict(skill_name="b", conflict_type=ConflictType.VERSION_MISMATCH,
                              severity=ConflictSeverity.WARNING, suggestion=""),
            ]
        )
        assert len(report.suggestions) == 1
        assert report.suggestions[0] == "do X"

    def test_summary_string(self):
        report = ConflictReport(
            profiles_checked=["sa_main", "sa_worker"],
            skills_checked=["skill_a", "skill_b"],
        )
        summary = report.summary()
        assert "sa_main" in summary
        assert "sa_worker" in summary
        assert "2" in summary  # 2 skills


# ═══════════════════════════════════════════════════════════════════════════════
# Test: SkillConflict string representation
# ═══════════════════════════════════════════════════════════════════════════════

class TestSkillConflict:
    def test_str_representation(self):
        conflict = SkillConflict(
            skill_name="my-skill",
            conflict_type=ConflictType.VERSION_MISMATCH,
            severity=ConflictSeverity.WARNING,
            profiles_involved=["sa_main", "sa_worker"],
            suggestion="Align versions",
        )
        s = str(conflict)
        assert "WARNING" in s
        assert "my-skill" in s
        assert "sa_main" in s
        assert "sa_worker" in s

    def test_version_conflict_inherits_skill_conflict(self):
        vc = VersionConflict(
            skill_name="x",
            versions={"sa_a": "1.0.0", "sa_b": "2.0.0"},
        )
        assert isinstance(vc, SkillConflict)
        assert vc.versions["sa_a"] == "1.0.0"

    def test_constraint_conflict_inherits_skill_conflict(self):
        cc = ConstraintConflict(
            skill_name="y",
            field_name="risk_level",
            values={"sa_a": "low", "sa_b": "high"},
        )
        assert isinstance(cc, SkillConflict)
        assert cc.field_name == "risk_level"


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Three-profile scenarios (split-brain prevention)
# ═══════════════════════════════════════════════════════════════════════════════

class TestThreeProfileSplitBrain:
    def test_three_profiles_version_divergence(self):
        """Three profiles: two agree, one diverges."""
        skill_v1 = UnifiedSkillConfig(
            name="search", version="1.0.0",
            lifecycle_state=SkillLifecycleState.ACTIVE,
        )
        skill_v2 = UnifiedSkillConfig(
            name="search", version="2.0.0",
            lifecycle_state=SkillLifecycleState.ACTIVE,
        )
        checker = ProfileConsistencyChecker()
        checker.register_profile("sa_a", {"search": skill_v1})
        checker.register_profile("sa_b", {"search": skill_v1})
        checker.register_profile("sa_c", {"search": skill_v2})

        report = checker.run_full_check()
        version_conflicts = [
            c for c in report.conflicts
            if c.conflict_type == ConflictType.VERSION_MISMATCH
        ]
        assert len(version_conflicts) == 1
        vc = version_conflicts[0]
        assert isinstance(vc, VersionConflict)
        # Should detect the conflict
        assert "sa_a" in vc.versions
        assert "sa_c" in vc.versions

    def test_three_profiles_lifecycle_chaos(self):
        """Three profiles with different lifecycle states."""
        skill_active = UnifiedSkillConfig(
            name="x", version="1.0.0",
            lifecycle_state=SkillLifecycleState.ACTIVE,
        )
        skill_deprecated = UnifiedSkillConfig(
            name="x", version="1.0.0",
            lifecycle_state=SkillLifecycleState.DEPRECATED,
        )
        skill_retired = UnifiedSkillConfig(
            name="x", version="1.0.0",
            lifecycle_state=SkillLifecycleState.RETIRED,
        )
        checker = ProfileConsistencyChecker()
        checker.register_profile("sa_a", {"x": skill_active})
        checker.register_profile("sa_b", {"x": skill_deprecated})
        checker.register_profile("sa_c", {"x": skill_retired})

        report = checker.run_full_check()
        lifecycle_conflicts = [
            c for c in report.conflicts
            if c.conflict_type == ConflictType.CONSTRAINT_DIVERGENCE
            and isinstance(c, ConstraintConflict)
            and c.field_name == "lifecycle_state"
        ]
        assert len(lifecycle_conflicts) >= 1
        # At least one should be CRITICAL (active vs retired)
        severities = {c.severity for c in lifecycle_conflicts}
        assert ConflictSeverity.CRITICAL in severities

    def test_three_profiles_partial_missing(self):
        """Skill exists in 2 of 3 profiles."""
        skill = UnifiedSkillConfig(
            name="shared", version="1.0.0",
            lifecycle_state=SkillLifecycleState.ACTIVE,
        )
        checker = ProfileConsistencyChecker()
        checker.register_profile("sa_a", {"shared": skill, "only_a": skill})
        checker.register_profile("sa_b", {"shared": skill})
        checker.register_profile("sa_c", {"shared": skill})

        report = checker.run_full_check()
        missing = [c for c in report.conflicts if c.conflict_type == ConflictType.MISSING_SKILL]
        assert len(missing) == 1
        assert missing[0].skill_name == "only_a"


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Deprecation path consistency
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeprecationPathConsistency:
    def test_deprecated_skill_dependency_mismatch(self):
        """Same skill deprecated in both profiles but different dependencies."""
        skill_a = UnifiedSkillConfig(
            name="old-skill", version="1.0.0",
            lifecycle_state=SkillLifecycleState.DEPRECATED,
            dependencies=["dep1", "dep2"],
        )
        skill_b = UnifiedSkillConfig(
            name="old-skill", version="1.0.0",
            lifecycle_state=SkillLifecycleState.DEPRECATED,
            dependencies=["dep1", "dep3"],
        )
        checker = ProfileConsistencyChecker()
        checker.register_profile("sa_main", {"old-skill": skill_a})
        checker.register_profile("sa_worker", {"old-skill": skill_b})

        report = checker.run_full_check()
        dep_conflicts = [
            c for c in report.conflicts
            if isinstance(c, ConstraintConflict) and c.field_name == "dependencies"
        ]
        assert len(dep_conflicts) == 1

    def test_deprecated_skill_composes_with_mismatch(self):
        """Same skill deprecated but different composes_with."""
        skill_a = UnifiedSkillConfig(
            name="old-skill", version="1.0.0",
            lifecycle_state=SkillLifecycleState.DEPRECATED,
            composes_with=["skill_x"],
        )
        skill_b = UnifiedSkillConfig(
            name="old-skill", version="1.0.0",
            lifecycle_state=SkillLifecycleState.DEPRECATED,
            composes_with=["skill_x", "skill_y"],
        )
        checker = ProfileConsistencyChecker()
        checker.register_profile("sa_main", {"old-skill": skill_a})
        checker.register_profile("sa_worker", {"old-skill": skill_b})

        report = checker.run_full_check()
        compose_conflicts = [
            c for c in report.conflicts
            if isinstance(c, ConstraintConflict) and c.field_name == "composes_with"
        ]
        assert len(compose_conflicts) == 1

    def test_active_skill_no_dependency_check(self):
        """Active skills don't trigger dependency checks."""
        skill_a = UnifiedSkillConfig(
            name="active-skill", version="1.0.0",
            lifecycle_state=SkillLifecycleState.ACTIVE,
            dependencies=["dep1"],
        )
        skill_b = UnifiedSkillConfig(
            name="active-skill", version="1.0.0",
            lifecycle_state=SkillLifecycleState.ACTIVE,
            dependencies=["dep1", "dep2"],
        )
        checker = ProfileConsistencyChecker()
        checker.register_profile("sa_main", {"active-skill": skill_a})
        checker.register_profile("sa_worker", {"active-skill": skill_b})

        report = checker.run_full_check()
        dep_conflicts = [
            c for c in report.conflicts
            if isinstance(c, ConstraintConflict) and c.field_name == "dependencies"
        ]
        assert len(dep_conflicts) == 0  # No check for active skills


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_no_profiles_at_all(self):
        checker = ProfileConsistencyChecker()
        report = checker.run_full_check()
        assert report.is_clean

    def test_many_profiles_same_config(self):
        """10 profiles all with the same config → no conflicts."""
        skill = UnifiedSkillConfig(
            name="shared", version="1.0.0",
            lifecycle_state=SkillLifecycleState.ACTIVE,
        )
        checker = ProfileConsistencyChecker()
        for i in range(10):
            checker.register_profile(f"sa_{i}", {"shared": skill})

        report = checker.run_full_check()
        assert report.is_clean

    def test_last_report_cached(self, two_profile_checker):
        assert two_profile_checker.last_report is None
        two_profile_checker.run_full_check()
        assert two_profile_checker.last_report is not None

    def test_rerun_overwrites_report(self, two_profile_checker):
        two_profile_checker.run_full_check()
        first_report = two_profile_checker.last_report
        two_profile_checker.run_full_check()
        second_report = two_profile_checker.last_report
        assert first_report is not second_report
