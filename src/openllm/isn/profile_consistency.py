"""
ISN Cross-Profile Consistency Checker — 防止分裂大脑
===================================================

当多个 SA 实例共享技能索引但运行不同模型上下文时，可能出现：
  - 同一技能在不同 profile 中版本不一致
  - 同一技能在不同 profile 中约束条件不一致（risk_level / lifecycle_state / contract）
  - 退出工程化路径在不同 profile 中状态不一致

本模块检测这些冲突，生成报告，并提供解决建议。

架构：
  - ProfileConsistencyChecker: 核心检查器
  - SkillConflict: 冲突基类
  - VersionConflict: 版本冲突
  - ConstraintConflict: 约束冲突
  - ConflictReport: 冲突报告

集成点：
  - UnifiedSkillConfig.validate() — 每个 skill 的单 profile 内验证
  - ProfileConsistencyChecker — 跨 profile 一致性验证

用法：
    from openllm.isn.profile_consistency import ProfileConsistencyChecker

    checker = ProfileConsistencyChecker()
    checker.register_profile("sa_main", {"skill_a": config_a, "skill_b": config_b})
    checker.register_profile("sa_worker", {"skill_a": config_a_v2, "skill_c": config_c})

    report = checker.run_full_check()
    if not report.is_clean:
        for suggestion in report.suggestions:
            print(suggestion)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from openllm.isn.unified_skill_config import (
    SkillLifecycleState,
    UnifiedSkillConfig,
)

logger = logging.getLogger("openllm.isn.profile_consistency")


# ═══════════════════════════════════════════════════════════════════════════════
# Conflict severity
# ═══════════════════════════════════════════════════════════════════════════════


class ConflictSeverity(Enum):
    """冲突严重程度。"""
    INFO = "info"           # 信息性差异，不需立即处理
    WARNING = "warning"     # 需关注，可能导致不一致行为
    ERROR = "error"         # 必须修复，会导致分裂大脑
    CRITICAL = "critical"   # 紧急：安全/数据完整性风险


class ConflictType(Enum):
    """冲突类型。"""
    VERSION_MISMATCH = "version_mismatch"
    LIFECYCLE_STATE_MISMATCH = "lifecycle_state_mismatch"
    RISK_LEVEL_MISMATCH = "risk_level_mismatch"
    CONSTRAINT_DIVERGENCE = "constraint_divergence"
    CONTRACT_INCOMPATIBILITY = "contract_incompatibility"
    MISSING_SKILL = "missing_skill"


# ═══════════════════════════════════════════════════════════════════════════════
# Conflict data classes
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SkillConflict:
    """技能冲突基类 — 记录跨 profile 的差异。"""
    skill_name: str
    conflict_type: ConflictType
    severity: ConflictSeverity
    profiles_involved: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    suggestion: str = ""

    def __str__(self) -> str:
        profiles_str = ", ".join(self.profiles_involved)
        return (
            f"[{self.severity.value.upper()}] {self.conflict_type.value} "
            f"on '{self.skill_name}' across [{profiles_str}]: {self.suggestion}"
        )


@dataclass
class VersionConflict(SkillConflict):
    """版本冲突 — 同一技能在不同 profile 中版本号不同。"""
    versions: dict[str, str] = field(default_factory=dict)  # profile → version

    def __init__(
        self,
        skill_name: str,
        versions: dict[str, str],
        severity: ConflictSeverity = ConflictSeverity.WARNING,
    ):
        profiles = list(versions.keys())
        suggestion = _suggest_version_resolution(versions)
        super().__init__(
            skill_name=skill_name,
            conflict_type=ConflictType.VERSION_MISMATCH,
            severity=severity,
            profiles_involved=profiles,
            details={"versions": dict(versions)},
            suggestion=suggestion,
        )
        self.versions = dict(versions)


@dataclass
class ConstraintConflict(SkillConflict):
    """约束冲突 — 同一技能在不同 profile 中约束条件不一致。"""
    field_name: str = ""
    values: dict[str, Any] = field(default_factory=dict)  # profile → value

    def __init__(
        self,
        skill_name: str,
        field_name: str,
        values: dict[str, Any],
        severity: ConflictSeverity = ConflictSeverity.WARNING,
    ):
        profiles = list(values.keys())
        suggestion = _suggest_constraint_resolution(field_name, values)
        super().__init__(
            skill_name=skill_name,
            conflict_type=ConflictType.CONSTRAINT_DIVERGENCE,
            severity=severity,
            profiles_involved=profiles,
            details={"field_name": field_name, "values": dict(values)},
            suggestion=suggestion,
        )
        self.field_name = field_name
        self.values = dict(values)


# ═══════════════════════════════════════════════════════════════════════════════
# Conflict report
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ConflictReport:
    """跨 profile 一致性检查报告。"""
    conflicts: list[SkillConflict] = field(default_factory=list)
    profiles_checked: list[str] = field(default_factory=list)
    skills_checked: list[str] = field(default_factory=list)
    validation_errors: dict[str, list[str]] = field(default_factory=dict)
    # profile → [error messages] for per-profile validation failures

    @property
    def is_clean(self) -> bool:
        """是否有冲突或错误。"""
        return (
            len(self.conflicts) == 0
            and len(self.validation_errors) == 0
        )

    @property
    def error_count(self) -> int:
        """严重错误和紧急数量。"""
        return sum(
            1 for c in self.conflicts
            if c.severity in (ConflictSeverity.ERROR, ConflictSeverity.CRITICAL)
        )

    @property
    def warning_count(self) -> int:
        return sum(
            1 for c in self.conflicts if c.severity == ConflictSeverity.WARNING
        )

    @property
    def suggestions(self) -> list[str]:
        """所有冲突的解决建议。"""
        return [c.suggestion for c in self.conflicts if c.suggestion]

    def summary(self) -> str:
        """人类可读的摘要。"""
        lines = [
            f"Cross-Profile Consistency Report",
            f"  Profiles: {', '.join(self.profiles_checked)}",
            f"  Skills: {len(self.skills_checked)}",
            f"  Conflicts: {len(self.conflicts)} "
            f"(errors={self.error_count}, warnings={self.warning_count})",
            f"  Validation errors: {len(self.validation_errors)} profiles",
        ]
        if not self.is_clean:
            lines.append("")
            for conflict in self.conflicts:
                lines.append(f"  {conflict}")
            if self.validation_errors:
                lines.append("")
                lines.append("  Per-profile validation errors:")
                for profile, errs in self.validation_errors.items():
                    for err in errs:
                        lines.append(f"    [{profile}] {err}")
        else:
            lines.append("  ✅ All consistent — no conflicts detected.")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Resolution suggestion helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _suggest_version_resolution(versions: dict[str, str]) -> str:
    """根据版本号差异生成解决建议。"""
    if len(versions) < 2:
        return ""

    # 用 SemVer 简单比较（major.minor.patch）
    def _parse_ver(v: str) -> tuple[int, ...]:
        parts = v.split(".")
        try:
            return tuple(int(p) for p in parts)
        except (ValueError, TypeError):
            return (0,)

    sorted_versions = sorted(versions.items(), key=lambda x: _parse_ver(x[1]))
    lowest_profile, lowest_ver = sorted_versions[0]
    highest_profile, highest_ver = sorted_versions[-1]

    if lowest_ver == highest_ver:
        return ""  # 其实没冲突

    # 判断是否是 major 版本差异
    lowest_parts = _parse_ver(lowest_ver)
    highest_parts = _parse_ver(highest_ver)
    is_major_diff = (
        len(lowest_parts) > 0
        and len(highest_parts) > 0
        and lowest_parts[0] != highest_parts[0]
    )

    if is_major_diff:
        return (
            f"MAJOR version gap ({lowest_ver} in {lowest_profile} vs "
            f"{highest_ver} in {highest_profile}): "
            f"Check breaking changes. Upgrade {lowest_profile} to {highest_ver} "
            f"or pin {highest_profile} to {lowest_ver}."
        )
    else:
        return (
            f"Minor version gap ({lowest_ver} in {lowest_profile} vs "
            f"{highest_ver} in {highest_profile}): "
            f"Align to latest ({highest_ver}). "
            f"Run: sync skill '{lowest_profile}:{lowest_ver}' → {highest_ver}."
        )


def _suggest_constraint_resolution(field_name: str, values: dict[str, Any]) -> str:
    """根据约束差异生成解决建议。"""
    profiles_vals = {p: str(v) for p, v in values.items()}
    unique_vals = set(profiles_vals.values())

    if len(unique_vals) <= 1:
        return ""

    # lifecycle_state 特殊处理
    if field_name == "lifecycle_state":
        states = set()
        for v in values.values():
            if isinstance(v, SkillLifecycleState):
                states.add(v)
            else:
                try:
                    states.add(SkillLifecycleState(v))
                except (ValueError, TypeError):
                    states.add(v)

        # 如果有任何 profile 在 retired 状态而其他不是
        if any(s == SkillLifecycleState.RETIRED for s in states if isinstance(s, SkillLifecycleState)):
            non_retired = [
                p for p, s in values.items()
                if not (isinstance(s, SkillLifecycleState) and s == SkillLifecycleState.RETIRED)
            ]
            return (
                f"Lifecycle state divergence: some profiles retired '{field_name}' "
                f"but {non_retired} still active/dormant. "
                f"Either propagate retirement to all profiles or revert."
            )

        # 如果有 deprecated 和 active 共存
        has_active = any(
            (isinstance(v, SkillLifecycleState) and v == SkillLifecycleState.ACTIVE)
            or v == "active"
            for v in values.values()
        )
        has_deprecated = any(
            (isinstance(v, SkillLifecycleState) and v == SkillLifecycleState.DEPRECATED)
            or v == "deprecated"
            for v in values.values()
        )
        if has_active and has_deprecated:
            return (
                f"Lifecycle state conflict: some profiles have skill as 'active' "
                f"while others mark it 'deprecated'. "
                f"Decide: promote all to active or deprecate across all profiles."
            )

    # risk_level 特殊处理
    if field_name == "risk_level":
        risk_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        max_risk = max(values.values(), key=lambda v: -risk_order.get(str(v), 2))
        min_risk = min(values.values(), key=lambda v: -risk_order.get(str(v), 2))
        return (
            f"Risk level mismatch: {field_name} ranges from "
            f"'{min_risk}' to '{max_risk}' across profiles. "
            f"Adopt the strictest (highest) risk level: '{max_risk}'."
        )

    # 通用处理
    val_descriptions = ", ".join(
        f"'{v}' in {p}" for p, v in profiles_vals.items()
    )
    return (
        f"Constraint divergence on '{field_name}': {val_descriptions}. "
        f"Align all profiles to a single value."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ProfileConsistencyChecker
# ═══════════════════════════════════════════════════════════════════════════════


class ProfileConsistencyChecker:
    """
    跨 Profile 一致性检查器 — 防止多 SA 实例分裂大脑。

    每个 profile 维护自己的 skill index（映射 skill name → UnifiedSkillConfig）。
    多个 SA 实例可能共享同一个 skill index，但运行在不同的模型上下文中。
    本检查器确保它们对同一技能的配置保持一致。

    检查维度：
      1. 单 profile 内验证（调用 UnifiedSkillConfig.validate()）
      2. 跨 profile 版本一致性
      3. 跨 profile 约束一致性（lifecycle_state, risk_level, contracts）
      4. 退出工程化路径一致性（deprecated/retired 状态同步）
    """

    # 需要跨 profile 检查的约束字段
    CONSTRAINT_FIELDS: list[str] = [
        "lifecycle_state",
        "risk_level",
    ]

    # 当 lifecycle_state=DEPRECATED 时额外检查的字段
    DEPRECATION_EXTRA_FIELDS: list[str] = [
        "dependencies",
        "composes_with",
    ]

    def __init__(self) -> None:
        self._profiles: dict[str, dict[str, UnifiedSkillConfig]] = {}
        self._last_report: Optional[ConflictReport] = None

    @property
    def profiles(self) -> dict[str, dict[str, UnifiedSkillConfig]]:
        """已注册的 profiles。"""
        return dict(self._profiles)

    @property
    def last_report(self) -> Optional[ConflictReport]:
        """上次检查的报告。"""
        return self._last_report

    def register_profile(
        self,
        profile_name: str,
        skills: dict[str, UnifiedSkillConfig],
    ) -> None:
        """
        注册一个 profile 的 skill index。

        Args:
            profile_name: profile 标识符（如 "sa_main", "sa_worker"）
            skills: 该 profile 的 skill 映射 (name → UnifiedSkillConfig)
        """
        self._profiles[profile_name] = dict(skills)
        logger.info(
            f"Registered profile '{profile_name}' with {len(skills)} skills"
        )

    def unregister_profile(self, profile_name: str) -> bool:
        """注销一个 profile。返回是否成功。"""
        if profile_name in self._profiles:
            del self._profiles[profile_name]
            logger.info(f"Unregistered profile '{profile_name}'")
            return True
        return False

    def get_all_skill_names(self) -> set[str]:
        """获取所有 profile 中出现过的技能名称。"""
        names: set[str] = set()
        for skills in self._profiles.values():
            names.update(skills.keys())
        return names

    def get_skill_across_profiles(
        self, skill_name: str
    ) -> dict[str, Optional[UnifiedSkillConfig]]:
        """获取某技能在所有 profile 中的配置（缺失的为 None）。"""
        result = {}
        for profile_name in self._profiles:
            result[profile_name] = self._profiles[profile_name].get(skill_name)
        return result

    # ── Check methods ────────────────────────────────────────────────────

    def check_version_conflicts(self) -> list[VersionConflict]:
        """
        检查跨 profile 的版本冲突。

        当同一技能在不同 profile 中有不同版本号时，检测并报告。
        版本号使用 SemVer 简单比较。

        Returns:
            版本冲突列表（空 = 无冲突）
        """
        conflicts: list[VersionConflict] = []
        all_skills = self.get_all_skill_names()

        for skill_name in sorted(all_skills):
            across = self.get_skill_across_profiles(skill_name)

            # 收集非 None 的版本
            versions: dict[str, str] = {}
            for profile_name, config in across.items():
                if config is not None and config.version:
                    versions[profile_name] = config.version

            # 检查是否有不同的版本
            unique_versions = set(versions.values())
            if len(unique_versions) <= 1:
                continue

            # 确定严重程度
            severity = self._assess_version_severity(versions)

            conflict = VersionConflict(
                skill_name=skill_name,
                versions=versions,
                severity=severity,
            )
            conflicts.append(conflict)
            logger.warning(f"Version conflict detected: {conflict}")

        return conflicts

    def check_constraint_conflicts(self) -> list[ConstraintConflict]:
        """
        检查跨 profile 的约束冲突。

        检查维度：
          - lifecycle_state: 生命周期状态不一致
          - risk_level: 风险等级不一致
          - dependencies / composes_with: 已废弃技能的依赖链不一致

        Returns:
            约束冲突列表（空 = 无冲突）
        """
        conflicts: list[ConstraintConflict] = []
        all_skills = self.get_all_skill_names()

        for skill_name in sorted(all_skills):
            across = self.get_skill_across_profiles(skill_name)

            # 检查标准约束字段
            for field_name in self.CONSTRAINT_FIELDS:
                values: dict[str, Any] = {}
                for profile_name, config in across.items():
                    if config is not None:
                        val = getattr(config, field_name, None)
                        if val is not None:
                            values[profile_name] = val

                if len(values) < 2:
                    continue

                unique_vals = set()
                for v in values.values():
                    if hasattr(v, "value"):
                        unique_vals.add(v.value)
                    else:
                        unique_vals.add(str(v))

                if len(unique_vals) > 1:
                    severity = self._assess_constraint_severity(
                        field_name, values
                    )
                    conflict = ConstraintConflict(
                        skill_name=skill_name,
                        field_name=field_name,
                        values=values,
                        severity=severity,
                    )
                    conflicts.append(conflict)
                    logger.warning(f"Constraint conflict detected: {conflict}")

            # 检查废弃技能的额外字段一致性
            lifecycle_vals = {
                p: getattr(c, "lifecycle_state", None)
                for p, c in across.items()
                if c is not None
            }
            has_deprecated = any(
                v in (SkillLifecycleState.DEPRECATED, SkillLifecycleState.RETIRED)
                for v in lifecycle_vals.values()
                if v is not None
            )
            if has_deprecated:
                for field_name in self.DEPRECATION_EXTRA_FIELDS:
                    dep_values: dict[str, Any] = {}
                    for profile_name, config in across.items():
                        if config is not None:
                            val = getattr(config, field_name, None)
                            if val is not None:
                                dep_values[profile_name] = val

                    if len(dep_values) < 2:
                        continue

                    # 比较列表是否一致
                    serialized = {
                        p: str(sorted(v)) if isinstance(v, list) else str(v)
                        for p, v in dep_values.items()
                    }
                    unique_ser = set(serialized.values())
                    if len(unique_ser) > 1:
                        conflict = ConstraintConflict(
                            skill_name=skill_name,
                            field_name=field_name,
                            values=dep_values,
                            severity=ConflictSeverity.WARNING,
                        )
                        conflicts.append(conflict)

        return conflicts

    def validate_profiles(self) -> dict[str, list[str]]:
        """
        对每个 profile 中的每个 skill 执行 UnifiedSkillConfig.validate()。

        Returns:
            profile → 错误消息列表的映射
        """
        errors: dict[str, list[str]] = {}
        for profile_name, skills in self._profiles.items():
            profile_errors: list[str] = []
            for skill_name, config in skills.items():
                skill_errors = config.validate()
                for err in skill_errors:
                    profile_errors.append(f"'{skill_name}': {err}")
            if profile_errors:
                errors[profile_name] = profile_errors
        return errors

    def suggest_resolution(self, conflict: SkillConflict) -> str:
        """
        为给定冲突生成具体的解决建议。

        Args:
            conflict: 任何 SkillConflict 子类实例

        Returns:
            人类可读的解决建议字符串
        """
        # 直接使用冲突对象已生成的建议
        if conflict.suggestion:
            return conflict.suggestion

        # 补充通用建议
        profiles_str = ", ".join(conflict.profiles_involved)

        if conflict.conflict_type == ConflictType.MISSING_SKILL:
            missing_in = conflict.details.get("missing_in", [])
            present_in = conflict.details.get("present_in", [])
            return (
                f"Skill '{conflict.skill_name}' exists in {present_in} "
                f"but not in {missing_in}. "
                f"Either install the skill in all profiles or "
                f"remove it from profiles where it's unused."
            )

        return (
            f"Review {conflict.conflict_type.value} on '{conflict.skill_name}' "
            f"across [{profiles_str}] and align configurations."
        )

    # ── Full check pipeline ──────────────────────────────────────────────

    def run_full_check(self) -> ConflictReport:
        """
        执行完整的跨 profile 一致性检查。

        检查顺序：
          1. 每个 profile 内的 UnifiedSkillConfig.validate()
          2. 跨 profile 版本冲突
          3. 跨 profile 约束冲突
          4. 生成报告

        Returns:
            完整的冲突报告
        """
        report = ConflictReport(
            profiles_checked=sorted(self._profiles.keys()),
            skills_checked=sorted(self.get_all_skill_names()),
        )

        # Step 1: Per-profile validation
        report.validation_errors = self.validate_profiles()

        # Step 2: Version conflicts
        version_conflicts = self.check_version_conflicts()
        report.conflicts.extend(version_conflicts)

        # Step 3: Constraint conflicts
        constraint_conflicts = self.check_constraint_conflicts()
        report.conflicts.extend(constraint_conflicts)

        # Step 4: Check for missing skills (skill in some profiles but not others)
        missing_conflicts = self._check_missing_skills()
        report.conflicts.extend(missing_conflicts)

        self._last_report = report

        logger.info(
            f"Full check complete: {len(report.conflicts)} conflicts, "
            f"{len(report.validation_errors)} profile errors, "
            f"clean={report.is_clean}"
        )
        return report

    # ── Private helpers ──────────────────────────────────────────────────

    def _check_missing_skills(self) -> list[SkillConflict]:
        """检测只存在于部分 profile 中的技能。"""
        if len(self._profiles) < 2:
            return []

        conflicts: list[SkillConflict] = []
        all_skills = self.get_all_skill_names()
        profile_names = sorted(self._profiles.keys())

        for skill_name in sorted(all_skills):
            present_in = [
                p for p in profile_names
                if skill_name in self._profiles[p]
            ]
            missing_in = [
                p for p in profile_names
                if skill_name not in self._profiles[p]
            ]

            if not missing_in:
                continue

            # 仅在多于一半的 profile 中存在时报 warning
            present_ratio = len(present_in) / len(profile_names)
            if present_ratio > 0.5:
                severity = ConflictSeverity.WARNING
            else:
                severity = ConflictSeverity.INFO

            conflict = SkillConflict(
                skill_name=skill_name,
                conflict_type=ConflictType.MISSING_SKILL,
                severity=severity,
                profiles_involved=profile_names,
                details={
                    "present_in": present_in,
                    "missing_in": missing_in,
                },
                suggestion=(
                    f"Skill '{skill_name}' present in {present_in} "
                    f"but missing from {missing_in}. "
                    f"Propagate or document the intentional absence."
                ),
            )
            conflicts.append(conflict)

        return conflicts

    def _assess_version_severity(
        self, versions: dict[str, str]
    ) -> ConflictSeverity:
        """评估版本冲突的严重程度。"""
        def _parse_ver(v: str) -> tuple[int, ...]:
            parts = v.split(".")
            try:
                return tuple(int(p) for p in parts)
            except (ValueError, TypeError):
                return (0,)

        parsed = {p: _parse_ver(v) for p, v in versions.items()}
        all_parsed = list(parsed.values())

        # major 版本差异 = ERROR
        majors = {p[0] if p else 0 for p in all_parsed}
        if len(majors) > 1:
            return ConflictSeverity.ERROR

        # minor 版本差异 = WARNING
        minors = {p[1] if len(p) > 1 else 0 for p in all_parsed}
        if len(minors) > 1:
            return ConflictSeverity.WARNING

        # patch 版本差异 = INFO
        return ConflictSeverity.INFO

    def _assess_constraint_severity(
        self, field_name: str, values: dict[str, Any]
    ) -> ConflictSeverity:
        """评估约束冲突的严重程度。"""
        if field_name == "lifecycle_state":
            # retired vs active = CRITICAL（有 profile 认为已退役）
            states = set()
            for v in values.values():
                if isinstance(v, SkillLifecycleState):
                    states.add(v)
                else:
                    try:
                        states.add(SkillLifecycleState(v))
                    except (ValueError, TypeError):
                        pass

            has_retired = SkillLifecycleState.RETIRED in states
            has_active = SkillLifecycleState.ACTIVE in states
            has_deprecated = SkillLifecycleState.DEPRECATED in states

            if has_retired and has_active:
                return ConflictSeverity.CRITICAL
            if has_retired and has_deprecated:
                return ConflictSeverity.ERROR
            if has_deprecated and has_active:
                return ConflictSeverity.ERROR
            if has_retired:
                return ConflictSeverity.WARNING
            return ConflictSeverity.WARNING

        if field_name == "risk_level":
            risk_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            risk_vals = set()
            for v in values.values():
                risk_vals.add(str(v))

            if "critical" in risk_vals and "low" in risk_vals:
                return ConflictSeverity.CRITICAL
            if "critical" in risk_vals:
                return ConflictSeverity.ERROR
            if "high" in risk_vals and len(risk_vals) > 1:
                return ConflictSeverity.WARNING
            return ConflictSeverity.WARNING

        return ConflictSeverity.WARNING
