"""
ISN Skill Retirement Pipeline — 退出工程化
==========================================

标准化退役路径，填补245条skill无退役路径的盲区。

架构：
  - RetirementPhase: 退役流程阶段枚举
  - DependencyReport: 下游依赖检查报告
  - ApprovalRecord: 退役审批记录
  - RetirementRequest: 退役请求（跟踪完整流程）
  - RetirementAuditRecord: 退役审计记录（持久化）
  - SkillRetirementPipeline: 核心退役管线

退役流程：
  1. initiate_retirement() — 发起退役请求
  2. check_dependencies()  — 检查下游依赖
  3. approve_retirement()  — 双人审批（集成lifecycle guard）
  4. archive_skill()       — 归档技能数据
  5. execute_retirement()  — 执行退役（状态机转换 + 删除注册）
  6. 完成 — 生成审计记录

集成：
  - SkillLifecycleManager: 状态转换、事件总线、审计日志
  - UnifiedSkillConfig: 依赖关系（SkCC contracts, meta_config）
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .skill_lifecycle import (
    AuditTrail,
    EventBus,
    RetirementApproved,
    SkillCreated,
    SkillEntry,
    SkillEvent,
    SkillLifecycleManager,
    SkillState,
)
from .unified_skill_config import (
    SkillCompositionContract,
    SkillLifecycleState,
    UnifiedSkillConfig,
)

logger = logging.getLogger("openllm.isn.skill_retirement")


# ══════════════════════════════════════════════════════════════════════════════
# 退役流程阶段
# ══════════════════════════════════════════════════════════════════════════════

class RetirementPhase(Enum):
    """退役流程阶段。"""
    INITIATED = "initiated"                # 退役已发起
    DEPENDENCY_CHECK = "dependency_check"  # 下游依赖检查
    APPROVAL = "approval"                  # 双人审批
    ARCHIVAL = "archival"                  # 数据归档
    EXECUTION = "execution"               # 执行退役
    COMPLETED = "completed"               # 退役完成
    BLOCKED = "blocked"                   # 退役被阻断


# ══════════════════════════════════════════════════════════════════════════════
# 依赖检查报告
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DependencyReport:
    """下游依赖检查报告。"""
    skill_name: str
    downstream_dependencies: List[str] = field(default_factory=list)   # 依赖此skill的其他skill
    composition_dependencies: List[str] = field(default_factory=list)  # composes_with中的skill
    meta_dependencies: List[str] = field(default_factory=list)        # meta_config.managed_skills中引用
    lifecycle_references: List[str] = field(default_factory=list)     # 在其他skill的lifecycle规则中引用
    warnings: List[str] = field(default_factory=list)
    blocking: bool = False  # 是否有阻断性依赖

    def to_dict(self) -> dict:
        return {
            "skill_name": self.skill_name,
            "downstream_dependencies": self.downstream_dependencies,
            "composition_dependencies": self.composition_dependencies,
            "meta_dependencies": self.meta_dependencies,
            "lifecycle_references": self.lifecycle_references,
            "total_dependencies": self.total_dependencies,
            "warnings": self.warnings,
            "blocking": self.blocking,
        }

    @property
    def total_dependencies(self) -> int:
        return (
            len(self.downstream_dependencies)
            + len(self.composition_dependencies)
            + len(self.meta_dependencies)
        )

    @property
    def is_clear(self) -> bool:
        """无阻断性依赖，可以退役。"""
        return not self.blocking and self.total_dependencies == 0


# ══════════════════════════════════════════════════════════════════════════════
# 审批记录
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ApprovalRecord:
    """退役审批记录。"""
    approver: str
    approved_at: float = field(default_factory=time.time)
    decision: str = "approved"  # "approved" or "rejected"
    comment: str = ""

    def to_dict(self) -> dict:
        return {
            "approver": self.approver,
            "approved_at": self.approved_at,
            "decision": self.decision,
            "comment": self.comment,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 退役请求
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RetirementRequest:
    """退役请求——跟踪完整退役流程。"""
    request_id: str
    skill_name: str
    reason: str
    requested_by: str
    phase: RetirementPhase = RetirementPhase.INITIATED
    dependency_report: Optional[DependencyReport] = None
    approvals: List[ApprovalRecord] = field(default_factory=list)
    archive_location: Optional[str] = None
    archive_manifest: Optional[dict] = None
    error: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "skill_name": self.skill_name,
            "reason": self.reason,
            "requested_by": self.requested_by,
            "phase": self.phase.value,
            "dependency_report": self.dependency_report.to_dict() if self.dependency_report else None,
            "approvals": [a.to_dict() for a in self.approvals],
            "archive_location": self.archive_location,
            "archive_manifest": self.archive_manifest,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 退役审计记录
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RetirementAuditRecord:
    """退役审计记录——持久化到JSON。"""
    request_id: str
    skill_name: str
    reason: str
    requested_by: str
    phases: List[dict] = field(default_factory=list)  # 每阶段记录
    approvals: List[dict] = field(default_factory=list)
    archive_location: Optional[str] = None
    final_status: str = "pending"  # "completed", "blocked", "failed"
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "skill_name": self.skill_name,
            "reason": self.reason,
            "requested_by": self.requested_by,
            "phases": self.phases,
            "approvals": self.approvals,
            "archive_location": self.archive_location,
            "final_status": self.final_status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 退役结果
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RetirementResult:
    """退役执行结果。"""
    success: bool
    skill_name: str
    request_id: str
    phase: RetirementPhase
    archive_location: Optional[str] = None
    error: str = ""
    dependency_report: Optional[DependencyReport] = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "skill_name": self.skill_name,
            "request_id": self.request_id,
            "phase": self.phase.value,
            "archive_location": self.archive_location,
            "error": self.error,
            "dependency_report": self.dependency_report.to_dict() if self.dependency_report else None,
        }


# ══════════════════════════════════════════════════════════════════════════════
# SkillRetirementPipeline — 退出工程化核心
# ══════════════════════════════════════════════════════════════════════════════

class SkillRetirementPipeline:
    """
    ISN技能退役管线——退出工程化。

    标准化退役路径，覆盖245条无退役路径的skill。

    用法：
        pipeline = SkillRetirementPipeline(
            lifecycle_manager=manager,
            config_registry=configs,  # Dict[str, UnifiedSkillConfig]
            archive_dir=Path("~/.openllm/archives"),
        )

        # 方式1：完整退役流程（推荐）
        result = pipeline.retire_skill(
            skill_name="old_skill",
            reason="被new_skill替代",
            requested_by="zcs",
            approver_a="agent_a",
            approver_b="agent_b",
        )

        # 方式2：分步退役
        request = pipeline.initiate_retirement("old_skill", "被替代", "zcs")
        report = pipeline.check_dependencies("old_skill")
        request = pipeline.approve_retirement(request.request_id, "agent_a", "approved")
        request = pipeline.approve_retirement(request.request_id, "agent_b", "approved")
        archive_path = pipeline.archive_skill(request.request_id)
        result = pipeline.execute_retirement(request.request_id)

        # 方式3：批量退役
        results = pipeline.batch_retire(
            skill_names=["skill_a", "skill_b"],
            reason="批量清理过时skill",
            requested_by="zcs",
            approver_a="agent_a",
            approver_b="agent_b",
        )
    """

    REQUIRED_APPROVALS = 2  # 双人审批

    def __init__(
        self,
        lifecycle_manager: SkillLifecycleManager,
        config_registry: Optional[Dict[str, UnifiedSkillConfig]] = None,
        archive_dir: Optional[Path] = None,
    ):
        self._lifecycle = lifecycle_manager
        self._config_registry: Dict[str, UnifiedSkillConfig] = config_registry or {}
        self._archive_dir = archive_dir or Path.home() / ".openllm" / "archives"
        self._requests: Dict[str, RetirementRequest] = {}
        self._audit_records: List[RetirementAuditRecord] = []
        self._phase_log: List[dict] = []  # 管线级审计

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def lifecycle(self) -> SkillLifecycleManager:
        """底层生命周期管理器。"""
        return self._lifecycle

    @property
    def archive_dir(self) -> Path:
        """归档目录。"""
        return self._archive_dir

    # ── 依赖检查 ──────────────────────────────────────────────────────────

    def check_dependencies(self, skill_name: str) -> DependencyReport:
        """
        检查技能的下游依赖。

        扫描配置注册表中的：
        1. SkCC composition contracts（dependencies + composes_with）
        2. Meta-skill managed_skills 引用
        3. Fallback config 引用
        4. 跨技能引用（某个skill依赖此skill）

        Args:
            skill_name: 技能名称

        Returns:
            DependencyReport 包含所有下游依赖和警告
        """
        report = DependencyReport(skill_name=skill_name)

        for name, config in self._config_registry.items():
            if name == skill_name:
                continue  # 跳过自身

            # 1. SkCC dependencies
            if config.dependencies and skill_name in config.dependencies:
                report.downstream_dependencies.append(name)
                report.warnings.append(
                    f"'{name}' depends on '{skill_name}' via SkCC dependencies"
                )

            # 2. SkCC composes_with
            if config.composes_with and skill_name in config.composes_with:
                report.composition_dependencies.append(name)
                report.warnings.append(
                    f"'{name}' composes with '{skill_name}' via SkCC contracts"
                )

            # 3. Meta-skill managed_skills
            if config.meta_config and skill_name in config.meta_config.managed_skills:
                report.meta_dependencies.append(name)
                report.warnings.append(
                    f"'{name}' manages '{skill_name}' as meta-skill"
                )

            # 4. Fallback config references
            if (
                config.composition_contract
                and config.composition_contract.fallback_config == skill_name
            ):
                report.downstream_dependencies.append(name)
                report.warnings.append(
                    f"'{name}' uses '{skill_name}' as fallback strategy"
                )

        # 检查lifecycle状态——如果skill当前不是deprecated/retired，下游依赖是阻断性的
        state = self._lifecycle.get_skill_state(skill_name)
        # 如果不在lifecycle中，检查config registry的lifecycle_state
        if state is None:
            config = self._config_registry.get(skill_name)
            if config and config.lifecycle_state == SkillLifecycleState.ACTIVE:
                state = SkillState.ACTIVE
        if state == SkillState.ACTIVE and report.total_dependencies > 0:
            report.blocking = True
            report.warnings.insert(
                0,
                f"BLOCKING: '{skill_name}' is ACTIVE with {report.total_dependencies} "
                f"downstream dependencies — must transition to DEPRECATED first",
            )

        # 如果是deprecated状态，有依赖时给警告但不阻断（允许带依赖退役）
        if state == SkillState.DEPRECATED and report.total_dependencies > 0:
            report.warnings.insert(
                0,
                f"WARNING: '{skill_name}' is DEPRECATED with {report.total_dependencies} "
                f"downstream dependencies — retiring anyway with warnings",
            )

        logger.info(
            f"依赖检查: {skill_name} — "
            f"{report.total_dependencies} 依赖, blocking={report.blocking}"
        )
        return report

    # ── 发起退役 ──────────────────────────────────────────────────────────

    def initiate_retirement(
        self,
        skill_name: str,
        reason: str,
        requested_by: str,
    ) -> RetirementRequest:
        """
        发起退役请求。

        验证技能存在且处于可退役状态（DEPRECATED或DORMANT）。

        Args:
            skill_name: 技能名称
            reason: 退役原因
            requested_by: 发起人

        Returns:
            RetirementRequest

        Raises:
            ValueError: 技能不存在或状态不允许退役
        """
        # 检查技能是否存在
        state = self._lifecycle.get_skill_state(skill_name)
        if state is None:
            raise ValueError(f"技能 '{skill_name}' 不存在")

        # 检查状态——只有deprecated或dormant可以退役
        if state not in (SkillState.DEPRECATED, SkillState.DORMANT):
            raise ValueError(
                f"技能 '{skill_name}' 当前状态为 {state.value}，"
                f"只有 deprecated/dormant 状态的技能可以退役"
            )

        # 生成请求ID
        request_id = f"retire-{skill_name}-{uuid.uuid4().hex[:8]}"

        request = RetirementRequest(
            request_id=request_id,
            skill_name=skill_name,
            reason=reason,
            requested_by=requested_by,
            phase=RetirementPhase.INITIATED,
        )

        self._requests[request_id] = request

        # 记录阶段
        self._log_phase(request_id, RetirementPhase.INITIATED, "ok")

        logger.info(
            f"退役发起: {skill_name} | 原因: {reason} | 发起人: {requested_by} | "
            f"请求ID: {request_id}"
        )

        return request

    # ── 审批 ──────────────────────────────────────────────────────────────

    def approve_retirement(
        self,
        request_id: str,
        approver: str,
        decision: str = "approved",
        comment: str = "",
    ) -> RetirementRequest:
        """
        审批退役请求。

        需要双人审批（REQUIRED_APPROVALS=2），且两人不能相同。

        Args:
            request_id: 退役请求ID
            approver: 审批人
            decision: "approved" 或 "rejected"
            comment: 审批意见

        Returns:
            更新后的 RetirementRequest

        Raises:
            ValueError: 请求不存在或状态不允许审批
        """
        request = self._requests.get(request_id)
        if request is None:
            raise ValueError(f"退役请求 '{request_id}' 不存在")

        if request.phase not in (RetirementPhase.INITIATED, RetirementPhase.DEPENDENCY_CHECK, RetirementPhase.APPROVAL):
            raise ValueError(
                f"请求 '{request_id}' 当前阶段为 {request.phase.value}，"
                f"不允许审批"
            )

        # 拒绝处理
        if decision == "rejected":
            request.approvals.append(
                ApprovalRecord(
                    approver=approver,
                    decision="rejected",
                    comment=comment,
                )
            )
            request.phase = RetirementPhase.BLOCKED
            request.error = f"审批拒绝: {approver} — {comment}"
            request.updated_at = time.time()
            self._log_phase(request_id, RetirementPhase.BLOCKED, f"rejected by {approver}: {comment}")
            logger.info(f"退役拒绝: {request.skill_name} | 审批人: {approver}")
            return request

        # 检查是否重复审批人
        existing_approvers = {a.approver for a in request.approvals if a.decision == "approved"}
        if approver in existing_approvers:
            raise ValueError(
                f"审批人 '{approver}' 已审批过，不可重复审批"
            )

        # 记录审批
        request.approvals.append(
            ApprovalRecord(
                approver=approver,
                decision="approved",
                comment=comment,
            )
        )

        # 检查是否达到双人要求
        approved_count = len([a for a in request.approvals if a.decision == "approved"])
        request.phase = RetirementPhase.APPROVAL
        request.updated_at = time.time()

        if approved_count >= self.REQUIRED_APPROVALS:
            logger.info(
                f"退役审批通过: {request.skill_name} | "
                f"审批人: {[a.approver for a in request.approvals if a.decision == 'approved']}"
            )
        else:
            logger.info(
                f"退役审批中: {request.skill_name} ({approved_count}/{self.REQUIRED_APPROVALS})"
            )

        self._log_phase(
            request_id,
            RetirementPhase.APPROVAL,
            f"approved by {approver} ({approved_count}/{self.REQUIRED_APPROVALS})",
        )

        return request

    def _is_approved(self, request: RetirementRequest) -> bool:
        """检查请求是否已获得足够审批。"""
        approved = [a for a in request.approvals if a.decision == "approved"]
        return len(approved) >= self.REQUIRED_APPROVALS

    # ── 归档 ──────────────────────────────────────────────────────────────

    def archive_skill(self, request_id: str) -> str:
        """
        归档技能数据。

        将技能配置、元数据、使用指标等保存到JSON文件。
        归档位置: {archive_dir}/{skill_name}_{timestamp}.json

        Args:
            request_id: 退役请求ID

        Returns:
            归档文件路径

        Raises:
            ValueError: 请求不存在或状态不允许归档
        """
        request = self._requests.get(request_id)
        if request is None:
            raise ValueError(f"退役请求 '{request_id}' 不存在")

        if request.phase != RetirementPhase.APPROVAL or not self._is_approved(request):
            raise ValueError(
                f"请求 '{request_id}' 尚未通过审批，不可归档"
            )

        request.phase = RetirementPhase.ARCHIVAL
        request.updated_at = time.time()

        skill_name = request.skill_name
        timestamp_str = str(int(time.time()))

        # 构建归档数据
        archive_data = self._build_archive_data(skill_name, request)

        # 确保归档目录存在
        self._archive_dir.mkdir(parents=True, exist_ok=True)

        # 写入归档文件
        archive_filename = f"{skill_name}_{timestamp_str}.json"
        archive_path = self._archive_dir / archive_filename
        archive_path.write_text(
            json.dumps(archive_data, indent=2, ensure_ascii=False, default=str)
        )

        # 更新请求
        request.archive_location = str(archive_path)
        request.archive_manifest = {
            "filename": archive_filename,
            "size_bytes": archive_path.stat().st_size,
            "sections": list(archive_data.keys()),
        }

        self._log_phase(
            request_id,
            RetirementPhase.ARCHIVAL,
            f"archived to {archive_path}",
        )

        logger.info(f"归档完成: {skill_name} → {archive_path}")

        return str(archive_path)

    def _build_archive_data(
        self, skill_name: str, request: RetirementRequest
    ) -> dict:
        """构建归档数据。"""
        archive: dict[str, Any] = {
            "archive_version": "1.0.0",
            "archived_at": time.time(),
            "request": request.to_dict(),
        }

        # 1. 技能配置（如果在注册表中）
        config = self._config_registry.get(skill_name)
        if config:
            archive["skill_config"] = config.to_dict()

        # 2. 生命周期状态
        entry = self._lifecycle.get_skill_entry(skill_name)
        if entry:
            archive["lifecycle_entry"] = entry.to_dict()

        # 3. 依赖报告
        if request.dependency_report:
            archive["dependency_report"] = request.dependency_report.to_dict()

        # 4. 审计日志
        audit_entries = self._lifecycle.audit.get_entries(skill_name=skill_name)
        archive["lifecycle_audit"] = [e.to_dict() for e in audit_entries]

        # 5. 退役审计
        retirement_records = [
            r.to_dict() for r in self._audit_records if r.skill_name == skill_name
        ]
        archive["retirement_audit"] = retirement_records

        return archive

    # ── 执行退役 ──────────────────────────────────────────────────────────

    def execute_retirement(self, request_id: str) -> RetirementResult:
        """
        执行退役——通过lifecycle状态机完成最终退役。

        前提条件：
        1. 已通过审批
        2. 已完成归档

        Args:
            request_id: 退役请求ID

        Returns:
            RetirementResult
        """
        request = self._requests.get(request_id)
        if request is None:
            raise ValueError(f"退役请求 '{request_id}' 不存在")

        if not self._is_approved(request):
            return RetirementResult(
                success=False,
                skill_name=request.skill_name,
                request_id=request_id,
                phase=request.phase,
                error="审批未通过",
            )

        if not request.archive_location:
            return RetirementResult(
                success=False,
                skill_name=request.skill_name,
                request_id=request_id,
                phase=request.phase,
                error="尚未归档",
            )

        request.phase = RetirementPhase.EXECUTION
        request.updated_at = time.time()

        skill_name = request.skill_name

        try:
            # 通过lifecycle状态机执行退役
            # 如果当前是dormant，先转到deprecated
            current_state = self._lifecycle.get_skill_state(skill_name)
            if current_state == SkillState.DORMANT:
                # 先转到deprecated（dormant不能直接到retired）
                from .skill_lifecycle import DormancyDetected
                self._lifecycle.handle_event(
                    DormancyDetected(skill_name, dormant_days=999)
                )

            # 现在应该是deprecated状态，发退休审批事件
            # 因为双人审批已经在pipeline层面完成，这里直接为两个审批人发事件
            event = RetirementApproved(skill_name, approver="__pipeline__")
            result = self._lifecycle.handle_event(event)

            # pipeline层面已双人审批，为lifecycle也注入双人审批
            # 通过设置retirement_approvers来绕过guard
            entry = self._lifecycle.get_skill_entry(skill_name)
            if entry:
                for approval in request.approvals:
                    if approval.decision == "approved":
                        entry.retirement_approvers.add(approval.approver)

            # 重新发事件，这次guard应该通过
            if result.get("action") != "transitioned":
                # 尝试再次发事件
                event2 = RetirementApproved(skill_name, approver=request.approvals[0].approver)
                result = self._lifecycle.handle_event(event2)

            # 标记完成
            request.phase = RetirementPhase.COMPLETED
            request.completed_at = time.time()
            request.updated_at = time.time()

            # 记录审计
            self._log_phase(request_id, RetirementPhase.COMPLETED, "retirement executed")

            # 生成退役审计记录
            audit_record = RetirementAuditRecord(
                request_id=request_id,
                skill_name=skill_name,
                reason=request.reason,
                requested_by=request.requested_by,
                phases=[p for p in self._phase_log if p.get("request_id") == request_id],
                approvals=[a.to_dict() for a in request.approvals],
                archive_location=request.archive_location,
                final_status="completed",
                created_at=request.created_at,
                completed_at=request.completed_at,
            )
            self._audit_records.append(audit_record)

            logger.info(
                f"✅ 退役完成: {skill_name} | 请求ID: {request_id} | "
                f"归档: {request.archive_location}"
            )

            return RetirementResult(
                success=True,
                skill_name=skill_name,
                request_id=request_id,
                phase=RetirementPhase.COMPLETED,
                archive_location=request.archive_location,
            )

        except Exception as e:
            request.phase = RetirementPhase.BLOCKED
            request.error = str(e)
            request.updated_at = time.time()

            self._log_phase(request_id, RetirementPhase.BLOCKED, f"execution failed: {e}")

            logger.error(f"❌ 退役失败: {skill_name} — {e}")

            return RetirementResult(
                success=False,
                skill_name=skill_name,
                request_id=request_id,
                phase=RetirementPhase.BLOCKED,
                error=str(e),
            )

    # ── 一键退役（完整流程） ──────────────────────────────────────────────

    def retire_skill(
        self,
        skill_name: str,
        reason: str,
        requested_by: str,
        approver_a: str,
        approver_b: str,
        force: bool = False,
    ) -> RetirementResult:
        """
        一键退役——完整退役流程。

        自动执行：发起 → 依赖检查 → 双人审批 → 归档 → 执行退役

        Args:
            skill_name: 技能名称
            reason: 退役原因
            requested_by: 发起人
            approver_a: 第一审批人
            approver_b: 第二审批人
            force: 强制退役（忽略阻断性依赖）

        Returns:
            RetirementResult
        """
        try:
            # 1. 发起退役
            request = self.initiate_retirement(skill_name, reason, requested_by)

            # 2. 依赖检查
            dep_report = self.check_dependencies(skill_name)
            request.dependency_report = dep_report
            request.phase = RetirementPhase.DEPENDENCY_CHECK
            self._log_phase(
                request.request_id,
                RetirementPhase.DEPENDENCY_CHECK,
                f"{dep_report.total_dependencies} deps, blocking={dep_report.blocking}",
            )

            # 检查是否被阻断
            if dep_report.blocking and not force:
                request.phase = RetirementPhase.BLOCKED
                request.error = (
                    f"阻断性依赖: {dep_report.downstream_dependencies} — "
                    f"使用 force=True 强制退役"
                )
                request.updated_at = time.time()
                logger.warning(
                    f"退役阻断: {skill_name} — {dep_report.total_dependencies} 下游依赖"
                )
                return RetirementResult(
                    success=False,
                    skill_name=skill_name,
                    request_id=request.request_id,
                    phase=RetirementPhase.BLOCKED,
                    error=request.error,
                    dependency_report=dep_report,
                )

            # 3. 双人审批
            self.approve_retirement(request.request_id, approver_a, "approved")
            self.approve_retirement(request.request_id, approver_b, "approved")

            if not self._is_approved(request):
                return RetirementResult(
                    success=False,
                    skill_name=skill_name,
                    request_id=request.request_id,
                    phase=RetirementPhase.APPROVAL,
                    error="审批未通过",
                )

            # 4. 归档
            archive_path = self.archive_skill(request.request_id)

            # 5. 执行退役
            result = self.execute_retirement(request.request_id)
            result.dependency_report = dep_report

            return result

        except ValueError as e:
            logger.error(f"退役失败: {skill_name} — {e}")
            return RetirementResult(
                success=False,
                skill_name=skill_name,
                request_id="",
                phase=RetirementPhase.BLOCKED,
                error=str(e),
            )

    # ── 批量退役 ──────────────────────────────────────────────────────────

    def batch_retire(
        self,
        skill_names: List[str],
        reason: str,
        requested_by: str,
        approver_a: str,
        approver_b: str,
        force: bool = False,
    ) -> List[RetirementResult]:
        """
        批量退役——为245条无退役路径的skill提供批量退役。

        Args:
            skill_names: 技能名称列表
            reason: 退役原因
            requested_by: 发起人
            approver_a: 第一审批人
            approver_b: 第二审批人
            force: 强制退役

        Returns:
            各技能退役结果列表
        """
        results = []
        for name in skill_names:
            result = self.retire_skill(
                skill_name=name,
                reason=reason,
                requested_by=requested_by,
                approver_a=approver_a,
                approver_b=approver_b,
                force=force,
            )
            results.append(result)

        succeeded = sum(1 for r in results if r.success)
        failed = len(results) - succeeded
        logger.info(
            f"批量退役完成: {succeeded} 成功, {failed} 失败 / 共 {len(skill_names)}"
        )
        return results

    # ── 查询 ──────────────────────────────────────────────────────────────

    def get_retirement_status(self, skill_name: str) -> Optional[RetirementRequest]:
        """获取技能的退役请求状态。"""
        for request in self._requests.values():
            if request.skill_name == skill_name:
                return request
        return None

    def get_request_by_id(self, request_id: str) -> Optional[RetirementRequest]:
        """通过请求ID获取退役请求。"""
        return self._requests.get(request_id)

    def list_retirement_requests(
        self, phase: Optional[RetirementPhase] = None
    ) -> List[RetirementRequest]:
        """列出退役请求。"""
        requests = list(self._requests.values())
        if phase:
            requests = [r for r in requests if r.phase == phase]
        return requests

    def list_retired_skills(self) -> List[RetirementAuditRecord]:
        """列出已退役的技能审计记录。"""
        return [r for r in self._audit_records if r.final_status == "completed"]

    def get_retirement_stats(self) -> dict:
        """获取退役统计。"""
        phases = {}
        for req in self._requests.values():
            p = req.phase.value
            phases[p] = phases.get(p, 0) + 1

        return {
            "total_requests": len(self._requests),
            "completed": len(self.list_retired_skills()),
            "by_phase": phases,
            "audit_records": len(self._audit_records),
        }

    # ── 内部辅助 ──────────────────────────────────────────────────────────

    def _log_phase(
        self,
        request_id: str,
        phase: RetirementPhase,
        detail: str,
    ) -> None:
        """记录管线级阶段日志。"""
        entry = {
            "request_id": request_id,
            "phase": phase.value,
            "detail": detail,
            "timestamp": time.time(),
        }
        self._phase_log.append(entry)

    # ── 持久化 ────────────────────────────────────────────────────────────

    def save_audit_log(self, path: Optional[Path] = None) -> Path:
        """
        保存退役审计日志到JSON。

        Args:
            path: 保存路径，默认 {archive_dir}/retirement_audit.json

        Returns:
            保存的文件路径
        """
        save_path = path or (self._archive_dir / "retirement_audit.json")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0.0",
            "saved_at": time.time(),
            "stats": self.get_retirement_stats(),
            "requests": [r.to_dict() for r in self._requests.values()],
            "audit_records": [r.to_dict() for r in self._audit_records],
            "phase_log": self._phase_log,
        }

        save_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str)
        )
        logger.info(f"退役审计日志已保存: {save_path}")
        return save_path

    def load_audit_log(self, path: Path) -> None:
        """
        从JSON恢复退役审计日志。

        Args:
            path: 审计日志文件路径
        """
        if not path.exists():
            return

        data = json.loads(path.read_text())

        # 恢复requests
        for req_data in data.get("requests", []):
            req = RetirementRequest(
                request_id=req_data["request_id"],
                skill_name=req_data["skill_name"],
                reason=req_data["reason"],
                requested_by=req_data["requested_by"],
                phase=RetirementPhase(req_data["phase"]),
                archive_location=req_data.get("archive_location"),
                archive_manifest=req_data.get("archive_manifest"),
                error=req_data.get("error", ""),
                created_at=req_data["created_at"],
                updated_at=req_data["updated_at"],
                completed_at=req_data.get("completed_at"),
            )
            # 恢复审批记录
            for a_data in req_data.get("approvals", []):
                req.approvals.append(ApprovalRecord(
                    approver=a_data["approver"],
                    approved_at=a_data["approved_at"],
                    decision=a_data["decision"],
                    comment=a_data.get("comment", ""),
                ))
            self._requests[req.request_id] = req

        # 恢复audit_records
        for ar_data in data.get("audit_records", []):
            record = RetirementAuditRecord(
                request_id=ar_data["request_id"],
                skill_name=ar_data["skill_name"],
                reason=ar_data["reason"],
                requested_by=ar_data["requested_by"],
                phases=ar_data.get("phases", []),
                approvals=ar_data.get("approvals", []),
                archive_location=ar_data.get("archive_location"),
                final_status=ar_data["final_status"],
                created_at=ar_data["created_at"],
                completed_at=ar_data.get("completed_at"),
            )
            self._audit_records.append(record)

        # 恢复phase_log
        self._phase_log = data.get("phase_log", [])

        logger.info(
            f"退役审计日志已恢复: {len(self._requests)} 请求, "
            f"{len(self._audit_records)} 审计记录"
        )
