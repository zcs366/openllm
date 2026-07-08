"""execution_broker.py — IOS动作主权Broker（Layer 4）

syscall与实际执行之间的策略层：
  - 执行前策略检查（参数合规→权限匹配→风险评估）
  - 执行后结果确认（类型匹配→无错误码→一致性）
  - 失败自动回滚（handler.undo()）

纯规则/函数调用，零LLM依赖。
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger("openllm.broker")

# ── 枚举 ──

class BrokerDecision(str, Enum):
    """Broker裁决类型"""
    ALLOW = "allow"       # 允许执行
    DENY = "deny"         # 拒绝执行
    ESCALATE = "escalate"  # 升级给人类
    MODIFIED = "modified"  # 修改参数后执行


class ExecutionPhase(str, Enum):
    """执行阶段"""
    PRE_CHECK = "pre_check"
    EXECUTING = "executing"
    POST_CONFIRM = "post_confirm"
    ROLLBACK = "rollback"


# ── 数据类 ──

@dataclass(frozen=True)
class SyscallRequest:
    """syscall请求"""
    syscall_name: str
    params: dict
    caller: str
    session_id: str = ""
    turn_id: str = ""


@dataclass(frozen=True)
class BrokerVerdict:
    """Broker裁决"""
    decision: BrokerDecision
    reason: str
    modified_params: Optional[dict] = None
    risk_level: str = "low"


@dataclass(frozen=True)
class AuditEntry:
    """审计条目（与immune.py格式兼容）"""
    timestamp: float = field(default_factory=time.time)
    action: str = ""
    memory_id: str = ""
    source: str = ""
    trust_level: str = ""
    threat_type: str = "none"
    details: str = ""
    session_id: str = ""
    checksum: str = ""


@dataclass(frozen=True)
class ExecutionResult:
    """执行结果"""
    success: bool
    result: Any = None
    error: Optional[str] = None
    phase: ExecutionPhase = ExecutionPhase.EXECUTING
    rollback_available: bool = False
    audit_entry: Optional[AuditEntry] = None


# ── 风险规则 ──

# 高风险syscall → 需要升级给人类
_ESCALATE_SYSCALLS: set[str] = {
    "delete_file", "drop_table", "revoke_cap", "shutdown",
}

# 参数必须存在
_REQUIRED_PARAMS: dict[str, list[str]] = {
    "write_file": ["path", "content"],
    "read_file": ["path"],
    "execute": ["command"],
    "delete_file": ["path"],
}

# 高风险参数关键词
_RISKY_KEYWORDS = ("rm ", "DROP ", "DELETE ", "FORMAT ", "shutdown", "reboot")


# ── 核心函数 ──

def _classify_risk(request: SyscallRequest) -> str:
    """评估请求风险等级：low / medium / high / critical"""
    name = request.syscall_name
    if name in _ESCALATE_SYSCALLS:
        return "critical"
    # 参数中含危险关键词
    params_str = str(request.params)
    if any(kw in params_str for kw in _RISKY_KEYWORDS):
        return "high"
    # 需要写入/执行操作
    if name in ("write_file", "execute", "network_request"):
        return "medium"
    return "low"


def _check_params(request: SyscallRequest) -> Optional[str]:
    """检查参数完整性，返回问题描述或None"""
    required = _REQUIRED_PARAMS.get(request.syscall_name)
    if not required:
        return None
    missing = [p for p in required if p not in request.params]
    if missing:
        return f"缺少必需参数: {missing}"
    return None


def pre_check(
    request: SyscallRequest,
    *,
    policy_rules: Optional[dict] = None,
    tool_validator: Optional[Any] = None,
) -> BrokerVerdict:
    """执行前策略检查。

    检查顺序：参数合规 → 风险分类 → 策略规则 → tool_validator
    返回 BrokerVerdict（ALLOW/DENY/ESCALATE/MODIFIED）。
    """
    # 1. 参数检查
    param_err = _check_params(request)
    if param_err:
        logger.warning(
            "Broker DENY: %s — %s", request.syscall_name, param_err
        )
        return BrokerVerdict(
            decision=BrokerDecision.DENY,
            reason=param_err,
            risk_level="low",
        )

    # 2. 风险分类
    risk = _classify_risk(request)

    # 3. 策略规则匹配
    if policy_rules and request.syscall_name in policy_rules.get("deny_list", set()):
        logger.warning(
            "Broker DENY: %s 在deny_list中", request.syscall_name
        )
        return BrokerVerdict(
            decision=BrokerDecision.DENY,
            reason=f"syscall '{request.syscall_name}' 被策略禁止",
            risk_level=risk,
        )

    # 参数修改规则
    if policy_rules and "param_overrides" in policy_rules:
        overrides = policy_rules["param_overrides"].get(request.syscall_name)
        if overrides:
            logger.info(
                "Broker MODIFIED: %s — 应用参数覆盖", request.syscall_name
            )
            merged = {**request.params, **overrides}
            return BrokerVerdict(
                decision=BrokerDecision.MODIFIED,
                reason="应用策略参数覆盖",
                modified_params=merged,
                risk_level=risk,
            )

    # 4. tool_validator可选调用（Layer 3依赖）
    if tool_validator is not None and hasattr(tool_validator, "validate"):
        try:
            vr = tool_validator.validate(request)
            if hasattr(vr, "should_block") and vr.should_block:
                logger.warning(
                    "Broker DENY: tool_validator拦截 %s",
                    request.syscall_name,
                )
                return BrokerVerdict(
                    decision=BrokerDecision.DENY,
                    reason="tool_validator拦截",
                    risk_level=risk,
                )
        except Exception as e:
            logger.warning(
                "tool_validator调用异常: %s — 跳过", e
            )

    # 5. critical风险 → 升级
    if risk == "critical":
        logger.warning(
            "Broker ESCALATE: %s — 风险等级critical",
            request.syscall_name,
        )
        return BrokerVerdict(
            decision=BrokerDecision.ESCALATE,
            reason=f"高风险syscall需人工确认: {request.syscall_name}",
            risk_level=risk,
        )

    # 6. 默认允许
    logger.debug(
        "Broker ALLOW: %s (risk=%s)", request.syscall_name, risk
    )
    return BrokerVerdict(
        decision=BrokerDecision.ALLOW,
        reason="通过策略检查",
        risk_level=risk,
    )


def post_confirm(
    request: SyscallRequest,
    result: ExecutionResult,
) -> ExecutionResult:
    """执行后确认。

    检查：结果有效性 → 一致性 → 审计条目生成。
    返回更新后的ExecutionResult。
    """
    # 已经失败的直接标记phase
    if not result.success:
        phase = ExecutionPhase.POST_CONFIRM
        audit = AuditEntry(
            action="execute_failed",
            source=request.caller,
            details=f"error={result.error}",
            session_id=request.session_id,
        )
        logger.warning(
            "post_confirm FAIL: %s — %s",
            request.syscall_name, result.error,
        )
        return ExecutionResult(
            success=False,
            result=result.result,
            error=result.error,
            phase=phase,
            rollback_available=result.rollback_available,
            audit_entry=audit,
        )

    # 成功路径：生成审计条目
    audit = AuditEntry(
        action="execute_ok",
        source=request.caller,
        details=f"syscall={request.syscall_name}",
        session_id=request.session_id,
    )
    logger.debug("post_confirm OK: %s", request.syscall_name)
    return ExecutionResult(
        success=True,
        result=result.result,
        phase=ExecutionPhase.POST_CONFIRM,
        rollback_available=result.rollback_available,
        audit_entry=audit,
    )


def _try_rollback(handler: Any, request: SyscallRequest) -> bool:
    """尝试回滚：调用handler.undo()（如果存在）。
    Returns: 是否成功回滚。
    """
    if not hasattr(handler, "undo"):
        logger.info("handler无undo方法，跳过回滚: %s", request.syscall_name)
        return False
    try:
        handler.undo()
        logger.info("回滚成功: %s", request.syscall_name)
        return True
    except Exception as e:
        logger.error("回滚失败: %s — %s", request.syscall_name, e)
        return False


def execute_with_broker(
    request: SyscallRequest,
    handler: Callable,
    *,
    pre_check_fn: Callable = pre_check,
    post_confirm_fn: Optional[Callable] = post_confirm,
) -> ExecutionResult:
    """带Broker的完整执行流程。

    1. pre_check → 裁决
    2. ALLOW/MODIFIED → 执行handler
    3. post_confirm → 确认结果
    4. 失败且rollback_available → 自动回滚

    Args:
        request: syscall请求
        handler: 实际执行函数（接受 **params）
        pre_check_fn: 执行前检查函数
        post_confirm_fn: 执行后确认函数（None则跳过）

    Returns:
        ExecutionResult — 包含执行结果、阶段、审计条目
    """
    # Phase 1: pre_check
    verdict = pre_check_fn(request)

    if verdict.decision == BrokerDecision.DENY:
        audit = AuditEntry(
            action="denied",
            source=request.caller,
            details=f"reason={verdict.reason}",
            session_id=request.session_id,
        )
        return ExecutionResult(
            success=False,
            error=verdict.reason,
            phase=ExecutionPhase.PRE_CHECK,
            audit_entry=audit,
        )

    if verdict.decision == BrokerDecision.ESCALATE:
        audit = AuditEntry(
            action="escalated",
            source=request.caller,
            details=f"reason={verdict.reason} risk={verdict.risk_level}",
            session_id=request.session_id,
        )
        return ExecutionResult(
            success=False,
            error=f"需人工确认: {verdict.reason}",
            phase=ExecutionPhase.PRE_CHECK,
            audit_entry=audit,
        )

    # Phase 2: 执行
    exec_params = verdict.modified_params or request.params
    try:
        raw_result = handler(**exec_params)
        result = ExecutionResult(
            success=True,
            result=raw_result,
            phase=ExecutionPhase.EXECUTING,
            rollback_available=hasattr(handler, "undo"),
        )
    except Exception as e:
        logger.error("执行异常: %s — %s", request.syscall_name, e)
        result = ExecutionResult(
            success=False,
            error=str(e),
            phase=ExecutionPhase.EXECUTING,
            rollback_available=hasattr(handler, "undo"),
        )

    # Phase 3: post_confirm
    if post_confirm_fn is not None:
        result = post_confirm_fn(request, result)

    # Phase 4: 回滚
    if not result.success and result.rollback_available:
        rolled_back = _try_rollback(handler, request)
        if rolled_back:
            result = ExecutionResult(
                success=False,
                result=None,
                error=f"已回滚: {result.error}",
                phase=ExecutionPhase.ROLLBACK,
                rollback_available=False,
                audit_entry=result.audit_entry,
            )

    return result
