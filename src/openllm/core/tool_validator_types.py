"""
openLLM 工具验证类型 — 从main_loop.py提取

避免agent_heartbeat.py与main_loop.py的循环导入。
"""
from dataclasses import dataclass
from typing import Any, Optional
from enum import Enum


class ValidationResult(Enum):
    """验证结果"""
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"


@dataclass
class ToolCall:
    """工具调用记录"""
    tool_name: str
    tool_type: str  # read/write/search/execute
    params: dict
    result: str
    duration_ms: float
    session_id: str
    turn_id: str


@dataclass
class AuditEntry:
    """审计条目"""
    reason: str
    severity: str = "medium"


@dataclass
class ValidationReport:
    """验证报告"""
    overall: ValidationResult
    checks: list
    should_retry: bool = False
    should_block: bool = False
    audit_entry: Optional[AuditEntry] = None


def validate_tool_result(call: ToolCall) -> ValidationReport:
    """验证工具调用结果"""
    checks = []
    
    # 基本检查
    if not call.result:
        checks.append(("empty_result", "warn"))
        return ValidationReport(
            overall=ValidationResult.WARN,
            checks=checks,
            should_retry=True
        )
    
    # 检查错误
    if "[错误]" in call.result or "[拦截]" in call.result:
        checks.append(("error_in_result", "fail"))
        return ValidationReport(
            overall=ValidationResult.FAIL,
            checks=checks,
            should_block=True,
            audit_entry=AuditEntry(reason=f"工具返回错误: {call.result[:100]}")
        )
    
    checks.append(("result_ok", "pass"))
    return ValidationReport(
        overall=ValidationResult.PASS,
        checks=checks
    )
