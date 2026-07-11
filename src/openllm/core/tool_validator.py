"""tool_validator.py — ISN工具结果验证器（Layer 3）

工具调用后结果质量检查，纯规则/零LLM调用。
五种检查：NON_EMPTY / TYPE_MATCH / ERROR_CODE / LOGICAL_CONSISTENCY / SIZE_BUDGET
由 main_loop.py 调用；BLOCKED时自动创建AuditEntry。
"""

import json, logging, time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger("openllm.tool_validator")


class ValidationResult(str, Enum):
    PASS, WARN, FAIL, BLOCKED = "pass", "warn", "fail", "blocked"


class ValidationCheck(str, Enum):
    NON_EMPTY, TYPE_MATCH, ERROR_CODE = "non_empty", "type_match", "error_code"
    LOGICAL_CONSISTENCY, SIZE_BUDGET = "logical_consistency", "size_budget"


@dataclass(frozen=True)
class ToolCall:
    tool_name: str; tool_type: str; params: dict; result: Any  # noqa: E702
    duration_ms: float = 0.0; session_id: str = ""; turn_id: str = ""  # noqa: E702


@dataclass(frozen=True)
class ValidationDetail:
    check: ValidationCheck; result: ValidationResult  # noqa: E702
    message: str = ""; details: dict = field(default_factory=dict)  # noqa: E702


@dataclass(frozen=True)
class AuditEntry:
    timestamp: float; tool_name: str; tool_type: str; reason: str  # noqa: E702
    session_id: str = ""; turn_id: str = ""  # noqa: E702


@dataclass(frozen=True)
class ToolValidationReport:
    tool_call: ToolCall; checks: list; overall: ValidationResult  # noqa: E702
    should_retry: bool = False; should_block: bool = False  # noqa: E702
    audit_entry: Optional[AuditEntry] = None  # noqa: E702


_EXPECTED_TYPES: dict[str, tuple[type, ...]] = {
    "search": (list,), "read": (str, bytes),
    "write": (dict,), "execute": (dict,), "network": (str, dict, list),
}
_ERROR_FIELDS = {"error", "errno", "exception", "error_code", "error_msg"}
_SEVERITY = {ValidationResult.PASS: 0, ValidationResult.WARN: 1,
             ValidationResult.FAIL: 2, ValidationResult.BLOCKED: 3}


def _d(check, result, msg="", details=None):
    return ValidationDetail(check=check, result=result,
                            message=msg, details=details or {})


def _is_empty(r):
    if r is None: return True
    if isinstance(r, str) and not r.strip(): return True
    if isinstance(r, (list, tuple, set, dict)) and len(r) == 0: return True
    return False


def _check_non_empty(tc):
    empty = _is_empty(tc.result)
    return _d(ValidationCheck.NON_EMPTY,
              ValidationResult.FAIL if empty else ValidationResult.PASS,
              "返回结果为空" if empty else "返回结果非空")


def _check_type_match(tc):
    expected = _EXPECTED_TYPES.get(tc.tool_type)
    if expected is None:
        return _d(ValidationCheck.TYPE_MATCH, ValidationResult.WARN,
                  f"未知tool_type: {tc.tool_type}", {"tool_type": tc.tool_type})
    ok = isinstance(tc.result, expected)
    names = [t.__name__ for t in expected]
    return _d(ValidationCheck.TYPE_MATCH,
              ValidationResult.PASS if ok else ValidationResult.FAIL,
              f"类型匹配: {type(tc.result).__name__}" if ok
              else f"类型不匹配: 期望{names}，实际{type(tc.result).__name__}",
              {} if ok else {"expected": names, "actual": type(tc.result).__name__})


def _check_error_code(tc):
    if not isinstance(tc.result, dict):
        return _d(ValidationCheck.ERROR_CODE, ValidationResult.PASS, "结果非dict，跳过")
    found = [k for k in _ERROR_FIELDS if k in tc.result]
    if not found:
        return _d(ValidationCheck.ERROR_CODE, ValidationResult.PASS, "未发现错误码字段")
    real = [k for k in found if tc.result[k] not in (None, "", 0, False, [])]
    if real:
        return _d(ValidationCheck.ERROR_CODE, ValidationResult.FAIL,
                  f"结果包含错误字段: {real}", {"error_fields": real})
    return _d(ValidationCheck.ERROR_CODE, ValidationResult.WARN,
              f"错误码字段值为空: {found}", {"found_fields": found})


def _check_logical_consistency(tc, consistency_rules=None):
    if consistency_rules:
        for name, fn in consistency_rules.items():
            if callable(fn) and not fn(tc):
                return _d(ValidationCheck.LOGICAL_CONSISTENCY,
                          ValidationResult.FAIL, f"自定义规则不通过: {name}")
    if tc.tool_type == "search" and isinstance(tc.result, list):
        if tc.result and not isinstance(tc.result[0], dict):
            return _d(ValidationCheck.LOGICAL_CONSISTENCY,
                      ValidationResult.WARN, "search列表项非dict")
        bad = [i for i, item in enumerate(tc.result)
               if isinstance(item, dict) and not any(
                   k in item for k in ("url", "title", "link", "content"))]
        if bad:
            return _d(ValidationCheck.LOGICAL_CONSISTENCY,
                      ValidationResult.WARN, f"第{bad}项缺url/title/content字段")
    return _d(ValidationCheck.LOGICAL_CONSISTENCY, ValidationResult.PASS, "通过")


def _check_size_budget(tc, budget):
    try:
        size = len(json.dumps(tc.result, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        size = len(str(tc.result))
    if size > budget:
        ratio = round(size / budget, 1)
        sev = ValidationResult.FAIL if ratio > 2.0 else ValidationResult.WARN
        return _d(ValidationCheck.SIZE_BUDGET, sev,
                  f"大小超标: {size}/{budget} ({ratio}x)",
                  {"size": size, "budget": budget, "ratio": ratio})
    return _d(ValidationCheck.SIZE_BUDGET, ValidationResult.PASS,
              f"合规: {size}/{budget}", {"size": size, "budget": budget})


def validate_tool_result(
    tool_call: ToolCall, *,
    size_budget: int = 100_000,
    allowed_error_codes: Optional[set] = None,
    consistency_rules: Optional[dict] = None,
) -> ToolValidationReport:
    """验证工具调用结果。PASS/WARN→继续, FAIL→重试, BLOCKED→丢弃+审计。"""
    checks = []
    # 1. 非空（空结果直接返回）
    checks.append(_check_non_empty(tool_call))
    if checks[-1].result == ValidationResult.FAIL:
        return ToolValidationReport(tool_call=tool_call, checks=checks,
                                    overall=ValidationResult.FAIL, should_retry=True)
    # 2. 类型匹配
    checks.append(_check_type_match(tool_call))
    # 3. 错误码（FAIL时检查allowed_error_codes决定BLOCKED或WARN）
    checks.append(_check_error_code(tool_call))
    if checks[-1].result == ValidationResult.FAIL:
        fields = checks[-1].details.get("error_fields", [])
        all_ok = (allowed_error_codes is not None
                  and all(f in allowed_error_codes for f in fields))
        if all_ok:
            checks[-1] = _d(ValidationCheck.ERROR_CODE, ValidationResult.WARN,
                            f"错误字段在允许列表内: {fields}", checks[-1].details)
        else:
            bad = [f for f in fields
                   if allowed_error_codes is None or f not in allowed_error_codes]
            checks[-1] = _d(ValidationCheck.ERROR_CODE, ValidationResult.BLOCKED,
                            f"未授权的错误字段: {bad}", {"disallowed_fields": bad})
    # 4. 逻辑一致性
    checks.append(_check_logical_consistency(tool_call, consistency_rules))
    # 5. 大小预算
    checks.append(_check_size_budget(tool_call, size_budget))
    # 综合判定
    worst = max(checks, key=lambda c: _SEVERITY[c.result])
    overall, blocked = worst.result, worst.result == ValidationResult.BLOCKED
    audit = None
    if blocked:
        reason = "; ".join(c.message for c in checks
                           if c.result == ValidationResult.BLOCKED)
        audit = AuditEntry(timestamp=time.time(), tool_name=tool_call.tool_name,
                           tool_type=tool_call.tool_type, reason=reason,
                           session_id=tool_call.session_id, turn_id=tool_call.turn_id)
        logger.warning("BLOCKED tool=%s reason=%s", tool_call.tool_name, reason)
    elif overall != ValidationResult.PASS:
        logger.info("validate tool=%s → %s", tool_call.tool_name, overall.value)
    return ToolValidationReport(tool_call=tool_call, checks=checks, overall=overall,
                                should_retry=overall == ValidationResult.FAIL,
                                should_block=blocked, audit_entry=audit)


# ═══════════════════════════════════════════════════════
# Engine级verify钩子（从engine.py提取）
# ═══════════════════════════════════════════════════════

import re as _re

TOOL_PARAM_SCHEMAS = {
    "read_file": {"required": ["path"], "check": None},
    "write_file": {"required": ["path", "content"], "check": None},
    "shell": {"required": ["command"], "check": None},
    "search": {"required": ["pattern"], "check": None},
    "list_dir": {"required": [], "check": None},
    "python_exec": {"required": ["code"], "check": None},
    "octopus_search": {"required": ["query"], "check": None},
    "octopus_self_model": {"required": [], "check": None},
}

RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}
DEFAULT_MAX_RISK = "high"


def verify_tool_params(tool_name: str, schemas: dict = None, **kwargs) -> dict:
    """工具调用前参数验证（规则引擎，零LLM）。"""
    schemas = schemas or TOOL_PARAM_SCHEMAS
    schema = schemas.get(tool_name)
    if not schema:
        return {"pass": True, "reason": "无schema验证"}
    for req in schema["required"]:
        if req not in kwargs or kwargs[req] is None:
            return {"pass": False, "reason": f"缺少必填参数: {req}"}
        val = kwargs[req]
        if isinstance(val, str) and len(val) == 0:
            return {"pass": False, "reason": f"参数{req}为空字符串"}
    if tool_name == "shell":
        cmd = kwargs.get("command", "")
        dangerous_patterns = [
            (r'\brm\s+[-/][^;]*\brf\b', "rm -rf 删除操作"),
            (r'\bsudo\s', "sudo 提权操作"),
            (r'\bdestro[y5]\s+', "破坏性命令"),
            (r'>\s*/dev/(sda|sdb|sdc|nvme|mmcblk)', "磁盘写入"),
            (r'\bmkf[sz]\s', "格式化操作"),
            (r'\bdd\s+if=\s*/dev/', "dd 磁盘写入"),
            (r':\(\)\s*\{', "fork炸弹"),
            (r'\b(?:curl|wget)\s+.*?\|', "下载并执行"),
            (r'\bchmod\s+777\s', "权限放开"),
            (r'exec\s+.*[;<`]', "exec 执行注入"),
        ]
        for pattern, reason in dangerous_patterns:
            if _re.search(pattern, cmd):
                return {"pass": False, "reason": f"Shell命令包含危险模式: {reason}"}
    return {"pass": True, "reason": ""}


def verify_tool_result(tool_name: str, result) -> dict:
    """工具调用后结果验证。"""
    if not result.success:
        return {"pass": False, "reason": f"工具执行失败: {result.error}"}
    if not result.output:
        return {"pass": True, "reason": "成功（无输出）"}
    if len(result.output) > 100000:
        return {"pass": False, "reason": f"输出过长({len(result.output)}字节)，需截断"}
    return {"pass": True, "reason": ""}


def check_tool_risk(tool_name: str, meta_map: dict = None,
                    max_risk: str = DEFAULT_MAX_RISK) -> dict:
    """检查工具的ISN风险等级。"""
    from .engine_integrations import get_isn_metadata
    if meta_map is None:
        _, meta_map = get_isn_metadata()
    if not meta_map:
        return {"pass": True, "reason": "ISN元数据未加载", "risk_level": "unknown"}
    tool_meta = meta_map.get(tool_name)
    if not tool_meta:
        return {"pass": True, "reason": f"工具'{tool_name}'不在ISN索引中", "risk_level": "unknown"}
    risk = tool_meta.get("risk_level", "low")
    if RISK_ORDER.get(risk, 0) > RISK_ORDER.get(max_risk, 0):
        return {"pass": False, "reason": f"风险等级'{risk}'超过允许上限'{max_risk}'", "risk_level": risk}
    return {"pass": True, "reason": "", "risk_level": risk}

