"""
openLLM 工具验证类型 — 从main_loop.py提取

避免agent_heartbeat.py与main_loop.py的循环导入。
"""
from dataclasses import dataclass
import re
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


# ══════════════════════════════════════════════════════════════════════
# 工具失败标记的**唯一解析点**（加固⑤·第一步，2026-10-01）
# ══════════════════════════════════════════════════════════════════════
# 病（真库实证，同一形态犯两次）：
#   RECALL.jsonl 第170行（10-01 03:28）status="ok"，
#     内容却是 `/bin/sh: 1: 线路：宜宾→成都: not found … [exit_code=127]`；
#   第188行（10-01 21:10）同型：`│: not found` ×4、`└─: not found`、`[exit_code=127]`。
#   根因链：tools/executor.py 只在**文本**里附 `[exit_code=N]`（不抛异常）
#   → core/isn_impl.py:133 `ActionResult(success=True, …)` 没抛异常就算成功
#   → 本文件下方 validate_tool_result 只认 `[错误]`/`[拦截]`，**漏了退出码与超时**
#   ⇒ 命令失败被记成成功，且加固④（判据是 result.success）对它不响。
#
# 第一步只做**观测**：本函数先立"唯一判定源"，接线留到第二步（要改 success 语义，
# 须逐点审下游：evolve／IKO.has_side_effects／DPO 原料／加固④）。
# 故本步**零行为变更**：不改 validate_tool_result 的返回值，不阻断、不改正文。

_EXIT_CODE_MARKER = re.compile(r"\[exit_code=(-?\d+)\]")

FAILURE_KINDS = ("exit_code", "timeout", "error", "blocked")


def detect_tool_failure(result_text: str) -> Optional[dict]:
    """从工具返回的**文本**里识别失败标记。识别到返回 dict，否则 None。

    标记优先级：退出码 > 超时 > 错误 > 拦截（同一条文本可能同时含多种）。
    退出码为 0 视为成功（executor 只在非 0 时才附标记，此处仍显式判 0 以防误报）。
    """
    text = result_text or ""
    m = _EXIT_CODE_MARKER.search(text)
    if m is not None:
        code = int(m.group(1))
        if code != 0:
            return {"kind": "exit_code", "exit_code": code,
                    "detail": f"工具进程退出码 {code}（非 0）"}
    if "[超时]" in text:
        return {"kind": "timeout", "detail": "工具执行超时"}
    if "[错误]" in text:
        return {"kind": "error", "detail": "工具返回 [错误] 标记"}
    if "[拦截]" in text:
        return {"kind": "blocked", "detail": "工具返回 [拦截] 标记"}
    return None


# 命令形态检查（同一批：治理"把展示块当命令执行"）
_SHELL_TOOL_NAMES = {"shell", "terminal", "bash", "sh", "exec", "cmd", "subprocess"}
_BOX_CHARS = "│└┌┐┘├┤┬┴┼─"


def looks_like_non_command(command: str) -> Optional[str]:
    """判一条 shell 命令是否**显然不像命令**。像，返回 None；不像，返回原因。

    保守设计（只报**高置信度**的形态，避免狼来了）：
      · 含框线字符 │└┌┐┘├┤┬┴┼─（展示块的典型特征；ASCII 的 | 不在此列）
      · 首词整词非 ASCII（如「线路：宜宾→成都」），且命令含换行（纯参数里带中文不算）
    不做判定：多行命令（heredoc 合法）、参数里含中文（grep 中文 file 合法）。
    """
    cmd = (command or "").strip()
    if not cmd:
        return "空命令"
    if any(ch in cmd for ch in _BOX_CHARS):
        return "含框线字符（疑似展示块，非命令）"
    first = cmd.split()[0] if cmd.split() else ""
    if first and all(ord(c) > 127 for c in first) and "\n" in cmd:
        return f"首词整词非 ASCII 且含换行（疑似展示文本）：{first[:12]}"
    return None
