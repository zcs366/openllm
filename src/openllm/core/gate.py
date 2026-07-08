"""gate.py — 权限门控模块

来自老IO-S syscall/gate.py：
  - 5种权限模式: READONLY / DEV / RESEARCH / PLAN / ADMIN
  - 分级审批: 低风险自动、高风险人工
  - 审计日志不可篡改
"""

import json
import logging
import time
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger("openllm.gate")

OPENLLM_HOME = Path.home() / ".openllm"
AUDIT_LOG_PATH = OPENLLM_HOME / "audit.jsonl"


class PermissionMode(str, Enum):
    READONLY = "readonly"
    DEV = "dev"
    RESEARCH = "research"
    PLAN = "plan"
    ADMIN = "admin"


PERMISSION_CONFIG = {
    PermissionMode.READONLY: {
        "description": "只读模式 — 搜索、查询、读取",
        "allowed_operations": ["search", "read", "query", "list"],
        "blocked_operations": ["write", "execute", "delete", "network"],
        "risk_level": "low",
        "requires_approval": False,
        "auto_approve": True,
    },
    PermissionMode.DEV: {
        "description": "开发模式 — 写文件、执行代码",
        "allowed_operations": [
            "search", "read", "query", "list", "write", "execute"],
        "blocked_operations": ["delete", "network"],
        "risk_level": "medium",
        "requires_approval": False,
        "auto_approve": True,
    },
    PermissionMode.RESEARCH: {
        "description": "研究模式 — 读+写分析文件、网络请求",
        "allowed_operations": [
            "search", "read", "query", "list", "write", "network"],
        "blocked_operations": ["delete"],
        "risk_level": "high",
        "requires_approval": False,
        "auto_approve": True,
    },
    PermissionMode.PLAN: {
        "description": "计划模式 — 只读+规划（不执行）",
        "allowed_operations": [
            "search", "read", "query", "list", "plan"],
        "blocked_operations": [
            "write", "execute", "delete", "network"],
        "risk_level": "low",
        "requires_approval": False,
        "auto_approve": True,
    },
    PermissionMode.ADMIN: {
        "description": "管理模式 — 全部权限（需人工确认）",
        "allowed_operations": [
            "search", "read", "query", "list",
            "write", "execute", "delete", "network"],
        "blocked_operations": [],
        "risk_level": "critical",
        "requires_approval": True,
        "auto_approve": False,
    },
}


def audit_log(operation: str, tool_name: str, params: dict,
              result: str, approved: bool = False,
              approver: str = ""):
    entry = {
        "timestamp": time.time(),
        "operation": operation,
        "tool_name": tool_name,
        "params": params,
        "result": result,
        "approved": approved,
        "approver": approver,
    }
    try:
        AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Failed to write audit log: {e}")


class PermissionGate:
    """权限门控 — 5模式分级管理。"""

    def __init__(self, mode: PermissionMode = PermissionMode.DEV):
        self._mode = mode
        self._config = PERMISSION_CONFIG[mode]

    @property
    def mode(self) -> PermissionMode:
        return self._mode

    def switch_mode(self, mode: PermissionMode):
        self._mode = mode
        self._config = PERMISSION_CONFIG[mode]
        logger.info(f"权限模式切换: {mode.value}")

    def check(self, operation: str) -> tuple[bool, str]:
        """检查操作是否允许。

        Returns: (允许与否, 原因)
        """
        allowed = self._config["allowed_operations"]
        blocked = self._config["blocked_operations"]

        if operation in blocked:
            return False, (
                f"操作 '{operation}' 在 {self._mode.value} 模式下被禁止")
        if operation in allowed:
            return True, ""
        # 未明确列出→拒绝
        return False, (
            f"操作 '{operation}' 未在 {self._mode.value} "
            f"模式的允许列表中")

    def requires_approval(self) -> bool:
        return self._config.get("requires_approval", False)

    def get_risk_level(self) -> str:
        return self._config.get("risk_level", "low")

    def get_status(self) -> dict:
        return {
            "mode": self._mode.value,
            "description": self._config["description"],
            "risk_level": self._config["risk_level"],
            "requires_approval": self._config["requires_approval"],
            "allowed_count": len(
                self._config["allowed_operations"]),
            "blocked_count": len(
                self._config["blocked_operations"]),
        }
