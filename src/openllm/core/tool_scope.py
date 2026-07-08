"""tool_scope.py — 工具作用域治理

来自老IO-S syscall/tool_scope.py：
  - 4种作用域: readonly / dev / research / admin
  - 按任务阶段动态切换工具集
  - 对接ISN元数据（risk_level + permission_level + side_effects）
"""

import logging
from enum import Enum
from typing import Optional

logger = logging.getLogger("openllm.tool_scope")


class ToolScope(str, Enum):
    READONLY = "readonly"
    DEV = "dev"
    RESEARCH = "research"
    ADMIN = "admin"


SCOPE_CONFIG = {
    ToolScope.READONLY: {
        "description": "只读模式 — 搜索、查询、读取文件",
        "allowed_types": ["search", "read", "query"],
        "blocked_types": [
            "write", "execute", "network", "delete"],
        "risk_threshold": "low",
        "requires_approval": False,
    },
    ToolScope.DEV: {
        "description": "开发模式 — 写文件、执行代码、终端命令",
        "allowed_types": [
            "search", "read", "query", "write", "execute"],
        "blocked_types": ["delete", "network"],
        "risk_threshold": "medium",
        "requires_approval": False,
    },
    ToolScope.RESEARCH: {
        "description": "研究模式 — 读+写分析文件、网络请求",
        "allowed_types": [
            "search", "read", "query", "write", "network"],
        "blocked_types": ["delete"],
        "risk_threshold": "high",
        "requires_approval": False,
    },
    ToolScope.ADMIN: {
        "description": "管理模式 — 全部权限（需人工确认）",
        "allowed_types": [
            "search", "read", "query", "write",
            "execute", "network", "delete"],
        "blocked_types": [],
        "risk_threshold": "critical",
        "requires_approval": True,
    },
}

# 风险等级排序
RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


class ToolScopeManager:
    """工具作用域管理器。"""

    def __init__(self, scope: ToolScope = ToolScope.DEV):
        self._scope = scope
        self._config = SCOPE_CONFIG[scope]
        self._isn_metadata: dict = {}  # tool_name → metadata

    @property
    def scope(self) -> ToolScope:
        return self._scope

    def switch_scope(self, scope: ToolScope):
        self._scope = scope
        self._config = SCOPE_CONFIG[scope]
        logger.info(f"工具作用域切换: {scope.value}")

    def is_tool_allowed(self, tool_name: str,
                        tool_type: Optional[str] = None,
                        tool_risk: Optional[str] = None
                        ) -> tuple[bool, str]:
        """检查工具是否在当前作用域内。

        优先使用ISN元数据，fallback到tool_type。
        """
        # 查ISN元数据
        meta = self._isn_metadata.get(tool_name, {})
        effective_type = tool_type or meta.get("type", "unknown")
        effective_risk = tool_risk or meta.get("risk_level", "low")

        # 检查风险阈值
        threshold = self._config.get("risk_threshold", "low")
        if RISK_ORDER.get(effective_risk, 0) > RISK_ORDER.get(
                threshold, 0):
            return False, (
                f"工具 '{tool_name}' 风险等级 {effective_risk} "
                f"超过作用域阈值 {threshold}")

        # 检查类型
        blocked = self._config.get("blocked_types", [])
        if effective_type in blocked:
            return False, (
                f"工具类型 '{effective_type}' 在 "
                f"{self._scope.value} 作用域被禁止")

        allowed = self._config.get("allowed_types", [])
        if effective_type in allowed:
            return True, ""

        return False, (
            f"工具类型 '{effective_type}' 未在 "
            f"{self._scope.value} 作用域允许列表中")

    def load_isn_metadata(self, metadata_list: list[dict]):
        """加载ISN工具元数据。"""
        for meta in metadata_list:
            name = meta.get("name") or meta.get("skill_id", "")
            if name:
                self._isn_metadata[name] = meta
        logger.info(
            f"加载 {len(metadata_list)} 个ISN工具元数据")

    def get_status(self) -> dict:
        return {
            "scope": self._scope.value,
            "description": self._config["description"],
            "risk_threshold": self._config["risk_threshold"],
            "allowed_types": self._config["allowed_types"],
            "blocked_types": self._config["blocked_types"],
            "isn_tools_loaded": len(self._isn_metadata),
        }
