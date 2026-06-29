"""ToolScope — IO-S工具作用域治理

根据任务阶段、上下文、权限级别动态裁剪可用工具集。
"给Agent一长串工具会降低性能"（Vercel实证：移除80%工具后结果改善）。

预定义作用域（4种）：
  - readonly：只读工具
  - development：读写+执行
  - research：搜索+读取
  - admin：全部工具（需人工确认破坏性操作）
"""

from __future__ import annotations
from typing import Optional

# ── 预定义作用域 ─────────────────────────────────────

# 作用域定义：{scope_name: [tool_names]}
_SCOPES: dict[str, list[str]] = {
    "readonly": [
        "read_file",
        "search",
        "search_files",
        "web_search",
        "session_search",
    ],
    "development": [
        "read_file",
        "write_file",
        "patch",
        "shell",
        "search",
        "search_files",
        "list_dir",
        "execute_tool",
    ],
    "research": [
        "web_search",
        "web_extract",
        "bing_search",
        "read_file",
        "search",
        "search_files",
        "session_search",
        "read_memory",
    ],
    "admin": [
        # 全部工具 — 运行时动态组装
    ],
}

# 危险(破坏性)操作——admin模式下也需人工确认
_DESTRUCTIVE_TOOLS: set[str] = {
    "shell",           # 可执行任意命令
    "write_file",      # 可覆盖文件
    "publish",         # 可发布信息
    "delete_file",     # 可删除文件
    "self_modify",     # 可修改自身代码
    "disable_security",  # 可关闭安全
}


def get_scope_tools(scope: str) -> list[str]:
    """获取指定作用域的可用工具列表。"""
    if scope not in _SCOPES:
        raise ValueError(f"未知作用域: {scope}。可用: {list(_SCOPES.keys())}")
    return list(_SCOPES[scope])


def list_scopes() -> list[dict]:
    """列出所有可用作用域。"""
    return [
        {
            "name": name,
            "tool_count": len(tools) if tools else "all",
            "destructive": any(t in _DESTRUCTIVE_TOOLS for t in tools) if tools else True,
        }
        for name, tools in _SCOPES.items()
    ]


def filter_by_scope(tools: list[str], scope: str) -> list[str]:
    """根据作用域过滤工具列表。"""
    if scope == "admin":
        return list(tools)
    allowed = set(_SCOPES.get(scope, []))
    return [t for t in tools if t in allowed]


def is_scope_allowed(tool_name: str, scope: str) -> bool:
    """检查工具是否在当前作用域内。"""
    if scope == "admin":
        return True
    allowed = set(_SCOPES.get(scope, []))
    return tool_name in allowed


def is_destructive(tool_name: str) -> bool:
    """检查工具是否为破坏性操作。"""
    return tool_name in _DESTRUCTIVE_TOOLS


# ── 作用域建议（根据任务阶段） ─────────────────────

_TASK_SCOPE_MAP: dict[str, str] = {
    "planning": "readonly",        # 规划阶段：只读
    "code_generation": "development",  # 编码阶段：读写
    "code_review": "readonly",     # 审查阶段：只读
    "research": "research",        # 研究阶段：搜索+读取
    "debugging": "development",    # 调试阶段：读写+执行
    "documentation": "development",  # 文档阶段：读写
    "architecture_design": "readonly",  # 架构设计：只读
    "tool_call": "development",    # 工具调用：按需
}


def suggest_scope(task_type: str) -> str:
    """根据任务类型建议作用域。"""
    return _TASK_SCOPE_MAP.get(task_type, "readonly")


def get_tools_for_task(task_type: str, all_tools: list[str]) -> list[str]:
    """为指定任务类型获取裁剪后的工具列表。"""
    scope = suggest_scope(task_type)
    return filter_by_scope(all_tools, scope)
