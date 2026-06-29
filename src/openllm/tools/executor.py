"""
OpenLLM Tool Executor — 工具执行层。

对标CC/CX的工具系统。Phase 1最小工具集：
  read_file  — 读取文件
  write_file — 写入文件  
  shell      — 执行Shell命令
  search     — 搜索文件内容
"""

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass
class ToolResult:
    """工具执行结果。"""
    tool_name: str
    success: bool
    output: str = ""
    error: str = ""
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


class ToolRegistry:
    """
    工具注册中心。

    每注册一个工具 = 一个可被Agent调用的函数。
    对标MCP协议——未来可对接外部MCP Server。
    """

    def __init__(self):
        self._tools: dict[str, Callable] = {}
        self._descriptions: dict[str, str] = {}
        self._schemas: dict[str, dict] = {}

    def register(
        self,
        name: str,
        func: Callable,
        description: str = "",
        schema: Optional[dict] = None,
    ) -> None:
        """注册一个工具。"""
        self._tools[name] = func
        self._descriptions[name] = description
        self._schemas[name] = schema or {}

    def list_tools(self) -> list[dict]:
        """列出所有可用工具。"""
        return [
            {"name": name, "description": desc}
            for name, desc in self._descriptions.items()
        ]

    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """执行工具调用。"""
        func = self._tools.get(tool_name)
        if func is None:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"未知工具: {tool_name}。可用工具: {list(self._tools.keys())}",
            )

        t0 = time.time()
        try:
            output = func(**kwargs)
            return ToolResult(
                tool_name=tool_name,
                success=True,
                output=str(output),
                latency_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=str(e),
                latency_ms=(time.time() - t0) * 1000,
            )


# ── 内置工具 ──────────────────────────────────────

def tool_read_file(path: str, offset: int = 0, limit: int = 500) -> str:
    """读取文件内容。"""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return f"[错误] 文件不存在: {path}"
    if p.is_dir():
        return f"[错误] 是目录: {path}"

    try:
        lines = p.read_text(encoding="utf-8").split("\n")
        total = len(lines)
        if offset >= total:
            return f"[提示] offset={offset}超出文件行数{total}"
        chunk = lines[offset:offset + limit]
        return f"文件: {path}\n行数: {total}\n片段: {offset}-{min(offset+limit, total)}\n" + "\n".join(
            f"{i+offset+1:4d}| {line}" for i, line in enumerate(chunk)
        )
    except UnicodeDecodeError:
        return f"[错误] 无法以UTF-8读取: {path}"


def tool_write_file(path: str, content: str) -> str:
    """写入文件内容。"""
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        p.write_text(content, encoding="utf-8")
        size = len(content)
        return f"已写入: {path} ({size} 字节)"
    except Exception as e:
        return f"[错误] 写入失败: {e}"


def tool_shell(command: str, timeout: int = 30, workdir: Optional[str] = None) -> str:
    """执行Shell命令。"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workdir,
        )
        output = result.stdout
        if result.stderr:
            output += "\n[stderr]\n" + result.stderr[:500]
        if result.returncode != 0:
            output += f"\n[exit_code={result.returncode}]"
        return output[:5000]  # 截断长输出
    except subprocess.TimeoutExpired:
        return f"[超时] 命令执行超过{timeout}秒"
    except Exception as e:
        return f"[错误] {e}"


def tool_search_files(pattern: str, path: str = ".", file_glob: Optional[str] = None) -> str:
    """
    搜索文件内容（ripgrep风格）。
    简化版：使用grep。
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return f"[错误] 路径不存在: {path}"

    cmd = f"grep -rn --include='{file_glob or '*'}' '{pattern}' {p} 2>/dev/null | head -20"
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        output = result.stdout.strip()
        return output if output else f"未找到匹配 '{pattern}' 的内容"
    except Exception as e:
        return f"[错误] {e}"


def tool_list_dir(path: str = ".") -> str:
    """列出目录内容。"""
    from pathlib import Path
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return f"[错误] 路径不存在: {path}"
    if not p.is_dir():
        return f"[错误] 不是目录: {path}"
    entries = []
    for item in sorted(p.iterdir()):
        prefix = "📁" if item.is_dir() else "📄"
        size = f"{item.stat().st_size}" if item.is_file() else ""
        entries.append(f"{prefix} {item.name} ({size})" if size else f"{prefix} {item.name}")
    return "\n".join(entries[:50]) + (f"\n... ({len(entries)} 项)" if len(entries) > 50 else "")


def tool_python_exec(code: str, timeout: int = 10) -> str:
    """执行 Python 代码并返回输出。"""
    import subprocess
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True, text=True, timeout=timeout,
        )
        output = result.stdout
        if result.stderr:
            output += "\n[stderr]\n" + result.stderr[:500]
        return output[:5000] if output else "(无输出)"
    except subprocess.TimeoutExpired:
        return f"[超时] Python 执行超过 {timeout} 秒"
    except Exception as e:
        return f"[错误] {e}"


# ── 创建默认工具集 ─────────────────────────────────

def create_default_tools() -> ToolRegistry:
    """创建Phase 1默认工具集。"""
    registry = ToolRegistry()
    registry.register("read_file", tool_read_file, "读取文件内容")
    registry.register("write_file", tool_write_file, "写入文件")
    registry.register("shell", tool_shell, "执行Shell命令")
    registry.register("search", tool_search_files, "搜索文件内容")
    registry.register("list_dir", tool_list_dir, "列出目录内容")
    registry.register("python_exec", tool_python_exec, "执行Python代码")
    
    # 章鱼记忆系统
    from .octopus import tool_octopus_search, tool_octopus_self_model
    registry.register("octopus_search", tool_octopus_search, "搜索章鱼记忆（RECALL+jiak cards）")
    registry.register("octopus_self_model", tool_octopus_self_model, "查看章鱼自省状态")
    
    return registry
