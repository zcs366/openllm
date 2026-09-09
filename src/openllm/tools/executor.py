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
        self._verify_hook: Optional[Callable] = None  # verify-before-complete
        self._requires_verify: set[str] = {"write_file", "shell"}  # 需要验证的工具

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

    def set_verify_hook(self, hook: Callable):
        """设置verify-before-complete钩子。用于写操作前验证。"""
        self._verify_hook = hook

    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """执行工具调用。写操作前触发verify钩子。"""
        func = self._tools.get(tool_name)
        if func is None:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"未知工具: {tool_name}。可用工具: {list(self._tools.keys())}",
            )

        # Verify-before-complete: 写操作前验证
        if tool_name in self._requires_verify and self._verify_hook:
            try:
                verified = self._verify_hook(tool_name, **kwargs)
                # 支持两种返回格式：bool 或 {"pass": bool, "reason": str}
                if isinstance(verified, dict):
                    ok = verified.get("pass", True)
                    reason = verified.get("reason", "")
                else:
                    ok = bool(verified)
                    reason = ""
                if not ok:
                    return ToolResult(
                        tool_name=tool_name,
                        success=False,
                        error=f"验证失败: {tool_name} — {reason}" if reason else f"验证失败: {tool_name}",
                    )
            except Exception as e:
                return ToolResult(
                    tool_name=tool_name,
                    success=False,
                    error=f"验证异常: {e}",
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
    from .octopus import (
        tool_octopus_search, tool_octopus_self_model,
        tool_octopus_search_stats, tool_octopus_route, tool_octopus_health,
    )
    registry.register("octopus_search", tool_octopus_search, "搜索章鱼记忆+外部搜索（v4黑匣子记录）")
    registry.register("octopus_self_model", tool_octopus_self_model, "查看章鱼自省状态")
    registry.register("octopus_search_stats", tool_octopus_search_stats, "搜索黑匣子统计——后端成功率/延迟分布")
    registry.register("octopus_route", tool_octopus_route, "根据查询推荐最优搜索后端（MAB策略）")
    registry.register("octopus_health", tool_octopus_health, "章鱼搜索系统健康检查")

    # ── curl-impersonate（浏览器TLS指纹模拟）──────────────────────
    def tool_curl_impersonate(action: str = "health", **kwargs) -> str:
        """浏览器TLS指纹模拟——绕过反爬检测。模拟Chrome/Firefox/Safari的TLS握手。"""
        import json as _json
        import sys as _sys
        _sys.path.insert(0, "/home/zcs/.hermes/hermes-agent")
        try:
            from tools.curl_impersonate_tool import fetch as _fetch, batch as _batch, list_targets as _list
        except ImportError as e:
            return _json.dumps({"error": f"curl_impersonate模块未安装: {e}"})

        if action == "fetch":
            url = kwargs.get("url", "")
            if not url:
                return _json.dumps({"error": "url参数不能为空"})
            target = kwargs.get("target", "chrome116")
            timeout = int(kwargs.get("timeout", 15))
            result = _fetch(url, target=target, timeout=timeout)
            # 不返回content字段（太大），只返回元数据
            result.pop("content", None)
            return _json.dumps(result, ensure_ascii=False)
        elif action == "batch":
            urls_str = kwargs.get("urls", "")
            if not urls_str:
                return _json.dumps({"error": "urls参数不能为空"})
            result = _batch(urls_str.split(","), target=kwargs.get("target", "chrome116"))
            for r in result["results"]:
                r.pop("content", None)
            return _json.dumps(result, ensure_ascii=False)
        elif action == "list_targets":
            return _json.dumps(_list(), ensure_ascii=False)
        elif action == "health":
            import json as _json2
            from tools.curl_impersonate_tool import health as _health
            return _json2.dumps(_health())
        return _json.dumps({"error": f"Unknown action: {action}"})

    registry.register("curl_impersonate", tool_curl_impersonate,
        "浏览器TLS指纹模拟——绕过反爬检测(Chrome/Firefox/Safari)",
        schema={"type": "object", "properties": {
            "action": {"type": "string", "enum": ["fetch","batch","list_targets","health"]},
            "url": {"type": "string"}, "urls": {"type": "string"},
            "target": {"type": "string"}, "timeout": {"type": "integer"},
        }, "required": ["action"]})

    # ── hermes_search v7.0（主力搜索引擎）──────────────────────────
    def tool_hermes_search(query: str = "", sources: str = "", max_results: int = 5, **kwargs) -> str:
        """Hermes Search v7.0——12后端多源聚合搜索引擎。支持知乎/B站/arXiv/Google Scholar等。"""
        import json as _json
        import sys as _sys
        _sys.path.insert(0, "/home/zcs/search-engine")
        try:
            from hermes_search import search_web, search_arxiv, search_cnscrape
        except ImportError as e:
            return _json.dumps({"error": f"hermes_search模块未安装: {e}"})

        if not query:
            return _json.dumps({"error": "query参数不能为空"})

        source_list = [s.strip() for s in sources.split(",") if s.strip()] if sources else ["cnscrape"]
        all_results = []

        for src in source_list:
            try:
                if src == "arxiv":
                    results = search_arxiv(query, max_results=max_results)
                elif src == "cnscrape":
                    results = search_cnscrape(query, max_results=max_results)
                else:
                    results = search_web(query, max_results=max_results)
                all_results.extend(results)
            except Exception as e:
                all_results.append({"title": f"[{src}错误]", "url": "", "description": str(e)})

        # 去重（按URL）
        seen_urls = set()
        unique = []
        for r in all_results:
            url = r.url if hasattr(r, 'url') else r.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                if hasattr(r, '__dict__'):
                    unique.append({"title": r.title, "url": r.url, "description": r.snippet[:200]})
                else:
                    unique.append(r)

        return _json.dumps({"query": query, "sources": source_list, "results": unique[:max_results]}, ensure_ascii=False)

    registry.register("hermes_search", tool_hermes_search,
        "Hermes Search v7.0——多源聚合搜索(cnscrape/arXiv/web)",
        schema={"type": "object", "properties": {
            "query": {"type": "string", "description": "搜索关键词"},
            "sources": {"type": "string", "description": "搜索源，逗号分隔(cnscrape,arxiv,web)"},
            "max_results": {"type": "integer", "description": "最大结果数", "default": 5},
        }, "required": ["query"]})

    # ── tool_failure_log（失败驱动工具发现）──────────────────────────
    def tool_failure_log(action: str = "health", **kwargs) -> str:
        """工具失败日志——记录失败→统计频率→提取需求信号。只有失败的才是真需求。"""
        import json as _json
        import sys as _sys
        _sys.path.insert(0, "/home/zcs/.hermes/hermes-agent")
        try:
            from tools.tool_failure_log import log as _log, stats as _stats, signals as _signals, health as _health
        except ImportError as e:
            return _json.dumps({"error": f"tool_failure_log模块未安装: {e}"})

        if action == "log":
            tool_name = kwargs.get("tool_name", "")
            if not tool_name:
                return _json.dumps({"error": "tool_name不能为空"})
            result = _log(tool_name, kwargs.get("failure_type", "L1_hard"),
                          kwargs.get("error_msg", ""), kwargs.get("context", ""))
            return _json.dumps(result, ensure_ascii=False)
        elif action == "stats":
            return _json.dumps(_stats(int(kwargs.get("days", 7))), ensure_ascii=False)
        elif action == "signals":
            return _json.dumps(_signals(), ensure_ascii=False)
        elif action == "health":
            return _json.dumps(_health(), ensure_ascii=False)
        return _json.dumps({"error": f"Unknown action: {action}"})

    registry.register("tool_failure_log", tool_failure_log,
        "工具失败日志——记录失败→统计→需求信号(失败驱动锻造)",
        schema={"type": "object", "properties": {
            "action": {"type": "string", "enum": ["log","stats","signals","health"]},
            "tool_name": {"type": "string"},
            "failure_type": {"type": "string"},
            "error_msg": {"type": "string"},
            "context": {"type": "string"},
        }, "required": ["action"]})

    # ── daily_health_check（每日健康报告）──────────────────────────
    def tool_daily_health_check(action: str = "health", **kwargs) -> str:
        """搜索后端每日健康报告——检查所有后端状态。"""
        import json as _json
        import sys as _sys
        _sys.path.insert(0, "/home/zcs/.hermes/hermes-agent")
        try:
            from tools.daily_health_check import run_check as _run, get_report as _report, get_history as _history
        except ImportError as e:
            return _json.dumps({"error": f"daily_health_check模块未安装: {e}"})

        if action == "run":
            return _json.dumps(_run(), ensure_ascii=False)
        elif action == "report":
            return _json.dumps(_report(), ensure_ascii=False)
        elif action == "history":
            return _json.dumps(_history(), ensure_ascii=False)
        return _json.dumps({"status": "ok"})

    registry.register("daily_health_check", tool_daily_health_check,
        "搜索后端每日健康报告——检查bing/cnscrape/ddgs/arxiv等状态")

    # ── tool_hunter（失败驱动工具猎手）──────────────────────────────
    def tool_tool_hunter(action: str = "health", **kwargs) -> str:
        """失败驱动工具猎手——从失败信号自动搜GitHub找替代品。"""
        import json as _json
        import sys as _sys
        _sys.path.insert(0, "/home/zcs/.hermes/hermes-agent")
        try:
            from tools.tool_hunter import hunt as _hunt, search as _search
        except ImportError as e:
            return _json.dumps({"error": f"tool_hunter模块未安装: {e}"})

        if action == "hunt":
            return _json.dumps(_hunt(), ensure_ascii=False)
        elif action == "search":
            tool_name = kwargs.get("tool_name", "")
            if not tool_name:
                return _json.dumps({"error": "tool_name不能为空"})
            return _json.dumps(_search(tool_name), ensure_ascii=False)
        return _json.dumps({"status": "ok"})

    registry.register("tool_hunter", tool_tool_hunter,
        "失败驱动工具猎手——从失败信号搜GitHub找替代品",
        schema={"type": "object", "properties": {
            "action": {"type": "string", "enum": ["hunt","search","health"]},
            "tool_name": {"type": "string"},
        }, "required": ["action"]})

    # ── auto_forge（工厂自动锻造）──────────────────────────────────
    def tool_auto_forge(action: str = "health", **kwargs) -> str:
        """工厂自动锻造——评估GitHub项目是否可锻造为Hermes工具。"""
        import json as _json
        import sys as _sys
        _sys.path.insert(0, "/home/zcs/.hermes/hermes-agent")
        try:
            from tools.auto_forge import evaluate as _eval, forge as _forge
        except ImportError as e:
            return _json.dumps({"error": f"auto_forge模块未安装: {e}"})

        if action == "evaluate":
            url = kwargs.get("url", "")
            if not url:
                return _json.dumps({"error": "url不能为空"})
            return _json.dumps(_eval(url), ensure_ascii=False)
        elif action == "forge":
            url = kwargs.get("url", "")
            if not url:
                return _json.dumps({"error": "url不能为空"})
            return _json.dumps(_forge(url), ensure_ascii=False)
        return _json.dumps({"status": "ok"})

    registry.register("auto_forge", tool_auto_forge,
        "工厂自动锻造——评估GitHub项目→生成Hermes工具框架",
        schema={"type": "object", "properties": {
            "action": {"type": "string", "enum": ["evaluate","forge","health"]},
            "url": {"type": "string"},
        }, "required": ["action"]})

    # ── Resilience（从Crawlee移植的韧性模块）─────────────────────
    def tool_resilience(action: str = "health", **kwargs) -> str:
        """韧性模块——错误分类+指数退避重试+URL去重。从Crawlee源码提取的轻量机制。"""
        import json as _json
        import sys as _sys
        _sys.path.insert(0, "/home/zcs/.hermes/hermes-agent")
        try:
            from tools.resilience import (
                classify_error, retry_with_backoff, normalize_url,
                URLDeduplicator, ErrorType, parse_retry_after
            )
        except ImportError as e:
            return _json.dumps({"error": f"resilience模块未安装: {e}"})

        if action == "classify":
            status = kwargs.get("status_code")
            timeout = kwargs.get("timeout", False)
            result = classify_error(
                status_code=int(status) if status else None,
                timeout=bool(timeout)
            )
            return _json.dumps({"error_type": result.value, "action": "classify"})
        elif action == "dedup_check":
            url = kwargs.get("url", "")
            dedup = URLDeduplicator()
            return _json.dumps({"url": url, "is_new": dedup.is_new(url)})
        elif action == "dedup_mark":
            url = kwargs.get("url", "")
            dedup = URLDeduplicator()
            dedup.mark_seen(url)
            return _json.dumps({"url": url, "marked": True})
        elif action == "normalize":
            url = kwargs.get("url", "")
            return _json.dumps({"original": url, "normalized": normalize_url(url)})
        elif action == "health":
            return _json.dumps({"status": "ok", "module": "resilience", "version": "1.0.0"})
        return _json.dumps({"error": f"Unknown action: {action}"})

    registry.register("resilience", tool_resilience,
        "韧性模块——错误分类+指数退避+URL去重(从Crawlee移植)",
        schema={"type": "object", "properties": {
            "action": {"type": "string", "enum": ["classify","dedup_check","dedup_mark","normalize","health"]},
            "status_code": {"type": "integer"}, "timeout": {"type": "boolean"},
            "url": {"type": "string"},
        }, "required": ["action"]})

    # ── Loop Detector（从browser-use移植的循环检测）────────────────
    def tool_loop_detector(action: str = "health", **kwargs) -> str:
        """循环检测器——防止Agent卡死无限重试。从browser-use源码移植。"""
        import json as _json
        import sys as _sys
        _sys.path.insert(0, "/home/zcs/.hermes/hermes-agent")
        try:
            from tools.loop_detector import LoopDetector
        except ImportError as e:
            return _json.dumps({"error": f"loop_detector模块未安装: {e}"})

        if action == "record":
            action_type = kwargs.get("action_type", "unknown")
            params = kwargs.get("params", {})
            if isinstance(params, str):
                params = {"raw": params}
            d = LoopDetector(window_size=20, repeat_nudge_1=5)
            d.record_action(action_type, params)
            nudge = d.get_nudge()
            return _json.dumps({"action": "record", "action_type": action_type,
                                "nudge": nudge, "rep_count": d.max_repetition_count})
        elif action == "health":
            return _json.dumps({"status": "ok", "module": "loop_detector", "version": "1.0.0"})
        return _json.dumps({"error": f"Unknown action: {action}"})

    registry.register("loop_detector", tool_loop_detector,
        "循环检测器——防止Agent卡死无限重试(从browser-use移植)",
        schema={"type": "object", "properties": {
            "action": {"type": "string", "enum": ["record","check","stats","reset","health"]},
            "action_type": {"type": "string"}, "params": {"type": "object"},
        }, "required": ["action"]})

    # ── Snapshot Enhancer（从browser-use移植的结构化元素索引）──────
    def tool_snapshot_enhancer(action: str = "health", **kwargs) -> str:
        """结构化元素索引——解析ariaSnapshot生成可交互元素列表+操作提示。"""
        import json as _json
        import sys as _sys
        _sys.path.insert(0, "/home/zcs/.hermes/hermes-agent")
        try:
            from tools.snapshot_enhancer import enhance_snapshot, count_interactive, parse_snapshot_elements
        except ImportError as e:
            return _json.dumps({"error": f"snapshot_enhancer模块未安装: {e}"})

        if action == "enhance":
            snapshot = kwargs.get("snapshot", "")
            if not snapshot:
                return _json.dumps({"error": "snapshot参数不能为空"})
            result = enhance_snapshot(snapshot)
            return _json.dumps({"action": "enhance", "enhanced": result})
        elif action == "count":
            snapshot = kwargs.get("snapshot", "")
            return _json.dumps({"action": "count", "interactive_count": count_interactive(snapshot)})
        elif action == "health":
            return _json.dumps({"status": "ok", "module": "snapshot_enhancer", "version": "1.0.0"})
        return _json.dumps({"error": f"Unknown action: {action}"})

    registry.register("snapshot_enhancer", tool_snapshot_enhancer,
        "结构化元素索引——解析ariaSnapshot生成可交互元素列表+操作提示(从browser-use移植)",
        schema={"type": "object", "properties": {
            "action": {"type": "string", "enum": ["enhance","count","parse","health"]},
            "snapshot": {"type": "string"}, "line": {"type": "string"},
        }, "required": ["action"]})

    # ── Doc Parser（PaddleOCR-VL文档结构化解析）─────────────────────
    from .doc_parser import tool_doc_parser
    registry.register("doc_parser", tool_doc_parser,
        "文档结构化解析——表格→Markdown、公式→LaTeX、图表→语义描述(PaddleOCR-VL)",
        schema={"type": "object", "properties": {
            "file_path": {"type": "string", "description": "PDF或图片文件路径"},
            "pages": {"type": "string", "description": "页码范围（仅PDF），如 1-5,8"},
            "elements": {"type": "string", "enum": ["table","formula","chart","text"],
                         "description": "聚焦元素类型，省略则自动识别全部"},
            "output_format": {"type": "string", "enum": ["markdown","json"],
                              "description": "输出格式（默认markdown）"},
        }, "required": ["file_path"]})

    # ── Fcrawl（HTTP级网页抓取——本地免费零API）────────────────────
    from .fcrawl import handle as _fcrawl_handle
    registry.register("fcrawl", _fcrawl_handle,
        "Fcrawl网页爬虫引擎——本地免费零API Key。六大功能: scrape(单页→Markdown), crawl(整站BFS), map(URL发现), search(搜索+完整内容), batch(批量并行), agent(自然语言搜索+抓取)。反爬自动降级，BM25焦点提取。",
        schema={"type": "object", "properties": {
            "action": {"type": "string", "enum": ["scrape","crawl","map","search","batch","agent","health"],
                       "description": "功能: scrape=单页抓取, crawl=整站爬取, map=URL发现, search=搜索+内容, batch=批量, agent=自然语言搜索, health=健康检查"},
            "url": {"type": "string", "description": "目标URL (scrape/crawl/map必填)"},
            "urls": {"type": "array", "items": {"type": "string"}, "description": "URL列表 (batch必填)"},
            "query": {"type": "string", "description": "搜索查询 (search必填)"},
            "prompt": {"type": "string", "description": "自然语言任务 (agent必填)"},
            "focus": {"type": "string", "description": "BM25焦点提取关键词"},
            "format": {"type": "string", "enum": ["markdown","html","json"], "description": "输出格式"},
            "limit": {"type": "integer", "description": "最大数量 (crawl默认50)"},
            "depth": {"type": "integer", "description": "爬取深度 (crawl默认3)"},
        }, "required": ["action"]})

    # ── Crawl4AI（浏览器级抓取——JS渲染/SPA，subprocess桥接Hermes venv）─
    from .crawl4ai import tool_crawl4ai
    registry.register("crawl4ai", tool_crawl4ai,
        "浏览器级网页抓取——Playwright驱动JS渲染+反爬，输出LLM原生Markdown。适用SPA/动态加载/无限滚动页面。降级链: crawl4ai(JS渲染) → fcrawl(HTTP级)。",
        schema={"type": "object", "properties": {
            "action": {"type": "string", "enum": ["scrape","batch","deep_crawl","extract"],
                       "description": "功能: scrape=单页, batch=批量, deep_crawl=多页爬取, extract=结构化抽取"},
            "url": {"type": "string", "description": "目标URL"},
            "urls": {"type": "string", "description": "逗号分隔URL列表 (batch)"},
            "query": {"type": "string", "description": "深爬过滤关键词"},
            "depth": {"type": "integer", "description": "爬取深度 (deep_crawl默认2)"},
            "max_pages": {"type": "integer", "description": "最大页数 (deep_crawl默认10)"},
            "css_selector": {"type": "string", "description": "CSS选择器聚焦提取"},
        }, "required": ["action"]})

    # ── OCR（HTTP桥接→Hermes端Unlimited-OCR 9872）──────────────────
    from .ocr import tool_ocr
    registry.register("ocr", tool_ocr,
        "长文档OCR——百度Unlimited-OCR(R-SWA机制)，几十页PDF一次解析。file_type=auto/image/pdf。",
        schema={"type": "object", "properties": {
            "file_path": {"type": "string", "description": "图片或PDF文件绝对路径"},
            "file_type": {"type": "string", "enum": ["auto","image","pdf"], "description": "文件类型(默认auto)"},
            "prompt": {"type": "string", "description": "OCR提示词"},
        }, "required": ["file_path"]})

    return registry
