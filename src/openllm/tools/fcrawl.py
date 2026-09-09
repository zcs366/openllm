#!/usr/bin/env python3
"""
fcrawl.py — openLLM侧fcrawl工具包装器
通过HTTP调用Hermes端fcrawl服务或直接调用CLI。

Action-based wrapper模式：一个tool + action参数路由。
"""

import json
import subprocess
import sys
from typing import Any

SCRIPT_PATH = "/home/zcs/.hermes/scripts/fcrawl.py"
# 钉死Hermes venv python——fcrawl.py依赖(curl_cffi等)装在Hermes venv，
# openLLM venv零重依赖（2026-09-10 鲁班修：sys.executable在openLLM venv缺依赖必崩）
HERMES_PY = "/home/zcs/.hermes/hermes-agent/venv/bin/python"

SCHEMA = {
    "name": "fcrawl",
    "description": "Fcrawl网页爬虫引擎——本地免费。六大功能: scrape(单页), crawl(整站), map(URL发现), search(搜索+内容), batch(批量), agent(自然语言搜索)",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["scrape", "crawl", "map", "search", "batch", "agent", "health"],
                "description": "功能选择"
            },
            "url": {"type": "string", "description": "目标URL"},
            "urls": {"type": "array", "items": {"type": "string"}, "description": "URL列表(batch)"},
            "query": {"type": "string", "description": "搜索查询"},
            "prompt": {"type": "string", "description": "自然语言任务(agent)"},
            "focus": {"type": "string", "description": "BM25焦点查询"},
            "format": {"type": "string", "enum": ["markdown", "html", "json"]},
            "limit": {"type": "integer", "description": "最大数量"},
            "depth": {"type": "integer", "description": "爬取深度"}
        },
        "required": ["action"]
    }
}


def handle(action: str, **kwargs) -> str:
    """Action-based wrapper"""
    cmd = [HERMES_PY, SCRIPT_PATH, action, "--json"]

    if action == "scrape":
        url = kwargs.get("url")
        if not url:
            return json.dumps({"error": "url required"})
        cmd.insert(3, url)
        fmt = kwargs.get("format", "markdown")
        cmd.extend(["--format", fmt])
        if kwargs.get("focus"):
            cmd.extend(["--focus", kwargs["focus"]])

    elif action == "crawl":
        url = kwargs.get("url")
        if not url:
            return json.dumps({"error": "url required"})
        cmd.insert(3, url)
        cmd.extend(["--limit", str(kwargs.get("limit", 50))])
        cmd.extend(["--depth", str(kwargs.get("depth", 3))])

    elif action == "map":
        url = kwargs.get("url")
        if not url:
            return json.dumps({"error": "url required"})
        cmd.insert(3, url)
        if kwargs.get("query"):
            cmd.extend(["--search", kwargs["query"]])

    elif action == "search":
        query = kwargs.get("query")
        if not query:
            return json.dumps({"error": "query required"})
        cmd.insert(3, query)
        cmd.extend(["--limit", str(kwargs.get("limit", 5))])

    elif action == "batch":
        urls = kwargs.get("urls", [])
        if not urls:
            return json.dumps({"error": "urls required"})
        cmd.extend(urls)

    elif action == "agent":
        prompt = kwargs.get("prompt")
        if not prompt:
            return json.dumps({"error": "prompt required"})
        cmd.insert(3, prompt)
        if kwargs.get("urls"):
            cmd.extend(["--urls"] + kwargs["urls"])

    elif action == "health":
        pass  # just health check
    else:
        return json.dumps({"error": f"Unknown action: {action}"})

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return result.stdout if result.returncode == 0 else json.dumps({"error": result.stderr[:500]})
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "Timeout 120s"})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── openLLM注册 ──────────────────────────────────────────────────
def register():
    """注册到openLLM工具系统"""
    try:
        from openllm.tools.executor import register_tool
        register_tool(SCHEMA, handle)
    except ImportError:
        pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["scrape", "crawl", "map", "search", "batch", "agent", "health"])
    parser.add_argument("--url", default="")
    parser.add_argument("--urls", nargs="*", default=[])
    parser.add_argument("--query", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--focus", default="")
    parser.add_argument("--format", default="markdown")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--depth", type=int, default=3)
    args = parser.parse_args()
    print(handle(args.action, url=args.url, urls=args.urls, query=args.query,
                 prompt=args.prompt, focus=args.focus, format=args.format,
                 limit=args.limit, depth=args.depth))
