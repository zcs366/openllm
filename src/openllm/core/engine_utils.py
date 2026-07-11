"""
engine_utils.py — engine小型工具函数

从engine.py提取的独立工具函数：
- startup_audit: 启动安全审计
- get_api_key: 获取API key
- build_system_prompt: 构建system prompt
- trim_history: 限制history长度
"""
import os
import logging
from typing import Any

logger = logging.getLogger("openllm.engine.utils")


def startup_audit():
    """启动安全审计（不阻塞，不抛异常）。"""
    try:
        from ..security.startup_audit import run_startup_audit
        report = run_startup_audit()
        if "❌" in report or "⚠️" in report:
            print(report)
    except Exception as e:
        logger.debug(f"启动审计跳过: {e}")


def get_api_key(provider: str) -> str:
    """获取当前provider的API key。"""
    if provider == "ollama":
        return "ollama"
    elif provider == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY", "")
    elif provider == "gemini":
        return os.environ.get("GEMINI_API_KEY", "")
    return os.environ.get("DEEPSEEK_API_KEY", "")


def build_system_prompt(tools_list: list, identity: str, mem_ctx: dict) -> str:
    """构建完整system prompt。"""
    parts = [identity]
    if tools_list:
        parts.append("\n## 可用工具")
        for t in tools_list:
            parts.append(f"- {t['name']}: {t['description']}")
    if mem_ctx.get("status") != "empty":
        decisions = mem_ctx.get("decisions", [])
        if decisions:
            parts.append(f"\n## 上次决策\n" + "; ".join(decisions[:3]))
    return "\n".join(parts)


def trim_history(history: list, max_turns: int = 10):
    """限制history长度，保留system prompt + 最近N轮对话。"""
    if len(history) <= 1:
        return
    system = [history[0]] if history[0].role == "system" else []
    recent = history[-max_turns * 2:]
    return system + recent
