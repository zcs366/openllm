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
        parts.append("当用户请求涉及文件操作、代码执行、搜索等任务时，你必须使用工具而非自行编写代码。")
        parts.append("调用方式：在```python代码块中调用工具函数，系统会自动执行。\n")
        for t in tools_list:
            parts.append(f"- {t['name']}: {t['description']}")
        parts.append("\n示例：用户说'读取某文件'→ 用read_file(\"路径\")；用户说'运行命令'→ 用terminal(\"命令\")")
    if mem_ctx.get("status") != "empty":
        decisions = mem_ctx.get("decisions", [])
        if decisions:
            parts.append(f"\n## 上次决策\n" + "; ".join(decisions[:3]))
    # === 源引用强制 + 不知道机制 (T-HALL-2+3) ===
    parts.append("\n## 源引用与诚实性约束")
    parts.append("1. 每个事实性声明后标注 [来源: ...]，格式如 [来源: 当前对话上下文]、[来源: 工具返回结果]、[来源: 已知知识]。")
    parts.append("2. 无法确认的事实标注 ⚠️未验证。")
    parts.append("3. 不确定时直接说「我需要查证」或「我不确定」，不要编造答案。")
    parts.append("4. 优先使用工具获取信息，而非凭记忆作答。")

    return "\n".join(parts)


def trim_history(history: list, max_turns: int = 10):
    """限制history长度，保留system prompt + 最近N轮对话。"""
    if len(history) <= 1:
        return
    system = [history[0]] if history[0].role == "system" else []
    recent = history[-max_turns * 2:]
    return system + recent
