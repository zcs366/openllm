"""llm_layer.py — 统一 LLM 调用层（G0 真裸：HTTP 直调 ollama，零 openllm import）

所有四级 harness 共用此层，保证公平：同模型、同温度、同超时。
不 import 任何 openllm.* 模块——连 provider 层都绕过。

用法:
    from llm_layer import llm_chat
    resp = llm_chat("1+1=?", system="你是一个数学助手")
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from typing import Optional

logger = logging.getLogger("ablation.llm_layer")

# ── 配置 ──
# 2026-09-01 修正：本地 ollama 被 tui_gateway 占满 GPU（17GB/22GB，93% util），
# 1+1 需 44s——G0-G3 统一改用云端 deepseek-v4-flash（1.3s/题，同模型同温度）。
# 仍为零 harness：urllib 直调，不 import 任何 openllm.* 模块。
import os
from pathlib import Path

BACKEND: str = os.environ.get("ABLATION_LLM_BACKEND", "cloud")  # cloud | ollama
DEFAULT_TEMPERATURE: float = 0.2
DEFAULT_TIMEOUT: int = 60  # 秒
MAX_RETRIES: int = 2
RETRY_DELAY: float = 1.0  # 秒（指数退避基数）

OLLAMA_URL: str = "http://localhost:11434/api/chat"
OLLAMA_MODEL: str = "huihui_ai/qwen3.5-abliterated:9b"

_CFG = json.loads(
    (Path.home() / ".openllm" / "config.json").read_text(encoding="utf-8")
)["providers"]["deepseek"]
CLOUD_ENDPOINT: str = _CFG["endpoint"]
CLOUD_API_KEY: str = _CFG["api_key"]
CLOUD_MODEL: str = _CFG["model"]


def llm_chat(
    prompt: str,
    system: str = "",
    temperature: float = DEFAULT_TEMPERATURE,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = MAX_RETRIES,
) -> str:
    """调用 ollama chat API，返回助手文本回复。

    Args:
        prompt: 用户消息。
        system: 系统提示（可选），注入为 system role。
        temperature: 生成温度。
        timeout: 单次请求超时秒数。
        max_retries: 最大重试次数（含首次）。

    Returns:
        模型回复的纯文本。

    Raises:
        RuntimeError: 超过最大重试次数后仍失败。
    """
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    if BACKEND == "cloud":
        payload = json.dumps({
            "model": CLOUD_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 512,
        }).encode("utf-8")
    else:  # ollama
        payload = json.dumps({
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
        }).encode("utf-8")

    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            headers = {"Content-Type": "application/json"}
            url = OLLAMA_URL
            if BACKEND == "cloud":
                headers["Authorization"] = f"Bearer {CLOUD_API_KEY}"
                url = CLOUD_ENDPOINT
            req = urllib.request.Request(
                url,
                data=payload,
                headers=headers,
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if BACKEND == "cloud":
                content = data["choices"][0]["message"]["content"]
            else:
                content = data["message"]["content"]
            if attempt > 0:
                logger.info(f"llm_chat 第 {attempt + 1} 次重试成功")
            return content
        except Exception as e:
            last_error = e
            logger.warning(f"llm_chat 第 {attempt + 1} 次失败: {e}")
            if attempt < max_retries - 1:
                delay = RETRY_DELAY * (2 ** attempt)
                time.sleep(delay)

    raise RuntimeError(
        f"llm_chat 在 {max_retries} 次尝试后仍失败: {last_error}"
    )
