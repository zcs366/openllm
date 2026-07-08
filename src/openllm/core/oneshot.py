"""
openLLM One-shot LLM Helper — 不走完整Agent循环的轻量级LLM调用。

从 Hermes v0.18.0 的 llm.oneshot RPC 学来。
有些任务不需要完整的ISA→IO-S→ISN→IKO管线：
  - 分类、打标签
  - 翻译
  - 摘要
  - 结构化提取
  - 简单问答

One-shot直接调provider，跳过agent loop。
"""

import logging
from typing import Optional

from .provider import Message, create_provider

logger = logging.getLogger("openllm.oneshot")


def oneshot(
    prompt: str,
    system: str = "你是一个有帮助的助手。",
    model: str = "deepseek-chat",
    provider: str = "deepseek",
    temperature: float = 0.3,
    max_tokens: int = 1024,
    api_key: str = "",
    endpoint: str = "",
) -> str:
    """
    一次性LLM调用。不走agent loop，不加载记忆/工具/身份。

    Args:
        prompt: 用户提示
        system: 系统提示（默认简短助手）
        model: 模型名
        provider: provider类型 (deepseek/openai/ollama/local)
        temperature: 温度（低=确定性，高=创造性）
        max_tokens: 最大输出token
        api_key: API key（空则从config读）
        endpoint: endpoint URL（空则从config读）

    Returns:
        LLM响应文本

    Raises:
        RuntimeError: API调用失败
    """
    try:
        llm = create_provider(provider, model=model, temperature=temperature,
                              max_tokens=max_tokens, api_key=api_key or None,
                              endpoint=endpoint or None)
    except Exception as e:
        raise RuntimeError(f"Provider初始化失败: {e}") from e

    messages = [
        Message(role="system", content=system),
        Message(role="user", content=prompt),
    ]

    logger.debug("oneshot: model=%s tokens=%d prompt=%s...",
                 model, max_tokens, prompt[:50])

    response = llm.chat(messages)

    logger.debug("oneshot: %d tokens, %.0fms",
                 response.usage.get("total_tokens", 0),
                 response.latency_ms)

    return response.content


def classify(text: str, categories: list[str], model: str = "deepseek-chat",
             provider: str = "deepseek", api_key: str = "") -> str:
    """
    用LLM做分类。返回categories中的一个。

    Args:
        text: 待分类文本
        categories: 可选类别列表
        model: 模型名
        provider: provider类型
        api_key: API key
    """
    cat_list = ", ".join(f'"{c}"' for c in categories)
    prompt = f"将以下文本分类到这些类别之一: [{cat_list}]\n\n只返回类别名，不解释。\n\n文本: {text}"
    return oneshot(prompt, temperature=0.0, max_tokens=20, model=model,
                   provider=provider, api_key=api_key)


def extract(text: str, schema: str, model: str = "deepseek-chat",
            provider: str = "deepseek", api_key: str = "") -> str:
    """
    用LLM做结构化提取。返回JSON字符串。

    Args:
        text: 待提取文本
        schema: JSON schema描述
        model: 模型名
        provider: provider类型
        api_key: API key
    """
    prompt = f"从以下文本中提取信息，按JSON格式返回。\n\nSchema: {schema}\n\n文本: {text}\n\n只返回JSON，不解释。"
    return oneshot(prompt, temperature=0.0, max_tokens=512, model=model,
                   provider=provider, api_key=api_key)


def summarize(text: str, max_words: int = 100, model: str = "deepseek-chat",
              provider: str = "deepseek", api_key: str = "") -> str:
    """
    用LLM做摘要。

    Args:
        text: 待摘要文本
        max_words: 摘要最大字数
        model: 模型名
        provider: provider类型
        api_key: API key
    """
    prompt = f"将以下文本压缩为{max_words}字以内的摘要:\n\n{text}"
    return oneshot(prompt, temperature=0.3, max_tokens=max_words * 2, model=model,
                   provider=provider, api_key=api_key)
