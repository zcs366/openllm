"""
Tool Search — 元工具搜索接口
=============================

Agent调用此模块发现其他可用工具。核心功能：
  - register_tools: 注册ToolDefinition列表到搜索索引
  - tool_search_handler: 按query搜索相关工具，返回top_k结果

搜索策略：BM25风格的关键词匹配（TF-IDF近似），支持中英文混合查询。
线程安全：用threading.Lock保护共享索引。

用法：
    from openllm.tool_search import (
        ToolDefinition, register_tools, tool_search_handler
    )

    register_tools([ToolDefinition(
        name="read_file",
        description="读取文件内容",
        parameters={"path": {"type": "string"}},
    )])

    results = tool_search_handler("读文件", top_k=5)
"""

import math
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── ToolDefinition ──────────────────────────────────

@dataclass
class ToolDefinition:
    """工具定义——搜索索引的基本单元。"""
    name: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "tags": self.tags,
            "effects": self.effects,
        }


# ── 搜索索引（线程安全） ────────────────────────────

_lock = threading.Lock()
_tools: Dict[str, ToolDefinition] = {}
# 预计算的倒排索引: token → set of tool names
_inverted_index: Dict[str, set] = {}
# 每个tool的token计数（用于TF）
_tool_token_counts: Dict[str, Counter] = {}


def _tokenize(text: str) -> List[str]:
    """简单的中英文分词。

    英文：按非字母数字字符分割，转小写。
    中文：按字符级别unigram + bigram（无依赖的简易分词）。
    """
    tokens = []
    # 英文token（连续字母数字序列）
    english_tokens = re.findall(r'[a-zA-Z0-9_]+', text.lower())
    tokens.extend(english_tokens)

    # 中文字符提取
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    # unigram
    tokens.extend(chinese_chars)
    # bigram
    for i in range(len(chinese_chars) - 1):
        tokens.append(chinese_chars[i] + chinese_chars[i + 1])

    return tokens


def _compute_idf(token: str) -> float:
    """计算token的IDF分数。"""
    n = len(_tools)
    if n == 0:
        return 0.0
    df = len(_inverted_index.get(token, set()))
    if df == 0:
        return 0.0
    return math.log((n + 1) / (df + 1)) + 1.0


def _score(tool_name: str, query_tokens: List[str]) -> float:
    """计算tool对query的BM25风格分数。"""
    tc = _tool_token_counts.get(tool_name, Counter())
    if not tc:
        return 0.0
    score = 0.0
    for token in query_tokens:
        tf = tc.get(token, 0)
        if tf > 0:
            idf = _compute_idf(token)
            # 简化BM25: TF-IDF with log normalization
            score += idf * (1 + math.log(tf))
    return score


def register_tools(tools: List[ToolDefinition]) -> None:
    """注册工具列表到搜索索引。线程安全。

    Args:
        tools: ToolDefinition列表，name必须唯一（后注册的覆盖先注册的）。
    """
    with _lock:
        for td in tools:
            # 构建索引文本：name + description + tags
            text = f"{td.name} {td.description} {' '.join(td.tags)}"
            tokens = _tokenize(text)
            tc = Counter(tokens)

            _tools[td.name] = td
            _tool_token_counts[td.name] = tc

            # 更新倒排索引
            for token in set(tokens):
                if token not in _inverted_index:
                    _inverted_index[token] = set()
                _inverted_index[token].add(td.name)


def tool_search_handler(query: str, top_k: int = 5) -> List[ToolDefinition]:
    """搜索工具——按query相关度返回top_k工具定义。

    Args:
        query: 搜索关键词（中英文混合）。
        top_k: 返回数量上限，默认5。

    Returns:
        按相关度降序排列的ToolDefinition列表。如果无工具注册，返回空列表。
    """
    with _lock:
        if not _tools:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        # 对所有tool计算分数
        scores: List[tuple] = []
        for name in _tools:
            s = _score(name, query_tokens)
            if s > 0:
                scores.append((name, s))

        # 按分数降序
        scores.sort(key=lambda x: x[1], reverse=True)

        # 返回top_k
        return [_tools[name] for name, _ in scores[:top_k]]
