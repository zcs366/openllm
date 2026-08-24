"""
跨session上下文污染防护——ContextGuard。

关键词匹配 + 身份锚点权重，实现简单而有效的相关性识别。
设计来源：context-guard-case-study.md（五人合议+七神启示）。
"""

import re
from typing import List, Set


# 停用词表——过滤无实际语义的高频词
_STOP_WORDS: Set[str] = {
    "的", "了", "在", "是", "有", "和", "与", "或", "但", "而",
    "如果", "那么", "就", "都", "也", "还", "这", "那", "被",
    "把", "从", "对", "到", "为", "能", "会", "可以", "没有",
    "一个", "我们", "你们", "他们", "自己", "什么", "怎么",
}


def _load_identity_keywords() -> Set[str]:
    """
    从openLLM的Soul锚点加载身份关键词集合。

    提取规则：
      - anchors['name'] → 单个词
      - anchors['values'] 列表中每个条目 → 按冒号/逗号拆分后取词
      - anchors['relationship'] 的键名 → 人物名
      - Soul顶层字段 name / creator → 追加

    加载失败或Soul不可用时优雅降级为空集。
    """
    try:
        from openllm.identity.soul import Soul
        soul = Soul()
    except Exception:
        return set()

    keywords: Set[str] = set()

    # 顶层字段
    if soul.name:
        keywords.add(soul.name)
    if soul.creator:
        keywords.add(soul.creator)

    # anchors 拆解
    anchors = soul.anchors
    if not isinstance(anchors, dict):
        return keywords

    # name 锚点
    name_val = anchors.get("name", "")
    if isinstance(name_val, str) and name_val:
        keywords.add(name_val)

    # values 列表——每个条目按冒号/逗号拆分成短语
    values = anchors.get("values", [])
    if isinstance(values, list):
        for entry in values:
            if not isinstance(entry, str):
                continue
            parts = re.split(r"[：:，,]", entry)
            for part in parts:
                word = part.strip()
                if word and len(word) >= 1:
                    keywords.add(word)

    # relationship 键名——人物名
    rel = anchors.get("relationship", {})
    if isinstance(rel, dict):
        for person_name in rel.keys():
            if isinstance(person_name, str) and person_name:
                keywords.add(person_name)

    return keywords


class ContextGuard:
    """
    跨session上下文污染防护。

    通过关键词匹配 + 身份锚点权重，判断两段上下文是否相关。
    不相关的内容被视为"污染"，应被清理。

    Attributes:
        threshold: 相关性阈值，分数 >= threshold 则判定为相关。
        _identity_keywords: 身份锚点关键词集合，用于加权。
    """

    def __init__(self, threshold: float = 0.1) -> None:
        """
        初始化ContextGuard。

        Args:
            threshold: 相关性判定阈值，默认0.1。
        """
        self.threshold: float = threshold
        self._identity_keywords: Set[str] = _load_identity_keywords()

    def extract_keywords(self, text: str) -> Set[str]:
        """
        从文本中提取关键词集合。

        提取规则（忠实于设计文档，补全工程细节）：
          - 中文词组：2-4个汉字的连续子串（重叠提取，覆盖更多词组）
          - 英文单词：字母数字组合（长度>1）
          - 单个中文字符：用于身份锚点匹配
          - 过滤停用词

        设计文档原始正则为 r'[\\u4e00-\\u9fa5]{2,4}'（贪婪匹配），
        实际使用重叠提取以避免贪婪匹配在连续中文文本上切割出无意义子串。

        Args:
            text: 输入文本。

        Returns:
            关键词集合。
        """
        keywords: Set[str] = set()

        # 中文词组（2-4字）——重叠提取，覆盖所有子串
        chinese_runs: List[str] = re.findall(r"[\u4e00-\u9fa5]+", text)
        for run in chinese_runs:
            for length in range(2, min(5, len(run) + 1)):
                for i in range(len(run) - length + 1):
                    candidate = run[i : i + length]
                    if candidate not in _STOP_WORDS:
                        keywords.add(candidate)

        # 英文单词（字母数字组合）
        english_words: List[str] = re.findall(r"[a-zA-Z0-9]+", text)
        for w in english_words:
            if len(w) > 1:
                keywords.add(w)

        # 添加单个中文字符（用于身份锚点匹配）
        single_chars: List[str] = re.findall(r"[\u4e00-\u9fa5]", text)
        keywords.update(single_chars)

        return keywords

    def calculate_relevance(
        self,
        current_keywords: Set[str],
        previous_keywords: Set[str],
    ) -> float:
        """
        计算两组关键词的相关性分数。

        公式：
          jaccard = |intersection| / |union|
          identity_weight = |intersection ∩ identity_keywords| × 2 × 0.1
          relevance = jaccard + identity_weight（封顶1.0）

        Args:
            current_keywords: 当前上下文的关键词集合。
            previous_keywords: 历史上下文的关键词集合。

        Returns:
            相关性分数，范围 [0.0, 1.0]。
        """
        if not current_keywords or not previous_keywords:
            return 0.0

        # Jaccard相似度
        intersection: Set[str] = current_keywords & previous_keywords
        union: Set[str] = current_keywords | previous_keywords
        jaccard: float = len(intersection) / len(union) if union else 0.0

        # 身份锚点权重：交集中属于身份关键词的数量 × 2 × 0.1
        identity_intersection: Set[str] = intersection & self._identity_keywords
        identity_weight: float = len(identity_intersection) * 2 * 0.1

        relevance: float = jaccard + identity_weight
        return min(relevance, 1.0)

    def is_relevant(self, current_context: str, previous_context: str) -> bool:
        """
        判断两段上下文是否相关。

        Args:
            current_context: 当前上下文文本。
            previous_context: 历史上下文文本。

        Returns:
            True 表示相关，False 表示为污染。
        """
        current_keywords = self.extract_keywords(current_context)
        previous_keywords = self.extract_keywords(previous_context)
        relevance = self.calculate_relevance(current_keywords, previous_keywords)
        return relevance >= self.threshold

    def clean_context(
        self,
        current_context: str,
        candidate_contexts: List[str],
    ) -> List[str]:
        """
        从候选上下文列表中过滤掉污染内容，只保留相关项。

        对每个候选上下文跑 is_relevant 判定。

        Args:
            current_context: 当前上下文文本。
            candidate_contexts: 候选上下文列表（来自历史/其他session的注入片段）。

        Returns:
            仅保留相关项的列表。
        """
        return [
            ctx for ctx in candidate_contexts
            if self.is_relevant(current_context, ctx)
        ]
