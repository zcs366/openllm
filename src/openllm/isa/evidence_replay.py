"""
Evidence Replay — 上下文证据回放
================================

来源：arXiv:2607.02509 "ReContext: Recursive Evidence Replay"

核心洞察：
- 128K上下文中，top 0.1%的token（128个）覆盖50-80%的question-conditioned relevance
- 递归证据回放可以单调改善证据利用率（cos(h^(j), y)严格递增）
- 不需要改模型权重——纯inference-time包装

本模块实现简化版Evidence Replay：
1. 从build_context输出的七要素中提取相关证据
2. 用query-conditioned scoring选择top-K证据
3. 将证据追加到prompt尾部（replay scaffold）
4. 与StatefulAudit联动：累积风险超阈值时自动启用
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("openllm.isa.evidence_replay")


@dataclass
class EvidenceSpan:
    """一段证据span。"""
    text: str
    score: float           # query-conditioned relevance score
    source: str            # 来源（"memory" | "causal" | "search" | "tool_result"）
    token_count: int = 0   # 近似token数

    def __post_init__(self):
        if self.token_count == 0:
            # 粗略估计：中文1字≈2token，英文1词≈1.3token
            self.token_count = max(1, len(self.text) // 2)


@dataclass
class ReplayResult:
    """证据回放结果。"""
    evidence_spans: list[EvidenceSpan]
    replay_text: str       # 追加到prompt的文本
    total_tokens: int      # replay增加的token数
    sources_used: list[str]  # 使用了哪些来源

    @property
    def count(self) -> int:
        return len(self.evidence_spans)


class EvidenceReplay:
    """上下文证据回放器。

    核心逻辑：
    - 从Context的七要素中提取候选证据
    - 用query-conditioned scoring排序
    - 选择top-K追加到prompt尾部
    - 保留完整原始上下文（不裁剪）

    用法：
        replay = EvidenceReplay(top_k=5, max_tokens=512)
        result = replay.replay(user_message, context)
        # result.replay_text 追加到prompt
    """

    def __init__(self, top_k: int = 5, max_tokens: int = 512):
        """
        Args:
            top_k: 选择top-K个证据span
            max_tokens: replay增加的最大token数
        """
        self.top_k = top_k
        self.max_tokens = max_tokens

    def replay(self, query: str, context: Any) -> ReplayResult:
        """从Context中提取证据并生成replay scaffold。

        Args:
            query: 用户原始query
            context: Context对象（七要素上下文）

        Returns:
            ReplayResult 包含replay文本和元数据
        """
        # Step 1: 从七要素中提取候选证据
        candidates = self._extract_candidates(query, context)

        # Step 2: query-conditioned scoring
        scored = self._score_candidates(query, candidates)

        # Step 3: top-K selection
        top_spans = self._select_top_k(scored, self.top_k, self.max_tokens)

        # Step 4: materialize replay scaffold
        replay_text = self._materialize(top_spans)

        sources_used = list(set(s.source for s in top_spans))
        total_tokens = sum(s.token_count for s in top_spans)

        result = ReplayResult(
            evidence_spans=top_spans,
            replay_text=replay_text,
            total_tokens=total_tokens,
            sources_used=sources_used,
        )

        logger.info(
            f"Evidence Replay: {result.count} spans, "
            f"{result.total_tokens} tokens, "
            f"sources={sources_used}"
        )

        return result

    def _extract_candidates(self, query: str, context: Any) -> list[EvidenceSpan]:
        """从Context的七要素中提取候选证据span。"""
        candidates = []

        # ② 记忆——最近决策、因果教训
        memory = getattr(context, 'memory', {})
        if isinstance(memory, dict):
            for key, value in memory.items():
                if isinstance(value, str) and len(value) > 10:
                    candidates.append(EvidenceSpan(
                        text=value, score=0.0, source="memory"
                    ))
                elif isinstance(value, list):
                    for item in value[:3]:  # 最多取3条
                        if isinstance(item, str) and len(item) > 10:
                            candidates.append(EvidenceSpan(
                                text=item, score=0.0, source="memory"
                            ))
                        elif isinstance(item, dict):
                            text = item.get("content", item.get("text", str(item)))
                            if len(text) > 10:
                                candidates.append(EvidenceSpan(
                                    text=text, score=0.0, source="memory"
                                ))

        # ④ 因果提示——上次类似操作的因果历史
        causal = getattr(context, 'causal_hints', [])
        if isinstance(causal, list):
            for hint in causal[:5]:
                if isinstance(hint, str) and len(hint) > 10:
                    candidates.append(EvidenceSpan(
                        text=hint, score=0.0, source="causal"
                    ))
                elif isinstance(hint, dict):
                    text = hint.get("lesson", hint.get("summary", str(hint)))
                    if len(text) > 10:
                        candidates.append(EvidenceSpan(
                            text=text, score=0.0, source="causal"
                        ))

        # ⑦ 搜索结果——触手脑检索结果
        search = getattr(context, 'search_results', [])
        if isinstance(search, list):
            for result in search[:5]:
                if isinstance(result, str) and len(result) > 10:
                    candidates.append(EvidenceSpan(
                        text=result, score=0.0, source="search"
                    ))
                elif isinstance(result, dict):
                    text = result.get("content", result.get("snippet", str(result)))
                    if len(text) > 10:
                        candidates.append(EvidenceSpan(
                            text=text, score=0.0, source="search"
                        ))

        return candidates

    def _score_candidates(self, query: str,
                          candidates: list[EvidenceSpan]) -> list[EvidenceSpan]:
        """用query-conditioned scoring给候选证据打分。

        简化版：基于关键词重叠 + 来源权重。
        完整版（Phase 2）应该用模型内部attention scores。
        """
        query_words = set(query.lower().split())
        # 中文分词简化：按字切分
        query_chars = set(query)

        source_weights = {
            "causal": 1.5,    # 因果历史权重最高
            "memory": 1.2,    # 记忆次之
            "search": 1.0,    # 搜索结果
            "tool_result": 0.8,
        }

        for span in candidates:
            span_text_lower = span.text.lower()
            span_words = set(span_text_lower.split())
            span_chars = set(span.text)

            # 关键词重叠得分
            word_overlap = len(query_words & span_words) / max(len(query_words), 1)
            char_overlap = len(query_chars & span_chars) / max(len(query_chars), 1)

            # 综合得分 = 0.6×词重叠 + 0.4×字重叠 × 来源权重
            base_score = 0.6 * word_overlap + 0.4 * char_overlap
            span.score = base_score * source_weights.get(span.source, 1.0)

        return candidates

    def _select_top_k(self, scored: list[EvidenceSpan],
                      k: int, max_tokens: int) -> list[EvidenceSpan]:
        """选择top-K证据，不超过max_tokens。"""
        # 按score降序排序
        sorted_spans = sorted(scored, key=lambda s: s.score, reverse=True)

        selected = []
        total_tokens = 0

        for span in sorted_spans:
            if len(selected) >= k:
                break
            if total_tokens + span.token_count > max_tokens:
                continue
            if span.score < 0.01:  # 得分太低的跳过
                continue
            selected.append(span)
            total_tokens += span.token_count

        return selected

    def _materialize(self, spans: list[EvidenceSpan]) -> str:
        """将选中的证据span生成replay scaffold文本。"""
        if not spans:
            return ""

        lines = ["[Evidence Replay — 相关证据回放]"]
        for i, span in enumerate(spans, 1):
            # 截断过长的span
            text = span.text[:300]
            if len(span.text) > 300:
                text += "..."
            lines.append(f"证据{i} [{span.source}] (相关度={span.score:.3f}): {text}")

        return "\n".join(lines)


def create_replay_for_context(query: str, context: Any,
                               top_k: int = 5,
                               max_tokens: int = 512) -> str:
    """便捷函数：为给定query和context生成evidence replay文本。

    Returns:
        replay_text: 追加到prompt尾部的文本，空字符串表示无需replay
    """
    replay = EvidenceReplay(top_k=top_k, max_tokens=max_tokens)
    result = replay.replay(query, context)
    return result.replay_text
