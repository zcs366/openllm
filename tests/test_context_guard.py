"""
ContextGuard 单元测试。

覆盖场景：
  1. 相关上下文（同主题）分数 >= 0.1，is_relevant=True
  2. 污染上下文（完全无关）分数 < 0.1，is_relevant=False
  3. 身份锚点加权：含身份词的文本分数更高
  4. clean_context 过滤：混合列表只保留相关项
  5. threshold 可配置：提高阈值后原本相关的变不相关
  6. Soul 加载失败时降级为空集不崩
"""

from unittest.mock import patch
import pytest

from openllm.memory.context_guard import ContextGuard, _load_identity_keywords


# ── 辅助文本 ──────────────────────────────────────────

# 同主题：围绕"记忆系统"，共享相同核心短语
TEXT_MEMORY_A = "openLLM的记忆系统采用多层架构。短期记忆和长期记忆。"
TEXT_MEMORY_B = "openLLM的记忆系统采用多层架构。管理短期记忆和长期记忆。"

# 污染：完全无关主题（烹饪）
TEXT_POLLUTED = "红烧肉的做法：五花肉切块，冷水下锅焯水，加入冰糖和酱油炖煮。"

# 身份锚点相关：包含"OpenLLM"、"张成市"等身份关键词
TEXT_IDENTITY_A = "OpenLLM的张成市设计了记忆系统，让AI从交互中学习。"
TEXT_IDENTITY_B = "张子和OpenLLM的创造者一起构建了经验积累框架。"


# ── 测试1：相关上下文 ──────────────────────────────────

class TestRelevantContext:
    """同主题文本应判定为相关。"""

    def test_same_topic_is_relevant(self):
        guard = ContextGuard()
        assert guard.is_relevant(TEXT_MEMORY_A, TEXT_MEMORY_B) is True

    def test_relevant_score_above_threshold(self):
        guard = ContextGuard()
        kw_a = guard.extract_keywords(TEXT_MEMORY_A)
        kw_b = guard.extract_keywords(TEXT_MEMORY_B)
        score = guard.calculate_relevance(kw_a, kw_b)
        assert score >= 0.1, f"同主题分数应 >= 0.1，实际 {score}"


# ── 测试2：污染上下文 ──────────────────────────────────

class TestPollutedContext:
    """完全无关的文本应判定为污染。"""

    def test_unrelated_is_not_relevant(self):
        guard = ContextGuard()
        assert guard.is_relevant(TEXT_MEMORY_A, TEXT_POLLUTED) is False

    def test_polluted_score_below_threshold(self):
        guard = ContextGuard()
        kw_mem = guard.extract_keywords(TEXT_MEMORY_A)
        kw_cook = guard.extract_keywords(TEXT_POLLUTED)
        score = guard.calculate_relevance(kw_mem, kw_cook)
        assert score < 0.1, f"污染分数应 < 0.1，实际 {score}"


# ── 测试3：身份锚点加权 ────────────────────────────────

class TestIdentityWeighting:
    """身份锚点关键词在交集中时应提升相关性分数。"""

    def test_identity_weight_increases_score(self):
        """同一对文本，有身份加权 vs 无身份加权，有加权时分数更高。"""
        guard_with = ContextGuard()
        guard_without = ContextGuard()
        guard_without._identity_keywords = set()  # 清空身份关键词

        # 含身份关键词的文本对
        kw_a = guard_with.extract_keywords(TEXT_IDENTITY_A)
        kw_b = guard_with.extract_keywords(TEXT_IDENTITY_B)

        score_with = guard_with.calculate_relevance(kw_a, kw_b)
        score_without = guard_without.calculate_relevance(kw_a, kw_b)

        # 有身份加权时分数更高（identity_weight > 0）
        assert score_with >= score_without, (
            f"身份加权后分数应更高：with={score_with} >= without={score_without}"
        )
        # 验证身份权重确实贡献了正向增量
        assert score_with > score_without, (
            f"身份权重应产生正向增量：{score_with} > {score_without}"
        )

    def test_identity_keywords_loaded(self):
        """Soul能加载时，身份关键词集合非空。"""
        keywords = _load_identity_keywords()
        assert len(keywords) > 0, "身份关键词集合不应为空"
        # Soul默认name是"OpenLLM"，creator是"张成市"
        assert "OpenLLM" in keywords or "张成市" in keywords


# ── 测试4：clean_context 过滤 ───────────────────────────

class TestCleanContext:
    """clean_context 应只保留相关项。"""

    def test_filters_out_polluted(self):
        guard = ContextGuard()
        candidates = [
            TEXT_MEMORY_B,       # 相关
            TEXT_POLLUTED,       # 污染
            TEXT_IDENTITY_A,     # 相关（含身份词）
            "宫保鸡丁是一道经典川菜。",  # 污染
        ]
        cleaned = guard.clean_context(TEXT_MEMORY_A, candidates)
        assert TEXT_MEMORY_B in cleaned
        assert TEXT_IDENTITY_A in cleaned
        assert TEXT_POLLUTED not in cleaned

    def test_empty_candidates(self):
        guard = ContextGuard()
        assert guard.clean_context(TEXT_MEMORY_A, []) == []


# ── 测试5：threshold 可配置 ─────────────────────────────

class TestConfigurableThreshold:
    """提高阈值后，原本相关的变为不相关。"""

    def test_high_threshold_makes_relevant_irrelevant(self):
        guard_low = ContextGuard(threshold=0.1)
        guard_high = ContextGuard(threshold=0.9)

        # 低阈值下相关
        assert guard_low.is_relevant(TEXT_MEMORY_A, TEXT_MEMORY_B) is True

        # 高阈值下不相关
        assert guard_high.is_relevant(TEXT_MEMORY_A, TEXT_MEMORY_B) is False

    def test_zero_threshold_accepts_everything(self):
        guard = ContextGuard(threshold=0.0)
        assert guard.is_relevant(TEXT_MEMORY_A, TEXT_POLLUTED) is True


# ── 测试6：Soul 加载失败降级 ─────────────────────────────

class TestSoulDegradation:
    """Soul加载失败时应降级为空集，不抛异常。"""

    def test_load_failure_returns_empty_set(self):
        with patch(
            "openllm.memory.context_guard._load_identity_keywords",
            return_value=set(),
        ):
            guard = ContextGuard()
            assert guard._identity_keywords == set()

    def test_guard_works_without_identity_keywords(self):
        """即使无身份关键词，基本功能仍正常。"""
        guard = ContextGuard()
        guard._identity_keywords = set()  # 强制清空
        kw_a = guard.extract_keywords(TEXT_MEMORY_A)
        kw_b = guard.extract_keywords(TEXT_MEMORY_B)
        score = guard.calculate_relevance(kw_a, kw_b)
        # 无身份加权，纯Jaccard，应该 > 0（有重叠词）
        assert score > 0.0

    def test_import_error_in_load_identity(self):
        """模拟Soul模块导入失败，应返回空集。"""
        with patch.dict("sys.modules", {"openllm.identity.soul": None}):
            result = _load_identity_keywords()
            assert isinstance(result, set)


# ── 测试7：边界情况 ─────────────────────────────────────

class TestEdgeCases:
    """空输入、极短文本等边界情况。"""

    def test_empty_text(self):
        guard = ContextGuard()
        assert guard.extract_keywords("") == set()

    def test_relevance_with_empty_keywords(self):
        guard = ContextGuard()
        assert guard.calculate_relevance(set(), {"a", "b"}) == 0.0
        assert guard.calculate_relevance({"a", "b"}, set()) == 0.0

    def test_identical_text_scores_high(self):
        guard = ContextGuard()
        text = "openLLM记忆系统架构设计"
        score = guard.calculate_relevance(
            guard.extract_keywords(text),
            guard.extract_keywords(text),
        )
        assert score >= 0.1
