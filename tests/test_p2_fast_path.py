"""P2-7 简单输入快路径测试 — octopus_impl.py

测试项：
 a. 简单输入走快路径：predict跳过LLM、review跳过LLM（verdict=approve）
 b. 复杂输入走全链路：predict/review正常调用
 c. 边界：长度阈值、工具关键词、风险信号三重守门
"""
from unittest.mock import MagicMock, patch

import pytest

from openllm.core.models import Context, Message, Prediction
from openllm.core import octopus_impl
from openllm.core.octopus_impl import 章鱼I, _FAST_PATH_MAX_LEN, _TOOL_KEYWORDS


def _ctx(text: str) -> Context:
    return Context(user_message=text)


# ═══════════════════════════════════════════════════════
# a. 简单输入 → 快路径
# ═══════════════════════════════════════════════════════

class TestFastPathSimpleInput:
    def test_is_simple_input_short_chat(self):
        assert 章鱼I._is_simple_input(_ctx("你好"), None) is True

    def test_predict_fast_path_skips_llm(self):
        """简单输入predict不发LLM请求"""
        o = 章鱼I()
        o.left.predict = MagicMock(side_effect=AssertionError("不应调用left.predict"))
        p = o.predict_consequences(_ctx("你好"))
        assert isinstance(p, Prediction)
        assert "快路径" in p.summary
        o.left.predict.assert_not_called()

    def test_reason_fast_path_skips_review(self):
        """简单输入reason不发右脑LLM请求"""
        o = 章鱼I()
        o.right.review = MagicMock(side_effect=AssertionError("不应调用right.review"))
        o.left.think = MagicMock(return_value=MagicMock(content="答复"))
        proposal, critique = o.reason(_ctx("谢谢"), None, None)
        o.right.review.assert_not_called()
        assert critique.verdict == "approve"
        assert "快路径" in critique.content


# ═══════════════════════════════════════════════════════
# b. 复杂输入 → 全链路
# ═══════════════════════════════════════════════════════

class TestFullChainComplexInput:
    def test_reason_full_chain_invokes_review(self):
        o = 章鱼I()
        o.left.think = MagicMock(return_value=MagicMock(content="方案"))
        o.right.review = MagicMock(return_value=MagicMock(verdict="approve"))
        long_text = "请帮我深入分析这个系统的架构设计，逐个模块审查代码质量、耦合度和测试覆盖，然后给出重构方案并执行修改" * 2
        _, critique = o.reason(_ctx(long_text), None, None)
        o.right.review.assert_called_once()

    def test_predict_full_chain_invokes_predict(self):
        o = 章鱼I()
        o.left.predict = MagicMock(return_value=Prediction(summary="预测"))
        o.predict_consequences(_ctx("帮我搜索最新的agent框架论文并整理对比表格"))
        o.left.predict.assert_called_once()


# ═══════════════════════════════════════════════════════
# c. 边界：三重守门
# ═══════════════════════════════════════════════════════

class TestGuardConditions:
    def test_empty_message_not_simple(self):
        assert 章鱼I._is_simple_input(_ctx(""), None) is False
        assert 章鱼I._is_simple_input(_ctx("   "), None) is False

    def test_length_boundary(self):
        # 恰好等于阈值→快路径
        assert 章鱼I._is_simple_input(_ctx("好" * _FAST_PATH_MAX_LEN), None) is True
        # 超阈值→全链路
        assert 章鱼I._is_simple_input(_ctx("好" * (_FAST_PATH_MAX_LEN + 1)), None) is False

    def test_tool_keywords_block_fast_path(self):
        for kw in ["搜索", "列出", "帮我查", "执行命令", "terminal", "SEARCH"]:
            assert _TOOL_KEYWORDS.search(kw), f"关键词{kw}应命中工具正则"
            assert 章鱼I._is_simple_input(_ctx(f"帮我{kw}一下"), None) is False

    def test_risk_signals_block_fast_path(self):
        risky = Prediction(summary="x", risk_signals=["不可逆操作"])
        assert 章鱼I._is_simple_input(_ctx("你好"), risky) is False

    def test_none_user_message_not_crash(self):
        ctx = Context(user_message=None)
        assert 章鱼I._is_simple_input(ctx, None) is False
