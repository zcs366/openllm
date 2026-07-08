"""
Tests for IKO IntentClassifier
===============================

覆盖全部6种 OutputIntent 的分类逻辑，共10个测试用例。
"""

import pytest

from openllm.iko.intent_classifier import (
    ClassificationResult,
    IntentClassifier,
    OutputIntent,
    RISK_HIGH,
    RISK_LOW,
    RISK_MEDIUM,
)


@pytest.fixture
def classifier() -> IntentClassifier:
    """创建 IntentClassifier 实例。"""
    return IntentClassifier()


# ── 辅助函数 ──

def _ctx(
    risk_level: str = RISK_LOW,
    has_tool_calls: bool = False,
    has_side_effects: bool = False,
    option_count: int = 0,
) -> dict:
    """构建 context 字典。"""
    return {
        "risk_level": risk_level,
        "has_tool_calls": has_tool_calls,
        "has_side_effects": has_side_effects,
        "option_count": option_count,
    }


def _dec(content: str = "hello", dtype: str = "answer") -> dict:
    """构建 decision 字典。"""
    return {"type": dtype, "content": content}


# ── 测试用例 ──


class TestErrorIntent:
    """ERROR 意图：risk_level == HIGH。"""

    def test_high_risk_error(self, classifier: IntentClassifier) -> None:
        """HIGH风险 → ERROR。"""
        result = classifier.classify(_ctx(risk_level=RISK_HIGH), _dec("error occurred"))
        assert result.intent == OutputIntent.ERROR
        assert "HIGH" in result.reason

    def test_high_risk_overrides_tool_calls(self, classifier: IntentClassifier) -> None:
        """HIGH风险 + tool_calls 仍为 ERROR（优先级最高）。"""
        result = classifier.classify(
            _ctx(risk_level=RISK_HIGH, has_tool_calls=True),
            _dec("critical error"),
        )
        assert result.intent == OutputIntent.ERROR


class TestConfirmIntent:
    """CONFIRM 意图：risk_level == MEDIUM。"""

    def test_medium_risk_confirm(self, classifier: IntentClassifier) -> None:
        """MEDIUM风险 → CONFIRM。"""
        result = classifier.classify(_ctx(risk_level=RISK_MEDIUM), _dec("confirm?"))
        assert result.intent == OutputIntent.CONFIRM
        assert "MEDIUM" in result.reason

    def test_medium_risk_overrides_tool_calls(self, classifier: IntentClassifier) -> None:
        """MEDIUM风险 + tool_calls 仍为 CONFIRM（优先级高于 ACT）。"""
        result = classifier.classify(
            _ctx(risk_level=RISK_MEDIUM, has_tool_calls=True),
            _dec("are you sure?"),
        )
        assert result.intent == OutputIntent.CONFIRM


class TestActIntent:
    """ACT 意图：has_tool_calls 或 has_side_effects。"""

    def test_tool_calls_act(self, classifier: IntentClassifier) -> None:
        """有 tool_calls → ACT。"""
        result = classifier.classify(_ctx(has_tool_calls=True), _dec("running tool"))
        assert result.intent == OutputIntent.ACT
        assert "has_tool_calls" in result.reason

    def test_side_effects_act(self, classifier: IntentClassifier) -> None:
        """有 side_effects → ACT。"""
        result = classifier.classify(_ctx(has_side_effects=True), _dec("file modified"))
        assert result.intent == OutputIntent.ACT
        assert "has_side_effects" in result.reason

    def test_both_tool_calls_and_side_effects(self, classifier: IntentClassifier) -> None:
        """同时有 tool_calls + side_effects → ACT，reason 包含两者。"""
        result = classifier.classify(
            _ctx(has_tool_calls=True, has_side_effects=True), _dec("executed")
        )
        assert result.intent == OutputIntent.ACT
        assert "has_tool_calls" in result.reason
        assert "has_side_effects" in result.reason


class TestDecideIntent:
    """DECIDE 意图：option_count >= 2。"""

    def test_two_options_decide(self, classifier: IntentClassifier) -> None:
        """2个选项 → DECIDE。"""
        result = classifier.classify(_ctx(option_count=2), _dec("choose one"))
        assert result.intent == OutputIntent.DECIDE
        assert "option_count=2" in result.reason

    def test_many_options_decide(self, classifier: IntentClassifier) -> None:
        """5个选项 → DECIDE。"""
        result = classifier.classify(_ctx(option_count=5), _dec("many choices"))
        assert result.intent == OutputIntent.DECIDE


class TestInformIntent:
    """INFORM 意图：decision.content 非空。"""

    def test_inform_with_content(self, classifier: IntentClassifier) -> None:
        """content 非空 → INFORM。"""
        result = classifier.classify(_ctx(), _dec("The answer is 42"))
        assert result.intent == OutputIntent.INFORM
        assert "non-empty" in result.reason


class TestSilentIntent:
    """SILENT 意图：默认兜底。"""

    def test_empty_content_silent(self, classifier: IntentClassifier) -> None:
        """空 content → SILENT。"""
        result = classifier.classify(_ctx(), _dec(""))
        assert result.intent == OutputIntent.SILENT
        assert "default SILENT" in result.reason


class TestValidation:
    """输入校验。"""

    def test_missing_context_field_raises(self, classifier: IntentClassifier) -> None:
        """context 缺少字段 → ValueError。"""
        with pytest.raises(ValueError, match="context missing required fields"):
            classifier.classify({"risk_level": "LOW"}, _dec())

    def test_missing_decision_content_raises(self, classifier: IntentClassifier) -> None:
        """decision 缺少 content → ValueError。"""
        with pytest.raises(ValueError, match="decision missing required field: content"):
            classifier.classify(_ctx(), {"type": "answer"})


class TestClassificationResult:
    """ClassificationResult 对象行为。"""

    def test_equality_by_intent(self) -> None:
        """同 intent 的两个 result 相等。"""
        r1 = ClassificationResult(OutputIntent.ERROR, "a")
        r2 = ClassificationResult(OutputIntent.ERROR, "b")
        assert r1 == r2

    def test_inequality(self) -> None:
        """不同 intent 的两个 result 不等。"""
        r1 = ClassificationResult(OutputIntent.ERROR, "a")
        r2 = ClassificationResult(OutputIntent.INFORM, "a")
        assert r1 != r2

    def test_repr(self) -> None:
        """repr 包含 intent 和 reason。"""
        r = ClassificationResult(OutputIntent.SILENT, "no content")
        assert "SILENT" in repr(r)
        assert "no content" in repr(r)

    def test_str(self) -> None:
        """str 格式为 'intent: reason'。"""
        r = ClassificationResult(OutputIntent.ACT, "has_tool_calls=True")
        s = str(r)
        assert s == "act: has_tool_calls=True"


class TestPriorityChain:
    """优先级验证——确保规则链顺序正确。"""

    def test_priority_high_over_medium(self, classifier: IntentClassifier) -> None:
        """HIGH > MEDIUM：即使同时满足，HIGH 优先。"""
        # 不可能同时为 HIGH 和 MEDIUM，但验证 HIGH 被先匹配
        result = classifier.classify(_ctx(risk_level=RISK_HIGH), _dec())
        assert result.intent == OutputIntent.ERROR

    def test_priority_medium_over_act(self, classifier: IntentClassifier) -> None:
        """MEDIUM > ACT：MEDIUM风险 + tool_calls 仍为 CONFIRM。"""
        result = classifier.classify(
            _ctx(risk_level=RISK_MEDIUM, has_tool_calls=True, has_side_effects=True),
            _dec(),
        )
        assert result.intent == OutputIntent.CONFIRM

    def test_priority_act_over_decide(self, classifier: IntentClassifier) -> None:
        """ACT > DECIDE：tool_calls + 多选项仍为 ACT。"""
        result = classifier.classify(
            _ctx(has_tool_calls=True, option_count=3),
            _dec(),
        )
        assert result.intent == OutputIntent.ACT

    def test_priority_decide_over_inform(self, classifier: IntentClassifier) -> None:
        """DECIDE > INFORM：多选项 + content 仍为 DECIDE。"""
        result = classifier.classify(
            _ctx(option_count=2),
            _dec("choose A or B"),
        )
        assert result.intent == OutputIntent.DECIDE

    def test_priority_inform_over_silent(self, classifier: IntentClassifier) -> None:
        """INFORM > SILENT：有 content 为 INFORM。"""
        result = classifier.classify(_ctx(), _dec("some content"))
        assert result.intent == OutputIntent.INFORM
