"""test_validator.py — 异源验证器测试。

验证规则：
1. 规则验证：已知事实、数值范围、代码语法、观点声明
2. LLM验证：用mock provider模拟第二个LLM
3. 组合验证：规则+LLM一致/不一致时的置信度变化
"""

import json
import pytest
from unittest.mock import MagicMock, patch


# ═══════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════

@pytest.fixture
def rule_validator():
    from openllm.core.validator import RuleValidator
    return RuleValidator()


@pytest.fixture
def mock_llm_provider():
    """模拟LLM provider——返回验证JSON。"""
    provider = MagicMock()

    def _chat(messages, **kwargs):
        content = messages[0].content if messages else ""
        # 根据输入返回不同响应
        if "地球" in content or "Earth" in content:
            resp = json.dumps({
                "verified": True,
                "confidence": 0.95,
                "evidence": "地球确实是椭球体",
            })
        elif "太阳" in content and "质量" in content:
            resp = json.dumps({
                "verified": True,
                "confidence": 0.90,
                "evidence": "太阳质量约1.989×10^30 kg",
            })
        elif "错误" in content or "虚假" in content:
            resp = json.dumps({
                "verified": False,
                "confidence": 0.85,
                "evidence": "此声明与已知事实不符",
            })
        else:
            resp = json.dumps({
                "verified": None,
                "confidence": 0.4,
                "evidence": "无法确定",
            })

        from openllm.core.provider import ModelResponse
        return ModelResponse(content=resp, model="mock-model")

    provider.chat = _chat
    return provider


@pytest.fixture
def full_validator(mock_llm_provider):
    """有secondary_provider的完整验证器。"""
    from openllm.core.validator import Validator
    return Validator(secondary_provider=mock_llm_provider)


# ═══════════════════════════════════════════════════════
# 规则验证测试
# ═══════════════════════════════════════════════════════

class TestRuleValidator:

    def test_known_fact_earth(self, rule_validator):
        result = rule_validator.validate("地球是一个球体")
        assert result.verified is True
        assert result.confidence > 0.9
        assert result.method.value == "rule"

    def test_known_fact_python(self, rule_validator):
        result = rule_validator.validate("Python发布于1991年")
        assert result.verified is True
        assert result.confidence > 0.9

    def test_known_fact_light_speed(self, rule_validator):
        result = rule_validator.validate("光速大约300000000米每秒")
        assert result.verified is True
        assert result.confidence > 0.9

    def test_unknown_fact(self, rule_validator):
        # 不含数值、不含代码、不含已知事实的声明
        result = rule_validator.validate("量子计算机已经完全取代经典计算机")
        # 规则引擎无法验证
        assert result.verified is False
        assert result.confidence == 0.0

    def test_opinion_declaration(self, rule_validator):
        result = rule_validator.validate("我认为Python比Java更好")
        # 观点声明——标记为不可验证
        assert result.verified is True  # 观点无需事实验证
        assert result.claim_type.value == "opinion"

    def test_claim_classification(self, rule_validator):
        assert rule_validator.classify_claim("地球是圆的").value == "factual"
        assert rule_validator.classify_claim("Python发布于1991年").value == "numerical"
        assert rule_validator.classify_claim("def foo(): pass").value == "code_syntax"
        assert rule_validator.classify_claim("我认为AI很好").value == "opinion"

    def test_code_syntax_valid(self, rule_validator):
        claim = "这段代码能运行：\n```python\ndef foo():\n    return 42\n```"
        result = rule_validator.validate(claim)
        assert result.verified is True
        assert result.claim_type.value == "code_syntax"

    def test_code_syntax_invalid(self, rule_validator):
        claim = "这段代码能运行：\n```python\ndef foo(\n    return 42\n```"
        result = rule_validator.validate(claim)
        assert result.verified is False
        assert "语法错误" in result.evidence

    def test_numerical_claim(self, rule_validator):
        result = rule_validator.validate("人体正常体温是36.5°C")
        assert result.claim_type.value == "numerical"

    def test_numerical_negative_temperature(self, rule_validator):
        result = rule_validator.validate("温度是-40度")
        assert result.claim_type.value == "numerical"

    def test_empty_claim(self, rule_validator):
        result = rule_validator.validate("")
        # 空声明应被处理
        assert result.verified is False


# ═══════════════════════════════════════════════════════
# LLM验证测试
# ═══════════════════════════════════════════════════════

class TestLLMValidator:

    def test_llm_verifies_fact(self, full_validator):
        result = full_validator.validate("地球是一个球体")
        # 规则验证通过 + LLM验证通过 → 高置信度
        assert result.verified is True
        assert result.confidence > 0.8
        assert result.method.value == "combined"

    def test_llm_rejects_false(self, full_validator):
        result = full_validator.validate("这是一个错误的虚假声明")
        # 规则: 无法验证 | LLM: 拒绝 → 综合拒绝
        assert result.verified is False

    def test_llm_uncertain(self, full_validator):
        result = full_validator.validate("某个随机声明的内容")
        # LLM不确定 + 规则无法验证 → 低置信度
        assert result.confidence < 0.5

    def test_history_tracking(self, full_validator):
        full_validator.validate("地球是圆的")
        full_validator.validate("太阳很大")
        assert len(full_validator.history) == 2

    def test_summary(self, full_validator):
        full_validator.validate("地球是圆的")
        full_validator.validate("这是一个错误的虚假声明")
        s = full_validator.summary()
        assert s["total"] == 2
        assert s["verified"] == 1
        assert s["rejected"] == 1


# ═══════════════════════════════════════════════════════
# 组合验证测试
# ═══════════════════════════════════════════════════════

class TestCombinedValidation:

    def test_rule_only_mode(self):
        """无secondary_provider → 纯规则验证。"""
        from openllm.core.validator import Validator
        v = Validator()
        result = v.validate("地球是一个球体")
        assert result.verified is True
        assert result.method.value == "rule"

    def test_consistency_boost(self, full_validator):
        """规则和LLM一致时，置信度提升。"""
        result = full_validator.validate("地球是一个球体")
        # 规则conf=0.99, LLM conf=0.95 → combined conf > 0.9
        assert result.confidence > 0.9
        details = result.details
        assert "rule" in details
        assert "llm" in details
        assert details["rule"]["verified"] == details["llm"]["verified"]

    def test_conflict_detection(self, mock_llm_provider):
        """规则和LLM不一致时，检测冲突。"""
        from openllm.core.validator import Validator

        # 自定义LLM返回与规则相反的结果
        def wrong_chat(messages, **kwargs):
            from openllm.core.provider import ModelResponse
            return ModelResponse(
                content=json.dumps({"verified": False, "confidence": 0.8, "evidence": "我认为不是"}),
                model="wrong-model",
            )
        mock_llm_provider.chat = wrong_chat

        v = Validator(secondary_provider=mock_llm_provider)
        result = v.validate("地球是一个球体")
        # 规则说真，LLM说假 → 冲突
        assert result.details.get("conflict") is True

    def test_custom_validator_priority(self):
        """自定义验证器优先级最高。"""
        from openllm.core.validator import Validator

        def custom(claim, context):
            from openllm.core.validator import ValidationResult, VerificationMethod
            return ValidationResult(
                claim=claim,
                verified=True,
                confidence=1.0,
                evidence="自定义验证通过",
                method=VerificationMethod.COMBINED,
            )

        v = Validator(custom_validator=custom)
        result = v.validate("任意声明")
        assert result.verified is True
        assert result.confidence == 1.0

    def test_custom_validator_returns_none_fallback(self):
        """自定义验证器返回None时，fallback到默认验证。"""
        from openllm.core.validator import Validator

        v = Validator(custom_validator=lambda c, ctx: None)
        result = v.validate("地球是一个球体")
        assert result.verified is True  # 规则验证通过

    def test_custom_validator_exception_fallback(self):
        """自定义验证器异常时，fallback到默认验证。"""
        from openllm.core.validator import Validator

        def broken(claim, context):
            raise RuntimeError("自定义验证器崩溃")

        v = Validator(custom_validator=broken)
        result = v.validate("地球是一个球体")
        assert result.verified is True  # 规则验证通过


# ═══════════════════════════════════════════════════════
# 边界条件测试
# ═══════════════════════════════════════════════════════

class TestEdgeCases:

    def test_empty_claim(self):
        from openllm.core.validator import Validator
        v = Validator()
        result = v.validate("")
        assert result.verified is False

    def test_very_long_claim(self):
        from openllm.core.validator import Validator
        v = Validator()
        long_claim = "这是一个很长的声明" * 1000
        result = v.validate(long_claim)
        assert result.verified is False  # 规则引擎无法验证

    def test_special_characters(self):
        from openllm.core.validator import Validator
        v = Validator()
        result = v.validate("声明包含特殊字符：<>&\"'")
        # 不应崩溃
        assert isinstance(result.verified, bool)

    def test_history_limit(self):
        """验证历史不超过100条。"""
        from openllm.core.validator import Validator
        v = Validator()
        for i in range(120):
            v.validate(f"声明{i}")
        assert len(v.history) == 100

    def test_llm_provider_exception(self):
        """LLM provider异常时降级到规则验证。"""
        from openllm.core.validator import Validator

        broken_provider = MagicMock()
        broken_provider.chat.side_effect = RuntimeError("API调用失败")

        v = Validator(secondary_provider=broken_provider)
        result = v.validate("地球是一个球体")
        # 降级到规则验证
        assert result.verified is True
        assert result.method.value == "combined"  # 综合了规则+LLM（LLM失败）
        assert result.details.get("llm", {}).get("error") is not None


# ═══════════════════════════════════════════════════════
# 自定义事实库测试
# ═══════════════════════════════════════════════════════

class TestCustomFacts:

    def test_custom_known_facts(self):
        from openllm.core.validator import Validator, KnownFact

        custom_facts = [
            KnownFact(
                pattern=r"openLLM是.*Agent引擎",
                response="openLLM确实是Agent引擎",
                confidence=0.99,
            ),
        ]
        v = Validator(known_facts=custom_facts)
        result = v.validate("openLLM是一个Agent引擎")
        assert result.verified is True
        assert result.confidence > 0.9
