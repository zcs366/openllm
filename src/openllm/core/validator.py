"""validator.py — 异源验证器（Cross-Source Validator）

阿瑞斯天启：验证壳是盾但不能防内部攻击。
核心思路：关键声明用第二个LLM独立判断，或用规则引擎回查。

两种验证模式：
  1. 规则验证（无secondary_provider时）——正则匹配/工具回查/已知事实库
  2. LLM验证（有secondary_provider时）——第二个模型独立判断

设计原则：
  - 独立模块，不修改engine.py
  - 不依赖沉默代码
  - ValidationResult: claim, verified, confidence, evidence, method
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger("openllm.validator")


# ═══════════════════════════════════════════════════════
# 数据类型
# ═══════════════════════════════════════════════════════

class VerificationMethod(str, Enum):
    """验证方法。"""
    RULE = "rule"
    LLM = "llm"
    COMBINED = "combined"


class ClaimType(str, Enum):
    """声明类型——决定走哪条验证路径。"""
    FACTUAL = "factual"          # 事实声明（"地球是圆的"）
    NUMERICAL = "numerical"      # 数值声明（"Python发布于1991年"）
    CODE_SYNTAX = "code_syntax"  # 代码声明（"这段代码能运行"）
    OPINION = "opinion"          # 观点声明（不可验证）


@dataclass
class ValidationResult:
    """异源验证结果。"""
    claim: str
    verified: bool
    confidence: float  # 0.0 ~ 1.0
    evidence: str
    method: VerificationMethod
    claim_type: ClaimType = ClaimType.FACTUAL
    details: dict = field(default_factory=dict)
    latency_ms: float = 0.0


@dataclass
class KnownFact:
    """已知事实条目（规则验证用）。"""
    pattern: str  # 正则模式
    response: str  # 验证回复
    confidence: float  # 基础置信度


# ═══════════════════════════════════════════════════════
# 已知事实库（规则验证用）
# ═══════════════════════════════════════════════════════

_DEFAULT_KNOWN_FACTS: list[KnownFact] = [
    # 基础事实
    KnownFact(
        pattern=r"地球(?:是|为)(?:一个?|颗?)(?:球|圆|椭圆)",
        response="地球确实是椭球体，赤道半径约6378km，极半径约6357km。",
        confidence=0.99,
    ),
    KnownFact(
        pattern=r"Python\s*(?:编程语言?)?\s*(?:发布|发行|诞生|创立)(?:于|于)?\s*19\d{2}",
        response="Python由Guido van Rossum于1991年首次发布。",
        confidence=0.99,
    ),
    KnownFact(
        pattern=r"(?:太阳|Sun)\s*(?:表面?)?\s*(?:温度|热度)\s*(?:大约|约为|约|是)\s*\d+",
        response="太阳表面温度约5778K（约5505°C）。",
        confidence=0.95,
    ),
    KnownFact(
        pattern=r"(?:光速|光的传播速度)\s*(?:大约|约为|约|是)\s*\d+",
        response="真空中的光速约299,792,458 m/s（约3×10^8 m/s）。",
        confidence=0.99,
    ),
    KnownFact(
        pattern=r"(?:水的沸点|水在常压下沸腾)\s*(?:是|为|大约|约为)\s*\d+",
        response="在标准大气压下，水的沸点是100°C。",
        confidence=0.99,
    ),
]

# 数值范围验证（检测明显异常的数值声明）
_NUMERICAL_RANGES: dict[str, tuple[float, float, str]] = {
    "地球到太阳的距离": (1.4e8, 1.6e8, "km"),
    "地球半径": (6350, 6400, "km"),
    "人体正常体温": (35.5, 37.5, "°C"),
    "大气压": (99000, 103000, "Pa"),
}


# ═══════════════════════════════════════════════════════
# 规则验证器
# ═══════════════════════════════════════════════════════

class RuleValidator:
    """纯规则验证——零LLM调用。

    策略：
    1. 已知事实正则匹配
    2. 数值范围检查
    3. 代码语法检查（py_compile）
    4. 逻辑矛盾检测
    """

    def __init__(self, known_facts: Optional[list[KnownFact]] = None):
        self.known_facts = known_facts or _DEFAULT_KNOWN_FACTS
        self._pattern_cache: list[tuple[re.Pattern, str, float]] = []
        for fact in self.known_facts:
            self._pattern_cache.append((
                re.compile(fact.pattern, re.IGNORECASE),
                fact.response,
                fact.confidence,
            ))

    def classify_claim(self, claim: str) -> ClaimType:
        """自动分类声明类型。"""
        claim_lower = claim.lower()

        # 代码声明检测
        code_symbols = ("def ", "class ", "import ", "```", "print(", "return ", "if __name__")
        has_code_keyword = any(kw in claim_lower for kw in ("代码", "code", "运行", "run", "编译", "compile"))
        has_code_symbol = any(sym in claim for sym in code_symbols)
        if has_code_keyword and has_code_symbol:
            return ClaimType.CODE_SYNTAX
        # 纯代码片段（无关键词但有明显代码模式）
        if has_code_symbol and len(claim.splitlines()) <= 5:
            return ClaimType.CODE_SYNTAX

        # 数值声明检测（含数字+单位）
        if re.search(r'-?\d+(?:\.\d+)?\s*(?:年|米|km|°C|°F|秒|Hz|Mbps|GB|TB|度)', claim):
            return ClaimType.NUMERICAL

        # 观点声明检测
        opinion_markers = ("我认为", "我觉得", "应该是", "可能", "也许", "in my opinion", "I think")
        if any(m in claim_lower for m in opinion_markers):
            return ClaimType.OPINION

        return ClaimType.FACTUAL

    def validate(self, claim: str, context: str = "") -> ValidationResult:
        """规则验证。"""
        t0 = time.time()
        claim_type = self.classify_claim(claim)

        # 观点不可验证
        if claim_type == ClaimType.OPINION:
            return ValidationResult(
                claim=claim,
                verified=True,
                confidence=0.5,
                evidence="观点声明，无需事实验证",
                method=VerificationMethod.RULE,
                claim_type=claim_type,
                latency_ms=(time.time() - t0) * 1000,
            )

        # 代码语法验证
        if claim_type == ClaimType.CODE_SYNTAX:
            return self._validate_code_claim(claim, context, t0)

        # 正则匹配已知事实
        for pattern, response, conf in self._pattern_cache:
            if pattern.search(claim):
                return ValidationResult(
                    claim=claim,
                    verified=True,
                    confidence=conf,
                    evidence=response,
                    method=VerificationMethod.RULE,
                    claim_type=claim_type,
                    latency_ms=(time.time() - t0) * 1000,
                )

        # 数值范围检查
        if claim_type == ClaimType.NUMERICAL:
            result = self._validate_numerical(claim)
            result.latency_ms = (time.time() - t0) * 1000
            return result

        # 无法规则验证——返回低置信度
        return ValidationResult(
            claim=claim,
            verified=False,
            confidence=0.0,
            evidence="规则引擎无法验证此声明，需要LLM验证或人工确认",
            method=VerificationMethod.RULE,
            claim_type=claim_type,
            latency_ms=(time.time() - t0) * 1000,
        )

    def _validate_code_claim(self, claim: str, context: str, t0: float) -> ValidationResult:
        """验证代码相关声明。"""
        # 从声明或上下文中提取代码块
        code_blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', claim, re.DOTALL)
        if not code_blocks and context:
            code_blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', context, re.DOTALL)

        if not code_blocks:
            return ValidationResult(
                claim=claim,
                verified=False,
                confidence=0.3,
                evidence="未找到代码块进行语法检查",
                method=VerificationMethod.RULE,
                claim_type=ClaimType.CODE_SYNTAX,
                latency_ms=(time.time() - t0) * 1000,
            )

        import py_compile
        import tempfile
        import os

        errors = []
        for code in code_blocks:
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(code)
                    tmp_path = f.name
                py_compile.compile(tmp_path, doraise=True)
            except py_compile.PyCompileError as e:
                errors.append(str(e))
            except SyntaxError as e:
                errors.append(f"SyntaxError: {e.msg} (line {e.lineno})")
            finally:
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

        if errors:
            return ValidationResult(
                claim=claim,
                verified=False,
                confidence=0.95,
                evidence=f"代码语法错误: {'; '.join(errors)}",
                method=VerificationMethod.RULE,
                claim_type=ClaimType.CODE_SYNTAX,
                details={"errors": errors, "code_blocks_checked": len(code_blocks)},
                latency_ms=(time.time() - t0) * 1000,
            )

        return ValidationResult(
            claim=claim,
            verified=True,
            confidence=0.95,
            evidence=f"{len(code_blocks)}个代码块语法检查通过",
            method=VerificationMethod.RULE,
            claim_type=ClaimType.CODE_SYNTAX,
            details={"code_blocks_checked": len(code_blocks)},
            latency_ms=(time.time() - t0) * 1000,
        )

    def _validate_numerical(self, claim: str) -> ValidationResult:
        """数值范围验证。"""
        # 提取数字
        numbers = re.findall(r'-?[\d,]+(?:\.\d+)?', claim.replace(',', ''))
        if not numbers:
            return ValidationResult(
                claim=claim,
                verified=False,
                confidence=0.2,
                evidence="无法从声明中提取数值",
                method=VerificationMethod.RULE,
                claim_type=ClaimType.NUMERICAL,
            )

        # 检查是否在合理范围内
        evidence_parts = []
        for num_str in numbers:
            try:
                num = float(num_str)
                if num < 0 and "温度" in claim:
                    evidence_parts.append(f"负温度值 {num_str} 需要上下文确认")
                elif num > 1e15 and "距离" not in claim:
                    evidence_parts.append(f"极大数值 {num_str} 需要确认单位")
            except ValueError:
                continue

        if evidence_parts:
            return ValidationResult(
                claim=claim,
                verified=False,
                confidence=0.4,
                evidence="; ".join(evidence_parts),
                method=VerificationMethod.RULE,
                claim_type=ClaimType.NUMERICAL,
                details={"extracted_numbers": numbers},
            )

        return ValidationResult(
            claim=claim,
            verified=False,
            confidence=0.3,
            evidence=f"数值提取成功({numbers})，但规则引擎无法独立验证",
            method=VerificationMethod.RULE,
            claim_type=ClaimType.NUMERICAL,
            details={"extracted_numbers": numbers},
        )


# ═══════════════════════════════════════════════════════
# LLM验证器
# ═══════════════════════════════════════════════════════

class LLMValidator:
    """LLM异源验证——用第二个模型独立判断。

    核心理念：
    - 主模型生成内容
    -  secondary模型独立验证
    - 两个模型不共享权重，实现真正的"异源"
    """

    VALIDATION_PROMPT = """你是一个事实验证专家。请独立判断以下声明是否正确。

声明: {claim}

上下文（可选参考）: {context}

要求：
1. 只判断事实正确性，不考虑表述风格
2. 如果无法确定，标注"不确定"
3. 返回JSON格式：
   {{"verified": true/false/null, "confidence": 0.0-1.0, "evidence": "你的判断依据"}}

只返回JSON，不解释。"""

    def __init__(self, provider: Any, model: str = ""):
        """
        Args:
            provider: LLM provider实例（需支持chat方法）
            model: 模型名（用于日志）
        """
        self.provider = provider
        self.model = model or "secondary"

    def validate(self, claim: str, context: str = "") -> ValidationResult:
        """LLM验证。"""
        t0 = time.time()

        try:
            from .provider import ChatMessage

            prompt = self.VALIDATION_PROMPT.format(
                claim=claim,
                context=context or "无",
            )

            messages = [ChatMessage(role="user", content=prompt)]
            response = self.provider.chat(messages)

            # 解析JSON响应
            result = self._parse_response(response.content)
            result.claim = claim
            result.method = VerificationMethod.LLM
            result.latency_ms = (time.time() - t0) * 1000
            result.details["model"] = self.model
            return result

        except Exception as e:
            logger.warning("LLM验证失败: %s", e)
            return ValidationResult(
                claim=claim,
                verified=False,
                confidence=0.0,
                evidence=f"LLM验证异常: {e}",
                method=VerificationMethod.LLM,
                latency_ms=(time.time() - t0) * 1000,
                details={"error": str(e)},
            )

    def _parse_response(self, content: str) -> ValidationResult:
        """解析LLM的JSON响应。"""
        # 尝试提取JSON
        json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        if not json_match:
            # 从文本中推断
            verified = None
            if any(w in content.lower() for w in ("true", "正确", "是", "verified")):
                verified = True
            elif any(w in content.lower() for w in ("false", "错误", "否", "incorrect")):
                verified = False

            return ValidationResult(
                claim="",
                verified=verified is True,
                confidence=0.3 if verified is None else 0.5,
                evidence=content[:500],
                method=VerificationMethod.LLM,
            )

        try:
            data = json.loads(json_match.group())
            verified = data.get("verified")
            confidence = float(data.get("confidence", 0.5))
            evidence = data.get("evidence", "")

            # null表示不确定
            if verified is None:
                verified = False
                confidence = min(confidence, 0.4)

            return ValidationResult(
                claim="",
                verified=bool(verified),
                confidence=max(0.0, min(1.0, confidence)),
                evidence=evidence,
                method=VerificationMethod.LLM,
            )
        except (json.JSONDecodeError, ValueError) as e:
            return ValidationResult(
                claim="",
                verified=False,
                confidence=0.2,
                evidence=f"JSON解析失败: {e}. 原文: {content[:300]}",
                method=VerificationMethod.LLM,
            )


# ═══════════════════════════════════════════════════════
# 统一验证器
# ═══════════════════════════════════════════════════════

class Validator:
    """异源验证器——统一入口。

    - 无secondary_provider → 规则验证
    - 有secondary_provider → LLM验证（异源）
    - 有callback → 自定义验证钩子
    """

    def __init__(
        self,
        primary_provider: Any = None,
        secondary_provider: Any = None,
        known_facts: Optional[list[KnownFact]] = None,
        custom_validator: Optional[Callable[[str, str], Optional[ValidationResult]]] = None,
    ):
        """
        Args:
            primary_provider: 主模型provider（预留，当前不用于验证）
            secondary_provider: 第二模型provider（用于异源LLM验证）
            known_facts: 已知事实库（规则验证用）
            custom_validator: 自定义验证回调 (claim, context) -> ValidationResult | None
        """
        self.primary_provider = primary_provider
        self.secondary_provider = secondary_provider
        self.rule_validator = RuleValidator(known_facts)
        self.llm_validator = LLMValidator(secondary_provider) if secondary_provider else None
        self.custom_validator = custom_validator
        self._history: list[ValidationResult] = []

    def validate(self, claim: str, context: str = "") -> ValidationResult:
        """异源验证入口。

        验证策略（按优先级）：
        1. 自定义验证器（如果有）
        2. LLM验证（如果有secondary_provider）
        3. 规则验证（兜底）

        如果LLM验证和规则验证结果一致，置信度提升。
        """
        t0 = time.time()

        # 1. 自定义验证器
        if self.custom_validator:
            try:
                result = self.custom_validator(claim, context)
                if result is not None:
                    result.method = VerificationMethod.COMBINED
                    self._record(result)
                    return result
            except Exception as e:
                logger.warning("自定义验证器异常: %s，继续默认验证", e)

        # 2. LLM验证（有secondary_provider时）
        llm_result = None
        if self.llm_validator:
            llm_result = self.llm_validator.validate(claim, context)

        # 3. 规则验证
        rule_result = self.rule_validator.validate(claim, context)

        # 4. 综合判定
        if llm_result is None:
            # 纯规则模式
            result = rule_result
        else:
            result = self._combine_results(claim, rule_result, llm_result)

        result.latency_ms = (time.time() - t0) * 1000
        self._record(result)
        return result

    def _combine_results(
        self,
        claim: str,
        rule_result: ValidationResult,
        llm_result: ValidationResult,
    ) -> ValidationResult:
        """综合规则验证和LLM验证的结果。"""
        # 两者一致 → 高置信度
        if rule_result.verified == llm_result.verified:
            combined_confidence = min(
                1.0,
                (rule_result.confidence + llm_result.confidence) / 2 + 0.1,  # 一致性加成
            )
            return ValidationResult(
                claim=claim,
                verified=rule_result.verified,
                confidence=combined_confidence,
                evidence=f"规则+LLM一致: {rule_result.evidence} | {llm_result.evidence}",
                method=VerificationMethod.COMBINED,
                claim_type=rule_result.claim_type,
                details={
                    "rule": {"verified": rule_result.verified, "confidence": rule_result.confidence},
                    "llm": {"verified": llm_result.verified, "confidence": llm_result.confidence},
                },
            )

        # 两者不一致 → 取LLM结果，但降低置信度
        # （规则有硬证据时LLM可能出错，但LLM理解更灵活）
        # 如果LLM验证失败（异常降级），尊重规则结果
        llm_failed = llm_result.details.get("error") is not None
        if llm_failed:
            # LLM异常降级——信任规则验证
            return ValidationResult(
                claim=claim,
                verified=rule_result.verified,
                confidence=rule_result.confidence * 0.9,  # 轻微降权
                evidence=f"LLM验证失败，降级到规则: {rule_result.evidence}",
                method=VerificationMethod.COMBINED,
                claim_type=rule_result.claim_type,
                details={
                    "rule": {"verified": rule_result.verified, "confidence": rule_result.confidence},
                    "llm": {"verified": False, "confidence": 0.0, "error": llm_result.details.get("error")},
                    "llm_degraded": True,
                },
            )

        return ValidationResult(
            claim=claim,
            verified=llm_result.verified,
            confidence=min(llm_result.confidence, rule_result.confidence) * 0.8,
            evidence=f"规则与LLM结论不一致。规则: {rule_result.evidence} | LLM: {llm_result.evidence}",
            method=VerificationMethod.COMBINED,
            claim_type=rule_result.claim_type,
            details={
                "rule": {"verified": rule_result.verified, "confidence": rule_result.confidence},
                "llm": {"verified": llm_result.verified, "confidence": llm_result.confidence},
                "conflict": True,
            },
        )

    def _record(self, result: ValidationResult):
        """记录验证历史。"""
        self._history.append(result)
        # 保留最近100条
        if len(self._history) > 100:
            self._history = self._history[-100:]

    @property
    def history(self) -> list[ValidationResult]:
        """验证历史。"""
        return list(self._history)

    def summary(self) -> dict:
        """验证统计摘要。"""
        if not self._history:
            return {"total": 0}

        verified = sum(1 for r in self._history if r.verified)
        methods = {}
        for r in self._history:
            m = r.method.value
            methods[m] = methods.get(m, 0) + 1

        avg_confidence = (
            sum(r.confidence for r in self._history) / len(self._history)
        )

        return {
            "total": len(self._history),
            "verified": verified,
            "rejected": len(self._history) - verified,
            "avg_confidence": round(avg_confidence, 3),
            "methods": methods,
        }
