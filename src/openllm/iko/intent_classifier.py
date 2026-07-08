"""
IKO Intent Classifier — 输出意图分类器
=======================================

纯规则、零LLM调用的输出意图分类器。

根据上下文（risk_level, has_tool_calls, has_side_effects, option_count）
和决策内容（decision）判定输出意图类型，驱动下游渲染逻辑。

分类优先级（从高到低）：
1. ERROR  — risk_level == "HIGH"
2. CONFIRM — risk_level == "MEDIUM"
3. ACT    — has_tool_calls 或 has_side_effects
4. DECIDE — option_count >= 2
5. INFORM — 纯查询，无特殊条件
6. SILENT — 默认兜底（无内容/空决策）

设计约束：
- 零LLM调用：所有分类逻辑均为确定性规则
- 可解释性：每次分类附带 reason 字段
- 预留 SilenceAuditor 校验接口（赫尔墨斯约束）

用法：
    classifier = IntentClassifier()
    intent = classifier.classify(
        context={"risk_level": "LOW", "has_tool_calls": False, "has_side_effects": False, "option_count": 0},
        decision={"type": "answer", "content": "Hello"},
    )
    # OutputIntent.INFORM
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ── 风险等级常量 ──

RISK_HIGH = "HIGH"
RISK_MEDIUM = "MEDIUM"
RISK_LOW = "LOW"


class OutputIntent(Enum):
    """输出意图枚举——覆盖IKO输出层全部意图类型。

    优先级：ERROR > CONFIRM > ACT > DECIDE > INFORM > SILENT
    """

    ERROR = "error"
    """错误输出：风险等级HIGH，需要向用户报告错误。"""

    CONFIRM = "confirm"
    """确认请求：风险等级MEDIUM，需要用户确认后才执行。"""

    ACT = "act"
    """执行输出：包含工具调用或副作用，需要展示执行动作。"""

    DECIDE = "decide"
    """决策输出：提供2+选项供用户选择。"""

    INFORM = "inform"
    """信息输出：纯查询结果，无特殊交互需求。"""

    SILENT = "silent"
    """静默输出：无实质内容，默认兜底，不产生可见输出。

    注意：SILENT意图必须通过SilenceAuditor校验（赫尔墨斯约束）。
    当SilenceAuditor尚未实现时，此意图直接通过。
    """


class IntentClassifier:
    """输出意图分类器——纯规则、零LLM调用。

    根据 context 和 decision 的字段判定 OutputIntent。
    所有分类逻辑均为确定性 if-elif 链，保证可预测性和可审计性。

    分类规则（按优先级从高到低）：

    1. ERROR:   context.risk_level == "HIGH"
    2. CONFIRM: context.risk_level == "MEDIUM"
    3. ACT:     context.has_tool_calls or context.has_side_effects
    4. DECIDE:  context.option_count >= 2
    5. INFORM:  decision.content 非空
    6. SILENT:  其他所有情况（默认兜底）

    示例：
        >>> classifier = IntentClassifier()
        >>> ctx = {"risk_level": "HIGH", "has_tool_calls": False,
        ...        "has_side_effects": False, "option_count": 0}
        >>> dec = {"type": "error", "content": "Connection failed"}
        >>> result = classifier.classify(ctx, dec)
        >>> result.intent
        <OutputIntent.ERROR: 'error'>
        >>> result.reason
        'risk_level=HIGH'
    """

    def classify(self, context: dict[str, Any], decision: dict[str, Any]) -> ClassificationResult:
        """分类决策入口——纯规则链。

        Args:
            context: 上下文信息，包含：
                - risk_level (str): 风险等级，"HIGH"/"MEDIUM"/"LOW"
                - has_tool_calls (bool): 是否包含工具调用
                - has_side_effects (bool): 是否有副作用
                - option_count (int): 选项数量
            decision: 决策内容，包含：
                - type (str): 决策类型
                - content (str): 决策内容文本
                - 可选的其他字段

        Returns:
            ClassificationResult 包含 intent 和 reason。

        Raises:
            ValueError: context 缺少必需字段时。
        """
        # ── 校验输入 ──
        self._validate_context(context)
        self._validate_decision(decision)

        risk_level = context.get("risk_level", RISK_LOW)
        has_tool_calls = context.get("has_tool_calls", False)
        has_side_effects = context.get("has_side_effects", False)
        option_count = context.get("option_count", 0)
        decision_content = decision.get("content", "")

        # ── 规则链（优先级从高到低）──

        # 1. ERROR: risk_level == HIGH
        if risk_level == RISK_HIGH:
            intent = OutputIntent.ERROR
            reason = f"risk_level={risk_level}"
            logger.debug("Intent classified as %s: %s", intent.value, reason)
            return ClassificationResult(intent=intent, reason=reason)

        # 2. CONFIRM: risk_level == MEDIUM
        if risk_level == RISK_MEDIUM:
            intent = OutputIntent.CONFIRM
            reason = f"risk_level={risk_level}"
            logger.debug("Intent classified as %s: %s", intent.value, reason)
            return ClassificationResult(intent=intent, reason=reason)

        # 3. ACT: has_tool_calls 或 has_side_effects
        if has_tool_calls or has_side_effects:
            intent = OutputIntent.ACT
            flags = []
            if has_tool_calls:
                flags.append("has_tool_calls=True")
            if has_side_effects:
                flags.append("has_side_effects=True")
            reason = ", ".join(flags)
            logger.debug("Intent classified as %s: %s", intent.value, reason)
            return ClassificationResult(intent=intent, reason=reason)

        # 4. DECIDE: option_count >= 2
        if option_count >= 2:
            intent = OutputIntent.DECIDE
            reason = f"option_count={option_count}"
            logger.debug("Intent classified as %s: %s", intent.value, reason)
            return ClassificationResult(intent=intent, reason=reason)

        # 5. INFORM: decision.content 非空
        if decision_content:
            intent = OutputIntent.INFORM
            reason = "decision.content is non-empty"
            logger.debug("Intent classified as %s: %s", intent.value, reason)
            return ClassificationResult(intent=intent, reason=reason)

        # 6. SILENT: 默认兜底
        intent = OutputIntent.SILENT
        reason = "no conditions met, default SILENT"
        logger.debug("Intent classified as %s: %s", intent.value, reason)
        return ClassificationResult(intent=intent, reason=reason)

    def _validate_context(self, context: dict[str, Any]) -> None:
        """校验 context 必需字段。"""
        required_fields = {"risk_level", "has_tool_calls", "has_side_effects", "option_count"}
        missing = required_fields - set(context.keys())
        if missing:
            raise ValueError(f"context missing required fields: {sorted(missing)}")

    def _validate_decision(self, decision: dict[str, Any]) -> None:
        """校验 decision 必需字段。"""
        if "content" not in decision:
            raise ValueError("decision missing required field: content")


class ClassificationResult:
    """分类结果——意图 + 可解释性原因。

    Attributes:
        intent: 输出意图类型。
        reason: 分类原因（可解释性字段，雅典娜约束）。
    """

    __slots__ = ("intent", "reason")

    def __init__(self, intent: OutputIntent, reason: str) -> None:
        self.intent = intent
        self.reason = reason

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ClassificationResult):
            return NotImplemented
        return self.intent == other.intent

    def __repr__(self) -> str:
        return f"ClassificationResult(intent={self.intent!r}, reason={self.reason!r})"

    def __str__(self) -> str:
        return f"{self.intent.value}: {self.reason}"
