"""
Silence Auditor — IKO 沉默审计器
==================================

审计 SILENT 意图是否合理，防止不当沉默。

七神约束：
- 赫尔墨斯：不能自证的沉默必须开口（reversible=False 时回退为 INFORM）
- 阿瑞斯：高风险决策不能沉默（risk_level > LOW 时回退为 CONFIRM）

审计规则（按优先级从高到低）：
1. 非 SILENT 意图直接放行
2. SILENT + reasoning_chain 不可逆（reversible=False）→ 回退为 INFORM
3. SILENT + 风险等级 > LOW → 回退为 CONFIRM
4. SILENT + 用户首次在此领域交互 → 回退为 INFORM
5. 否则 SILENT 通过

用法：
    auditor = SilenceAuditor()
    result = auditor.audit(
        intent=OutputIntent.SILENT,
        context={"risk_level": "LOW", "domain": "general"},
        reversible=True,
    )
    # result == OutputIntent.SILENT

    reason = auditor.explain_silence("out-001")
    # "系统判断此输出无需呈现，原因：{reason}。如需查看完整推理链，请追问。"
"""

from __future__ import annotations

import logging
from typing import Optional

from openllm.iko.intent_classifier import OutputIntent

logger = logging.getLogger(__name__)

# ── 风险等级常量（与 intent_classifier.py 保持一致）──
RISK_HIGH = "HIGH"
RISK_MEDIUM = "MEDIUM"
RISK_LOW = "LOW"

# 风险等级到数值的映射，用于 > LOW 的判断
_RISK_LEVEL_ORDER: dict[str, int] = {
    RISK_LOW: 0,
    RISK_MEDIUM: 1,
    RISK_HIGH: 2,
}


class SilenceAuditor:
    """沉默审计器——审计 SILENT 意图是否合理。

    防止系统在不恰当的场景下保持沉默，确保用户始终能获得
    关键信息。审计结果通过 OutputIntent 枚举返回。

    审计规则：
    1. 非 SILENT 意图直接放行（不做干预）
    2. SILENT + reversible=False → 回退为 INFORM（赫尔墨斯约束）
    3. SILENT + risk_level > LOW → 回退为 CONFIRM（阿瑞斯约束）
    4. SILENT + 用户首次在此领域交互 → 回退为 INFORM
    5. 否则 SILENT 通过

    Attributes:
        _domain_history: 记录用户在各领域的交互次数。
        _audit_log: 审计日志，记录每次审计的 output_id → reason 映射。
    """

    def __init__(self) -> None:
        """初始化沉默审计器。"""
        self._domain_history: dict[str, int] = {}
        self._audit_log: dict[str, str] = {}

    def audit(
        self,
        intent: OutputIntent,
        context: dict,
        reversible: bool = True,
        output_id: str = "",
    ) -> OutputIntent:
        """审计输出意图，必要时将 SILENT 回退为更合适的意图。

        Args:
            intent: 待审计的输出意图。
            context: 上下文信息，期望包含：
                - risk_level (str): 风险等级 "HIGH"/"MEDIUM"/"LOW"
                - domain (str, optional): 领域标识，用于判断首次交互
            reversible: 推理链是否可逆（SymmetricCodec 占位参数）。
                False 表示推理不可逆，根据赫尔墨斯约束必须开口。
            output_id: 输出唯一标识，用于审计日志记录。

        Returns:
            审计后的输出意图。可能从 SILENT 回退为 INFORM 或 CONFIRM。

        Examples:
            >>> auditor = SilenceAuditor()
            >>> auditor.audit(OutputIntent.SILENT, {"risk_level": "LOW"})
            <OutputIntent.SILENT: 'silent'>

            >>> auditor.audit(OutputIntent.SILENT, {"risk_level": "MEDIUM"})
            <OutputIntent.CONFIRM: 'confirm'>
        """
        # ── 规则 0：非 SILENT 直接放行 ──
        if intent != OutputIntent.SILENT:
            logger.debug("Non-SILENT intent %s passed through", intent.value)
            return intent

        risk_level = context.get("risk_level", RISK_LOW)
        domain = context.get("domain", "unknown")

        # ── 规则 1：不可逆推理链 → 回退为 INFORM（赫尔墨斯约束）──
        if not reversible:
            reason = "reasoning_chain is irreversible (赫尔墨斯约束)"
            self._record(output_id, reason)
            logger.debug("SILENT → INFORM: %s", reason)
            return OutputIntent.INFORM

        # ── 规则 2：高风险 → 回退为 CONFIRM（阿瑞斯约束）──
        risk_value = _RISK_LEVEL_ORDER.get(risk_level, 0)
        if risk_value > 0:  # > LOW
            reason = f"risk_level={risk_level} > LOW (阿瑞斯约束)"
            self._record(output_id, reason)
            logger.debug("SILENT → CONFIRM: %s", reason)
            return OutputIntent.CONFIRM

        # ── 规则 3：首次领域交互 → 回退为 INFORM ──
        if self._is_first_interaction(domain):
            reason = f"first interaction in domain '{domain}'"
            self._record(output_id, reason)
            logger.debug("SILENT → INFORM: %s", reason)
            return OutputIntent.INFORM

        # ── 规则 4：SILENT 通过 ──
        reason = "all checks passed, SILENT is appropriate"
        self._record(output_id, reason)
        logger.debug("SILENT passed: %s", reason)
        return OutputIntent.SILENT

    def explain_silence(self, output_id: str = "") -> str:
        """向用户解释沉默原因。

        Args:
            output_id: 要解释的输出标识。为空时返回通用说明。

        Returns:
            格式化的解释文本：
            "系统判断此输出无需呈现，原因：{reason}。如需查看完整推理链，请追问。"
        """
        if output_id and output_id in self._audit_log:
            reason = self._audit_log[output_id]
        else:
            reason = "该输出未触发审计记录"

        return f"系统判断此输出无需呈现，原因：{reason}。如需查看完整推理链，请追问。"

    def _is_first_interaction(self, domain: str) -> bool:
        """判断用户是否首次在指定领域交互。

        Args:
            domain: 领域标识符。

        Returns:
            如果该领域交互次数为 0 返回 True，否则 False。
        """
        return self._domain_history.get(domain, 0) == 0

    def record_interaction(self, domain: str) -> None:
        """记录用户在指定领域的一次交互。

        调用方应在用户交互后调用此方法，以更新领域历史。

        Args:
            domain: 领域标识符。
        """
        self._domain_history[domain] = self._domain_history.get(domain, 0) + 1

    def _record(self, output_id: str, reason: str) -> None:
        """记录审计日志。

        Args:
            output_id: 输出标识。
            reason: 审计原因。
        """
        if output_id:
            self._audit_log[output_id] = reason
