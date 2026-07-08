"""
SilenceAuditor 单元测试
========================

覆盖场景：
1. 非 SILENT 意图直接放行
2. SILENT + 不可逆推理链 → INFORM（赫尔墨斯约束）
3. SILENT + 高风险 → CONFIRM（阿瑞斯约束）
4. SILENT + 中风险 → CONFIRM（阿瑞斯约束）
5. SILENT + 首次领域交互 → INFORM
6. SILENT + 非首次交互 + 低风险 + 可逆 → SILENT 通过
7. explain_silence 带 output_id 返回正确格式
8. explain_silence 无 output_id 返回通用说明
9. context 缺少字段时使用默认值
10. 多领域交互跟踪
"""

from __future__ import annotations

import pytest

from openllm.iko.intent_classifier import OutputIntent
from openllm.iko.silence_auditor import SilenceAuditor


class TestSilenceAuditorAudit:
    """audit() 方法测试集。"""

    def test_non_silent_passthrough(self) -> None:
        """非 SILENT 意图直接放行，不做干预。"""
        auditor = SilenceAuditor()
        result = auditor.audit(
            intent=OutputIntent.INFORM,
            context={"risk_level": "LOW"},
        )
        assert result == OutputIntent.INFORM

    def test_non_silent_error_passthrough(self) -> None:
        """ERROR 意图也直接放行。"""
        auditor = SilenceAuditor()
        result = auditor.audit(
            intent=OutputIntent.ERROR,
            context={"risk_level": "HIGH"},
        )
        assert result == OutputIntent.ERROR

    def test_silent_irreversible_fallback_to_inform(self) -> None:
        """SILENT + 不可逆推理链 → 回退为 INFORM（赫尔墨斯约束）。"""
        auditor = SilenceAuditor()
        result = auditor.audit(
            intent=OutputIntent.SILENT,
            context={"risk_level": "LOW"},
            reversible=False,
        )
        assert result == OutputIntent.INFORM

    def test_silent_high_risk_fallback_to_confirm(self) -> None:
        """SILENT + HIGH 风险 → 回退为 CONFIRM（阿瑞斯约束）。"""
        auditor = SilenceAuditor()
        result = auditor.audit(
            intent=OutputIntent.SILENT,
            context={"risk_level": "HIGH"},
            reversible=True,
        )
        assert result == OutputIntent.CONFIRM

    def test_silent_medium_risk_fallback_to_confirm(self) -> None:
        """SILENT + MEDIUM 风险 → 回退为 CONFIRM（阿瑞斯约束）。"""
        auditor = SilenceAuditor()
        result = auditor.audit(
            intent=OutputIntent.SILENT,
            context={"risk_level": "MEDIUM"},
            reversible=True,
        )
        assert result == OutputIntent.CONFIRM

    def test_silent_first_domain_interaction_fallback(self) -> None:
        """SILENT + 首次领域交互 → 回退为 INFORM。"""
        auditor = SilenceAuditor()
        result = auditor.audit(
            intent=OutputIntent.SILENT,
            context={"risk_level": "LOW", "domain": "finance"},
            reversible=True,
        )
        assert result == OutputIntent.INFORM

    def test_silent_passes_when_all_conditions_met(self) -> None:
        """SILENT + 非首次交互 + 低风险 + 可逆 → SILENT 通过。"""
        auditor = SilenceAuditor()
        # 先记录一次交互，使 "finance" 领域不再是首次
        auditor.record_interaction("finance")
        result = auditor.audit(
            intent=OutputIntent.SILENT,
            context={"risk_level": "LOW", "domain": "finance"},
            reversible=True,
        )
        assert result == OutputIntent.SILENT

    def test_silent_irreversible_overrides_risk(self) -> None:
        """不可逆推理链优先于风险等级判断。"""
        auditor = SilenceAuditor()
        # 即使 risk_level=HIGH，不可逆规则优先 → INFORM
        result = auditor.audit(
            intent=OutputIntent.SILENT,
            context={"risk_level": "HIGH"},
            reversible=False,
        )
        assert result == OutputIntent.INFORM

    def test_context_defaults_when_missing(self) -> None:
        """context 缺少字段时使用默认值（risk_level=LOW, domain=unknown）。"""
        auditor = SilenceAuditor()
        # 空 context，首次交互 unknown 领域 → INFORM
        result = auditor.audit(
            intent=OutputIntent.SILENT,
            context={},
        )
        assert result == OutputIntent.INFORM

    def test_audit_log_recorded(self) -> None:
        """审计日志正确记录 output_id → reason 映射。"""
        auditor = SilenceAuditor()
        auditor.audit(
            intent=OutputIntent.SILENT,
            context={"risk_level": "MEDIUM"},
            output_id="out-001",
        )
        assert "out-001" in auditor._audit_log
        assert "risk_level=MEDIUM" in auditor._audit_log["out-001"]


class TestSilenceAuditorExplain:
    """explain_silence() 方法测试集。"""

    def test_explain_with_known_output_id(self) -> None:
        """已知 output_id 返回包含原因的解释。"""
        auditor = SilenceAuditor()
        auditor.audit(
            intent=OutputIntent.SILENT,
            context={"risk_level": "MEDIUM"},
            output_id="out-042",
        )
        result = auditor.explain_silence("out-042")
        assert "系统判断此输出无需呈现" in result
        assert "risk_level=MEDIUM" in result
        assert "如需查看完整推理链，请追问" in result

    def test_explain_with_unknown_output_id(self) -> None:
        """未知 output_id 返回通用说明。"""
        auditor = SilenceAuditor()
        result = auditor.explain_silence("unknown-id")
        assert "该输出未触发审计记录" in result
        assert "如需查看完整推理链，请追问" in result

    def test_explain_without_output_id(self) -> None:
        """空 output_id 返回通用说明。"""
        auditor = SilenceAuditor()
        result = auditor.explain_silence()
        assert "该输出未触发审计记录" in result

    def test_explain_format_matches_spec(self) -> None:
        """解释文本格式严格匹配规范。"""
        auditor = SilenceAuditor()
        result = auditor.explain_silence()
        expected_prefix = "系统判断此输出无需呈现，原因："
        expected_suffix = "。如需查看完整推理链，请追问。"
        assert result.startswith(expected_prefix)
        assert result.endswith(expected_suffix)


class TestDomainTracking:
    """领域交互跟踪测试。"""

    def test_multi_domain_tracking(self) -> None:
        """多领域独立跟踪交互次数。"""
        auditor = SilenceAuditor()
        auditor.record_interaction("finance")
        auditor.record_interaction("finance")

        # finance 领域已交互 2 次，不再是首次
        result_finance = auditor.audit(
            intent=OutputIntent.SILENT,
            context={"risk_level": "LOW", "domain": "finance"},
            reversible=True,
        )
        assert result_finance == OutputIntent.SILENT

        # medical 领域首次交互 → INFORM
        result_medical = auditor.audit(
            intent=OutputIntent.SILENT,
            context={"risk_level": "LOW", "domain": "medical"},
            reversible=True,
        )
        assert result_medical == OutputIntent.INFORM

    def test_record_interaction_increments(self) -> None:
        """record_interaction 正确递增计数。"""
        auditor = SilenceAuditor()
        assert auditor._domain_history.get("test", 0) == 0
        auditor.record_interaction("test")
        assert auditor._domain_history["test"] == 1
        auditor.record_interaction("test")
        assert auditor._domain_history["test"] == 2
