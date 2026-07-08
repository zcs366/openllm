"""
IKO End-to-End Tests — 全管线端到端测试
========================================

测试场景覆盖 IKO 完整生命周期（classify→audit→route→render→audit_chain→feedback→λ校准），
使用全部真实模块，零 mock。

测试矩阵：
1. 5轮用户对话完整管线（每步输出验证）
2. 高风险场景（risk_level=HIGH → ERROR → 审计链记录）
3. 用户拒绝 → REJECTED → λ下降
4. 用户追问 → CLARIFIED → λ下降 + rollback检查
5. 审计链完整性（verify=True）
6. λ范围始终在 [0, 1]
"""

from __future__ import annotations

import pytest

# ── 真实模块导入（零 mock） ──
from openllm.iko.intent_classifier import (
    IntentClassifier,
    OutputIntent,
    ClassificationResult,
)
from openllm.iko.output_router import (
    OutputRouter,
    RenderPlan,
    OutputFormat,
)
from openllm.iko.silence_auditor import SilenceAuditor
from openllm.iko.feedback_collector import (
    OutputFeedbackCollector,
    FeedbackSignal,
)
from openllm.iko.lambda_calibrator import (
    LambdaCalibrator,
    LambdaState,
    LAMBDA_MIN,
    LAMBDA_MAX,
    LAMBDA_DEFAULT,
    ROLLBACK_THRESHOLD,
    CLARIFIED_STREAK_LIMIT,
)
from openllm.iko.output_audit import (
    OutputAuditChain,
    OutputAuditEntry,
    compute_entry_hash,
)
from openllm.iko.symmetric_codec import (
    SymmetricCodec,
    CompressedReasoning,
    FullReasoningChain,
)


# ═══════════════════════════════════════════════════════════════════════
# 辅助工具
# ═══════════════════════════════════════════════════════════════════════

def _ctx(
    risk: str = "LOW",
    tool_calls: bool = False,
    side_effects: bool = False,
    options: int = 0,
) -> dict:
    """构建标准 context 字典。"""
    return {
        "risk_level": risk,
        "has_tool_calls": tool_calls,
        "has_side_effects": side_effects,
        "option_count": options,
    }


def _dec(content: str = "", dtype: str = "answer") -> dict:
    """构建标准 decision 字典。"""
    return {"type": dtype, "content": content}


class IKOPipeline:
    """IKO 全管线封装——每轮走完 classify→audit→route→render→audit_chain→feedback→λ校准。"""

    def __init__(self) -> None:
        self.classifier = IntentClassifier()
        self.router = OutputRouter()
        self.auditor = SilenceAuditor()
        self.collector = OutputFeedbackCollector()
        self.calibrator = LambdaCalibrator()
        self.audit_chain = OutputAuditChain()
        self.codec = SymmetricCodec()

    def run_round(
        self,
        round_num: int,
        context: dict,
        decision: dict,
        user_action: dict,
        output_confidence: float,
        domain: str = "general",
    ) -> dict:
        """执行一轮完整管线，返回各步骤结果。

        Returns:
            dict 包含:
                - round: 轮次编号
                - classified_intent: 分类器原始意图
                - audited_intent: 审计后意图
                - render_plan: 路由引擎产出的 RenderPlan
                - rendered: 渲染输出字符串
                - audit_entry: 审计链条目
                - signal: 反馈信号
                - lambda_value: 校准后 λ 值
        """
        output_id = f"e2e-out-r{round_num}"

        # ── Step 1: 意图分类 ──
        cls_result = self.classifier.classify(context, decision)
        intent = cls_result.intent

        # ── Step 2: 沉默审计（仅 SILENT 需要） ──
        audited_intent = intent
        if intent == OutputIntent.SILENT:
            audited_intent = self.auditor.audit(
                intent, context, reversible=True, output_id=output_id,
            )

        # ── Step 3: 路由 ──
        # 根据意图类型构造完整的 content 字典
        raw_content = decision.get("content", "")
        if audited_intent == OutputIntent.ACT:
            content = {
                "changes": [{"file": "output.txt", "old": "", "new": raw_content}],
                "reason": raw_content,
            }
        elif audited_intent == OutputIntent.ERROR:
            content = {"error": raw_content, "cause": raw_content, "recovery": ["重试"]}
        elif audited_intent == OutputIntent.CONFIRM:
            content = {"action": raw_content, "risk_level": context.get("risk_level", "LOW"), "options": ["确认", "取消"]}
        elif audited_intent == OutputIntent.DECIDE:
            content = {"items": [{"label": f"选项{i}"} for i in range(max(2, context.get("option_count", 2)))]}
        else:
            content = {"text": raw_content, "error": raw_content}
        render_plan = self.router.route(audited_intent, content, {})

        # ── Step 4: 渲染 ──
        rendered = self.router.render(
            audited_intent, content, {}, confidence=output_confidence,
        )

        # ── Step 5: 审计链记录 ──
        compressed = self.codec.compress(
            FullReasoningChain(
                phases=[{"description": f"Round {round_num} pipeline", "confidence": output_confidence}],
                risk_assessments=[{"decision": "proceed"}],
            )
        )
        risk_float = 1.0 if context.get("risk_level") == "HIGH" else 0.1
        audit_entry = self.audit_chain.append(
            output_id=output_id,
            intent=audited_intent.value,
            content=rendered.encode() if rendered else b"",
            decision_source="IKO",
            risk_level=risk_float,
            confidence=output_confidence,
            reasoning_chain_hash=compressed.full_chain_ref,
        )

        # ── Step 6: 反馈收集 ──
        signal = self.collector.detect_signal(user_action, output_id)

        # ── Step 7: λ 校准 ──
        self.calibrator.update(signal, output_confidence)
        self.auditor.record_interaction(domain)

        return {
            "round": round_num,
            "classified_intent": cls_result.intent,
            "audited_intent": audited_intent,
            "render_plan": render_plan,
            "rendered": rendered,
            "audit_entry": audit_entry,
            "signal": signal,
            "lambda_value": self.calibrator.get_current_lambda(),
        }


# ═══════════════════════════════════════════════════════════════════════
# 测试场景 1：5轮用户对话完整管线
# ═══════════════════════════════════════════════════════════════════════

class TestFiveRoundPipeline:
    """模拟5轮用户对话，每轮走完整 IKO 管线，验证每步输出正确。"""

    def setup_method(self) -> None:
        self.pipe = IKOPipeline()

    def test_five_rounds_all_steps(self) -> None:
        """5轮对话：INFORM → CONFIRM → ACT → DECIDE → INFORM，验证每步。"""
        rounds = [
            # Round 0: 简单查询 → INFORM + ACCEPTED
            {
                "context": _ctx(risk="LOW"),
                "decision": _dec("Python是解释型语言"),
                "user_action": {"action_type": "execute", "user_id": "e2e-u1"},
                "confidence": 0.9,
            },
            # Round 1: MEDIUM风险 → CONFIRM + ACCEPTED
            {
                "context": _ctx(risk="MEDIUM"),
                "decision": _dec("删除临时文件"),
                "user_action": {"action_type": "execute", "user_id": "e2e-u1"},
                "confidence": 0.8,
            },
            # Round 2: 有工具调用 → ACT + CLARIFIED（追问）
            {
                "context": _ctx(tool_calls=True),
                "decision": _dec("执行git push"),
                "user_action": {"text": "推送前需要确认", "time_delta": 2.0, "user_id": "e2e-u1"},
                "confidence": 0.75,
            },
            # Round 3: 3选项 → DECIDE + REJECTED（拒绝）
            {
                "context": _ctx(options=3),
                "decision": _dec("选择部署方案"),
                "user_action": {"action_type": "reject", "user_id": "e2e-u1"},
                "confidence": 0.85,
            },
            # Round 4: 简单查询 → INFORM + ACCEPTED
            {
                "context": _ctx(risk="LOW"),
                "decision": _dec("当前版本号是v2.1.0"),
                "user_action": {"action_type": "execute", "user_id": "e2e-u1"},
                "confidence": 0.95,
            },
        ]

        results = []
        for i, r in enumerate(rounds):
            result = self.pipe.run_round(
                round_num=i,
                context=r["context"],
                decision=r["decision"],
                user_action=r["user_action"],
                output_confidence=r["confidence"],
            )
            results.append(result)

        # ── 验证5轮全部执行成功 ──
        assert len(results) == 5

        # ── 验证每步输出 ──
        for i, r in enumerate(results):
            # Step 1: classify 产出 ClassificationResult
            assert isinstance(r["classified_intent"], OutputIntent)
            # Step 2: audited_intent 存在
            assert isinstance(r["audited_intent"], OutputIntent)
            # Step 3: render_plan 是 RenderPlan
            assert isinstance(r["render_plan"], RenderPlan)
            assert r["render_plan"].renderer_name != ""
            # Step 4: rendered 是 str
            assert isinstance(r["rendered"], str)
            # Step 5: audit_entry 存在
            assert isinstance(r["audit_entry"], OutputAuditEntry)
            assert r["audit_entry"].output_id == f"e2e-out-r{i}"
            # Step 6: signal 是 FeedbackSignal
            assert isinstance(r["signal"], FeedbackSignal)
            # Step 7: λ 在合法范围
            assert LAMBDA_MIN <= r["lambda_value"] <= LAMBDA_MAX

        # ── 验证意图分类 ──
        assert results[0]["classified_intent"] == OutputIntent.INFORM
        assert results[1]["classified_intent"] == OutputIntent.CONFIRM
        assert results[2]["classified_intent"] == OutputIntent.ACT
        assert results[3]["classified_intent"] == OutputIntent.DECIDE
        assert results[4]["classified_intent"] == OutputIntent.INFORM

        # ── 验证渲染输出 ──
        assert results[0]["rendered"] == "Python是解释型语言"
        assert "操作确认" in results[1]["rendered"]
        assert "git push" in results[2]["rendered"]
        assert "选项" in results[3]["rendered"]  # DECIDE renders items with labels
        assert results[4]["rendered"] == "当前版本号是v2.1.0"

        # ── 验证路由计划 ──
        assert results[0]["render_plan"].renderer_name == "text"
        assert results[1]["render_plan"].renderer_name == "confirmation"
        assert results[2]["render_plan"].renderer_name == "diff"
        assert results[3]["render_plan"].renderer_name == "structured"
        assert results[4]["render_plan"].renderer_name == "text"

        # ── 验证反馈信号 ──
        assert results[0]["signal"] == FeedbackSignal.ACCEPTED
        assert results[1]["signal"] == FeedbackSignal.ACCEPTED
        assert results[2]["signal"] == FeedbackSignal.CLARIFIED
        assert results[3]["signal"] == FeedbackSignal.REJECTED
        assert results[4]["signal"] == FeedbackSignal.ACCEPTED

        # ── 验证λ值变化 ──
        # ACCEPTED(0.9→+0.02) → ACCEPTED(0.8→+0.02) → CLARIFIED(0.75→-0.03)
        # → REJECTED(0.85→-0.05, 高置信度>0.7) → ACCEPTED(0.95→+0.02)
        # 0.5 + 0.02 + 0.02 - 0.03 - 0.05 + 0.02 = 0.48
        expected_lambda = LAMBDA_DEFAULT + 0.02 + 0.02 - 0.03 - 0.05 + 0.02
        assert results[-1]["lambda_value"] == pytest.approx(expected_lambda, abs=0.001)

        # ── 验证审计链 ──
        assert len(self.pipe.audit_chain) == 5
        assert self.pipe.audit_chain.verify() is True

        # ── 验证审计链溯源 ──
        provenance = self.pipe.audit_chain.get_provenance("e2e-out-r4")
        assert len(provenance) == 5
        assert provenance[0].output_id == "e2e-out-r0"
        assert provenance[-1].output_id == "e2e-out-r4"

    def test_lambda_always_in_bounds(self) -> None:
        """每轮后 λ 始终在 [0, 1]。"""
        for i in range(5):
            self.pipe.run_round(
                round_num=i,
                context=_ctx(risk="LOW"),
                decision=_dec(f"内容{i}"),
                user_action={"action_type": "execute", "user_id": "e2e-bounds"},
                output_confidence=0.9,
            )
            lam = self.pipe.calibrator.get_current_lambda()
            assert LAMBDA_MIN <= lam <= LAMBDA_MAX, f"Round {i}: λ={lam} out of bounds"


# ═══════════════════════════════════════════════════════════════════════
# 测试场景 2：高风险场景（risk_level=HIGH → ERROR → 审计链记录）
# ═══════════════════════════════════════════════════════════════════════

class TestHighRiskScenario:
    """模拟高风险场景，验证 ERROR 意图分类和审计链记录。"""

    def setup_method(self) -> None:
        self.pipe = IKOPipeline()

    def test_high_risk_classifies_as_error(self) -> None:
        """risk_level=HIGH → 分类为 ERROR。"""
        result = self.pipe.run_round(
            round_num=0,
            context=_ctx(risk="HIGH"),
            decision=_dec("Connection timeout to database"),
            user_action={"action_type": "reject", "user_id": "hr-user"},
            output_confidence=0.95,
        )
        assert result["classified_intent"] == OutputIntent.ERROR
        assert result["audited_intent"] == OutputIntent.ERROR

    def test_high_risk_renders_error_card(self) -> None:
        """ERROR 意图渲染为错误卡片，包含 ❌。"""
        result = self.pipe.run_round(
            round_num=0,
            context=_ctx(risk="HIGH"),
            decision=_dec("磁盘空间不足"),
            user_action={"action_type": "execute", "user_id": "hr-user2"},
            output_confidence=0.9,
        )
        assert "❌" in result["rendered"]
        assert "磁盘空间不足" in result["rendered"]

    def test_high_risk_recorded_in_audit_chain(self) -> None:
        """高风险审计条目 risk_level=1.0 且链完整。"""
        result = self.pipe.run_round(
            round_num=0,
            context=_ctx(risk="HIGH"),
            decision=_dec("API密钥泄露"),
            user_action={"action_type": "execute", "user_id": "hr-user3"},
            output_confidence=0.88,
        )
        assert result["audit_entry"].risk_level == 1.0
        assert result["audit_entry"].confidence == 0.88
        assert result["audit_entry"].intent == "error"
        assert self.pipe.audit_chain.verify() is True

    def test_high_risk_error_plan(self) -> None:
        """ERROR 意图的 render_plan 为 error_card 渲染器。"""
        result = self.pipe.run_round(
            round_num=0,
            context=_ctx(risk="HIGH"),
            decision=_dec("网络异常"),
            user_action={"action_type": "execute", "user_id": "hr-user4"},
            output_confidence=0.92,
        )
        assert result["render_plan"].renderer_name == "error_card"
        assert result["render_plan"].format_hint == OutputFormat.CARD
        assert result["render_plan"].trace_reveal == "full"


# ═══════════════════════════════════════════════════════════════════════
# 测试场景 3：用户拒绝 → REJECTED → λ 下降
# ═══════════════════════════════════════════════════════════════════════

class TestRejectionLambdaDecrease:
    """模拟用户拒绝场景，验证 REJECTED 信号导致 λ 下降。"""

    def setup_method(self) -> None:
        self.pipe = IKOPipeline()

    def test_single_rejection_high_conf_decreases_lambda(self) -> None:
        """单次 REJECTED + 高置信度(>0.7) → λ -= 0.05。"""
        initial_lambda = self.pipe.calibrator.get_current_lambda()
        result = self.pipe.run_round(
            round_num=0,
            context=_ctx(risk="LOW"),
            decision=_dec("建议方案A"),
            user_action={"action_type": "reject", "user_id": "rej-user"},
            output_confidence=0.9,
        )
        assert result["signal"] == FeedbackSignal.REJECTED
        assert result["lambda_value"] == pytest.approx(initial_lambda - 0.05, abs=0.001)

    def test_single_rejection_low_conf_decreases_lambda_slightly(self) -> None:
        """单次 REJECTED + 低置信度(≤0.7) → λ -= 0.02。"""
        initial_lambda = self.pipe.calibrator.get_current_lambda()
        result = self.pipe.run_round(
            round_num=0,
            context=_ctx(risk="LOW"),
            decision=_dec("建议方案B"),
            user_action={"action_type": "reject", "user_id": "rej-user2"},
            output_confidence=0.6,
        )
        assert result["signal"] == FeedbackSignal.REJECTED
        assert result["lambda_value"] == pytest.approx(initial_lambda - 0.02, abs=0.001)

    def test_rejection_triggers_rollback_check(self) -> None:
        """高置信度 REJECTED → should_rollback() == True。"""
        self.pipe.run_round(
            round_num=0,
            context=_ctx(risk="LOW"),
            decision=_dec("建议方案"),
            user_action={"action_type": "reject", "user_id": "rej-rb"},
            output_confidence=0.9,
        )
        assert self.pipe.calibrator.should_rollback() is True

    def test_lambda_always_in_bounds_after_rejection(self) -> None:
        """连续多次 REJECTED，λ 始终在 [0, 1]。"""
        for i in range(10):
            self.pipe.run_round(
                round_num=i,
                context=_ctx(risk="LOW"),
                decision=_dec(f"建议{i}"),
                user_action={"action_type": "reject", "user_id": "rej-stress"},
                output_confidence=0.95,
            )
            lam = self.pipe.calibrator.get_current_lambda()
            assert LAMBDA_MIN <= lam <= LAMBDA_MAX, f"Round {i}: λ={lam} out of bounds"


# ═══════════════════════════════════════════════════════════════════════
# 测试场景 4：用户追问 → CLARIFIED → λ 下降 + rollback 检查
# ═══════════════════════════════════════════════════════════════════════

class TestClarificationLambdaDecrease:
    """模拟用户追问场景，验证 CLARIFIED 信号导致 λ 下降和回滚。"""

    def setup_method(self) -> None:
        self.pipe = IKOPipeline()

    def test_single_clarified_decreases_lambda(self) -> None:
        """单次 CLARIFIED → λ -= 0.03。"""
        initial_lambda = self.pipe.calibrator.get_current_lambda()
        result = self.pipe.run_round(
            round_num=0,
            context=_ctx(risk="LOW"),
            decision=_dec("Python是解释型语言"),
            user_action={"text": "什么意思？", "time_delta": 2.0, "user_id": "cli-user"},
            output_confidence=0.7,
        )
        assert result["signal"] == FeedbackSignal.CLARIFIED
        assert result["lambda_value"] == pytest.approx(initial_lambda - 0.03, abs=0.001)

    def test_consecutive_clarified_triggers_rollback(self) -> None:
        """连续3次 CLARIFIED → should_rollback() == True。"""
        for i in range(CLARIFIED_STREAK_LIMIT):
            self.pipe.run_round(
                round_num=i,
                context=_ctx(risk="LOW"),
                decision=_dec(f"解释{i}"),
                user_action={"text": "追问", "time_delta": 1.0, "user_id": "cli-streak"},
                output_confidence=0.6,
            )
        assert self.pipe.calibrator.should_rollback() is True

    def test_clarified_then_transparency_rollback(self) -> None:
        """3次 CLARIFIED 后触发透明回滚，λ 提升至 ≥ 0.6。"""
        for i in range(3):
            self.pipe.run_round(
                round_num=i,
                context=_ctx(risk="LOW"),
                decision=_dec(f"追问{i}"),
                user_action={"text": "为什么？", "time_delta": 2.0, "user_id": "cli-rollback"},
                output_confidence=0.6,
            )
        assert self.pipe.calibrator.should_rollback() is True

        # 触发透明回滚
        self.pipe.calibrator.trigger_transparency_rollback(domain="general")
        lam = self.pipe.calibrator.get_current_lambda()
        assert lam >= 0.6

    def test_clarified_decreases_lambda_each_time(self) -> None:
        """每次 CLARIFIED 都使 λ 下降 0.03。"""
        initial = self.pipe.calibrator.get_current_lambda()
        for i in range(3):
            self.pipe.run_round(
                round_num=i,
                context=_ctx(risk="LOW"),
                decision=_dec(f"内容{i}"),
                user_action={"text": "请解释", "time_delta": 3.0, "user_id": "cli-dec"},
                output_confidence=0.7,
            )
            expected = initial - 0.03 * (i + 1)
            assert self.pipe.calibrator.get_current_lambda() == pytest.approx(expected, abs=0.001)

    def test_lambda_always_in_bounds_after_clarified(self) -> None:
        """连续多次 CLARIFIED，λ 始终在 [0, 1]。"""
        for i in range(20):
            self.pipe.run_round(
                round_num=i,
                context=_ctx(risk="LOW"),
                decision=_dec(f"追问{i}"),
                user_action={"text": "不清楚", "time_delta": 1.0, "user_id": "cli-stress"},
                output_confidence=0.6,
            )
            lam = self.pipe.calibrator.get_current_lambda()
            assert LAMBDA_MIN <= lam <= LAMBDA_MAX, f"Round {i}: λ={lam} out of bounds"


# ═══════════════════════════════════════════════════════════════════════
# 测试场景 5：审计链完整性（verify=True）
# ═══════════════════════════════════════════════════════════════════════

class TestAuditChainIntegrity:
    """验证审计链完整性，包含哈希链接验证和溯源。"""

    def setup_method(self) -> None:
        self.pipe = IKOPipeline()

    def test_empty_chain_is_valid(self) -> None:
        """空审计链也是完整的。"""
        chain = OutputAuditChain()
        assert chain.verify() is True
        assert len(chain) == 0

    def test_single_entry_chain_valid(self) -> None:
        """单条审计链完整。"""
        self.pipe.run_round(
            round_num=0,
            context=_ctx(risk="LOW"),
            decision=_dec("单条测试"),
            user_action={"action_type": "execute", "user_id": "ac-user"},
            output_confidence=0.9,
        )
        assert len(self.pipe.audit_chain) == 1
        assert self.pipe.audit_chain.verify() is True

    def test_multi_round_chain_valid(self) -> None:
        """多轮后审计链完整。"""
        for i in range(5):
            self.pipe.run_round(
                round_num=i,
                context=_ctx(risk="LOW"),
                decision=_dec(f"链测试{i}"),
                user_action={"action_type": "execute", "user_id": "ac-multi"},
                output_confidence=0.85,
            )
        assert len(self.pipe.audit_chain) == 5
        assert self.pipe.audit_chain.verify() is True

    def test_high_risk_chain_valid(self) -> None:
        """高风险审计条目不影响链完整性。"""
        self.pipe.run_round(
            round_num=0,
            context=_ctx(risk="HIGH"),
            decision=_dec("高风险操作"),
            user_action={"action_type": "execute", "user_id": "ac-hr"},
            output_confidence=0.9,
        )
        self.pipe.run_round(
            round_num=1,
            context=_ctx(risk="LOW"),
            decision=_dec("后续操作"),
            user_action={"action_type": "execute", "user_id": "ac-hr"},
            output_confidence=0.8,
        )
        assert self.pipe.audit_chain.verify() is True

    def test_provenance_links_back_to_genesis(self) -> None:
        """溯源链从 genesis 到最新条目。"""
        for i in range(4):
            self.pipe.run_round(
                round_num=i,
                context=_ctx(risk="LOW"),
                decision=_dec(f"溯源{i}"),
                user_action={"action_type": "execute", "user_id": "ac-prov"},
                output_confidence=0.88,
            )
        provenance = self.pipe.audit_chain.get_provenance("e2e-out-r3")
        assert len(provenance) == 4
        # genesis → r0 → r1 → r2 → r3
        assert provenance[0].prev_hash == "genesis"
        assert provenance[-1].output_id == "e2e-out-r3"

    def test_audit_entry_fields_correct(self) -> None:
        """审计条目各字段值正确。"""
        result = self.pipe.run_round(
            round_num=0,
            context=_ctx(risk="HIGH"),
            decision=_dec("字段验证"),
            user_action={"action_type": "execute", "user_id": "ac-fields"},
            output_confidence=0.92,
        )
        entry = result["audit_entry"]
        assert entry.output_id == "e2e-out-r0"
        assert entry.intent == "error"
        assert entry.decision_source == "IKO"
        assert entry.risk_level == 1.0
        assert entry.confidence == 0.92
        assert entry.prev_hash == "genesis"
        assert len(entry.content_hash) == 16
        assert len(entry.reasoning_chain_hash) == 16

    def test_duplicate_output_id_raises(self) -> None:
        """重复 output_id 抛出 ValueError。"""
        chain = OutputAuditChain()
        chain.append(
            output_id="dup-001", intent="test", content=b"data",
            decision_source="IKO", risk_level=0.0, confidence=0.9,
            reasoning_chain_hash="hash1",
        )
        with pytest.raises(ValueError, match="Duplicate output_id"):
            chain.append(
                output_id="dup-001", intent="test2", content=b"data2",
                decision_source="IKO", risk_level=0.0, confidence=0.9,
                reasoning_chain_hash="hash2",
            )


# ═══════════════════════════════════════════════════════════════════════
# 测试场景 6：λ 范围始终在 [0, 1]
# ═══════════════════════════════════════════════════════════════════════

class TestLambdaBounds:
    """验证 λ 在各种操作后始终在 [0, 1]。"""

    def setup_method(self) -> None:
        self.calibrator = LambdaCalibrator()

    def test_initial_lambda_in_bounds(self) -> None:
        """初始 λ 在 [0, 1]。"""
        lam = self.calibrator.get_current_lambda()
        assert LAMBDA_MIN <= lam <= LAMBDA_MAX
        assert lam == LAMBDA_DEFAULT

    def test_accepted_increases_lambda(self) -> None:
        """ACCEPTED 增加 λ。"""
        before = self.calibrator.get_current_lambda()
        self.calibrator.update(FeedbackSignal.ACCEPTED, 0.8)
        after = self.calibrator.get_current_lambda()
        assert after > before
        assert LAMBDA_MIN <= after <= LAMBDA_MAX

    def test_rejected_decreases_lambda(self) -> None:
        """REJECTED 减少 λ。"""
        before = self.calibrator.get_current_lambda()
        self.calibrator.update(FeedbackSignal.REJECTED, 0.9)
        after = self.calibrator.get_current_lambda()
        assert after < before
        assert LAMBDA_MIN <= after <= LAMBDA_MAX

    def test_clarified_decreases_lambda(self) -> None:
        """CLARIFIED 减少 λ。"""
        before = self.calibrator.get_current_lambda()
        self.calibrator.update(FeedbackSignal.CLARIFIED, 0.7)
        after = self.calibrator.get_current_lambda()
        assert after < before
        assert LAMBDA_MIN <= after <= LAMBDA_MAX

    def test_ignored_no_change(self) -> None:
        """IGNORED 不改变 λ。"""
        before = self.calibrator.get_current_lambda()
        self.calibrator.update(FeedbackSignal.IGNORED, 0.5)
        after = self.calibrator.get_current_lambda()
        assert after == before

    def test_extreme_accepted_never_exceeds_1(self) -> None:
        """大量 ACCEPTED 不会超过 1.0。"""
        for _ in range(100):
            self.calibrator.update(FeedbackSignal.ACCEPTED, 0.9)
        lam = self.calibrator.get_current_lambda()
        assert lam <= LAMBDA_MAX

    def test_extreme_rejected_never_below_0(self) -> None:
        """大量 REJECTED 不会低于 0.0。"""
        for _ in range(100):
            self.calibrator.update(FeedbackSignal.REJECTED, 0.9)
        lam = self.calibrator.get_current_lambda()
        assert lam >= LAMBDA_MIN

    def test_mixed_signals_lambda_stays_bounded(self) -> None:
        """混合信号操作，λ 始终在 [0, 1]。"""
        signals = [
            (FeedbackSignal.ACCEPTED, 0.8),
            (FeedbackSignal.REJECTED, 0.9),
            (FeedbackSignal.CLARIFIED, 0.6),
            (FeedbackSignal.ACCEPTED, 0.7),
            (FeedbackSignal.IGNORED, 0.5),
            (FeedbackSignal.REJECTED, 0.8),
            (FeedbackSignal.CLARIFIED, 0.6),
            (FeedbackSignal.ACCEPTED, 0.9),
            (FeedbackSignal.CLARIFIED, 0.7),
            (FeedbackSignal.REJECTED, 0.95),
        ]
        for sig, conf in signals:
            self.calibrator.update(sig, conf)
            lam = self.calibrator.get_current_lambda()
            assert LAMBDA_MIN <= lam <= LAMBDA_MAX, (
                f"After {sig.value}: λ={lam} out of bounds"
            )

    def test_transparency_rollback_lambda_in_bounds(self) -> None:
        """透明回滚后 λ 在 [0, 1]。"""
        self.calibrator.trigger_transparency_rollback(domain="test")
        lam = self.calibrator.get_current_lambda()
        assert LAMBDA_MIN <= lam <= LAMBDA_MAX
        assert lam >= 0.6

    def test_lambda_state_snapshot_in_bounds(self) -> None:
        """LambdaState 快照的 value 也在 [0, 1]。"""
        for sig, conf in [
            (FeedbackSignal.ACCEPTED, 0.8),
            (FeedbackSignal.REJECTED, 0.9),
            (FeedbackSignal.CLARIFIED, 0.6),
        ]:
            self.calibrator.update(sig, conf)
        state = self.calibrator.get_state()
        assert LAMBDA_MIN <= state.value <= LAMBDA_MAX


# ═══════════════════════════════════════════════════════════════════════
# 补充：混合场景端到端压力测试
# ═══════════════════════════════════════════════════════════════════════

class TestMixedScenarioE2E:
    """混合场景端到端压力测试——高风险+拒绝+追问交叉。"""

    def setup_method(self) -> None:
        self.pipe = IKOPipeline()

    def test_high_risk_then_rejection_then_clarification(self) -> None:
        """高风险ERROR → REJECTED → CLARIFIED → 审计链完整 + λ 在界。"""
        # Round 0: 高风险 ERROR
        r0 = self.pipe.run_round(
            round_num=0,
            context=_ctx(risk="HIGH"),
            decision=_dec("数据库连接失败"),
            user_action={"action_type": "reject", "user_id": "mix-u1"},
            output_confidence=0.95,
        )
        assert r0["classified_intent"] == OutputIntent.ERROR
        assert r0["signal"] == FeedbackSignal.REJECTED

        # Round 1: CLARIFIED 追问
        r1 = self.pipe.run_round(
            round_num=1,
            context=_ctx(risk="LOW"),
            decision=_dec("重试方案"),
            user_action={"text": "什么是重试方案？", "time_delta": 3.0, "user_id": "mix-u1"},
            output_confidence=0.7,
        )
        assert r1["signal"] == FeedbackSignal.CLARIFIED

        # Round 2: ACCEPTED
        r2 = self.pipe.run_round(
            round_num=2,
            context=_ctx(risk="LOW"),
            decision=_dec("连接成功"),
            user_action={"action_type": "execute", "user_id": "mix-u1"},
            output_confidence=0.9,
        )
        assert r2["signal"] == FeedbackSignal.ACCEPTED

        # 验证审计链
        assert len(self.pipe.audit_chain) == 3
        assert self.pipe.audit_chain.verify() is True

        # 验证λ在界
        lam = self.pipe.calibrator.get_current_lambda()
        assert LAMBDA_MIN <= lam <= LAMBDA_MAX

    def test_ten_round_mixed_pipeline(self) -> None:
        """10轮混合管线，每轮验证 λ 在界 + 审计链完整。"""
        scenarios = [
            (_ctx(risk="LOW"), _dec("查询"), {"action_type": "execute", "user_id": "mix10"}, 0.9),
            (_ctx(risk="HIGH"), _dec("高危"), {"action_type": "reject", "user_id": "mix10"}, 0.88),
            (_ctx(risk="LOW"), _dec("重试"), {"text": "为什么？", "time_delta": 2.0, "user_id": "mix10"}, 0.7),
            (_ctx(risk="MEDIUM"), _dec("确认"), {"action_type": "execute", "user_id": "mix10"}, 0.85),
            (_ctx(options=3), _dec("选择"), {"action_type": "reject", "user_id": "mix10"}, 0.8),
            (_ctx(risk="LOW"), _dec("版本"), {"action_type": "execute", "user_id": "mix10"}, 0.95),
            (_ctx(tool_calls=True), _dec("提交"), {"text": "确认提交？", "time_delta": 1.5, "user_id": "mix10"}, 0.72),
            (_ctx(risk="LOW"), _dec("日志"), {"action_type": "execute", "user_id": "mix10"}, 0.88),
            (_ctx(risk="HIGH"), _dec("超时"), {"action_type": "reject", "user_id": "mix10"}, 0.91),
            (_ctx(risk="LOW"), _dec("完成"), {"action_type": "execute", "user_id": "mix10"}, 0.93),
        ]

        for i, (ctx, dec, action, conf) in enumerate(scenarios):
            result = self.pipe.run_round(
                round_num=i,
                context=ctx,
                decision=dec,
                user_action=action,
                output_confidence=conf,
            )
            lam = result["lambda_value"]
            assert LAMBDA_MIN <= lam <= LAMBDA_MAX, f"Round {i}: λ={lam}"

        assert len(self.pipe.audit_chain) == 10
        assert self.pipe.audit_chain.verify() is True
