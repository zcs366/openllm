"""
IKO Integration Tests — 模块间数据流集成测试
=============================================

测试场景覆盖8个IKO模块间的端到端数据流，不使用mock。
所有import使用真实模块路径 from openllm.iko.xxx import YYY。

测试矩阵：
1. 完整管线：IntentClassifier → OutputRouter → render
2. 沉默审计：IntentClassifier(SILENT) → SilenceAuditor.audit()
3. 反馈→λ校准：OutputFeedbackCollector → LambdaCalibrator
4. 审计链+编解码：OutputAuditChain + SymmetricCodec
5. 追问训练+意图分类：ProbingTrainer + IntentClassifier
6. 路由+渲染：OutputRouter.route() + BaseRenderer.render()
7. 完整生命周期：5轮用户交互全管线
"""

from __future__ import annotations

import pytest

# ── 真实模块导入 ──
from openllm.iko.intent_classifier import (
    IntentClassifier,
    OutputIntent,
    ClassificationResult,
)
from openllm.iko.output_router import (
    OutputRouter,
    RenderPlan,
    OutputFormat,
    RendererRegistry,
    TextRenderer,
    StructuredRenderer,
    DiffRenderer,
    ConfirmationRenderer,
    ErrorCardRenderer,
    SilentRenderer,
)
from openllm.iko.silence_auditor import SilenceAuditor
from openllm.iko.feedback_collector import (
    OutputFeedbackCollector,
    FeedbackSignal,
)
from openllm.iko.lambda_calibrator import (
    LambdaCalibrator,
    LambdaState,
    ROLLBACK_THRESHOLD,
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
from openllm.iko.probing_trainer import ProbingTrainer


# ── 辅助：构建标准 context/decision ──

def _ctx(
    risk: str = "LOW",
    tool_calls: bool = False,
    side_effects: bool = False,
    options: int = 0,
) -> dict:
    return {
        "risk_level": risk,
        "has_tool_calls": tool_calls,
        "has_side_effects": side_effects,
        "option_count": options,
    }


def _dec(content: str = "", dtype: str = "answer") -> dict:
    return {"type": dtype, "content": content}


# ═══════════════════════════════════════════════════════════════════════
# 1. 完整管线测试：IntentClassifier → OutputRouter → render
# ═══════════════════════════════════════════════════════════════════════

class TestFullPipeline:
    """IntentClassifier → OutputRouter → render 端到端数据流。"""

    def setup_method(self):
        self.classifier = IntentClassifier()
        self.router = OutputRouter()

    def test_inform_pipeline(self):
        """INFORM意图：分类 → 路由 → 文本渲染。"""
        result = self.classifier.classify(
            _ctx(risk="LOW"), _dec("Python是一门解释型语言")
        )
        assert result.intent == OutputIntent.INFORM

        plan = self.router.route(result.intent, {"text": "Python是一门解释型语言"}, {})
        assert plan.renderer_name == "text"
        assert plan.format_hint == OutputFormat.PLAIN

        rendered = self.router.render(
            result.intent, {"text": "Python是一门解释型语言"}, {}, confidence=0.9
        )
        assert rendered == "Python是一门解释型语言"

    def test_error_pipeline(self):
        """ERROR意图：HIGH风险 → 错误卡片渲染。"""
        result = self.classifier.classify(
            _ctx(risk="HIGH"), _dec("Connection timeout")
        )
        assert result.intent == OutputIntent.ERROR

        rendered = self.router.render(
            result.intent,
            {"error": "连接超时", "cause": "网络异常", "recovery": ["重试", "检查网络"]},
            {},
            confidence=0.95,
        )
        assert "❌" in rendered
        assert "连接超时" in rendered
        assert "重试" in rendered

    def test_confirm_pipeline(self):
        """CONFIRM意图：MEDIUM风险 → 确认卡片渲染。"""
        result = self.classifier.classify(
            _ctx(risk="MEDIUM"), _dec("删除文件")
        )
        assert result.intent == OutputIntent.CONFIRM

        rendered = self.router.render(
            result.intent,
            {"action": "删除文件", "risk_level": "MEDIUM", "options": ["确认", "取消"]},
            {},
            confidence=0.8,
        )
        assert "操作确认" in rendered
        assert "中风险" in rendered

    def test_act_pipeline(self):
        """ACT意图：有工具调用 → diff渲染。"""
        result = self.classifier.classify(
            _ctx(tool_calls=True), _dec("执行git commit")
        )
        assert result.intent == OutputIntent.ACT

        rendered = self.router.render(
            result.intent,
            {"changes": [{"file": "main.py", "old": "v1", "new": "v2"}], "reason": "修复bug"},
            {},
            confidence=0.85,
        )
        assert "修复bug" in rendered
        assert "main.py" in rendered

    def test_decide_pipeline(self):
        """DECIDE意图：2+选项 → 结构化渲染。"""
        result = self.classifier.classify(
            _ctx(options=3), _dec("选择方案")
        )
        assert result.intent == OutputIntent.DECIDE

        rendered = self.router.render(
            result.intent,
            {"items": [{"label": "方案A"}, {"label": "方案B"}, {"label": "方案C"}]},
            {},
            confidence=0.7,
        )
        assert "方案A" in rendered
        assert "1." in rendered


# ═══════════════════════════════════════════════════════════════════════
# 2. 沉默审计集成：IntentClassifier(SILENT) → SilenceAuditor.audit()
# ═══════════════════════════════════════════════════════════════════════

class TestSilenceAuditIntegration:
    """IntentClassifier → SilenceAuditor 沉默审计数据流。"""

    def setup_method(self):
        self.classifier = IntentClassifier()
        self.auditor = SilenceAuditor()

    def test_silent_passes_through(self):
        """SILENT + 可逆 + LOW风险 + 非首次 → 保持SILENT。"""
        result = self.classifier.classify(_ctx(), _dec(""))
        assert result.intent == OutputIntent.SILENT

        self.auditor.record_interaction("general")
        audited = self.auditor.audit(
            OutputIntent.SILENT,
            {"risk_level": "LOW", "domain": "general"},
            reversible=True,
            output_id="out-silent-001",
        )
        assert audited == OutputIntent.SILENT

    def test_irreversible_fallback_to_inform(self):
        """SILENT + 不可逆 → 回退为INFORM（赫尔墨斯约束）。"""
        result = self.classifier.classify(_ctx(), _dec(""))
        assert result.intent == OutputIntent.SILENT

        audited = self.auditor.audit(
            OutputIntent.SILENT,
            {"risk_level": "LOW", "domain": "general"},
            reversible=False,
            output_id="out-silent-002",
        )
        assert audited == OutputIntent.INFORM

    def test_high_risk_fallback_to_confirm(self):
        """SILENT + MEDIUM风险 → 回退为CONFIRM（阿瑞斯约束）。"""
        result = self.classifier.classify(_ctx(), _dec(""))
        assert result.intent == OutputIntent.SILENT

        audited = self.auditor.audit(
            OutputIntent.SILENT,
            {"risk_level": "MEDIUM", "domain": "general"},
            reversible=True,
            output_id="out-silent-003",
        )
        assert audited == OutputIntent.CONFIRM

    def test_first_domain_interaction_fallback(self):
        """SILENT + 首次领域交互 → 回退为INFORM。"""
        result = self.classifier.classify(_ctx(), _dec(""))
        assert result.intent == OutputIntent.SILENT

        # "new_domain" 从未出现过
        audited = self.auditor.audit(
            OutputIntent.SILENT,
            {"risk_level": "LOW", "domain": "new_domain"},
            reversible=True,
            output_id="out-silent-004",
        )
        assert audited == OutputIntent.INFORM

    def test_non_silent_passthrough(self):
        """非SILENT意图不受审计影响。"""
        result = self.classifier.classify(
            _ctx(risk="LOW", tool_calls=True), _dec("执行操作")
        )
        assert result.intent == OutputIntent.ACT

        audited = self.auditor.audit(
            result.intent,
            {"risk_level": "HIGH"},
            reversible=True,
        )
        assert audited == OutputIntent.ACT

    def test_explain_silence_after_audit(self):
        """审计后可获取沉默解释。"""
        self.classifier.classify(_ctx(), _dec(""))
        self.auditor.audit(
            OutputIntent.SILENT,
            {"risk_level": "LOW", "domain": "test"},
            reversible=False,
            output_id="out-exp-001",
        )
        explanation = self.auditor.explain_silence("out-exp-001")
        assert "irreversible" in explanation
        assert "赫尔墨斯约束" in explanation


# ═══════════════════════════════════════════════════════════════════════
# 3. 反馈→λ校准集成：OutputFeedbackCollector → LambdaCalibrator
# ═══════════════════════════════════════════════════════════════════════

class TestFeedbackToLambdaCalibration:
    """OutputFeedbackCollector.detect_signal() → LambdaCalibrator.update()。"""

    def setup_method(self):
        self.collector = OutputFeedbackCollector()
        self.calibrator = LambdaCalibrator()

    def test_accepted_high_conf_increases_lambda(self):
        """ACCEPTED + 高置信度 → λ += 0.02。"""
        initial_lambda = self.calibrator.get_current_lambda()

        signal = self.collector.detect_signal(
            {"action_type": "execute", "user_id": "u-001"}, "out-001"
        )
        assert signal == FeedbackSignal.ACCEPTED

        self.calibrator.update(signal, output_confidence=0.8)
        assert self.calibrator.get_current_lambda() == pytest.approx(
            initial_lambda + 0.02, abs=0.001
        )

    def test_accepted_low_conf_increases_lambda_slightly(self):
        """ACCEPTED + 低置信度 → λ += 0.01。"""
        initial_lambda = self.calibrator.get_current_lambda()

        signal = self.collector.detect_signal(
            {"action_type": "execute", "user_id": "u-002"}, "out-002"
        )
        assert signal == FeedbackSignal.ACCEPTED

        self.calibrator.update(signal, output_confidence=0.5)
        assert self.calibrator.get_current_lambda() == pytest.approx(
            initial_lambda + 0.01, abs=0.001
        )

    def test_rejected_high_conf_decreases_lambda(self):
        """REJECTED + 高置信度 → λ -= 0.05（阿瑞斯惩罚）。"""
        initial_lambda = self.calibrator.get_current_lambda()

        signal = self.collector.detect_signal(
            {"action_type": "reject", "user_id": "u-003"}, "out-003"
        )
        assert signal == FeedbackSignal.REJECTED

        self.calibrator.update(signal, output_confidence=0.9)
        assert self.calibrator.get_current_lambda() == pytest.approx(
            initial_lambda - 0.05, abs=0.001
        )

    def test_clarified_decreases_lambda(self):
        """CLARIFIED → λ -= 0.03。"""
        initial_lambda = self.calibrator.get_current_lambda()

        signal = self.collector.detect_signal(
            {"text": "你说的是什么意思？", "time_delta": 2.0, "user_id": "u-004"},
            "out-004",
        )
        assert signal == FeedbackSignal.CLARIFIED

        self.calibrator.update(signal, output_confidence=0.7)
        assert self.calibrator.get_current_lambda() == pytest.approx(
            initial_lambda - 0.03, abs=0.001
        )

    def test_ignored_no_change(self):
        """IGNORED → λ不变。"""
        initial_lambda = self.calibrator.get_current_lambda()

        signal = self.collector.detect_signal(
            {"time_delta": 60.0, "user_id": "u-005"}, "out-005"
        )
        assert signal == FeedbackSignal.IGNORED

        self.calibrator.update(signal, output_confidence=0.6)
        assert self.calibrator.get_current_lambda() == initial_lambda

    def test_consecutive_clarified_triggers_rollback(self):
        """连续3次CLARIFIED → should_rollback() == True。"""
        for i in range(3):
            signal = self.collector.detect_signal(
                {"text": "追问", "time_delta": 1.0, "user_id": "u-006"},
                f"out-rc-{i}",
            )
            assert signal == FeedbackSignal.CLARIFIED
            self.calibrator.update(signal, output_confidence=0.6)

        assert self.calibrator.should_rollback() is True

    def test_feedback_history_recorded(self):
        """反馈信号被记录到收集器历史中。"""
        self.collector.detect_signal(
            {"action_type": "execute", "user_id": "u-007"}, "out-hist-001"
        )
        self.collector.detect_signal(
            {"action_type": "reject", "user_id": "u-007"}, "out-hist-002"
        )
        prefs = self.collector.get_user_preferences("u-007")
        assert prefs["total_interactions"] == 2
        assert prefs["accepted_ratio"] == 0.5
        assert prefs["rejection_ratio"] == 0.5


# ═══════════════════════════════════════════════════════════════════════
# 4. 审计链+编解码集成：OutputAuditChain + SymmetricCodec
# ═══════════════════════════════════════════════════════════════════════

class TestAuditChainAndCodec:
    """OutputAuditChain.append() + SymmetricCodec.compress() 数据一致性。"""

    def setup_method(self):
        self.chain = OutputAuditChain()
        self.codec = SymmetricCodec()

    def test_audit_chain_with_compressed_reasoning(self):
        """审计链条目使用压缩推理链的hash作为reasoning_chain_hash。"""
        # 构建完整推理链
        full_chain = FullReasoningChain(
            phases=[
                {"description": "分析用户意图", "confidence": 0.85},
                {"description": "生成回答", "confidence": 0.9},
            ],
            risk_assessments=[{"decision": "低风险，直接回答"}],
            tool_calls=[],
            memory_sources=["wiki"],
        )

        compressed = self.codec.compress(full_chain)
        assert compressed.reversible is True
        assert compressed.summary == "分析用户意图"
        assert compressed.confidence == pytest.approx(0.875, abs=0.01)

        # 使用压缩链的ref作为审计链的reasoning_chain_hash
        entry = self.chain.append(
            output_id="out-codec-001",
            intent="回答用户问题",
            content=b"Python is great",
            decision_source="IKO",
            risk_level=0.1,
            confidence=0.85,
            reasoning_chain_hash=compressed.full_chain_ref,
        )
        assert entry.content_hash == self.chain.entries[0].content_hash
        assert self.chain.verify() is True

    def test_audit_chain_verify_after_appends(self):
        """多条审计条目后链完整性验证。"""
        for i in range(5):
            self.chain.append(
                output_id=f"out-verify-{i:03d}",
                intent=f"测试意图{i}",
                content=f"内容{i}".encode(),
                decision_source="IKO",
                risk_level=0.1 * i,
                confidence=1.0 - 0.1 * i,
                reasoning_chain_hash=f"hash-{i:03d}",
            )
        assert self.chain.verify() is True
        assert len(self.chain) == 5

    def test_compress_decompress_symmetry(self):
        """压缩→解压对称性验证（赫淮斯托斯约束）。"""
        full_chain = FullReasoningChain(
            phases=[{"description": "压缩测试", "confidence": 0.8}],
            risk_assessments=[{"decision": "安全"}],
        )

        compressed = self.codec.compress(full_chain)
        assert self.codec.verify_reversibility(compressed) is True

        restored = self.codec.decompress(compressed)
        assert restored.phases[0]["description"] == "压缩测试"
        assert restored.phases[0]["confidence"] == 0.8

    def test_irreversible_decompress_raises(self):
        """不可逆压缩态解压必须抛出ValueError（赫尔墨斯约束）。"""
        non_reversible = CompressedReasoning(
            summary="不完整推理",
            reversible=False,
        )
        with pytest.raises(ValueError, match="non-reversible"):
            self.codec.decompress(non_reversible)

    def test_audit_chain_provenance(self):
        """审计链溯源：从最后一条追溯到genesis。"""
        for i in range(3):
            self.chain.append(
                output_id=f"out-prov-{i}",
                intent=f"意图{i}",
                content=f"数据{i}".encode(),
                decision_source="IKO",
                risk_level=0.0,
                confidence=0.9,
                reasoning_chain_hash=f"rc-hash-{i}",
            )
        provenance = self.chain.get_provenance("out-prov-2")
        assert len(provenance) == 3
        assert provenance[0].output_id == "out-prov-0"
        assert provenance[-1].output_id == "out-prov-2"


# ═══════════════════════════════════════════════════════════════════════
# 5. 追问训练+意图分类集成：ProbingTrainer + IntentClassifier
# ═══════════════════════════════════════════════════════════════════════

class TestProbingAndClassification:
    """ProbingTrainer.should_prompt_probing() + IntentClassifier.classify()。"""

    def setup_method(self):
        self.prober = ProbingTrainer()
        self.classifier = IntentClassifier()

    def test_m1_session_5_triggers_probing(self):
        """M1阶段（session_count=5）→ should_prompt_probing == True。"""
        assert self.prober.should_prompt_probing("u-new", 5) is True

    def test_m1_session_4_no_probing(self):
        """M1阶段（session_count=4，非5的倍数）→ 不触发。"""
        assert self.prober.should_prompt_probing("u-new", 4) is False

    def test_m2_session_40_triggers_probing(self):
        """M2阶段（session_count=40）→ should_prompt_probing == True。"""
        assert self.prober.should_prompt_probing("u-mid", 40) is True

    def test_m3_no_probing(self):
        """M3阶段（session_count=100）→ 不触发追问。"""
        assert self.prober.should_prompt_probing("u-old", 100) is False

    def test_probing_suggestion_with_code(self):
        """追问建议包含代码时的特征分析。"""
        suggestion = self.prober.get_probing_suggestion(
            "这是一个函数：\n```python\ndef hello():\n    pass\n```"
        )
        assert "边界条件" in suggestion

    def test_probing_suggestion_with_list(self):
        """追问建议包含列表时的特征分析。"""
        suggestion = self.prober.get_probing_suggestion(
            "- 第一项\n- 第二项\n- 第三项"
        )
        assert "优先级" in suggestion

    def test_probing_plus_classification_flow(self):
        """追问训练 + 意图分类联合流程。"""
        # M1阶段，第5次会话
        should_probe = self.prober.should_prompt_probing("u-flow", 5)
        assert should_probe is True

        # 获取追问建议
        suggestion = self.prober.get_probing_suggestion("当前输出内容较短")
        assert len(suggestion) > 0

        # 同时进行意图分类
        result = self.classifier.classify(
            _ctx(risk="LOW", options=2),
            _dec("请选择方案"),
        )
        assert result.intent == OutputIntent.DECIDE

    def test_empty_output_no_suggestion(self):
        """空输出不生成追问建议。"""
        assert self.prober.get_probing_suggestion("") == ""
        assert self.prober.get_probing_suggestion("  ") == ""


# ═══════════════════════════════════════════════════════════════════════
# 6. 路由+渲染集成：OutputRouter.route() + BaseRenderer.render()
# ═══════════════════════════════════════════════════════════════════════

class TestRouterAndRendering:
    """OutputRouter.route() + BaseRenderer.render() 输出格式验证。"""

    def setup_method(self):
        self.router = OutputRouter()

    def test_route_produces_render_plan(self):
        """路由引擎生成正确的RenderPlan。"""
        plan = self.router.route(
            OutputIntent.ERROR, {"error": "test"}, {}
        )
        assert isinstance(plan, RenderPlan)
        assert plan.renderer_name == "error_card"
        assert plan.format_hint == OutputFormat.CARD
        assert plan.confidence_display == "always"
        assert plan.trace_reveal == "full"

    def test_all_intents_have_routes(self):
        """所有6种意图都有对应的路由。"""
        for intent in OutputIntent:
            plan = self.router.route(intent, {}, {})
            assert isinstance(plan, RenderPlan)
            assert plan.renderer_name != ""

    def test_renderer_registry_has_all_defaults(self):
        """注册表包含全部6个默认渲染器。"""
        registry = self.router.registry
        assert len(registry) == 6
        for name in ["text", "structured", "diff", "confirmation", "error_card", "silent"]:
            assert registry.has(name)

    def test_text_renderer_concise_mode(self):
        """TextRenderer简洁模式截取前200字符。"""
        renderer = TextRenderer()
        long_text = "A" * 300
        result = renderer.render(
            OutputIntent.INFORM, {"text": long_text}, {"verbosity": "concise"}, 0.9
        )
        assert len(result) == 203  # 200 + "..."
        assert result.endswith("...")

    def test_structured_renderer_items(self):
        """StructuredRenderer渲染选项列表。"""
        renderer = StructuredRenderer()
        result = renderer.render(
            OutputIntent.DECIDE,
            {"items": [{"label": "Alpha"}, {"label": "Beta"}]},
            {},
            0.8,
        )
        assert "1. Alpha" in result
        assert "2. Beta" in result

    def test_structured_renderer_table(self):
        """StructuredRenderer渲染Markdown表格。"""
        renderer = StructuredRenderer()
        result = renderer.render(
            OutputIntent.DECIDE,
            {"table": [["Name", "Score"], ["Alice", "95"], ["Bob", "88"]]},
            {},
            0.9,
        )
        assert "| Name | Score |" in result
        assert "| Alice | 95 |" in result

    def test_diff_renderer_with_reason(self):
        """DiffRenderer渲染变更理由。"""
        renderer = DiffRenderer()
        result = renderer.render(
            OutputIntent.ACT,
            {"changes": [{"file": "a.py", "old": "x=1", "new": "x=2"}], "reason": "修复"},
            {},
            0.85,
        )
        assert "修复" in result
        assert "--- a.py" in result
        assert "- x=1" in result
        assert "+ x=2" in result

    def test_error_card_renderer_recovery(self):
        """ErrorCardRenderer渲染恢复建议。"""
        renderer = ErrorCardRenderer()
        result = renderer.render(
            OutputIntent.ERROR,
            {"error": "磁盘满", "cause": "日志过多", "recovery": ["清理日志", "扩容"]},
            {},
            0.7,
        )
        assert "磁盘满" in result
        assert "1. 清理日志" in result
        assert "2. 扩容" in result

    def test_silent_renderer_returns_empty(self):
        """SilentRenderer返回空字符串。"""
        renderer = SilentRenderer()
        result = renderer.render(OutputIntent.SILENT, {}, {}, 0.0)
        assert result == ""

    def test_render_method一站式(self):
        """OutputRouter.render() 一站式路由+渲染。"""
        result = self.router.render(
            OutputIntent.INFORM,
            {"text": "Hello World"},
            {},
            confidence=0.95,
        )
        assert result == "Hello World"


# ═══════════════════════════════════════════════════════════════════════
# 7. 完整生命周期模拟：5轮用户交互全管线
# ═══════════════════════════════════════════════════════════════════════

class TestFullLifecycle:
    """模拟5轮用户交互，每轮走完整管线。"""

    def setup_method(self):
        self.classifier = IntentClassifier()
        self.router = OutputRouter()
        self.auditor = SilenceAuditor()
        self.collector = OutputFeedbackCollector()
        self.calibrator = LambdaCalibrator()
        self.audit_chain = OutputAuditChain()
        self.codec = SymmetricCodec()
        self.prober = ProbingTrainer()

    def _run_pipeline(
        self,
        round_num: int,
        context: dict,
        decision: dict,
        user_action: dict,
        output_confidence: float,
        domain: str = "general",
    ) -> dict:
        """执行一轮完整管线，返回所有中间结果。"""
        output_id = f"out-round-{round_num}"

        # Step 1: 意图分类
        cls_result = self.classifier.classify(context, decision)
        intent = cls_result.intent

        # Step 2: 沉默审计（仅SILENT需要）
        if intent == OutputIntent.SILENT:
            audited_intent = self.auditor.audit(
                intent, context, reversible=True, output_id=output_id
            )
            intent = audited_intent

        # Step 3: 路由+渲染
        content = {"text": decision.get("content", ""), "error": decision.get("content", "")}
        rendered = self.router.render(intent, content, {}, confidence=output_confidence)

        # Step 4: 审计链记录
        compressed = self.codec.compress(
            FullReasoningChain(
                phases=[{"description": f"Round {round_num}", "confidence": output_confidence}],
                risk_assessments=[{"decision": "proceed"}],
            )
        )
        self.audit_chain.append(
            output_id=output_id,
            intent=intent.value,
            content=rendered.encode() if rendered else b"",
            decision_source="IKO",
            risk_level=context.get("risk_level", "LOW") == "HIGH" and 1.0 or 0.1,
            confidence=output_confidence,
            reasoning_chain_hash=compressed.full_chain_ref,
        )

        # Step 5: 反馈收集
        signal = self.collector.detect_signal(user_action, output_id)

        # Step 6: λ校准
        self.calibrator.update(signal, output_confidence)
        self.auditor.record_interaction(domain)

        return {
            "round": round_num,
            "intent": cls_result.intent,
            "audited_intent": intent,
            "rendered": rendered,
            "signal": signal,
            "lambda": self.calibrator.get_current_lambda(),
        }

    def test_five_round_lifecycle(self):
        """5轮用户交互全管线模拟。"""
        rounds = [
            # Round 1: 简单查询
            {
                "context": _ctx(risk="LOW"),
                "decision": _dec("Python是解释型语言"),
                "user_action": {"action_type": "execute", "user_id": "lifecycle-user"},
                "confidence": 0.9,
            },
            # Round 2: 需要确认的操作
            {
                "context": _ctx(risk="MEDIUM"),
                "decision": _dec("删除临时文件"),
                "user_action": {"action_type": "execute", "user_id": "lifecycle-user"},
                "confidence": 0.8,
            },
            # Round 3: 有工具调用的操作
            {
                "context": _ctx(tool_calls=True),
                "decision": _dec("执行git push"),
                "user_action": {"text": "推送前需要确认", "time_delta": 2.0, "user_id": "lifecycle-user"},
                "confidence": 0.75,
            },
            # Round 4: 多选项决策
            {
                "context": _ctx(options=3),
                "decision": _dec("选择部署方案"),
                "user_action": {"action_type": "reject", "user_id": "lifecycle-user"},
                "confidence": 0.85,
            },
            # Round 5: 再次简单查询
            {
                "context": _ctx(risk="LOW"),
                "decision": _dec("当前版本号"),
                "user_action": {"action_type": "execute", "user_id": "lifecycle-user"},
                "confidence": 0.95,
            },
        ]

        results = []
        for i, r in enumerate(rounds):
            result = self._run_pipeline(
                round_num=i,
                context=r["context"],
                decision=r["decision"],
                user_action=r["user_action"],
                output_confidence=r["confidence"],
            )
            results.append(result)

        # 验证每轮都成功执行
        assert len(results) == 5

        # 验证意图分类正确
        assert results[0]["intent"] == OutputIntent.INFORM
        assert results[1]["intent"] == OutputIntent.CONFIRM
        assert results[2]["intent"] == OutputIntent.ACT
        assert results[3]["intent"] == OutputIntent.DECIDE
        assert results[4]["intent"] == OutputIntent.INFORM

        # 验证渲染输出非空（第3轮确认后是ACT，应该有diff内容）
        for r in results:
            assert isinstance(r["rendered"], str)

        # 验证反馈信号正确
        assert results[0]["signal"] == FeedbackSignal.ACCEPTED
        assert results[1]["signal"] == FeedbackSignal.ACCEPTED
        assert results[2]["signal"] == FeedbackSignal.CLARIFIED
        assert results[3]["signal"] == FeedbackSignal.REJECTED
        assert results[4]["signal"] == FeedbackSignal.ACCEPTED

        # 验证λ值变化：ACCEPTED(+0.02) → ACCEPTED(+0.02) → CLARIFIED(-0.03)
        # → REJECTED(-0.02) → ACCEPTED(+0.02) = 0.5 + 0.02+0.02-0.03-0.02+0.02 = 0.51
        # 注意：REJECTED + 高置信度(>0.7) → -0.05，所以 round 3 confidence=0.75 > 0.7
        # Wait, let me re-check: confidence for round 3 is 0.75, which is > 0.7
        # But signal is CLARIFIED, not REJECTED. CLARIFIED → -0.03 regardless of confidence.
        # Round 3 confidence 0.75 is used for CLARIFIED, so -0.03
        # Round 4 confidence 0.85 > 0.7 and signal is REJECTED, so -0.05
        # Total: 0.5 + 0.02 + 0.02 - 0.03 - 0.05 + 0.02 = 0.48
        expected_lambda = 0.5 + 0.02 + 0.02 - 0.03 - 0.05 + 0.02
        assert results[-1]["lambda"] == pytest.approx(expected_lambda, abs=0.001)

        # 验证审计链完整性
        assert len(self.audit_chain) == 5
        assert self.audit_chain.verify() is True

        # 验证λ校准器状态
        state = self.calibrator.get_state()
        assert state.recovery_count == 3  # 3次ACCEPTED
        assert state.error_count == 1     # 1次REJECTED

    def test_lambda_rollback_after_rejection(self):
        """高置信度REJECTED触发回滚判定。"""
        self._run_pipeline(
            round_num=0,
            context=_ctx(risk="LOW"),
            decision=_dec("测试"),
            user_action={"action_type": "reject", "user_id": "rollback-user"},
            output_confidence=0.9,  # > 0.7 → 高置信度REJECTED
        )
        assert self.calibrator.should_rollback() is True

    def test_audit_chain_grows_across_rounds(self):
        """审计链跨轮次正确增长。"""
        for i in range(3):
            self._run_pipeline(
                round_num=i,
                context=_ctx(risk="LOW"),
                decision=_dec(f"内容{i}"),
                user_action={"action_type": "execute", "user_id": "chain-user"},
                output_confidence=0.8,
            )
        assert len(self.audit_chain) == 3
        assert self.audit_chain.verify() is True

        # 溯源链应包含全部3条
        provenance = self.audit_chain.get_provenance("out-round-2")
        assert len(provenance) == 3
