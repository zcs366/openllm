"""IKO 公开API导入测试"""
from openllm.iko import (
    OutputIntent, IntentClassifier, ClassificationResult,
    FeedbackSignal, OutputFeedbackCollector,
    OutputAuditEntry, OutputAuditChain,
    ProbingTrainer,
    BaseRenderer, OutputRouter, RendererRegistry, RenderPlan,
    SilenceAuditor,
    CompressedReasoning, FullReasoningChain, SymmetricCodec,
    LambdaState, LambdaCalibrator,
)

def test_imports():
    """验证所有17个公开类可导入"""
    assert OutputIntent is not None
    assert IntentClassifier is not None
    assert ClassificationResult is not None
    assert FeedbackSignal is not None
    assert OutputFeedbackCollector is not None
    assert OutputAuditEntry is not None
    assert OutputAuditChain is not None
    assert ProbingTrainer is not None
    assert BaseRenderer is not None
    assert OutputRouter is not None
    assert RendererRegistry is not None
    assert RenderPlan is not None
    assert SilenceAuditor is not None
    assert CompressedReasoning is not None
    assert FullReasoningChain is not None
    assert SymmetricCodec is not None
    assert LambdaState is not None
    assert LambdaCalibrator is not None
    print("✓ All 17 public classes imported")

def test_version():
    """验证版本号"""
    import openllm.iko as iko
    assert hasattr(iko, '__version__')
    assert iko.__version__ == '0.1.0'
    print(f"✓ IKO version: {iko.__version__}")

def test_intent_classifier():
    """IntentClassifier 功能测试"""
    c = IntentClassifier()
    r = c.classify(
        context={"risk_level": "LOW", "has_tool_calls": False,
                 "has_side_effects": False, "option_count": 0},
        decision={"content": "Hello"},
    )
    assert r.intent == OutputIntent.INFORM
    print("✓ IntentClassifier.classify -> INFORM")

def test_output_router():
    """OutputRouter 功能测试"""
    r = OutputRouter()
    plan = r.route(OutputIntent.INFORM, {"text": "Hello"}, {})
    assert plan.renderer_name == "text"
    print("✓ OutputRouter.route -> text renderer")

def test_output_audit():
    """OutputAuditChain 功能测试"""
    chain = OutputAuditChain()
    chain.append("o1", "INFORM", b"hi", "IKO", 0.1, 0.9, "h1")
    chain.append("o2", "ACT", b"do", "ISN", 0.3, 0.8, "h2")
    assert chain.verify()
    assert len(chain) == 2
    print("✓ OutputAuditChain append+verify OK")

def test_symmetric_codec():
    """SymmetricCodec 功能测试"""
    codec = SymmetricCodec()
    chain = FullReasoningChain(
        phases=[{"description": "step1", "confidence": 0.9}],
        risk_assessments=[{"decision": "go"}],
    )
    compressed = codec.compress(chain)
    assert codec.verify_reversibility(compressed)
    restored = codec.decompress(compressed)
    assert len(restored.phases) == 1
    print("✓ SymmetricCodec compress/decompress OK")

def test_silence_auditor():
    """SilenceAuditor 功能测试"""
    a = SilenceAuditor()
    # SILENT + MEDIUM -> CONFIRM
    r = a.audit(OutputIntent.SILENT, {"risk_level": "MEDIUM"})
    assert r == OutputIntent.CONFIRM
    # SILENT + LOW + first interaction -> INFORM
    r2 = a.audit(OutputIntent.SILENT, {"risk_level": "LOW", "domain": "test"})
    assert r2 == OutputIntent.INFORM
    print("✓ SilenceAuditor audit rules OK")

def test_lambda_calibrator():
    """LambdaCalibrator 功能测试"""
    cal = LambdaCalibrator()
    initial = cal.get_current_lambda()
    cal.update(FeedbackSignal.ACCEPTED, 0.8)
    assert cal.get_current_lambda() > initial
    print(f"✓ LambdaCalibrator λ {initial} -> {cal.get_current_lambda()}")

def test_feedback_collector():
    """OutputFeedbackCollector 功能测试"""
    fc = OutputFeedbackCollector()
    s = fc.detect_signal({"action_type": "execute", "time_delta": 2.0}, "o1")
    assert s == FeedbackSignal.ACCEPTED
    prefs = fc.get_user_preferences("anonymous")
    assert prefs["total_interactions"] == 1
    print("✓ OutputFeedbackCollector detect+preferences OK")

def test_probing_trainer():
    """ProbingTrainer 功能测试"""
    pt = ProbingTrainer()
    assert pt.should_prompt_probing("u", 5) is True
    assert pt.should_prompt_probing("u", 7) is False
    assert pt.should_prompt_probing("u", 91) is False
    s = pt.get_probing_suggestion("```code```")
    assert len(s) > 0
    print("✓ ProbingTrainer probing rules OK")

if __name__ == "__main__":
    test_imports()
    test_version()
    test_intent_classifier()
    test_output_router()
    test_output_audit()
    test_symmetric_codec()
    test_silence_auditor()
    test_lambda_calibrator()
    test_feedback_collector()
    test_probing_trainer()
    print("\n🎉 ALL 10 TESTS PASSED")
