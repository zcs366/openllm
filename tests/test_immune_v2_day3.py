"""Day 3 测试: HTIR-Step1 harness_layer归因"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/projects/openllm/src"))

from openllm.core.failure_tracker import (
    FailureCategory, HarnessLayer, CATEGORY_TO_LAYER, FailureSignature
)
from openllm.core.governance_engine import GovernanceEngine


def test_harness_layer_enum():
    """七层枚举完整"""
    layers = [l.value for l in HarnessLayer]
    expected = ["execution_environment", "tool_interface", "context_and_memory",
                "lifecycle_orchestration", "observability", "verification", "governance"]
    for e in expected:
        assert e in layers, f"缺失层: {e}"
    print(f"  ✅ 七层枚举: {len(layers)}层完整")


def test_category_to_layer_mapping():
    """所有FailureCategory都有映射"""
    for cat in FailureCategory:
        layer = CATEGORY_TO_LAYER.get(cat)
        assert layer is not None, f"{cat.name}无harness_layer映射"
        assert isinstance(layer, HarnessLayer)
    print(f"  ✅ 映射完整: {len(CATEGORY_TO_LAYER)}个category→layer")


def test_capture_failure_populates_layer():
    """capture_failure自动填充harness_layer"""
    engine = GovernanceEngine()
    
    # TOOL_PARAM → tool_interface
    sig = engine.capture_failure({
        "tool_name": "read_file",
        "error": "缺少必填参数path",
        "mechanism": "tool_param"
    })
    assert sig.harness_layer == "tool_interface", f"期望tool_interface, 得到{sig.harness_layer}"
    
    # TOOL_TIMEOUT → lifecycle_orchestration
    sig = engine.capture_failure({
        "tool_name": "web_search",
        "error": "timeout after 30s",
        "mechanism": "tool_timeout"
    })
    assert sig.harness_layer == "lifecycle_orchestration"
    
    # LLM_HALLUCINATION → context_and_memory
    sig = engine.capture_failure({
        "tool_name": "unknown_tool",
        "error": "tool not found",
        "mechanism": "llm_hallucination"
    })
    assert sig.harness_layer == "context_and_memory"
    
    print("  ✅ capture_failure自动归因: 3/3正确")


def test_failure_signature_has_layer():
    """FailureSignature包含harness_layer字段"""
    sig = FailureSignature(
        category=FailureCategory.TOOL_PARAM,
        tool_name="test",
        error_pattern="test error",
        harness_layer="tool_interface"
    )
    assert sig.harness_layer == "tool_interface"
    print("  ✅ FailureSignature: harness_layer字段可用")


if __name__ == "__main__":
    print("=" * 60)
    print("Day 3 · HTIR-Step1 测试")
    print("=" * 60)
    tests = [
        test_harness_layer_enum,
        test_category_to_layer_mapping,
        test_capture_failure_populates_layer,
        test_failure_signature_has_layer,
    ]
    p, f = 0, 0
    for t in tests:
        try: t(); p += 1
        except Exception as e: print(f"  ❌ {t.__name__}: {e}"); f += 1
    print(f"\n结果: {p} passed, {f} failed")
    sys.exit(1 if f else 0)
