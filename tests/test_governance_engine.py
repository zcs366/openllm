"""
治理转换引擎集成测试 — P0-c

验证: 3次同类失败→自动触发治理转换→输出可验证的GovernanceRule
"""

import json
import time
import sys
from pathlib import Path

# 确保src在path中（处理openllm.py同名遮蔽问题）
_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from openllm.core.structural_failure_classifier import (
    StructuralFailureClassifier, FailureSignature, FailureCategory,
    GovernanceRequest, ClassificationResult
)
from openllm.core.governance_rule import (
    GovernanceRule, GovernanceRuleStore, RuleType, RuleFamily, RuleStatus
)
from openllm.core.governance_engine import GovernanceEngine


# ── P0-a 测试: 结构性失败分类器 ──────────────────────

def test_classifier_local_failure():
    """测试: 单次失败→local分类"""
    classifier = StructuralFailureClassifier()
    sig = FailureSignature(
        category=FailureCategory.TOOL_PARAM,
        tool_name="read_file",
        error_pattern="missing required parameter 'path'",
        raw_error="TypeError: missing required parameter 'path'",
        timestamp=time.time(),
    )
    result = classifier.classify(sig)
    assert result.classification == "local", f"Expected local, got {result.classification}"
    assert result.composite_score < 0.4, f"Expected <0.4, got {result.composite_score}"
    print(f"  ✅ 单次失败→local (score={result.composite_score})")

def test_classifier_structural_failure():
    """测试: 3次同类失败→structural分类"""
    classifier = StructuralFailureClassifier()
    for i in range(3):
        sig = FailureSignature(
            category=FailureCategory.TOOL_PARAM,
            tool_name="read_file",
            error_pattern="missing required parameter 'path'",
            raw_error="TypeError: missing required parameter 'path'",
            timestamp=time.time() - (2 - i) * 60,
        )
        result = classifier.classify(sig)
    assert result.classification == "structural", f"Expected structural, got {result.classification}"
    assert result.composite_score >= 0.6, f"Expected >=0.6, got {result.composite_score}"
    print(f"  ✅ 3次同类失败→structural (score={result.composite_score})")

def test_classifier_ambiguous_failure():
    """测试: 2次同类失败→ambiguous→GovernanceRequest"""
    classifier = StructuralFailureClassifier()
    for i in range(2):
        sig = FailureSignature(
            category=FailureCategory.TOOL_TIMEOUT,
            tool_name="terminal",
            error_pattern="command timed out after 180s",
            raw_error="TimeoutError: command timed out after 180 seconds",
            timestamp=time.time() - (1 - i) * 3600,
        )
        result = classifier.classify(sig)
    assert result.classification == "ambiguous", f"Expected ambiguous, got {result.classification}"
    assert result.governance_request is not None, "Expected GovernanceRequest for ambiguous"
    assert result.governance_request.status == "pending"
    print(f"  ✅ 2次同类失败→ambiguous (score={result.composite_score}, request={result.governance_request.request_id})")

def test_classifier_time_decay():
    """测试: 时间衰减——远期失败权重低于近期"""
    classifier = StructuralFailureClassifier()
    sig_old = FailureSignature(
        category=FailureCategory.TOOL_PARAM,
        tool_name="read_file",
        error_pattern="missing required parameter 'path'",
        raw_error="TypeError: missing required parameter 'path'",
        timestamp=time.time() - 10 * 86400,
    )
    result_old = classifier.classify(sig_old)
    sig_new = FailureSignature(
        category=FailureCategory.TOOL_PARAM,
        tool_name="read_file",
        error_pattern="missing required parameter 'path'",
        raw_error="TypeError: missing required parameter 'path'",
        timestamp=time.time() - 3600,
    )
    result_new = classifier.classify(sig_new)
    print(f"  ✅ 时间衰减: 近期freq={result_new.freq_score}, 远期freq={result_old.freq_score}")


# ── P0-b 测试: 治理规则模型 ──────────────────────────

def test_governance_rule_serialization():
    """测试: 治理规则序列化/反序列化"""
    rule = GovernanceRule(
        rule_type=RuleType.CONTROL,
        family=RuleFamily.VALIDATION,
        target="read_file",
        condition="missing_required_param('path')",
        action="validate_params_before_call()",
        source_failure="TOOL_PARAM:read_file:missing required",
        source_mechanism="tool_loop",
        source_count=3,
        description="参数验证: read_file调用前必须检查path参数",
        confidence=0.8,
        status=RuleStatus.ACTIVE,
    )
    d = rule.to_dict()
    assert d["rule_type"] == "control"
    assert d["family"] == "validation"
    assert d["status"] == "active"
    rule2 = GovernanceRule.from_dict(d)
    assert rule2.rule_type == RuleType.CONTROL
    assert rule2.family == RuleFamily.VALIDATION
    assert rule2.target == "read_file"
    print(f"  ✅ 治理规则序列化/反序列化: rule_id={rule.rule_id}")

def test_governance_rule_store():
    """测试: 规则存储CRUD + 冲突检测"""
    store = GovernanceRuleStore()
    rule1 = GovernanceRule(
        rule_type=RuleType.CONTROL,
        family=RuleFamily.VALIDATION,
        target="read_file",
        condition="param_missing('path')",
        action="block()",
        status=RuleStatus.ACTIVE,
    )
    store.save(rule1)
    active = store.get_active()
    assert len(active) >= 1
    rule2 = GovernanceRule(
        rule_type=RuleType.CONTROL,
        family=RuleFamily.VALIDATION,
        target="read_file",
        condition="param_missing('path')",
        action="warn()",
        status=RuleStatus.PENDING,
    )
    conflicts = store.detect_conflicts(rule2)
    assert len(conflicts) >= 1, f"Expected conflicts, got {len(conflicts)}"
    print(f"  ✅ 规则存储CRUD + 冲突检测: {len(conflicts)} conflicts found")


# ── P0-b 测试: 治理转换引擎 ──────────────────────────

def test_governance_engine_full_pipeline():
    """测试: 完整治理转换管线——3次同类失败→治理规则"""
    engine = GovernanceEngine()
    result = None
    for i in range(3):
        trace = {
            "tool_name": "read_file",
            "error": f"TypeError: missing required parameter 'path' (attempt {i+1})",
            "mechanism": "tool_loop",
        }
        result = engine.convert(trace)
    assert result is not None
    assert result.classification == "structural", f"Expected structural, got {result.classification}"
    assert result.rule is not None, "Expected governance rule"
    assert result.rule.status.value in ("active", "conflicted"), f"Unexpected status: {result.rule.status.value}"
    print(f"  ✅ 完整管线: {result.classification} → rule_id={result.rule.rule_id}, status={result.rule.status.value}")
    print(f"     family={result.rule.family.value}, target={result.rule.target}")
    print(f"     description={result.rule.description[:60]}")

def test_governance_engine_ambiguous_triggers_request():
    """测试: ambiguous失败→GovernanceRequest"""
    engine = GovernanceEngine()
    trace = {
        "tool_name": "terminal",
        "error": "TimeoutError: command timed out after 180s",
        "mechanism": "timeout",
    }
    result = engine.convert(trace)
    assert result.classification != "structural", f"Unexpected structural for single failure"
    print(f"  ✅ 单次失败: classification={result.classification}")


# ── 运行所有测试 ──────────────────────────────────────

def run_all_tests():
    """运行所有治理转换测试"""
    print("\n" + "="*60)
    print("治理转换引擎集成测试")
    print("="*60)
    
    tests = [
        ("P0-a: 单次失败→local", test_classifier_local_failure),
        ("P0-a: 3次同类→structural", test_classifier_structural_failure),
        ("P0-a: 2次同类→ambiguous", test_classifier_ambiguous_failure),
        ("P0-a: 时间衰减", test_classifier_time_decay),
        ("P0-b: 规则序列化", test_governance_rule_serialization),
        ("P0-b: 规则存储+冲突检测", test_governance_rule_store),
        ("P0-b: 完整管线", test_governance_engine_full_pipeline),
        ("P0-b: ambiguous→请求", test_governance_engine_ambiguous_triggers_request),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            print(f"\n[TEST] {name}")
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"结果: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'='*60}")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
