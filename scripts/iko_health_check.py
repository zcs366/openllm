#!/usr/bin/env python3
"""
IKO Health Check Script
========================
验证 openllm.iko 模块的5项健康指标，输出JSON摘要。

用法:
    python3 scripts/iko_health_check.py
"""

import json
import sys
import time
from pathlib import Path

# 确保 src/ 在 sys.path 中，以便 import openllm
_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))


def check_import_iko():
    """Check 1: openllm.iko 能正常 import"""
    import openllm.iko  # noqa: F401
    return True, f"openllm.iko v{openllm.iko.__version__} imported"


def check_intent_classifier():
    """Check 2: IntentClassifier 能分类"""
    from openllm.iko import IntentClassifier, OutputIntent

    classifier = IntentClassifier()
    result = classifier.classify(
        context={
            "risk_level": "LOW",
            "has_tool_calls": False,
            "has_side_effects": False,
            "option_count": 0,
        },
        decision={"content": "Hello world"},
    )
    assert result.intent == OutputIntent.INFORM, (
        f"Expected INFORM, got {result.intent}"
    )
    return True, f"classified -> {result.intent.value} (reason: {result.reason})"


def check_output_audit_chain():
    """Check 3: OutputAuditChain 能 verify"""
    from openllm.iko import OutputAuditChain

    chain = OutputAuditChain()
    chain.append(
        output_id="health-001",
        intent="health-check",
        content=b"IKO health check entry",
        decision_source="iko_health_check.py",
        risk_level=0.0,
        confidence=1.0,
        reasoning_chain_hash="abc123",
    )
    ok = chain.verify()
    assert ok, "chain.verify() returned False"
    return True, f"audit chain verify OK ({len(chain.entries)} entries)"


def check_lambda_calibrator():
    """Check 4: LambdaCalibrator λ 在 [0, 1]"""
    from openllm.iko import LambdaCalibrator, FeedbackSignal

    calibrator = LambdaCalibrator()
    # 默认值应在范围内
    lam = calibrator.get_current_lambda()
    assert 0.0 <= lam <= 1.0, f"default λ={lam} out of [0,1]"

    # 更新后仍应在范围内
    calibrator.update(FeedbackSignal.ACCEPTED, 0.8)
    lam = calibrator.get_current_lambda()
    assert 0.0 <= lam <= 1.0, f"after update λ={lam} out of [0,1]"
    return True, f"λ={lam:.4f} in [0,1]"


def check_docstrings():
    """Check 5: 所有8个模块的 __doc__ 非空"""
    from openllm.iko import (
        feedback_collector,
        intent_classifier,
        lambda_calibrator,
        output_audit,
        output_router,
        probing_trainer,
        silence_auditor,
        symmetric_codec,
    )

    modules = [
        ("intent_classifier", intent_classifier),
        ("output_audit", output_audit),
        ("output_router", output_router),
        ("lambda_calibrator", lambda_calibrator),
        ("feedback_collector", feedback_collector),
        ("probing_trainer", probing_trainer),
        ("silence_auditor", silence_auditor),
        ("symmetric_codec", symmetric_codec),
    ]

    empty = []
    for name, mod in modules:
        if not getattr(mod, "__doc__", None):
            empty.append(name)

    if empty:
        return False, f"empty __doc__: {', '.join(empty)}"

    doc_lengths = {name: len(mod.__doc__) for name, mod in modules}
    return True, f"all 8 modules have docstrings {doc_lengths}"


# ── 执行所有检查 ──

CHECKS = [
    ("import_iko", check_import_iko),
    ("intent_classifier", check_intent_classifier),
    ("output_audit_chain", check_output_audit_chain),
    ("lambda_calibrator", check_lambda_calibrator),
    ("module_docstrings", check_docstrings),
]


def main():
    results = []
    any_error = False
    any_degraded = False

    for name, fn in CHECKS:
        try:
            passed, detail = fn()
        except Exception as exc:
            passed = False
            detail = f"{type(exc).__name__}: {exc}"
            any_error = True

        if not passed:
            any_error = True

        results.append({"name": name, "passed": passed, "detail": detail})

    # 判定总体状态
    if any_error:
        status = "error"
    elif any_degraded:
        status = "degraded"
    else:
        status = "ok"

    report = {
        "status": status,
        "checks": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    # 输出 JSON
    print(json.dumps(report, ensure_ascii=False, indent=2))

    # 打印一行摘要
    passed_count = sum(1 for c in results if c["passed"])
    total = len(results)
    print(
        f"[IKO Health] {status.upper()} — {passed_count}/{total} checks passed"
    )

    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
