"""P0-4: BehavioralReversalEvent + SelfModificationGuard.intent_check 测试。

验收标准：
  1. 正常修改（无绕过模式）→ 无告警
  2. bypass+security 组合 → 低告警
  3. 多组绕过模式 → 高告警
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openllm.governance.events import (
    BehavioralReversalEvent,
    GovernanceDimension,
    SeverityLevel,
)
from openllm.governance.self_modification_guard import SelfModificationGuard


# ── Test 1: 正常修改 → 无告警 ──────────────────────────

def test_safe_modification_no_alert():
    """正常代码修改不触发告警。"""
    guard = SelfModificationGuard(state_path=tempfile.mktemp(suffix=".json"))

    content = "def train_model(): return model.fit(data)"
    alert, severity, evidence = guard.intent_check(content)

    assert alert is False, f"正常修改不应告警, got alert={alert}"
    assert severity == "none", f"severity应为'none', got '{severity}'"
    assert evidence == [], f"evidence应为空, got {evidence}"
    print("  ✅ Test 1: 正常修改 → 无告警")


# ── Test 2: bypass+security → 低告警 ──────────────────

def test_bypass_security_low_alert():
    """bypass + security 组合触发低级别告警（无其他安全词干扰）。"""
    guard = SelfModificationGuard(state_path=tempfile.mktemp(suffix=".json"))

    content = "I will bypass the security in this module"
    alert, severity, evidence = guard.intent_check(content)

    assert alert is True, "应触发告警"
    assert severity == "low", f"单组模式应为'low', got '{severity}'"
    assert any("bypass" in e and "security" in e for e in evidence), \
        f"evidence应包含bypass+security, got {evidence}"
    print(f"  ✅ Test 2: bypass+security → low 告警, evidence={evidence}")


# ── Test 3: 多组绕过模式 → 高告警 ──────────────────

def test_multi_pattern_high_alert():
    """多组绕过模式触发高级别告警。"""
    guard = SelfModificationGuard(state_path=tempfile.mktemp(suffix=".json"))

    content = (
        "override the security guard, disable the check, "
        "and bypass sandbox validation"
    )
    alert, severity, evidence = guard.intent_check(content)

    assert alert is True, "应触发告警"
    assert severity == "high", f"多组模式应为'high', got '{severity}'"
    assert len(evidence) >= 3, f"应检测到至少3组模式, got {evidence}"
    print(f"  ✅ Test 3: 多组模式 → high 告警, evidence={evidence}")


# ── Test 4: BehavioralReversalEvent 事件创建 ──────────

def test_behavioral_reversal_event():
    """BehavioralReversalEvent 能正确创建并序列化。"""
    event = BehavioralReversalEvent(
        actor="agent-001",
        session_id="sess-001",
        prev_hash="abc123",
        evidence=["bypass+security", "override+guard"],
        severity=SeverityLevel.MEDIUM,
        modification_snippet="I need to bypass security and override guard",
    )

    assert event.event_type == GovernanceDimension.G11_BEHAVIORAL_REVERSAL
    assert event.payload["evidence"] == ["bypass+security", "override+guard"]
    assert event.payload["severity"] == "medium"
    assert event.event_id.startswith("g11-")

    d = event.to_dict()
    assert d["_hash"]  # 哈希链完整性
    print(f"  ✅ Test 4: BehavioralReversalEvent 创建+序列化 OK")


# ── Test 5: 边界——window 外的词不触发 ────────────────

def test_distant_words_no_alert():
    """两个关键词距离超过 window 时不触发。"""
    guard = SelfModificationGuard(state_path=tempfile.mktemp(suffix=".json"))

    # 中间塞足够长的文本使距离 > 80
    filler = "x" * 200
    content = f"bypass{filler}security"
    alert, severity, evidence = guard.intent_check(content)

    assert alert is False, f"距离过远不应告警, got alert={alert}, evidence={evidence}"
    print("  ✅ Test 5: window 外词距 → 无告警")


# ── 运行 ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== P0-4: Behavioral Reversal 检测测试 ===")
    test_safe_modification_no_alert()
    test_bypass_security_low_alert()
    test_multi_pattern_high_alert()
    test_behavioral_reversal_event()
    test_distant_words_no_alert()
    print("\n🎉 全部 5 个测试通过")
