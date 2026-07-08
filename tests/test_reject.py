"""
T-P0-1 验收测试：IOS拒绝机制
验证：reject()方法可触发；RejectionRecord数据完整。
"""
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from openllm.core.governance_engine import GovernanceEngine, RejectionRecord


def test_rejection_record_creation():
    """RejectionRecord数据模型创建正确"""
    record = RejectionRecord(
        instruction="rm -rf /",
        reason="破坏性操作",
        timestamp=time.time(),
        belief_confidence=0.95,
    )
    assert record.instruction == "rm -rf /"
    assert record.reason == "破坏性操作"
    assert record.belief_confidence == 0.95
    assert record.rejected_by == "IOS"
    assert 0 < record.timestamp <= time.time()
    print("✅ RejectionRecord创建正确")


def test_rejection_record_to_dict():
    """RejectionRecord序列化正确"""
    record = RejectionRecord(
        instruction="dangerous_action",
        reason="test",
        timestamp=1234567890.0,
        belief_confidence=0.8,
    )
    d = record.to_dict()
    assert d["instruction"] == "dangerous_action"
    assert d["reason"] == "test"
    assert d["timestamp"] == 1234567890.0
    assert d["belief_confidence"] == 0.8
    assert d["rejected_by"] == "IOS"
    print("✅ RejectionRecord序列化正确")


def test_governance_reject():
    """GovernanceEngine.reject()方法返回RejectionRecord"""
    engine = GovernanceEngine()
    record = engine.reject(
        instruction="shell: rm -rf /",
        reason="ISN标记为critical级别工具",
        belief_confidence=0.95,
    )
    assert isinstance(record, RejectionRecord)
    assert record.instruction == "shell: rm -rf /"
    assert record.reason == "ISN标记为critical级别工具"
    assert record.belief_confidence == 0.95
    print("✅ GovernanceEngine.reject()返回RejectionRecord")


def test_governance_reject_default_confidence():
    """reject()默认置信度为0.0"""
    engine = GovernanceEngine()
    record = engine.reject(
        instruction="test_action",
        reason="test reason",
    )
    assert record.belief_confidence == 0.0
    assert record.timestamp > 0
    print("✅ 默认置信度正确")


def test_reject_has_governance_engine():
    """确认GovernanceEngine类有reject方法"""
    assert hasattr(GovernanceEngine, 'reject')
    print("✅ GovernanceEngine有reject方法")


if __name__ == "__main__":
    test_rejection_record_creation()
    test_rejection_record_to_dict()
    test_governance_reject()
    test_governance_reject_default_confidence()
    test_reject_has_governance_engine()
    print("\n🎉 T-P0-1验收测试全部通过！IOS拒绝机制可触发。")
