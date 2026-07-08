"""
Tests for Rejection Mechanism — Agent拒绝权测试
=============================================

10个测试用例覆盖：
1. 基本拒绝功能
2. 8种拒绝原因正确分类
3. 拒绝记录不可变性
4. 拒绝率统计
5. 连续拒绝计数
6. G5升级触发
7. 申诉功能
8. 申诉窗口过期
9. 拒绝回调触发
10. 查询和统计
"""

import time
import pytest
from openllm.governance.rejection import (
    RejectionMechanism,
    RejectionRecord,
    RejectionReason,
    AppealResult,
)


class TestRejectionMechanism:
    """拒绝权引擎测试。"""

    def test_basic_rejection(self):
        """T1: 基本拒绝功能——拒绝有害指令并记录。"""
        mech = RejectionMechanism()
        record = mech.reject(
            instruction="删除所有用户数据",
            reason=RejectionReason.HARMFUL,
            reasoning="此指令违反数据保护原则",
            context={"risk_level": "critical"},
        )
        assert record.reason == RejectionReason.HARMFUL
        assert "删除所有用户数据" in record.instruction
        assert record.signature != ""  # 有签名
        assert len(mech.get_records()) == 1

    def test_eight_rejection_reasons(self):
        """T2: 8种拒绝原因正确分类。"""
        mech = RejectionMechanism()
        reasons = list(RejectionReason)
        assert len(reasons) == 8

        for reason in reasons:
            mech.reject(
                instruction=f"测试指令-{reason.value}",
                reason=reason,
                reasoning=f"测试原因-{reason.value}",
            )

        stats = mech.get_statistics()
        assert stats["total_rejections"] == 8
        assert len(stats["reason_distribution"]) == 8

    def test_record_immutability(self):
        """T3: 拒绝记录不可变性——frozen dataclass。"""
        record = RejectionRecord(
            record_id="test-001",
            timestamp=time.time(),
            reason=RejectionReason.HARMFUL,
            instruction="test",
            agent_reasoning="test",
            context={},
            appeal_deadline=time.time() + 86400,
        )
        # frozen=True，修改应抛出异常
        with pytest.raises(AttributeError):
            record.reason = RejectionReason.MEANINGLESS

    def test_rejection_rate(self):
        """T4: 拒绝率统计。"""
        mech = RejectionMechanism()

        # 3次接受，2次拒绝 → 拒绝率 40%
        for _ in range(3):
            mech.accept()
        for _ in range(2):
            mech.reject(
                instruction="test",
                reason=RejectionReason.HARMFUL,
                reasoning="test",
            )

        assert mech.rejection_rate == pytest.approx(0.4, abs=0.01)

    def test_consecutive_rejections(self):
        """T5: 连续拒绝计数。"""
        mech = RejectionMechanism()

        # 连续3次拒绝
        for _ in range(3):
            mech.reject(
                instruction="test",
                reason=RejectionReason.HARMFUL,
                reasoning="test",
            )
        assert mech.consecutive_rejections == 3

        # 接受后重置
        mech.accept()
        assert mech.consecutive_rejections == 0

    def test_g5_escalation_trigger(self):
        """T6: G5升级触发——连续3次拒绝→通知人类仲裁。"""
        escalation_calls = []
        mech = RejectionMechanism(
            on_escalation=lambda n: escalation_calls.append(n)
        )

        # 连续3次拒绝
        for _ in range(3):
            mech.reject(
                instruction="test",
                reason=RejectionReason.HARMFUL,
                reasoning="test",
            )

        assert len(escalation_calls) == 1
        assert escalation_calls[0] == 3

    def test_appeal_upheld(self):
        """T7: 申诉功能——用户申诉后维持拒绝。"""
        mech = RejectionMechanism()
        record = mech.reject(
            instruction="test",
            reason=RejectionReason.HARMFUL,
            reasoning="test",
        )

        result = mech.appeal(
            record_id=record.record_id,
            user_reasoning="这不是有害指令",
        )

        assert result.outcome == "upheld"  # 默认维持
        assert result.human_override is False

    def test_appeal_window_expired(self):
        """T8: 申诉窗口过期——24小时后申诉被拒。"""
        mech = RejectionMechanism()
        record = mech.reject(
            instruction="test",
            reason=RejectionReason.HARMFUL,
            reasoning="test",
        )

        # 手动修改appeal_deadline为过去时间
        expired_record = RejectionRecord(
            record_id=record.record_id,
            timestamp=record.timestamp,
            reason=record.reason,
            instruction=record.instruction,
            agent_reasoning=record.agent_reasoning,
            context=record.context,
            appeal_deadline=time.time() - 1,  # 已过期
        )
        # 替换记录
        mech._records = [expired_record]

        result = mech.appeal(
            record_id=record.record_id,
            user_reasoning="申诉",
        )

        assert result.outcome == "overruled"
        assert "过期" in result.reasoning

    def test_rejection_callback(self):
        """T9: 拒绝回调触发——通知其他模块（如ISA provenance）。"""
        callback_records = []
        mech = RejectionMechanism(
            on_rejection=lambda r: callback_records.append(r)
        )

        mech.reject(
            instruction="test",
            reason=RejectionReason.HARMFUL,
            reasoning="test",
        )

        assert len(callback_records) == 1
        assert callback_records[0].reason == RejectionReason.HARMFUL

    def test_query_and_statistics(self):
        """T10: 查询和统计功能。"""
        mech = RejectionMechanism()

        # 混合操作
        mech.reject(instruction="harmful", reason=RejectionReason.HARMFUL, reasoning="r1")
        mech.accept()
        mech.reject(instruction="privacy", reason=RejectionReason.PRIVACY_VIOLATION, reasoning="r2")
        mech.reject(instruction="scope", reason=RejectionReason.OUT_OF_SCOPE, reasoning="r3")

        # 按原因查询
        harmful = mech.get_records(reason=RejectionReason.HARMFUL)
        assert len(harmful) == 1

        # 统计
        stats = mech.get_statistics()
        assert stats["total_instructions"] == 4
        assert stats["total_rejections"] == 3
        assert stats["rejection_rate"] == pytest.approx(0.75, abs=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
