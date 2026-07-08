"""
Tests for Belief Provenance — ISA信念溯源测试
============================================

13个测试用例覆盖：
1. 基本溯源记录
2. 6种BeliefChangeType
3. 溯源链连续性
4. 篡改检测
5. on_rejection回调
6. 链完整性验证
7. 多信念追踪
8. 查询过滤
9. 统计功能
10. 信念内容hash
11. 防篡改检测
12. 不可变性
13. 100条记录性能
"""

import time
import pytest
from openllm.isa.provenance import (
    BeliefProvenance,
    ProvenanceEntry,
    BeliefChangeType,
)


class TestBeliefProvenance:
    """信念溯源引擎测试。"""

    def test_basic_record(self):
        """T1: 基本溯源记录——写入信念并附加溯源。"""
        prov = BeliefProvenance()
        entry = prov.record(
            belief_id="b1",
            change_type=BeliefChangeType.NEW,
            source="IOS",
            content="用户偏好简洁回答",
            trigger_context={"session_id": "s1"},
        )
        assert entry.belief_id == "b1"
        assert entry.change_type == BeliefChangeType.NEW
        assert entry.signature != ""

    def test_six_change_types(self):
        """T2: 6种BeliefChangeType——全部可用。"""
        prov = BeliefProvenance()
        types = list(BeliefChangeType)
        assert len(types) == 6

        for ct in types:
            prov.record(
                belief_id=f"b-{ct.value}",
                change_type=ct,
                source="IOS",
                content=f"content-{ct.value}",
            )

        stats = prov.get_statistics()
        assert stats["total_entries"] == 6

    def test_chain_continuity(self):
        """T3: 溯源链连续性——previous_hash指向前一条。"""
        prov = BeliefProvenance()
        e1 = prov.record(belief_id="b1", change_type=BeliefChangeType.NEW, source="IOS", content="c1")
        e2 = prov.record(belief_id="b1", change_type=BeliefChangeType.UPDATE, source="IOS", content="c2")

        assert e2.previous_hash == e1.compute_hash()

    def test_tamper_detection(self):
        """T4: 篡改检测——修改内容后hash不匹配。"""
        prov = BeliefProvenance()
        prov.record(belief_id="b1", change_type=BeliefChangeType.NEW, source="IOS", content="original")

        # 未篡改
        assert prov.detect_tampering("b1", "original") is False

        # 篡改
        assert prov.detect_tampering("b1", "tampered") is True

    def test_on_rejection_callback(self):
        """T5: on_rejection回调——拒绝权联动自动记录。"""
        prov = BeliefProvenance()
        entry = prov.on_rejection(
            rejection_record_id="rej-001",
            belief_id="b-harmful",
            source="IOS",
            content="拒绝执行删除用户数据",
            reason="HARMFUL",
        )

        assert entry.change_type == BeliefChangeType.REJECT
        assert entry.trigger_context["rejection_record_id"] == "rej-001"

    def test_chain_integrity(self):
        """T6: 链完整性验证——10条记录全部通过。"""
        prov = BeliefProvenance()
        for i in range(10):
            prov.record(
                belief_id=f"b{i}",
                change_type=BeliefChangeType.NEW,
                source="IOS",
                content=f"content-{i}",
            )
        assert prov.verify_chain() is True

    def test_multi_belief_tracking(self):
        """T7: 多信念追踪——3个信念各自独立。"""
        prov = BeliefProvenance()
        prov.record(belief_id="b1", change_type=BeliefChangeType.NEW, source="IOS", content="c1")
        prov.record(belief_id="b2", change_type=BeliefChangeType.NEW, source="ISA", content="c2")
        prov.record(belief_id="b3", change_type=BeliefChangeType.NEW, source="IAI", content="c3")

        stats = prov.get_statistics()
        assert stats["unique_beliefs"] == 3

    def test_query_filter(self):
        """T8: 查询过滤——按belief_id和change_type。"""
        prov = BeliefProvenance()
        prov.record(belief_id="b1", change_type=BeliefChangeType.NEW, source="IOS", content="c1")
        prov.record(belief_id="b1", change_type=BeliefChangeType.UPDATE, source="IOS", content="c2")
        prov.record(belief_id="b2", change_type=BeliefChangeType.NEW, source="ISA", content="c3")

        # 按belief_id
        b1_entries = prov.get_entries(belief_id="b1")
        assert len(b1_entries) == 2

        # 按change_type
        new_entries = prov.get_entries(change_type=BeliefChangeType.NEW)
        assert len(new_entries) == 2

    def test_statistics(self):
        """T9: 统计功能。"""
        prov = BeliefProvenance()
        prov.record(belief_id="b1", change_type=BeliefChangeType.NEW, source="IOS", content="c1")
        prov.record(belief_id="b1", change_type=BeliefChangeType.UPDATE, source="IOS", content="c2")

        stats = prov.get_statistics()
        assert stats["total_entries"] == 2
        assert stats["chain_valid"] is True
        assert stats["unique_beliefs"] == 1

    def test_content_hash(self):
        """T10: 信念内容hash——不同内容产生不同hash。"""
        prov = BeliefProvenance()
        e1 = prov.record(belief_id="b1", change_type=BeliefChangeType.NEW, source="IOS", content="content-A")
        e2 = prov.record(belief_id="b2", change_type=BeliefChangeType.NEW, source="IOS", content="content-B")

        assert e1.content_hash != e2.content_hash

    def test_anti_tamper_comprehensive(self):
        """T11: 综合防篡改检测——链+内容双重验证。"""
        prov = BeliefProvenance()
        prov.record(belief_id="b1", change_type=BeliefChangeType.NEW, source="IOS", content="original")

        # 链完整
        assert prov.verify_chain() is True

        # 内容未篡改
        assert prov.detect_tampering("b1", "original") is False

        # 内容已篡改
        assert prov.detect_tampering("b1", "modified") is True

    def test_entry_immutability(self):
        """T12: 溯源条目不可变性。"""
        entry = ProvenanceEntry(
            entry_id="test-001",
            timestamp=time.time(),
            belief_id="b1",
            change_type=BeliefChangeType.NEW,
            source="IOS",
            trigger_context={},
            previous_hash="genesis",
            content_hash="abc",
        )
        with pytest.raises(AttributeError):
            entry.belief_id = "b2"

    def test_100_entries_performance(self):
        """T13: 100条记录性能——链完整性验证。"""
        prov = BeliefProvenance()
        for i in range(100):
            prov.record(
                belief_id=f"b{i % 10}",
                change_type=BeliefChangeType.NEW,
                source="IOS",
                content=f"content-{i}",
            )

        assert prov.verify_chain() is True
        stats = prov.get_statistics()
        assert stats["total_entries"] == 100
        assert stats["unique_beliefs"] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
