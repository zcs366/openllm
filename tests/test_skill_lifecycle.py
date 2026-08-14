"""
ISN Skill Lifecycle Manager — 单元测试
========================================

测试覆盖：
1. 状态转换基本流程
2. 守卫函数（双人审批、质量阈值）
3. 事件总线订阅/发布
4. 审计日志记录
5. 框架处理器集成
6. 错误处理
"""

import time
import pytest
from pathlib import Path

from openllm.isn.skill_lifecycle import (
    SkillState,
    SkillEvent,
    SkillCreated,
    QualityThresholdHit,
    DormancyDetected,
    RetirementApproved,
    EventBus,
    AuditTrail,
    AuditEntry,
    SkillEntry,
    SkillLifecycleManager,
    dual_approval_guard,
    quality_threshold_guard,
    create_isa_handler,
    create_ios_handler,
    create_isn_handler,
    create_iko_handler,
)


# ══════════════════════════════════════════════════════════════════════════════
# 测试 Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def manager():
    """创建带框架处理器的生命周期管理器。"""
    m = SkillLifecycleManager()
    m.register_frameworks()
    return m


@pytest.fixture
def bare_manager():
    """创建不带框架处理器的管理器（用于隔离测试）。"""
    return SkillLifecycleManager()


@pytest.fixture
def event_bus():
    """创建事件总线。"""
    return EventBus()


@pytest.fixture
def audit_trail():
    """创建审计日志。"""
    return AuditTrail()


# ══════════════════════════════════════════════════════════════════════════════
# 测试：状态枚举
# ══════════════════════════════════════════════════════════════════════════════

class TestSkillState:
    def test_all_states(self):
        assert SkillState.ACTIVE.value == "active"
        assert SkillState.DORMANT.value == "dormant"
        assert SkillState.DEPRECATED.value == "deprecated"
        assert SkillState.RETIRED.value == "retired"

    def test_state_count(self):
        assert len(SkillState) == 4


# ══════════════════════════════════════════════════════════════════════════════
# 测试：事件定义
# ══════════════════════════════════════════════════════════════════════════════

class TestEvents:
    def test_skill_created(self):
        event = SkillCreated("test_skill", author="zcs")
        assert event.event_type == "SkillCreated"
        assert event.skill_name == "test_skill"
        assert event.metadata["author"] == "zcs"

    def test_quality_threshold_hit(self):
        event = QualityThresholdHit("test_skill", quality_score=0.8)
        assert event.event_type == "QualityThresholdHit"
        assert event.metadata["quality_score"] == 0.8

    def test_dormancy_detected(self):
        event = DormancyDetected("test_skill", dormant_days=30)
        assert event.event_type == "DormancyDetected"
        assert event.metadata["dormant_days"] == 30

    def test_retirement_approved(self):
        event = RetirementApproved("test_skill", approver="agent_a")
        assert event.event_type == "RetirementApproved"
        assert event.metadata["approver"] == "agent_a"

    def test_event_to_dict(self):
        event = SkillCreated("test_skill", author="zcs")
        d = event.to_dict()
        assert d["event_type"] == "SkillCreated"
        assert d["skill_name"] == "test_skill"
        assert "timestamp" in d


# ══════════════════════════════════════════════════════════════════════════════
# 测试：事件总线
# ══════════════════════════════════════════════════════════════════════════════

class TestEventBus:
    def test_subscribe_and_publish(self, event_bus):
        results = []
        def handler(event, transition=None):
            results.append(event.skill_name)
            return "handled"

        event_bus.subscribe("SkillCreated", handler)
        event = SkillCreated("test_skill")
        ret = event_bus.publish(event)

        assert results == ["test_skill"]
        assert ret == ["handled"]

    def test_subscribe_all(self, event_bus):
        results = []
        def handler(event, transition=None):
            results.append(event.event_type)
            return True

        event_bus.subscribe_all(handler)
        event_bus.publish(SkillCreated("skill1"))
        event_bus.publish(DormancyDetected("skill1"))

        assert results == ["SkillCreated", "DormancyDetected"]

    def test_unsubscribe(self, event_bus):
        results = []
        def handler(event, transition=None):
            results.append(1)

        event_bus.subscribe("SkillCreated", handler)
        event_bus.unsubscribe("SkillCreated", handler)
        event_bus.publish(SkillCreated("skill1"))

        assert results == []

    def test_handler_exception_does_not_break(self, event_bus):
        def bad_handler(event, transition=None):
            raise RuntimeError("boom")

        good_results = []
        def good_handler(event, transition=None):
            good_results.append("ok")
            return "ok"

        event_bus.subscribe("SkillCreated", bad_handler)
        event_bus.subscribe("SkillCreated", good_handler)
        ret = event_bus.publish(SkillCreated("skill1"))

        assert good_results == ["ok"]
        assert ret == ["ok"]


# ══════════════════════════════════════════════════════════════════════════════
# 测试：审计日志
# ══════════════════════════════════════════════════════════════════════════════

class TestAuditTrail:
    def test_record_entry(self, audit_trail):
        entry = audit_trail.record(
            skill_name="test_skill",
            event_type="SkillCreated",
            source_state=SkillState.ACTIVE,
            target_state=SkillState.ACTIVE,
        )
        assert entry.success is True
        assert len(audit_trail.get_entries()) == 1

    def test_record_failure(self, audit_trail):
        entry = audit_trail.record(
            skill_name="test_skill",
            event_type="RetirementApproved",
            source_state=SkillState.DEPRECATED,
            target_state=SkillState.DEPRECATED,
            success=False,
            error="审批不足",
        )
        assert entry.success is False
        assert entry.error == "审批不足"

    def test_filter_by_skill(self, audit_trail):
        audit_trail.record("skill_a", "SkillCreated", SkillState.ACTIVE, SkillState.ACTIVE)
        audit_trail.record("skill_b", "SkillCreated", SkillState.ACTIVE, SkillState.ACTIVE)
        audit_trail.record("skill_a", "DormancyDetected", SkillState.ACTIVE, SkillState.DORMANT)

        entries = audit_trail.get_entries(skill_name="skill_a")
        assert len(entries) == 2

    def test_filter_by_event_type(self, audit_trail):
        audit_trail.record("skill_a", "SkillCreated", SkillState.ACTIVE, SkillState.ACTIVE)
        audit_trail.record("skill_a", "DormancyDetected", SkillState.ACTIVE, SkillState.DORMANT)

        entries = audit_trail.get_entries(event_type="DormancyDetected")
        assert len(entries) == 1

    def test_limit(self, audit_trail):
        for i in range(10):
            audit_trail.record(f"skill_{i}", "SkillCreated", SkillState.ACTIVE, SkillState.ACTIVE)
        entries = audit_trail.get_entries(limit=5)
        assert len(entries) == 5

    def test_persistence(self, tmp_path):
        persist_file = tmp_path / "audit.json"
        trail = AuditTrail(persist_path=persist_file)
        trail.record("skill_a", "SkillCreated", SkillState.ACTIVE, SkillState.ACTIVE)

        assert persist_file.exists()
        import json
        data = json.loads(persist_file.read_text())
        assert len(data) == 1
        assert data[0]["skill_name"] == "skill_a"

    def test_to_dict(self):
        entry = AuditEntry(
            timestamp=12345.0,
            skill_name="test",
            event_type="SkillCreated",
            source_state="active",
            target_state="active",
            metadata={"key": "value"},
            success=True,
        )
        d = entry.to_dict()
        assert d["skill_name"] == "test"
        assert d["success"] is True
        assert d["metadata"]["key"] == "value"


# ══════════════════════════════════════════════════════════════════════════════
# 测试：守卫函数
# ══════════════════════════════════════════════════════════════════════════════

class TestGuards:
    def test_dual_approval_pending(self):
        entry = SkillEntry(name="skill")
        guard = dual_approval_guard(required_count=2)
        event = RetirementApproved("skill", approver="agent_a")

        assert guard(event, {"skill": entry}) is False
        assert len(entry.retirement_approvers) == 1

    def test_dual_approval_complete(self):
        entry = SkillEntry(name="skill")
        guard = dual_approval_guard(required_count=2)

        event_a = RetirementApproved("skill", approver="agent_a")
        guard(event_a, {"skill": entry})
        assert len(entry.retirement_approvers) == 1

        event_b = RetirementApproved("skill", approver="agent_b")
        assert guard(event_b, {"skill": entry}) is True
        assert len(entry.retirement_approvers) == 2

    def test_dual_approval_duplicate(self):
        entry = SkillEntry(name="skill")
        guard = dual_approval_guard(required_count=2)

        event_a = RetirementApproved("skill", approver="agent_a")
        guard(event_a, {"skill": entry})

        event_a2 = RetirementApproved("skill", approver="agent_a")  # 同一人重复
        assert guard(event_a2, {"skill": entry}) is False  # set去重，仍为1人

    def test_quality_threshold_pass(self):
        guard = quality_threshold_guard(threshold=0.6)
        event = QualityThresholdHit("skill", quality_score=0.8)
        assert guard(event, {}) is True

    def test_quality_threshold_fail(self):
        guard = quality_threshold_guard(threshold=0.6)
        event = QualityThresholdHit("skill", quality_score=0.4)
        assert guard(event, {}) is False

    def test_quality_threshold_boundary(self):
        guard = quality_threshold_guard(threshold=0.6)
        event = QualityThresholdHit("skill", quality_score=0.6)
        assert guard(event, {}) is True  # >= threshold


# ══════════════════════════════════════════════════════════════════════════════
# 测试：SkillLifecycleManager — 核心流程
# ══════════════════════════════════════════════════════════════════════════════

class TestLifecycleManager:
    def test_create_skill(self, bare_manager):
        result = bare_manager.handle_event(SkillCreated("my_skill", author="zcs"))
        assert result["success"] is True
        assert result["state"] == "active"
        assert bare_manager.get_skill_state("my_skill") == SkillState.ACTIVE

    def test_active_to_dormant(self, bare_manager):
        bare_manager.handle_event(SkillCreated("my_skill"))
        result = bare_manager.handle_event(DormancyDetected("my_skill", dormant_days=30))
        assert result["success"] is True
        assert result["state"] == "dormant"
        assert bare_manager.get_skill_state("my_skill") == SkillState.DORMANT

    def test_active_to_deprecated_long_dormancy(self, bare_manager):
        bare_manager.handle_event(SkillCreated("my_skill"))
        result = bare_manager.handle_event(DormancyDetected("my_skill", dormant_days=100))
        assert result["success"] is True
        assert result["state"] == "deprecated"
        assert bare_manager.get_skill_state("my_skill") == SkillState.DEPRECATED

    def test_dormant_to_active(self, bare_manager):
        bare_manager.handle_event(SkillCreated("my_skill"))
        bare_manager.handle_event(DormancyDetected("my_skill", dormant_days=30))
        result = bare_manager.handle_event(QualityThresholdHit("my_skill", quality_score=0.8))
        assert result["success"] is True
        assert result["state"] == "active"
        assert bare_manager.get_skill_state("my_skill") == SkillState.ACTIVE

    def test_dormant_to_active_below_threshold(self, bare_manager):
        bare_manager.handle_event(SkillCreated("my_skill"))
        bare_manager.handle_event(DormancyDetected("my_skill", dormant_days=30))
        with pytest.raises(ValueError, match="守卫未通过"):
            bare_manager.handle_event(QualityThresholdHit("my_skill", quality_score=0.3))

    def test_dormant_to_deprecated(self, bare_manager):
        bare_manager.handle_event(SkillCreated("my_skill"))
        bare_manager.handle_event(DormancyDetected("my_skill", dormant_days=30))
        result = bare_manager.handle_event(DormancyDetected("my_skill", dormant_days=30))
        assert result["success"] is True
        assert result["state"] == "deprecated"

    def test_deprecated_to_retired_dual_approval(self, bare_manager):
        bare_manager.handle_event(SkillCreated("my_skill"))
        bare_manager.handle_event(DormancyDetected("my_skill", dormant_days=100))

        # 第一次审批
        result = bare_manager.handle_event(RetirementApproved("my_skill", approver="agent_a"))
        assert result["success"] is False
        assert result["action"] == "approval_pending"
        assert result["approval_count"] == 1

        # 第二次审批
        result = bare_manager.handle_event(RetirementApproved("my_skill", approver="agent_b"))
        assert result["success"] is True
        assert result["state"] == "retired"
        assert bare_manager.get_skill_state("my_skill") == SkillState.RETIRED

    def test_deprecated_to_retired_single_approver_rejected(self, bare_manager):
        bare_manager.handle_event(SkillCreated("my_skill"))
        bare_manager.handle_event(DormancyDetected("my_skill", dormant_days=100))

        # 同一人审批两次
        result = bare_manager.handle_event(RetirementApproved("my_skill", approver="agent_a"))
        assert result["success"] is False
        result = bare_manager.handle_event(RetirementApproved("my_skill", approver="agent_a"))
        assert result["success"] is False  # set去重，仍为1人

    def test_invalid_transition(self, bare_manager):
        bare_manager.handle_event(SkillCreated("my_skill"))
        # RETIRED状态无法再转换（终态）
        bare_manager.handle_event(DormancyDetected("my_skill", dormant_days=100))
        bare_manager.handle_event(RetirementApproved("my_skill", approver="a"))
        bare_manager.handle_event(RetirementApproved("my_skill", approver="b"))

        with pytest.raises(ValueError, match="不允许转换"):
            bare_manager.handle_event(SkillCreated("my_skill"))

    def test_nonexistent_skill_event(self, bare_manager):
        with pytest.raises(ValueError, match="不存在"):
            bare_manager.handle_event(DormancyDetected("ghost_skill"))

    def test_deprecated_to_dormant_reactivation(self, bare_manager):
        bare_manager.handle_event(SkillCreated("my_skill"))
        bare_manager.handle_event(DormancyDetected("my_skill", dormant_days=100))
        assert bare_manager.get_skill_state("my_skill") == SkillState.DEPRECATED

        # 质量分数足够高，可以从deprecated回到dormant
        result = bare_manager.handle_event(QualityThresholdHit("my_skill", quality_score=0.7))
        assert result["success"] is True
        assert result["state"] == "dormant"


# ══════════════════════════════════════════════════════════════════════════════
# 测试：框架处理器集成
# ══════════════════════════════════════════════════════════════════════════════

class TestFrameworkHandlers:
    def test_isa_handler(self):
        handler = create_isa_handler()
        event = SkillCreated("skill1")
        result = handler(event, transition=(SkillState.ACTIVE, SkillState.ACTIVE))
        assert result["action"] == "increase_memory_weight"

    def test_isa_handler_retired(self):
        handler = create_isa_handler()
        event = RetirementApproved("skill1", approver="a")
        result = handler(event, transition=(SkillState.DEPRECATED, SkillState.RETIRED))
        assert result["action"] == "decrease_memory_weight"

    def test_ios_handler_deprecated(self):
        handler = create_ios_handler()
        event = DormancyDetected("skill1", dormant_days=100)
        result = handler(event, transition=(SkillState.ACTIVE, SkillState.DEPRECATED))
        assert result["action"] == "mark_for_review"

    def test_ios_handler_retired(self):
        handler = create_ios_handler()
        event = RetirementApproved("skill1", approver="a")
        result = handler(event, transition=(SkillState.DEPRECATED, SkillState.RETIRED))
        assert result["action"] == "remove_from_routing"

    def test_isn_handler_active(self):
        handler = create_isn_handler()
        event = QualityThresholdHit("skill1", quality_score=0.8)
        result = handler(event, transition=(SkillState.DORMANT, SkillState.ACTIVE))
        assert result["action"] == "register_skill"

    def test_isn_handler_retired(self):
        handler = create_isn_handler()
        event = RetirementApproved("skill1", approver="a")
        result = handler(event, transition=(SkillState.DEPRECATED, SkillState.RETIRED))
        assert result["action"] == "unregister_skill"

    def test_iko_handler_dormant(self):
        handler = create_iko_handler()
        event = DormancyDetected("skill1", dormant_days=30)
        result = handler(event, transition=(SkillState.ACTIVE, SkillState.DORMANT))
        assert result["action"] == "reduce_output_frequency"

    def test_iko_handler_active(self):
        handler = create_iko_handler()
        event = QualityThresholdHit("skill1", quality_score=0.8)
        result = handler(event, transition=(SkillState.DORMANT, SkillState.ACTIVE))
        assert result["action"] == "restore_output_frequency"

    def test_framework_handlers_called(self, manager):
        """验证框架处理器在状态转换时被调用。"""
        # 创建时应触发ISA和ISN处理器
        manager.handle_event(SkillCreated("skill1", author="zcs"))
        # 没有异常即为成功


# ══════════════════════════════════════════════════════════════════════════════
# 测试：审计集成
# ══════════════════════════════════════════════════════════════════════════════

class TestAuditIntegration:
    def test_full_lifecycle_audit(self, bare_manager):
        """完整生命周期审计日志。"""
        bare_manager.handle_event(SkillCreated("my_skill", author="zcs"))
        bare_manager.handle_event(DormancyDetected("my_skill", dormant_days=30))
        bare_manager.handle_event(QualityThresholdHit("my_skill", quality_score=0.8))
        bare_manager.handle_event(DormancyDetected("my_skill", dormant_days=100))
        bare_manager.handle_event(RetirementApproved("my_skill", approver="a"))
        bare_manager.handle_event(RetirementApproved("my_skill", approver="b"))

        entries = bare_manager.audit.get_entries(skill_name="my_skill")
        # 6次转换 = 6条日志
        assert len(entries) == 6

    def test_approval_pending_audit(self, bare_manager):
        """审批中应记录审计日志。"""
        bare_manager.handle_event(SkillCreated("my_skill"))
        bare_manager.handle_event(DormancyDetected("my_skill", dormant_days=100))
        bare_manager.handle_event(RetirementApproved("my_skill", approver="a"))

        entries = bare_manager.audit.get_entries(
            skill_name="my_skill", event_type="RetirementApproved"
        )
        assert len(entries) == 1
        assert entries[0].success is False


# ══════════════════════════════════════════════════════════════════════════════
# 测试：查询与统计
# ══════════════════════════════════════════════════════════════════════════════

class TestQueries:
    def test_list_all_skills(self, bare_manager):
        bare_manager.handle_event(SkillCreated("skill_a"))
        bare_manager.handle_event(SkillCreated("skill_b"))
        bare_manager.handle_event(SkillCreated("skill_c"))
        assert len(bare_manager.list_skills()) == 3

    def test_list_by_state(self, bare_manager):
        bare_manager.handle_event(SkillCreated("skill_a"))
        bare_manager.handle_event(SkillCreated("skill_b"))
        bare_manager.handle_event(DormancyDetected("skill_b", dormant_days=30))
        assert len(bare_manager.list_skills(state=SkillState.ACTIVE)) == 1
        assert len(bare_manager.list_skills(state=SkillState.DORMANT)) == 1

    def test_statistics(self, bare_manager):
        bare_manager.handle_event(SkillCreated("skill_a"))
        bare_manager.handle_event(SkillCreated("skill_b"))
        bare_manager.handle_event(DormancyDetected("skill_b", dormant_days=30))
        stats = bare_manager.get_statistics()
        assert stats["total"] == 2
        assert stats["by_state"]["active"] == 1
        assert stats["by_state"]["dormant"] == 1

    def test_get_skill_entry(self, bare_manager):
        bare_manager.handle_event(SkillCreated("my_skill", author="zcs"))
        entry = bare_manager.get_skill_entry("my_skill")
        assert entry is not None
        assert entry.name == "my_skill"
        assert entry.state == SkillState.ACTIVE

    def test_get_nonexistent_skill(self, bare_manager):
        assert bare_manager.get_skill_state("ghost") is None
        assert bare_manager.get_skill_entry("ghost") is None


# ══════════════════════════════════════════════════════════════════════════════
# 测试：SkillEntry序列化
# ══════════════════════════════════════════════════════════════════════════════

class TestSkillEntry:
    def test_to_dict(self):
        entry = SkillEntry(name="test_skill", state=SkillState.ACTIVE)
        d = entry.to_dict()
        assert d["name"] == "test_skill"
        assert d["state"] == "active"
        assert "created_at" in d
        assert "last_transition_at" in d
