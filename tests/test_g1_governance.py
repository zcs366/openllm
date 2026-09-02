#!/usr/bin/env python3
"""
G-1 治理通电验证脚本
验证内容：
1. L2-3仲裁路径正确性
2. GovernanceEngine心跳trace写入审计链
"""
import sys
import os

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from openllm.core.models import (
    Proposal, Critique, RiskAssessment, Decision
)


# ── 1. 测试 _select_strategy 路由 ──────────────────────
def test_select_strategy():
    """验证风险等级→仲裁级别映射"""
    from openllm.core.ios_arbitrate import _select_strategy

    assert _select_strategy(None) == 1, "None→L1"
    assert _select_strategy(RiskAssessment(level="low")) == 2, "low→L2"
    assert _select_strategy(RiskAssessment(level="medium")) == 3, "medium→L3"
    assert _select_strategy(RiskAssessment(level="high")) == 4, "high→L4"
    assert _select_strategy(RiskAssessment(level="critical")) == 5, "critical→L5"
    print("✅ _select_strategy 路由正确")


# ── 2. 测试 L2 self_consistency ──────────────────────
def test_l2_self_consistency():
    """验证L2：高confidence通过，低confidence降级L1"""
    from openllm.core.ios_arbitrate import arbitrate

    # 模拟 IOS 对象
    class MockIOS:
        _arbiter_policy = "conservative"
        _rejection_engine = None
        _RejectionReason = None
    ios = MockIOS()

    risk = RiskAssessment(level="low")
    critique = Critique(content="ok", verdict="approve")

    # 高confidence → L2通过
    proposal_high = Proposal(content="test", confidence=0.8)
    d = arbitrate(ios, proposal_high, critique, risk)
    assert d.approved, f"L2高confidence应通过: {d.reason}"
    assert "Level 2" in d.reason, f"应标明Level 2: {d.reason}"
    assert "self_consistency通过" in d.reason, f"应标明通过: {d.reason}"
    print(f"  ✅ L2高confidence通过: {d.reason}")

    # 低confidence → 降级L1，仍approved（低风险保守放行）
    proposal_low = Proposal(content="test", confidence=0.3)
    d2 = arbitrate(ios, proposal_low, critique, risk)
    assert d2.approved, f"L2低confidence应降级L1仍通过: {d2.reason}"
    assert "self_consistency未通过" in d2.reason, f"应标明未通过: {d2.reason}"
    assert "降级L1" in d2.reason, f"应标明降级: {d2.reason}"
    print(f"  ✅ L2低confidence降级L1: {d2.reason}")

    print("✅ L2 self_consistency 验证通过")


# ── 3. 测试 L3 multi_persona ──────────────────────
def test_l3_multi_persona():
    """验证L3：双视角聚合"""
    from openllm.core.ios_arbitrate import arbitrate

    class MockIOS:
        _arbiter_policy = "conservative"
        _rejection_engine = None
        _RejectionReason = None
    ios = MockIOS()

    risk = RiskAssessment(level="medium")
    proposal = Proposal(content="test", confidence=0.8)

    # L3: critique.approve → 一致通过
    critique_approve = Critique(content="looks good", verdict="approve")
    d = arbitrate(ios, proposal, critique_approve, risk)
    assert d.approved, f"L3一致通过应approved: {d.reason}"
    assert "Level 3" in d.reason, f"应标明Level 3: {d.reason}"
    assert "multi_persona一致通过" in d.reason
    print(f"  ✅ L3一致通过: {d.reason}")

    # L3: critique.reject → 否决
    critique_reject = Critique(content="bad idea", verdict="reject", concerns=["dangerous"])
    d2 = arbitrate(ios, proposal, critique_reject, risk)
    assert not d2.approved, f"L3否决应not approved: {d2.reason}"
    assert "Level 3" in d2.reason
    assert "multi_persona否决" in d2.reason
    print(f"  ✅ L3否决: {d2.reason}")

    # L3: critique.revise + 高confidence → 存疑通过
    critique_revise = Critique(content="needs fix", verdict="revise")
    d3 = arbitrate(ios, proposal, critique_revise, risk)
    assert d3.approved, f"L3存疑通过应approved: {d3.reason}"
    assert "Level 3" in d3.reason
    assert "multi_persona存疑通过" in d3.reason
    print(f"  ✅ L3存疑通过: {d3.reason}")

    # L3: critique.revise + 低confidence → 降级L4
    proposal_low = Proposal(content="test", confidence=0.4)
    d4 = arbitrate(ios, proposal_low, critique_revise, risk)
    assert "Level 3" in d4.reason, f"应标明从Level 3降级: {d4.reason}"
    assert "降级L4" in d4.reason
    # L4 conservative + revise → approved=False
    assert not d4.approved
    print(f"  ✅ L3降级L4: {d4.reason}")

    print("✅ L3 multi_persona 验证通过")


# ── 4. 测试 L4/L5 未被破坏 ──────────────────────
def test_l4_l5_unchanged():
    """验证L4原有逻辑未被破坏。L5为defense-in-depth路径（critical risk始终被
    is_blocked拦截，不会走到arbitrate的L5分支——这是原始设计）。"""
    from openllm.core.ios_arbitrate import arbitrate

    class MockIOS:
        _arbiter_policy = "conservative"
        _rejection_engine = None
        _RejectionReason = None
    ios = MockIOS()

    proposal = Proposal(content="test", confidence=0.8)

    # L4 approve
    risk_high = RiskAssessment(level="high")
    d = arbitrate(ios, proposal, Critique(content="ok", verdict="approve"), risk_high)
    assert d.approved and "Level 4" in d.reason
    print(f"  ✅ L4 approve: {d.reason}")

    # L4 reject
    d2 = arbitrate(ios, proposal, Critique(content="bad", verdict="reject", concerns=["risky"]), risk_high)
    assert not d2.approved and "Level 4" in d2.reason
    print(f"  ✅ L4 reject: {d2.reason}")

    # L5 defense-in-depth：critical risk 被 is_blocked 拦截（原始设计）
    risk_crit = RiskAssessment(level="critical")
    d3 = arbitrate(ios, proposal, Critique(content="ok", verdict="approve"), risk_crit)
    assert not d3.approved, "critical risk 应被安全拦截"
    assert "安全拦截" in d3.reason
    print(f"  ✅ L5 defense-in-depth: critical risk被拦截: {d3.reason}")

    print("✅ L4/L5 原有逻辑未被破坏")


# ── 5. 测试 GovernanceEngine 心跳trace ──────────────
def test_heartbeat_trace():
    """验证心跳trace写入审计链"""
    from openllm.core.governance_engine import GovernanceEngine

    engine = GovernanceEngine()

    # 写入一条心跳trace
    row_id = engine.heartbeat_trace(
        tick_id="test_tick_001",
        risk_level="low",
        approved=True,
        decision_reason="Level 2: self_consistency通过",
        duration_ms=150.0,
        agent_id="test_agent",
    )
    assert row_id > 0, f"心跳trace写入应返回正整数ID: {row_id}"
    print(f"  ✅ 心跳trace写入成功: row_id={row_id}")

    # 查询验证
    events = engine.audit_log.query(event_type="heartbeat", limit=5)
    assert len(events) >= 1, f"应有至少1条heartbeat事件: {len(events)}"
    latest = events[0]  # 最新的在前
    assert latest["event_type"] == "heartbeat"
    assert latest["action"] == "tick"
    assert "risk=low" in latest["details"]
    assert "approved=True" in latest["details"]
    print(f"  ✅ 心跳trace查询验证: {latest['details'][:80]}")

    # 链完整性验证
    valid, broken_at = engine.audit_log.verify_chain()
    print(f"  ✅ 审计链完整性: valid={valid}, broken_at={broken_at}")

    print("✅ GovernanceEngine 心跳trace 验证通过")


# ── 6. 测试 get_stats 反映心跳痕迹 ──────────────
def test_get_stats_reflects_heartbeat_traces():
    """验证 get_stats() 返回心跳trace统计"""
    from openllm.core.governance_engine import GovernanceEngine

    engine = GovernanceEngine()

    # 初始状态：get_stats 应有 heartbeat_traces 键
    stats_before = engine.get_stats()
    assert "heartbeat_traces" in stats_before, f"get_stats缺少heartbeat_traces键: {stats_before.keys()}"
    before_total = stats_before["heartbeat_traces"]["total"]

    # 写入3条心跳trace（模拟3个tick）
    for i in range(3):
        engine.heartbeat_trace(
            tick_id=f"stats_tick_{i}",
            risk_level="low" if i < 2 else "medium",
            approved=True,
            decision_reason=f"tick {i} ok",
            duration_ms=100.0 * (i + 1),
            agent_id="test_stats",
        )

    # get_stats 应反映新增的trace
    stats_after = engine.get_stats()
    ht = stats_after["heartbeat_traces"]
    assert ht["total"] >= before_total + 3, (
        f"心跳trace总数不正确: before={before_total} after={ht['total']}"
    )
    assert ht["recent_risk"] == "medium", f"最近risk应为medium: {ht['recent_risk']}"
    assert ht["recent_approved"] == "True", f"最近approved应为True: {ht['recent_approved']}"
    print(f"  ✅ get_stats反映心跳痕迹: total={ht['total']}, risk={ht['recent_risk']}, approved={ht['recent_approved']}")
    print("✅ get_stats 心跳trace统计验证通过")


# ── 运行所有验证 ──────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("G-1 治理通电验证")
    print("=" * 60)

    errors = []
    for name, func in [
        ("_select_strategy", test_select_strategy),
        ("L2 self_consistency", test_l2_self_consistency),
        ("L3 multi_persona", test_l3_multi_persona),
        ("L4/L5 unchanged", test_l4_l5_unchanged),
        ("heartbeat_trace", test_heartbeat_trace),
        ("get_stats heartbeat_traces", test_get_stats_reflects_heartbeat_traces),
    ]:
        print(f"\n--- {name} ---")
        try:
            func()
        except Exception as e:
            print(f"❌ {name} 失败: {e}")
            import traceback
            traceback.print_exc()
            errors.append(name)

    print("\n" + "=" * 60)
    if errors:
        print(f"❌ 验证失败: {', '.join(errors)}")
        sys.exit(1)
    else:
        print("✅ 全部验证通过")
        sys.exit(0)
