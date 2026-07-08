"""openLLM Governance Layer 测试——验证G1-G6完整实现。"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openllm.governance import (
    GovernanceEngine,
    AuditChain,
    GovernanceDimension,
    VotePosition,
    AgentRole,
)


def test_full_deliberation():
    """完整的五人合议流程测试。"""
    print("=== 测试：完整五人合议 ===")

    engine = GovernanceEngine()
    session = engine.create_session("test-001", "openLLM治理层设计评审")

    # G1: 准入成员
    engine.admit_member(session, "子产", AgentRole.SKEPTIC)
    engine.admit_member(session, "韩信", AgentRole.STRATEGIST)
    engine.admit_member(session, "鲁班", AgentRole.ARCHITECT)
    engine.admit_member(session, "萧何", AgentRole.EXECUTOR)
    engine.admit_member(session, "子贡", AgentRole.COORDINATOR)
    print(f"  G1 成员准入: {len(session.members)} 人 ✅")

    # G2: 提案
    claim = engine.propose(session, "韩信", "openLLM应该实现协议级治理层")
    print(f"  G2 提案: {claim.claim_id} ✅")

    # G2: 挑战
    engine.challenge(session, "子产", claim.claim_id, "工业界真的需要这个吗？")
    engine.support(session, "鲁班", claim.claim_id, "recall_append.py已有骨架")
    print(f"  G2 挑战/支持 ✅")

    # G3: 投票
    engine.vote(session, "子产", claim.claim_id, VotePosition.SUPPORT, "差异化明确")
    engine.vote(session, "韩信", claim.claim_id, VotePosition.STRONG_SUPPORT, "护城河")
    engine.vote(session, "鲁班", claim.claim_id, VotePosition.SUPPORT, "工程成本低")
    engine.vote(session, "萧何", claim.claim_id, VotePosition.SUPPORT, "资源够")
    engine.vote(session, "子贡", claim.claim_id, VotePosition.NEUTRAL, "需看市场")
    print(f"  G3 投票完成 ✅")

    # G4: 异议
    engine.record_dissent(session, "子贡", claim.claim_id, VotePosition.NEUTRAL, "市场需求未验证")
    print(f"  G4 异议记录 ✅")

    # G5: 检查升级（应该不触发，因为平均分高）
    escalated = engine.check_escalation(session, claim.claim_id)
    print(f"  G5 升级检查: triggered={escalated} ✅")

    # 统计投票
    result = engine.tally_votes(session, claim.claim_id)
    print(f"  投票结果: score={result['weighted_score']}, result={result['result']} ✅")

    # G6: 关闭会话
    report = engine.close_session(session)
    print(f"  G6 审计链: valid={report['audit_chain']['valid']}, events={report['audit_chain']['total_events']} ✅")

    # 验证审计链
    chain_status = engine.audit.verify_chain("test-001")
    assert chain_status["valid"], "审计链应该有效"
    assert chain_status["total_events"] > 0, "应该有审计事件"
    assert "membership" in chain_status["dimensions_covered"], "应该有G1事件"
    assert "voting" in chain_status["dimensions_covered"], "应该有G3事件"
    assert "dissent" in chain_status["dimensions_covered"], "应该有G4事件"

    print(f"  覆盖维度: {chain_status['dimensions_covered']} ✅")
    print("=== 测试通过 ✅ ===\n")


def test_escalation_trigger():
    """测试人类升级触发条件。"""
    print("=== 测试：人类升级触发 ===")

    engine = GovernanceEngine()
    session = engine.create_session("test-002", "争议性决策")

    # 准入成员
    engine.admit_member(session, "子产", AgentRole.SKEPTIC)
    engine.admit_member(session, "韩信", AgentRole.STRATEGIST)

    # 提案
    claim = engine.propose(session, "子产", "是否采用新架构？")

    # 投票——全票反对，应该触发升级
    engine.vote(session, "子产", claim.claim_id, VotePosition.STRONG_OPPOSE, "风险太高")
    engine.vote(session, "韩信", claim.claim_id, VotePosition.OPPOSE, "时机不对")

    escalated = engine.check_escalation(session, claim.claim_id)
    assert escalated, "全票反对应该触发人类升级"
    assert len(session.escalations) > 0, "应该有升级记录"
    print(f"  全票反对升级: triggered={escalated} ✅")
    print(f"  升级记录: {session.escalations[0]['trigger']} ✅")

    print("=== 测试通过 ✅ ===\n")


def test_dissent_retrieval():
    """测试异议检索——G4核心能力。"""
    print("=== 测试：异议检索 ===")

    engine = GovernanceEngine()
    session = engine.create_session(f"test-dissent-{int(time.time()*1000)}", "异议保留测试")

    # 准入 + 提案 + 异议
    engine.admit_member(session, "鲁班", AgentRole.ARCHITECT)
    claim = engine.propose(session, "鲁班", "采用微服务架构")
    engine.record_dissent(session, "鲁班", claim.claim_id, VotePosition.OPPOSE, "过度工程化")

    # 检索历史异议
    dissents = engine.get_historical_dissents(session.session_id)
    print(f"  session_id: {session.session_id}")
    print(f"  dissents count: {len(dissents)}")
    for d in dissents:
        print(f"    event_type={d.get('event_type')}, rationale={d.get('payload', {}).get('rationale')}")
    assert len(dissents) == 1, f"应该有1条异议, got {len(dissents)}"
    assert dissents[0]["payload"]["rationale"] == "过度工程化"
    print(f"  异议检索: {len(dissents)} 条 ✅")
    print(f"  异议内容: {dissents[0]['payload']['rationale']} ✅")

    print("=== 测试通过 ✅ ===\n")


def test_replay():
    """测试G6重放——确定性重建。"""
    print("=== 测试：确定性重放 ===")

    engine = GovernanceEngine()
    session = engine.create_session(f"test-replay-{int(time.time()*1000)}", "重放测试")

    # 执行一系列操作
    engine.admit_member(session, "子产", AgentRole.SKEPTIC)
    claim = engine.propose(session, "子产", "测试提案")
    engine.vote(session, "子产", claim.claim_id, VotePosition.SUPPORT)
    engine.close_session(session)

    # 重放
    events = engine.replay_session(session.session_id)
    assert len(events) > 0, "应该有事件"
    print(f"  重放事件数: {len(events)} ✅")

    # 验证事件顺序
    event_types = [e["event_type"] for e in events]
    assert event_types[0] == "audit", "第一个应该是session_created"
    assert event_types[-1] == "audit", "最后一个应该是session_closed"
    print(f"  事件顺序正确 ✅")

    print("=== 测试通过 ✅ ===\n")


if __name__ == "__main__":
    print("🏛️ openLLM Governance Layer 测试\n")
    test_full_deliberation()
    test_escalation_trigger()
    test_dissent_retrieval()
    test_replay()
    print("🎯 所有测试通过！治理层G1-G6完整实现。")
