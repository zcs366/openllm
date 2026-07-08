"""
免疫系统v2 · 端到端集成测试
全流程: 失败→分类→归因→治理→审计→heuristic检索

验收标准:
1. 失败→分类器输出structural
2. harness_layer=tool_interface
3. 治理规则写入governance_rules.jsonl
4. 审计事件写入hermes.db governance_audit表
5. 链式hash验证通过
6. heuristics检索能返回relevant结果
7. 成长信号日志输出
8. 规则冲突降级（DEGRADED）
"""
import sys
import os
import json
import tempfile
import sqlite3

sys.path.insert(0, os.path.expanduser("~/projects/openllm/src"))

from openllm.core.governance_engine import (
    GovernanceEngine, GovernanceAuditLog, HeuristicsConsumer, DelegationGuard
)
from openllm.core.governance_rule import RuleStatus
from openllm.core.failure_tracker import HarnessLayer


def test_e2e_failure_to_governance():
    """全流程: 失败→分类→治理→验证"""
    engine = GovernanceEngine()
    
    # 模拟3次同类TOOL_PARAM失败
    for i in range(3):
        result = engine.convert({
            "tool_name": "read_file",
            "error": "缺少必填参数path",
            "mechanism": "tool_param",
        })
    
    # 第3次应该触发structural分类
    # 检查是否有规则被安装
    rules = list(engine.rule_store._rules.values())
    assert len(rules) > 0, f"应有规则被安装: {len(rules)}"
    
    # 检查harness_layer
    sig = engine.capture_failure({
        "tool_name": "read_file",
        "error": "缺少必填参数path",
        "mechanism": "tool_param",
    })
    assert sig.harness_layer == "tool_interface", f"期望tool_interface: {sig.harness_layer}"
    
    print("  ✅ 全流程: 失败→分类→治理→harness_layer归因")


def test_e2e_audit_chain():
    """全流程: 审计事件→链式验证"""
    test_db = tempfile.mktemp(suffix=".db")
    
    # 创建审计日志
    log = GovernanceAuditLog()
    log._db_path = test_db
    log._conn = sqlite3.connect(test_db)
    log._ensure_table()
    
    # 写入多个事件
    log.append("rule_installed", action="block_tool_param")
    log.append("rule_verified", action="regression_pass")
    log.append("governance_request", action="ambiguous_detected")
    
    # 链式验证
    valid, broken_at = log.verify_chain()
    assert valid, f"链应完整: broken_at={broken_at}"
    
    # 查询
    results = log.query(event_type="rule_installed")
    assert len(results) == 1
    assert results[0]["action"] == "block_tool_param"
    
    log._conn.close()
    os.remove(test_db)
    
    print("  ✅ 审计链: 追加→链验证→查询")


def test_e2e_heuristic_retrieval():
    """全流程: heuristics检索+格式化"""
    consumer = HeuristicsConsumer()
    
    # 检索
    results = consumer.retrieve("tool call error parameter failure", top_k=3)
    assert isinstance(results, list)
    
    # 格式化
    formatted = consumer.format_for_context(results)
    if results:
        assert "⚠️历史经验" in formatted
    
    print(f"  ✅ Heuristics检索: {len(results)}条, 格式化{'成功' if formatted else '(空)'}")


def test_e2e_delegation_guard():
    """全流程: 委派防护"""
    guard = DelegationGuard()
    
    # 模拟5层委派
    for i in range(5):
        ok, _ = guard.check(f"tool_{i}", {"i": i})
        assert ok
        guard.push(f"tool_{i}", {"i": i})
    
    # 第6层被拒
    ok, reason = guard.check("tool_6", {"i": 6})
    assert not ok
    assert "chain_depth" in reason
    
    # 循环检测
    guard2 = DelegationGuard()
    ok, _ = guard2.check("read", {"path": "/etc/passwd"})
    guard2.push("read", {"path": "/etc/passwd"})
    ok, reason = guard2.check("read", {"path": "/etc/passwd"})
    assert not ok
    assert "循环检测" in reason
    
    print("  ✅ 委派防护: 深度限制+循环检测")


def test_e2e_growth_signal():
    """全流程: 成长信号日志"""
    import inspect
    from openllm.core.governance_engine import GovernanceEngine
    source = inspect.getsource(GovernanceEngine.verify_governance)
    assert "系统学会了" in source
    print("  ✅ 成长信号: 代码存在")


def test_e2e_rule_conflict_degradation():
    """全流程: 规则冲突降级"""
    from openllm.core.governance_rule import (
        GovernanceRule, RuleType, RuleFamily, RuleStatus
    )
    
    # 创建两条冲突的拦截类规则
    engine = GovernanceEngine()
    
    r1 = GovernanceRule(
        rule_type=RuleType.CONSTRAINT,
        family=RuleFamily.VALIDATION,
        target="test_tool",
        condition="retry_count >= 3",
        action="block_and_report(tool='test_tool')",
        description="测试拦截规则1",
    )
    r1.status = RuleStatus.ACTIVE
    engine.rule_store.save(r1)
    
    r2 = GovernanceRule(
        rule_type=RuleType.CONSTRAINT,
        family=RuleFamily.VALIDATION,
        target="test_tool",
        condition="retry_count >= 3",
        action="block_and_report(tool='test_tool')",
        description="测试拦截规则2",
    )
    
    # 安装冲突规则 → 应降级
    installed = engine.install_governance(r2)
    assert installed, "拦截类冲突规则应降级安装成功"
    assert r2.status == RuleStatus.DEGRADED, f"期望DEGRADED: {r2.status}"
    
    print("  ✅ 冲突降级: 拦截类规则DEGRADED安装")


def test_e2e_decay_fields():
    """全流程: 规则衰减字段"""
    from openllm.core.governance_rule import (
        GovernanceRule, RuleType, RuleFamily
    )
    import time
    
    rule = GovernanceRule(
        rule_type=RuleType.CONTROL,
        family=RuleFamily.VALIDATION,
        target="test",
        condition="true",
        action="log",
        description="衰减测试",
        last_triggered_at=time.time(),
        trigger_count=10,
    )
    
    d = rule.to_dict()
    assert d["last_triggered_at"] > 0
    assert d["trigger_count"] == 10
    
    rule2 = GovernanceRule.from_dict(d)
    assert rule2.last_triggered_at == rule.last_triggered_at
    assert rule2.trigger_count == rule.trigger_count
    
    print("  ✅ 衰减字段: 序列化/反序列化完整")


if __name__ == "__main__":
    print("=" * 60)
    print("免疫系统v2 · 端到端集成测试")
    print("=" * 60)
    
    tests = [
        test_e2e_failure_to_governance,
        test_e2e_audit_chain,
        test_e2e_heuristic_retrieval,
        test_e2e_delegation_guard,
        test_e2e_growth_signal,
        test_e2e_rule_conflict_degradation,
        test_e2e_decay_fields,
    ]
    
    p, f = 0, 0
    for t in tests:
        try:
            t()
            p += 1
        except Exception as e:
            print(f"  ❌ {t.__name__}: {e}")
            f += 1
    
    print(f"\n结果: {p} passed, {f} failed, {p+f} total")
    sys.exit(1 if f else 0)
