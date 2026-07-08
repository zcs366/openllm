"""
免疫系统v2测试 — Day 1 交付物
测试项:
  ② DelegationGuard 三重防护
  ⑦ 规则冲突降级（DEGRADED状态）
  ⑨ 规则衰减字段预留
  ⑥ 成长信号日志
"""
import sys
import json
import time
import os

sys.path.insert(0, os.path.expanduser("~/projects/openllm/src"))

from openllm.core.governance_engine import DelegationGuard, GovernanceEngine
from openllm.core.governance_rule import (
    GovernanceRule, RuleStatus, RuleType, RuleFamily
)


def test_delegation_guard_depth():
    """② 深度限制: ≤5层"""
    guard = DelegationGuard()
    
    # 5层内允许
    for i in range(5):
        ok, reason = guard.check(f"tool_{i}", {"arg": i})
        assert ok, f"Layer {i} should be allowed: {reason}"
        guard.push(f"tool_{i}", {"arg": i})
    
    # 第6层拒绝
    ok, reason = guard.check("tool_6", {"arg": 6})
    assert not ok, f"Layer 6 should be rejected: {reason}"
    assert "chain_depth" in reason
    
    # 出栈后恢复
    guard.pop()
    ok, reason = guard.check("tool_6", {"arg": 6})
    assert ok, f"After pop, layer 6 should be allowed: {reason}"
    
    print("  ✅ 深度限制: ≤5层正确拦截")


def test_delegation_guard_cycle():
    """② 循环检测: 相同content hash"""
    guard = DelegationGuard()
    
    # 第一次调用
    ok, _ = guard.check("read_file", {"path": "/etc/passwd"})
    assert ok
    guard.push("read_file", {"path": "/etc/passwd"})
    
    # 相同参数再次调用 → 检测到循环
    ok, reason = guard.check("read_file", {"path": "/etc/passwd"})
    assert not ok, f"Cycle should be detected: {reason}"
    assert "循环检测" in reason
    
    # 不同参数 → 允许
    ok, _ = guard.check("read_file", {"path": "/etc/hosts"})
    assert ok, "Different params should be allowed"
    
    guard.pop()
    print("  ✅ 循环检测: 相同content hash正确拦截")


def test_delegation_guard_pop_safety():
    """② pop安全: 空栈不崩溃"""
    guard = DelegationGuard()
    guard.pop()  # 空栈pop
    assert guard.get_depth() == 0
    print("  ✅ pop安全: 空栈不崩溃")


def test_degraded_status():
    """⑦ DEGRADED状态存在"""
    assert hasattr(RuleStatus, 'DEGRADED')
    assert RuleStatus.DEGRADED.value == "degraded"
    print("  ✅ DEGRADED状态: 枚举存在")


def test_decay_fields():
    """⑨ 规则衰减字段存在"""
    rule = GovernanceRule(
        rule_type=RuleType.CONTROL,
        family=RuleFamily.VALIDATION,
        target="test",
        condition="true",
        action="log",
        description="test rule",
    )
    
    # 默认值
    assert rule.last_triggered_at == 0.0
    assert rule.trigger_count == 0
    
    # 可赋值
    rule.last_triggered_at = time.time()
    rule.trigger_count = 5
    assert rule.last_triggered_at > 0
    assert rule.trigger_count == 5
    
    # 序列化保留
    d = rule.to_dict()
    assert d["last_triggered_at"] > 0
    assert d["trigger_count"] == 5
    
    # 反序列化保留
    rule2 = GovernanceRule.from_dict(d)
    assert rule2.last_triggered_at == rule.last_triggered_at
    assert rule2.trigger_count == rule.trigger_count
    
    print("  ✅ 衰减字段: 序列化/反序列化完整")


def test_growth_signal():
    """⑥ 成长信号: verify_governance成功时记录"""
    # 检查代码中是否有成长信号日志
    import inspect
    from openllm.core.governance_engine import GovernanceEngine
    source = inspect.getsource(GovernanceEngine.verify_governance)
    assert "系统学会了" in source, "成长信号日志未找到"
    print("  ✅ 成长信号: verify_governance含'系统学会了'日志")


if __name__ == "__main__":
    print("=" * 60)
    print("免疫系统v2 · Day 1 单元测试")
    print("=" * 60)
    
    tests = [
        test_delegation_guard_depth,
        test_delegation_guard_cycle,
        test_delegation_guard_pop_safety,
        test_degraded_status,
        test_decay_fields,
        test_growth_signal,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ❌ {test.__name__}: {e}")
            failed += 1
    
    print(f"\n结果: {passed} passed, {failed} failed, {passed + failed} total")
    sys.exit(1 if failed else 0)
