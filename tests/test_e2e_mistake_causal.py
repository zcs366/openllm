"""
端到端集成测试: MistakeLedger→CausalMemory→检索
全流程: 记录错误→转换为因果教训→存储→检索验证

验收标准:
1. MistakeLedger记录成功写入
2. mistake_to_causal正确转换
3. CausalMemoryStore成功存储
4. 检索能返回relevant结果
5. 写门控正确拒绝重复信息
"""
import os
import sys
import tempfile
import json
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/projects/openllm/src"))

from openllm.memory.mistake_ledger import MistakeLedger
from openllm.memory.mistake_bridge import mistake_to_causal, find_recurring_patterns
from openllm.memory.causal_memory import CausalMemory, CausalMemoryStore, TrustLevel


def _store_causal(store, causal: CausalMemory):
    """辅助函数：存储CausalMemory到store"""
    store.store(
        action_signature=causal.action_signature,
        context_features=causal.context_features,
        prediction=causal.prediction,
        prediction_confidence=causal.prediction_confidence,
        actual_result=causal.actual_result,
        actual_success=causal.actual_success,
        delta=causal.delta,
        delta_magnitude=causal.delta_magnitude,
        lesson=causal.lesson,
        source=causal.source,
        trust_level=causal.trust_level,
        importance=causal.importance,
        tags=causal.tags,
        session_id=causal.session_id,
    )


def test_e2e_mistake_to_causal_pipeline():
    """全流程: MistakeLedger→CausalMemory→检索"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. 初始化组件
        ledger = MistakeLedger(path=os.path.join(tmpdir, "ledger.jsonl"))
        store = CausalMemoryStore(store_dir=Path(tmpdir))

        # 2. 记录错误
        mistake1 = ledger.append(
            what="端口配错，应该是8080",
            why="默认端口配置与实际环境不匹配",
            agent="tool_executor",
            severity="medium",
            tags=["config", "port"],
        )
        assert mistake1 is not None, "MistakeLedger记录失败"
        print("  ✅ Step 1: MistakeLedger记录成功")

        # 3. 转换为CausalMemory
        causal1 = mistake_to_causal(mistake1)
        assert causal1.actual_result == "端口配错，应该是8080"
        assert causal1.lesson == "默认端口配置与实际环境不匹配"
        assert causal1.trust_level.value == "internal"
        print("  ✅ Step 2: mistake_to_causal转换成功")

        # 4. 存储到CausalMemoryStore
        _store_causal(store, causal1)
        print("  ✅ Step 3: CausalMemoryStore存储成功")

        # 5. 检索验证
        results = store.search("端口配置")
        assert len(results) > 0, "检索无结果"
        print(f"  ✅ Step 4: 检索返回 {len(results)} 条结果")

        # 6. 记录第二个错误
        mistake2 = ledger.append(
            what="数据库连接超时",
            why="连接池配置过小导致高并发时耗尽",
            agent="db_connector",
            severity="high",
            tags=["database", "timeout"],
        )
        causal2 = mistake_to_causal(mistake2)
        _store_causal(store, causal2)

        # 7. 检索第二个错误
        results2 = store.search("数据库")
        assert len(results2) > 0, "数据库相关检索无结果"
        print(f"  ✅ Step 5: 第二条因果教训检索成功")

        # 8. 重复错误模式检测
        patterns = find_recurring_patterns(ledger)
        print(f"  ✅ Step 6: 检测到 {len(patterns)} 个重复模式")

        print("\n🎉 端到端集成测试通过!")
        return True


def test_write_gate_integration():
    """测试写门控在真实场景中的行为"""
    import sys as _sys
    _sys.path.insert(0, os.path.expanduser("~/.hermes/jiak"))
    from belief_update import write_gate

    print("\n测试写门控集成:")

    # 1. 相同信息应被拒绝
    result1 = write_gate(
        "端口配错应该是8080",
        "端口配错，应该是8080"
    )
    assert not result1["pass"], f"相同信息应被拒绝, got {result1}"
    print(f"  ✅ 相同信息拒绝: cosine={result1['g1_score']:.3f}")

    # 2. 有新信息应通过
    result2 = write_gate(
        "端口配错应该是8080，同时需要检查防火墙规则",
        "端口配错，应该是8080"
    )
    assert result2["pass"], f"有新信息应通过, got {result2}"
    print(f"  ✅ 新信息通过: cosine={result2['g1_score']:.3f}")

    # 3. 不相关文本应通过
    result3 = write_gate(
        "今天天气很好适合出门",
        "端口配错应该是8080"
    )
    assert result3["pass"], f"不相关文本应通过"
    print(f"  ✅ 不相关文本通过")

    print("  🎉 写门控集成测试通过!")
    return True


def test_causal_memory_temperature():
    """测试因果记忆的温度衰减"""
    print("\n测试因果记忆温度:")

    causal = CausalMemory(
        memory_id="test-temp",
        prediction="测试预测",
        actual_result="测试结果",
        delta="测试差距",
        lesson="测试教训",
        importance=0.8,
    )

    # 新记忆应该温度高
    temp_new = causal.temperature(decay_lambda=0.01)
    assert temp_new > 0.5, f"新记忆温度应>0.5, got {temp_new}"
    print(f"  ✅ 新记忆温度: {temp_new:.3f}")

    # 访问次数增加应该影响温度
    causal.access_count = 10
    temp_accessed = causal.temperature(decay_lambda=0.01)
    print(f"  ✅ 访问10次后温度: {temp_accessed:.3f}")

    print("  🎉 温度衰减测试通过!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("端到端集成测试: MistakeLedger→CausalMemory→检索")
    print("=" * 60)

    results = []
    results.append(("E2E Pipeline", test_e2e_mistake_to_causal_pipeline()))
    results.append(("Write Gate", test_write_gate_integration()))
    results.append(("Temperature", test_causal_memory_temperature()))

    print("\n" + "=" * 60)
    print("测试结果汇总:")
    print("=" * 60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")

    all_passed = all(p for _, p in results)
    print(f"\n{'🎉 All tests passed!' if all_passed else '❌ Some tests failed!'}")
