"""MistakeBridge桥接测试"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from openllm.memory.mistake_ledger import MistakeLedger
from openllm.memory.mistake_bridge import mistake_to_causal, find_recurring_patterns


def test_mistake_to_causal():
    """测试：MistakeLedger记录→CausalMemory转换"""
    mistake = {
        "id": "mistake-001",
        "what": "端口配错，应该是8080",
        "why": "默认端口配置与实际环境不匹配",
        "agent": "tool_executor",
        "severity": "medium",
        "tags": ["config", "port"],
    }

    causal = mistake_to_causal(mistake)

    assert causal.actual_result == "端口配错，应该是8080"
    assert causal.lesson == "默认端口配置与实际环境不匹配"
    assert causal.actual_success is False
    assert causal.delta_magnitude == 0.5  # medium → 0.5
    assert causal.trust_level.value == "internal"
    assert causal.source == "mistake_ledger:tool_executor"
    assert causal.context_features == ["config", "port"]

    print("✅ test_mistake_to_causal passed")


def test_severity_mapping():
    """测试：severity正确映射为delta_magnitude"""
    for sev, expected in [("low", 0.2), ("medium", 0.5), ("high", 0.8), ("critical", 1.0)]:
        causal = mistake_to_causal({"what": "x", "why": "y", "severity": sev})
        assert causal.delta_magnitude == expected, f"severity={sev} → {causal.delta_magnitude} != {expected}"

    print("✅ test_severity_mapping passed")


def test_find_recurring_patterns():
    """测试：重复错误模式检测"""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name

    try:
        ledger = MistakeLedger(path)
        # 同一个agent犯3次完全相同的错误
        for i in range(3):
            ledger.append(
                what="端口配错，默认8080实际3000",
                why="配置问题",
                agent="executor",
            )
        # 另一个agent犯1次不同错误
        ledger.append(what="文件找不到", why="路径错误", agent="file_op")

        patterns = find_recurring_patterns(ledger, min_count=3, window_hours=1)
        assert len(patterns) == 1
        assert patterns[0]["agent"] == "executor"
        assert patterns[0]["count"] == 3

        print("✅ test_find_recurring_patterns passed")
    finally:
        os.unlink(path)


if __name__ == "__main__":
    print("=== MistakeBridge 桥接测试 ===\n")
    test_mistake_to_causal()
    test_severity_mapping()
    test_find_recurring_patterns()
    print("\n🎉 全部桥接测试通过！")
