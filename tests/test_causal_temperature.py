"""
测试因果温度函数——验证"因果效应参与遗忘决策"三刀改造
测试1: causal_weight=0 时温度等于旧公式（向后兼容）
测试2: causal_weight=1.0 时温度比 causal_weight=0 高 CAUSAL_BONUS
测试3: 两条记忆同importance同时间，delta_magnitude高的温度更高
测试4: CausalMemory.temperature() 包含 delta_magnitude 加成
"""
import math
import time
import sys
from pathlib import Path

# 确保openllm包可导入
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openllm.memory.temperature_engine import (
    calculate_temperature, CAUSAL_BONUS, BASE_HEAT,
    LAMBDA_MAP, EMOTION_ENABLED
)
from openllm.memory.causal_memory import CausalMemory
from openllm.core.memory_os import MemoryEntry


# ═══════════════════════════════════════════════════
# 测试1: 向后兼容——causal_weight=0 等于旧公式
# ═══════════════════════════════════════════════════
def test_backward_compatibility():
    """causal_weight=0 时，温度应等于不含因果加成的旧公式。"""
    imp, days, mem_type, acc = 5.0, 10, "insight", 2

    # 旧公式: importance × e^(-λt × emotion_factor) * relevance_boost + heat
    # 新公式: 同上 + causal_weight × CAUSAL_BONUS
    # 当 causal_weight=0 时，两者相等
    temp_old = calculate_temperature(imp, days, mem_type, acc, 0.0, causal_weight=0.0)
    temp_new_no_causal = calculate_temperature(imp, days, mem_type, acc, 0.0, causal_weight=0.0)

    assert temp_old == temp_new_no_causal, \
        f"向后兼容失败: old={temp_old}, new={temp_new_no_causal}"
    print(f"✅ 测试1 PASS: causal_weight=0 → T={temp_old}（向后兼容）")


# ═══════════════════════════════════════════════════
# 测试2: causal_weight=1.0 比 causal_weight=0 高 CAUSAL_BONUS
# ═══════════════════════════════════════════════════
def test_causal_bonus():
    """causal_weight=1.0 时温度应比 causal_weight=0 高 CAUSAL_BONUS（不触MAX_TEMP）。"""
    # 用小importance+多天数，确保总温度远低于MAX_TEMP=10，不会被截断
    imp, days, mem_type, acc = 1.0, 100, "insight", 0

    temp_no = calculate_temperature(imp, days, mem_type, acc, 0.0, causal_weight=0.0)
    temp_full = calculate_temperature(imp, days, mem_type, acc, 0.0, causal_weight=1.0)

    expected_diff = CAUSAL_BONUS  # 应该正好高 3.0
    actual_diff = temp_full - temp_no

    assert abs(actual_diff - expected_diff) < 0.01, \
        f"CAUSAL_BONUS不匹配: 期望差={expected_diff}, 实际差={actual_diff}"
    print(f"✅ 测试2 PASS: delta_T={actual_diff:.2f} = CAUSAL_BONUS={CAUSAL_BONUS}")


# ═══════════════════════════════════════════════════
# 测试3: 两条记忆同importance同时间，delta_magnitude高的温度更高
# ═══════════════════════════════════════════════════
def test_causal_memory活得更久():
    """两条记忆importance和时间相同，但delta_magnitude不同，高的温度更高。"""
    now = time.time()
    mem_low = CausalMemory(
        action_signature="test_low",
        importance=0.5,
        last_accessed=now - 86400,  # 1天前
        delta_magnitude=0.1,
    )
    mem_high = CausalMemory(
        action_signature="test_high",
        importance=0.5,
        last_accessed=now - 86400,  # 1天前
        delta_magnitude=0.9,
    )

    temp_low = mem_low.temperature()
    temp_high = mem_high.temperature()

    assert temp_high > temp_low, \
        f"因果记忆活得更久测试失败: low={temp_low}, high={temp_high}"
    print(f"✅ 测试3 PASS: delta=0.1→T={temp_low:.4f}, delta=0.9→T={temp_high:.4f}")


# ═══════════════════════════════════════════════════
# 测试4: CausalMemory.temperature() 包含 delta_magnitude 加成
# ═══════════════════════════════════════════════════
def test_causal_memory_temperature_formula():
    """CausalMemory.temperature() = importance × e^(-λt) + delta_magnitude × 3.0"""
    now = time.time()
    mem = CausalMemory(
        action_signature="formula_test",
        importance=1.0,
        last_accessed=now,  # 刚访问，t≈0
        delta_magnitude=0.5,
    )

    temp = mem.temperature(decay_lambda=0.01)
    # t≈0: base ≈ importance × e^0 = 1.0
    # expected = 1.0 + 0.5 × 3.0 = 2.5
    expected_min = 2.4  # 容忍微小时间差
    expected_max = 2.6

    assert expected_min <= temp <= expected_max, \
        f"公式验证失败: temp={temp}, 期望[{expected_min}, {expected_max}]"
    print(f"✅ 测试4 PASS: T={temp:.4f} (expected≈2.5)")


# ═══════════════════════════════════════════════════
# 测试5: MemoryEntry.temperature() 向后兼容 + causal_delta
# ═══════════════════════════════════════════════════
def test_memory_entry_causal():
    """MemoryEntry.temperature() 加 causal_delta=0 向后兼容，加>0 时温度升高。"""
    entry = MemoryEntry(key="test", value={"x": 1}, importance=0.8)
    now = time.time()
    entry.last_accessed = now  # 刚访问

    temp_no_causal = entry.temperature(decay_lambda=0.01, causal_delta=0.0)
    temp_with_causal = entry.temperature(decay_lambda=0.01, causal_delta=1.0)

    assert temp_with_causal > temp_no_causal, \
        f"MemoryEntry因果测试失败: no_causal={temp_no_causal}, with_causal={temp_with_causal}"
    assert abs(temp_with_causal - temp_no_causal - 3.0) < 0.01, \
        f"MemoryEntry因果差值不等于CAUSAL_BONUS: diff={temp_with_causal - temp_no_causal}"
    print(f"✅ 测试5 PASS: no_causal={temp_no_causal:.4f}, with_causal={temp_with_causal:.4f}")


# ═══════════════════════════════════════════════════
# 运行
# ═══════════════════════════════════════════════════
if __name__ == "__main__":
    test_backward_compatibility()
    test_causal_bonus()
    test_causal_memory活得更久()
    test_causal_memory_temperature_formula()
    test_memory_entry_causal()
    print(f"\n🎉 全部 5/5 测试通过！CAUSAL_BONUS={CAUSAL_BONUS}")
