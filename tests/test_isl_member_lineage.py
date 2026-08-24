"""
ISL lineage_weight 衰减函数测试（2026-08-25）

L1 单调衰减：lineage_weight(0)=1.0；lineage_weight(n) > lineage_weight(n+1)
L2 永不归零：lineage_weight(10000) > 0
L3 负数拒绝：lineage_weight(-1) 抛 ValueError
"""
import pytest
from openllm.isn.isl_member import lineage_weight


# ═══════════════════════════════════════════════════════════════
# L1 单调衰减
# ═══════════════════════════════════════════════════════════════

class TestL1MonotonicDecay:
    def test_zero_is_one(self):
        """L1-a: lineage_weight(0) == 1.0"""
        assert lineage_weight(0) == 1.0

    def test_monotonic(self):
        """L1-b: lineage_weight(n) > lineage_weight(n+1) 对所有 n >= 0"""
        prev = lineage_weight(0)
        for n in range(1, 100):
            curr = lineage_weight(n)
            assert curr < prev, f"lineage_weight({n-1})={prev} <= lineage_weight({n})={curr}"
            prev = curr

    def test_custom_decay(self):
        """L1-c: 自定义decay参数生效"""
        assert lineage_weight(1, decay=0.5) == 0.5
        assert lineage_weight(2, decay=0.5) == 0.25


# ═══════════════════════════════════════════════════════════════
# L2 永不归零
# ═══════════════════════════════════════════════════════════════

class TestL2NeverZero:
    def test_large_n_positive(self):
        """L2: lineage_weight(10000) > 0"""
        w = lineage_weight(10000)
        assert w > 0
        assert w < 1.0  # 但也确实衰减了

    def test_very_large_n(self):
        """L2-b: lineage_weight(10000) > 0（指数衰减永不归零）"""
        w = lineage_weight(10000)
        assert w > 0


# ═══════════════════════════════════════════════════════════════
# L3 负数拒绝
# ═══════════════════════════════════════════════════════════════

class TestL3NegativeReject:
    def test_negative_raises(self):
        """L3: lineage_weight(-1) 抛 ValueError"""
        with pytest.raises(ValueError, match="n_sessions 不能为负"):
            lineage_weight(-1)

    def test_negative_large_raises(self):
        """L3-b: lineage_weight(-100) 抛 ValueError"""
        with pytest.raises(ValueError, match="n_sessions 不能为负"):
            lineage_weight(-100)
