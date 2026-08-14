"""test_bias_monitor.py — 偏差监控器测试。

验证规则：
1. record 记录验证判决并持久化
2. get_bias 计算某来源的偏差率
3. is_biased 判断是否有显著偏差
4. summary 统计摘要
5. 多来源独立统计
6. 持久化（写入/读取JSON）
7. 空记录处理
"""

import json
import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def tmp_state(tmp_path):
    """临时状态文件路径。"""
    return tmp_path / "bias_state.json"


@pytest.fixture
def monitor(tmp_state):
    """初始化 BiasMonitor（使用临时文件）。"""
    from openllm.core.bias_monitor import BiasMonitor
    return BiasMonitor(state_path=tmp_state)


# ═══════════════════════════════════════════════════════
# record 测试
# ═══════════════════════════════════════════════════════

class TestRecord:
    """记录功能测试。"""

    def test_record_basic(self, monitor):
        """基本记录操作。"""
        monitor.record("地球是圆的", True, "rule")
        monitor.record("太阳是方的", False, "rule")
        summary = monitor.summary()
        assert summary["total_records"] == 2

    def test_record_persists(self, tmp_state):
        """记录持久化到文件。"""
        from openllm.core.bias_monitor import BiasMonitor

        m1 = BiasMonitor(state_path=tmp_state)
        m1.record("测试声明", True, "test_source")
        # 重新加载
        m2 = BiasMonitor(state_path=tmp_state)
        summary = m2.summary()
        assert summary["total_records"] == 1

    def test_record_multiple_sources(self, monitor):
        """多来源记录。"""
        monitor.record("声明A", True, "source_a")
        monitor.record("声明B", False, "source_b")
        summary = monitor.summary()
        assert len(summary["sources"]) == 2


# ═══════════════════════════════════════════════════════
# get_bias 测试
# ═══════════════════════════════════════════════════════

class TestGetBias:
    """偏差率计算测试。"""

    def test_no_records_returns_zero(self, monitor):
        """无记录时偏差率为0。"""
        assert monitor.get_bias("unknown") == 0.0

    def test_balanced_bias_zero(self, monitor):
        """完全平衡时偏差率为0。"""
        monitor.record("A", True, "balanced")
        monitor.record("B", False, "balanced")
        assert monitor.get_bias("balanced") == 0.0

    def test_completely_biased(self, monitor):
        """完全偏向时偏差率为1.0。"""
        for i in range(5):
            monitor.record(f"声明{i}", True, "always_true")
        assert monitor.get_bias("always_true") == 1.0

    def test_completely_biased_false(self, monitor):
        """完全拒绝时偏差率为1.0。"""
        for i in range(5):
            monitor.record(f"声明{i}", False, "always_false")
        assert monitor.get_bias("always_false") == 1.0

    def test_partial_bias(self, monitor):
        """部分偏向——75%通过。"""
        for i in range(3):
            monitor.record(f"声明{i}", True, "partial")
        monitor.record("声明X", False, "partial")
        # approval_rate = 3/4 = 0.75
        # bias = |0.75 - 0.5| * 2 = 0.5
        assert abs(monitor.get_bias("partial") - 0.5) < 1e-9

    def test_source_independence(self, monitor):
        """不同来源的偏差率独立计算。"""
        # source_a: 全true
        for i in range(5):
            monitor.record(f"A{i}", True, "source_a")
        # source_b: 全false
        for i in range(5):
            monitor.record(f"B{i}", False, "source_b")

        assert monitor.get_bias("source_a") == 1.0
        assert monitor.get_bias("source_b") == 1.0


# ═══════════════════════════════════════════════════════
# is_biased 测试
# ═══════════════════════════════════════════════════════

class TestIsBiased:
    """偏差判断测试。"""

    def test_balanced_not_biased(self, monitor):
        """平衡来源无偏差。"""
        monitor.record("A", True, "fair")
        monitor.record("B", False, "fair")
        assert not monitor.is_biased("fair")

    def test_biased_detected(self, monitor):
        """偏差来源被检测。"""
        for i in range(5):
            monitor.record(f"声明{i}", True, "biased_src")
        assert monitor.is_biased("biased_src")

    def test_custom_threshold(self, monitor):
        """自定义阈值。"""
        for i in range(3):
            monitor.record(f"声明{i}", True, "mild")
        # approval_rate = 1.0, bias = 1.0, threshold=0.5 → biased
        assert monitor.is_biased("mild", threshold=0.5)
        # threshold=1.0 → not biased
        assert not monitor.is_biased("mild", threshold=1.0)

    def test_no_records_not_biased(self, monitor):
        """无记录不认为偏差。"""
        assert not monitor.is_biased("nonexistent")


# ═══════════════════════════════════════════════════════
# summary 测试
# ═══════════════════════════════════════════════════════

class TestSummary:
    """摘要测试。"""

    def test_empty_summary(self, monitor):
        """空记录摘要。"""
        s = monitor.summary()
        assert s["total_records"] == 0
        assert s["sources"] == {}

    def test_summary_structure(self, monitor):
        """摘要结构完整。"""
        monitor.record("A", True, "src")
        s = monitor.summary()
        assert "total_records" in s
        assert "global_approval_rate" in s
        assert "global_bias_rate" in s
        assert "sources" in s

    def test_summary_source_stats(self, monitor):
        """来源统计准确。"""
        monitor.record("A", True, "my_src")
        monitor.record("B", True, "my_src")
        monitor.record("C", False, "my_src")
        s = monitor.summary()
        src = s["sources"]["my_src"]
        assert src["total"] == 3
        assert src["true_count"] == 2
        assert src["false_count"] == 1
        assert src["approval_rate"] == pytest.approx(2 / 3, abs=1e-4)
        assert src["bias_rate"] == pytest.approx(abs(2 / 3 - 0.5) * 2, abs=1e-4)

    def test_global_stats(self, monitor):
        """全局统计准确。"""
        monitor.record("A", True, "x")
        monitor.record("B", False, "y")
        s = monitor.summary()
        assert s["total_records"] == 2
        assert s["global_approval_rate"] == 0.5


# ═══════════════════════════════════════════════════════
# clear 测试
# ═══════════════════════════════════════════════════════

class TestClear:
    """清空测试。"""

    def test_clear_removes_records(self, monitor):
        """清空后无记录。"""
        monitor.record("A", True, "src")
        monitor.clear()
        assert monitor.summary()["total_records"] == 0

    def test_clear_removes_file(self, tmp_state):
        """清空后删除文件。"""
        from openllm.core.bias_monitor import BiasMonitor

        m = BiasMonitor(state_path=tmp_state)
        m.record("A", True, "src")
        assert tmp_state.exists()
        m.clear()
        assert not tmp_state.exists()


# ═══════════════════════════════════════════════════════
# 边界情况
# ═══════════════════════════════════════════════════════

class TestEdgeCases:
    """边界情况测试。"""

    def test_single_record_bias(self, monitor):
        """单条记录偏差率为1.0。"""
        monitor.record("only", True, "single")
        assert monitor.get_bias("single") == 1.0

    def test_many_records(self, monitor):
        """大量记录性能。"""
        for i in range(1000):
            monitor.record(f"声明{i}", i % 3 != 0, "bulk")
        # 2/3 true → approval_rate ≈ 0.667 → bias ≈ 0.333
        bias = monitor.get_bias("bulk")
        assert 0.33 < bias < 0.34

    def test_corrupt_state_file(self, tmp_state):
        """损坏的状态文件不崩溃。"""
        from openllm.core.bias_monitor import BiasMonitor

        tmp_state.parent.mkdir(parents=True, exist_ok=True)
        tmp_state.write_text("NOT JSON!!!", encoding="utf-8")
        m = BiasMonitor(state_path=tmp_state)
        assert m.summary()["total_records"] == 0

    def test_empty_source(self, monitor):
        """空来源标识正常工作。"""
        monitor.record("声明", True, "")
        assert monitor.get_bias("") == 1.0
