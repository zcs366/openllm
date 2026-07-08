"""tests/test_topology.py — 六体拓扑单元测试。"""

import sys
from pathlib import Path

# 确保 importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from openllm.iai.topology import (
    AI_BODIES,
    DUTIES,
    Body,
    BodyTopology,
    get_signals,
    signal_graph,
)


class TestBodyEnum:
    def test_has_six_bodies(self):
        assert len(Body) == 6

    def test_expected_members(self):
        expected = {"IAI", "IAX", "ISA", "IOS", "ISN", "USER"}
        assert {b.value for b in Body} == expected


class TestDuties:
    def test_all_bodies_have_duties(self):
        for body in Body:
            assert body in DUTIES, f"{body.value} 缺少职责定义"

    def test_all_duties_are_strings(self):
        for body, duty in DUTIES.items():
            assert isinstance(duty, str) and len(duty) > 5


class TestSignals:
    def test_signal_count(self):
        signals = get_signals()
        assert len(signals) >= 7  # 至少覆盖核心通路

    def test_all_sources_are_bodies(self):
        for src, dst, _desc in get_signals():
            assert isinstance(src, Body)
            assert isinstance(dst, Body)

    def test_user_can_send_to_iai(self):
        graph = signal_graph()
        assert Body.IAI in graph[Body.USER]

    def test_ios_can_send_to_user(self):
        graph = signal_graph()
        assert Body.USER in graph[Body.IOS]


class TestTopology:
    def setup_method(self):
        self.topo = BodyTopology()

    def test_six_bodies(self):
        assert len(self.topo.bodies) == 6

    def test_ai_bodies_excludes_user(self):
        assert Body.USER not in self.topo.ai_bodies
        assert len(self.topo.ai_bodies) == 5

    def test_red_line_holds(self):
        """赫尔墨斯红线：AI体无终端对话权。"""
        assert self.topo.verify_red_line()

    def test_only_user_has_terminal_power(self):
        assert self.topo.has_terminal_dialogue_power(Body.USER)
        for body in self.topo.ai_bodies:
            assert not self.topo.has_terminal_dialogue_power(body)

    def test_in_degree_non_negative(self):
        for body in Body:
            assert self.topo.in_degree(body) >= 0

    def test_out_degree_non_negative(self):
        for body in Body:
            assert self.topo.out_degree(body) >= 0

    def test_user_has_nonzero_in_out(self):
        """USER 既是输入源也是输出目标。"""
        assert self.topo.in_degree(Body.USER) > 0
        assert self.topo.out_degree(Body.USER) > 0
