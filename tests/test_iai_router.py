"""test_iai_router.py — IAI快路径路由引擎测试"""
import tempfile
import asyncio
from pathlib import Path

import pytest
from openllm.core.router import RuleRouter, PhaseAction, RouteDecision, RoutingContext
from openllm.iai.event_bus import EventBus, Event
from openllm.iai.router import IAIRouter, EVENT_ROUTING_DECIDED


class TestIAIRouterBasics:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.bus = EventBus(log_dir=Path(self.tmp), ttl_hours=24)
        self.router = IAIRouter(bus=self.bus)

    def test_returns_4_phase_decisions(self):
        ctx = RoutingContext(user_input="帮我写个脚本")
        decisions = self.router.route(ctx)
        assert len(decisions) == 4
        phases = [d.phase for d in decisions]
        assert phases == ["PLAN", "ACT", "OBSERVE", "REFLECT"]

    def test_actions_are_valid_enum(self):
        ctx = RoutingContext(user_input="你好")
        decisions = self.router.route(ctx)
        for d in decisions:
            assert d.action in (PhaseAction.RUN, PhaseAction.SKIP, PhaseAction.DEGRADED)

    def test_greeting_skips_act_and_observe(self):
        ctx = RoutingContext(user_input="hello")
        decisions = self.router.route(ctx)
        act = next(d for d in decisions if d.phase == "ACT")
        observe = next(d for d in decisions if d.phase == "OBSERVE")
        assert act.action == PhaseAction.SKIP
        assert observe.action == PhaseAction.SKIP

    def test_file_operation_runs_all(self):
        ctx = RoutingContext(user_input="读取文件config.yaml")
        decisions = self.router.route(ctx)
        for d in decisions:
            assert d.action == PhaseAction.RUN

    def test_high_risk_runs_all(self):
        ctx = RoutingContext(user_input="删除所有数据", risk_level="critical")
        decisions = self.router.route(ctx)
        for d in decisions:
            assert d.action == PhaseAction.RUN


class TestRouteEventType:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.bus = EventBus(log_dir=Path(self.tmp), ttl_hours=24)
        self.router = IAIRouter(bus=self.bus)

    def test_simple_event_skips_tools(self):
        ctx = RoutingContext(user_input="谢谢")
        decisions = self.router.route_event_type("thanks", ctx)
        assert len(decisions) == 4
        act = next(d for d in decisions if d.phase == "ACT")
        assert act.action == PhaseAction.SKIP

    def test_complex_event_runs_all(self):
        ctx = RoutingContext(user_input="执行部署")
        decisions = self.router.route_event_type("deploy", ctx)
        for d in decisions:
            assert d.action == PhaseAction.RUN

    def test_unknown_event_falls_through(self):
        ctx = RoutingContext(user_input="这是普通消息")
        decisions = self.router.route_event_type("unknown_type", ctx)
        # 未知事件类型应走标准 route()
        assert len(decisions) == 4


class TestRedline:
    def test_redline_blocks_non_standard_actions(self):
        router = IAIRouter()
        # PhaseAction 只有 RUN/SKIP/DEGRADED，所以直接传入不会触发
        # 但 _redline 可以被测试逻辑覆盖
        fake_decisions = [RouteDecision("PLAN", PhaseAction.RUN, "test")]
        result = IAIRouter._redline(fake_decisions)
        assert len(result) == 1
        assert result[0].action == PhaseAction.RUN

    def test_terminate_action_becomes_degraded(self):
        """模拟一个非法action被红线拦截"""
        class FakeAction:
            name = "terminate"
        d = RouteDecision("PLAN", FakeAction(), "dangerous")
        result = IAIRouter._redline([d])
        assert result[0].action == PhaseAction.DEGRADED
        assert "红线覆盖" in result[0].reason


class TestOverride:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.bus = EventBus(log_dir=Path(self.tmp), ttl_hours=24)
        self.router = IAIRouter(bus=self.bus)

    def test_override_applies_once(self):
        custom = [
            RouteDecision("PLAN", PhaseAction.SKIP, "override plan"),
            RouteDecision("ACT", PhaseAction.SKIP, "override act"),
            RouteDecision("OBSERVE", PhaseAction.SKIP, "override obs"),
            RouteDecision("REFLECT", PhaseAction.SKIP, "override refl"),
        ]
        self.router.set_override("test input", custom)
        ctx = RoutingContext(user_input="test input")
        decisions = self.router.route(ctx)
        # Should use override
        assert all(d.action == PhaseAction.SKIP for d in decisions)
        # Second call: override consumed, back to normal rules
        ctx2 = RoutingContext(user_input="test input")
        decisions2 = self.router.route(ctx2)
        # "test input" doesn't match greeting/file/code patterns, so default
        assert any(d.action == PhaseAction.RUN for d in decisions2)


class TestEventBusIntegration:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.bus = EventBus(log_dir=Path(self.tmp), ttl_hours=24)
        self.router = IAIRouter(bus=self.bus)

    def test_publishes_routing_decided_event(self):
        received = []
        self.bus.subscribe(
            lambda e: received.append(e),
            source_filter="IAI",
            type_filter=EVENT_ROUTING_DECIDED,
        )
        ctx = RoutingContext(user_input="写代码")
        self.router.route(ctx)
        assert len(received) == 1
        evt = received[0]
        assert evt.source == "IAI"
        assert evt.type == EVENT_ROUTING_DECIDED
        assert "decisions" in evt.payload
        assert "PLAN" in evt.payload["decisions"]

    def test_event_payload_contains_risk_level(self):
        received = []
        self.bus.subscribe(lambda e: received.append(e), source_filter="IAI")
        ctx = RoutingContext(user_input="你好", risk_level="high")
        self.router.route(ctx)
        assert received[0].payload["risk_level"] == "high"

    def test_route_event_type_publishes_event(self):
        received = []
        self.bus.subscribe(lambda e: received.append(e), source_filter="IAI")
        ctx = RoutingContext(user_input="ack")
        self.router.route_event_type("ack", ctx)
        assert len(received) == 1


class TestAsyncSlowPath:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.bus = EventBus(log_dir=Path(self.tmp), ttl_hours=24)
        self.router = IAIRouter(bus=self.bus)

    def test_analyze_async_returns_none(self):
        ctx = RoutingContext(user_input="complex question")
        result = asyncio.get_event_loop().run_until_complete(
            self.router.analyze_async(ctx)
        )
        assert result is None


class TestStats:
    def test_stats_include_iai_route_count(self):
        router = IAIRouter()
        ctx = RoutingContext(user_input="hello")
        router.route(ctx)
        router.route(ctx)
        stats = router.get_stats()
        assert stats["iai_route_count"] == 2
        assert stats["total_decisions"] == 2


class TestStandalone:
    def test_no_bus_creates_own(self):
        router = IAIRouter()
        ctx = RoutingContext(user_input="hi")
        decisions = router.route(ctx)
        assert len(decisions) == 4
