"""
OpenLLM 元认知层测试。
"""

import time
from openllm.core.meta import (
    CognitiveDashboard, MetaSnapshot, CognitiveState, SelfRescue,
)


class TestCognitiveDashboard:
    def test_initial_state(self):
        dash = CognitiveDashboard()
        assert dash.success_rate == 1.0
        assert dash.avg_latency == 0.0

    def test_record_tool_call(self):
        dash = CognitiveDashboard()
        dash.record_tool_call(True)
        dash.record_tool_call(True)
        dash.record_tool_call(False)
        assert dash.tool_total == 3
        assert dash.tool_success == 2
        assert dash.success_rate == 2/3

    def test_record_latency(self):
        dash = CognitiveDashboard()
        dash.record_latency(100)
        dash.record_latency(200)
        assert dash.avg_latency == 150.0

    def test_healthy_at_low_context(self):
        dash = CognitiveDashboard()
        snap = dash.evaluate(context_used=2000, turn_count=5)
        assert snap.cognitive_state == CognitiveState.HEALTHY
        assert not snap.overloaded
        assert not snap.should_compress

    def test_fatigued_at_70pct(self):
        dash = CognitiveDashboard(max_context=8000)
        snap = dash.evaluate(context_used=6000, turn_count=20)
        assert snap.cognitive_state == CognitiveState.FATIGUED
        assert snap.should_compress

    def test_overloaded_at_85pct(self):
        dash = CognitiveDashboard(max_context=8000)
        snap = dash.evaluate(context_used=7000, turn_count=30)
        assert snap.cognitive_state == CognitiveState.OVERLOADED
        assert snap.overloaded
        assert snap.should_compress

    def test_degraded_when_tools_fail(self):
        dash = CognitiveDashboard()
        for _ in range(10):
            dash.record_tool_call(False)
        snap = dash.evaluate(context_used=1000, turn_count=5)
        assert snap.cognitive_state == CognitiveState.DEGRADED
        assert snap.should_rest

    def test_trend_detection(self):
        dash = CognitiveDashboard(max_context=8000)
        dash.evaluate(context_used=4000, turn_count=5)  # 50%, healthy
        snap = dash.evaluate(context_used=6500, turn_count=15)  # 81%, +31pp
        assert snap.trend == "declining"

    def test_self_awareness_report(self):
        dash = CognitiveDashboard()
        dash.evaluate(context_used=1000, turn_count=3)
        report = dash.self_awareness_report()
        assert "认知状态" in report
        assert "🟢" in report

    def test_history_tracking(self):
        dash = CognitiveDashboard()
        for i in range(5):
            dash.evaluate(context_used=i*1000, turn_count=i)
        assert len(dash.history) == 5

    def test_reset_ephemeral(self):
        dash = CognitiveDashboard()
        dash.record_tool_call(True)
        dash.record_latency(100)
        dash.evaluate(context_used=1000, turn_count=1)
        dash.reset_ephemeral()
        # 延迟归零
        assert dash.avg_latency == 0.0
        # 工具统计保留
        assert dash.tool_total == 1


class TestSelfRescue:
    def test_overload_rescue(self):
        dash = CognitiveDashboard(max_context=1000)
        rescue = SelfRescue(dash)
        snap = dash.evaluate(context_used=900, turn_count=10)
        msg = rescue.on_overload(snap)
        assert "过载" in msg
        assert rescue.rescue_count == 1

    def test_fatigue_rescue(self):
        dash = CognitiveDashboard(max_context=1000)
        rescue = SelfRescue(dash)
        snap = dash.evaluate(context_used=750, turn_count=15)
        msg = rescue.on_fatigue(snap)
        assert "疲劳" in msg

    def test_callback_integration(self):
        """自救回调与Dashboard集成。"""
        dash = CognitiveDashboard(max_context=1000)
        rescue = SelfRescue(dash)
        dash.on_overload = rescue.on_overload
        dash.on_fatigue = rescue.on_fatigue

        snap = dash.evaluate(context_used=900, turn_count=10)
        assert snap.overloaded
        # 回调后自救计数增加
        assert rescue.rescue_count >= 1


class TestMetaSnapshot:
    def test_defaults(self):
        snap = MetaSnapshot()
        assert snap.cognitive_state == CognitiveState.HEALTHY
        assert snap.tool_success_rate == 1.0
        assert not snap.overloaded

    def test_custom(self):
        snap = MetaSnapshot(
            context_used_pct=85.0,
            tool_success_rate=0.65,
            cognitive_state=CognitiveState.OVERLOADED,
            overloaded=True,
            should_compress=True,
        )
        assert snap.context_used_pct == 85.0
        assert snap.should_compress


# ── Engine 集成测试 ─────────────────────────────────

class TestEngineWithMeta:
    def test_engine_has_dashboard(self):
        import tempfile
        from pathlib import Path
        from openllm.core.engine import OpenLLMEngine, AgentConfig

        with tempfile.TemporaryDirectory() as tmp:
            config = AgentConfig(capsule_dir=tmp)
            engine = OpenLLMEngine(config)
            assert engine.dashboard is not None
            assert engine.rescue is not None

    def test_status_includes_cognitive(self):
        import tempfile
        from pathlib import Path
        from openllm.core.engine import OpenLLMEngine, AgentConfig

        with tempfile.TemporaryDirectory() as tmp:
            config = AgentConfig(capsule_dir=tmp)
            engine = OpenLLMEngine(config)
            s = engine.status()
            assert "cognitive" in s
            assert "tool_success_rate" in s
            assert "should_compress" in s

    def test_cognitive_report(self):
        import tempfile
        from pathlib import Path
        from openllm.core.engine import OpenLLMEngine, AgentConfig

        with tempfile.TemporaryDirectory() as tmp:
            config = AgentConfig(capsule_dir=tmp)
            engine = OpenLLMEngine(config)
            engine.wake()
            report = engine.cognitive_report()
            assert "认知状态" in report or "苏醒" in report
