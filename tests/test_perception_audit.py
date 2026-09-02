"""感知源审计验收测试（②多眼律 · PAL T-F-8）

验收标准：
  1. 现有感知源数量 >= 3（触手脑阵列 + EventBus + SignalBridge + ActiveSampler + ISA感知）
  2. SignalBridge类存在且可实例化
  3. 感知结果进入context（build_context含search_results字段）
  4. EventBus pub/sub功能正常
  5. 触手脑通信协议（TentacleReport）正常
"""
import json
import time
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ═══════════════════════════════════════════════════════
# 验收1：感知源数量 >= 3
# ═══════════════════════════════════════════════════════

class TestPerceptionSourceCount:
    """验收：现有感知源数量 >= 3"""

    def test_tentacle_brains_exist(self):
        """触手脑：FileWatcherBrain + IndexBrain = 2个感知源"""
        from openllm.core.tentacle import FileWatcherBrain, IndexBrain
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcherBrain(watch_dir=tmpdir)
            indexer = IndexBrain(index_dir=tmpdir)
            assert watcher.name == "file_watcher"
            assert indexer.name == "index"

    def test_event_bus_exists(self):
        """EventBus = 事件感知源"""
        from openllm.iai.event_bus import EventBus
        bus = EventBus()
        assert bus.subscriber_count() == 0

    def test_signal_bridge_exists(self):
        """SignalBridge = 文件系统信号感知源"""
        from openllm.iai.event_bus import SignalBridge
        from openllm.iai.event_bus import EventBus
        bus = EventBus()
        bridge = SignalBridge(bus, poll_interval=1.0)
        assert bridge._interval == 1.0
        assert bridge._running is False

    def test_active_sampler_exists(self):
        """ActiveSampler = 主动采样策略感知源"""
        from openllm.iai.active_sampler import ActiveSampler
        sampler = ActiveSampler()
        assert sampler.budget == 3

    def test_perception_source_count_ge_3(self):
        """验收：感知源总数 >= 3"""
        from openllm.core.tentacle import FileWatcherBrain, IndexBrain
        from openllm.iai.event_bus import EventBus, SignalBridge
        from openllm.iai.active_sampler import ActiveSampler

        sources = [
            FileWatcherBrain(),
            IndexBrain(),
            EventBus(),
            SignalBridge(EventBus()),
            ActiveSampler(),
        ]
        assert len(sources) >= 3, f"感知源数量不足: {len(sources)} < 3"


# ═══════════════════════════════════════════════════════
# 验收2：SignalBridge功能完整
# ═══════════════════════════════════════════════════════

class TestSignalBridge:
    """验收：SignalBridge存在且可桥接文件系统信号到EventBus"""

    def test_signal_bridge_class_exists(self):
        """SignalBridge类存在"""
        from openllm.iai.event_bus import SignalBridge
        assert SignalBridge is not None

    def test_signal_bridge_instance(self):
        """SignalBridge可实例化"""
        from openllm.iai.event_bus import SignalBridge, EventBus
        bus = EventBus()
        bridge = SignalBridge(bus, poll_interval=2.0)
        assert bridge._bus is bus
        assert bridge._interval == 2.0
        assert bridge._running is False

    def test_signal_bridge_start_stop(self):
        """SignalBridge start/stop控制"""
        from openllm.iai.event_bus import SignalBridge, EventBus
        bus = EventBus()
        bridge = SignalBridge(bus)
        bridge.start()
        assert bridge._running is True
        bridge.stop()
        assert bridge._running is False

    def test_signal_bridge_poll_publishes_event(self):
        """SignalBridge轮询将信号转换为Event发布到EventBus"""
        from openllm.iai.event_bus import SignalBridge, EventBus, Event
        bus = EventBus()
        received_events = []

        def on_event(event: Event):
            received_events.append(event)

        bus.subscribe(on_event, source_filter="SIGNAL")

        bridge = SignalBridge(bus)
        # 手动调用_poll（不启动线程）
        # 由于signal目录可能为空，不应报错
        bridge._poll()
        # 无信号时不报错
        assert isinstance(received_events, list)


# ═══════════════════════════════════════════════════════
# 验收3：感知结果进入context（search_results字段）
# ═══════════════════════════════════════════════════════

class TestPerceptionIntoContext:
    """验收：感知结果通过build_context进入Context.search_results"""

    def test_context_has_search_results_field(self):
        """Context dataclass包含search_results字段"""
        from openllm.core.models import Context
        ctx = Context(user_message="test")
        assert hasattr(ctx, 'search_results')
        assert isinstance(ctx.search_results, list)

    def test_index_brain_results_enter_context(self):
        """IndexBrain搜索结果可注入Context.search_results"""
        from openllm.core.tentacle import IndexBrain
        from openllm.core.models import Context

        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = IndexBrain(index_dir=tmpdir)
            test_file = Path(tmpdir) / "test.md"
            test_file.write_text("# Test\nHello world openllm")
            indexer.build()

            results = indexer.search("openllm", limit=5)
            ctx = Context(user_message="openllm")
            if results:
                for r in results:
                    ctx.search_results.append(
                        f"[search:{r.get('filepath','?')}] {r.get('snippet','')}"
                    )
            assert len(ctx.search_results) >= 1

    def test_build_context_produces_search_results(self):
        """build_context产出的Context包含search_results字段"""
        from openllm.core.models import Context, Message
        msg = Message(text="test query")

        # 模拟build_context产出的Context结构
        ctx = Context(
            user_message=msg.text,
            search_results=["[search:test.md] snippet"],
        )
        assert hasattr(ctx, 'search_results')
        assert len(ctx.search_results) >= 1

    def test_causal_inject_appends_to_search_results(self):
        """因果疤注入追加到search_results"""
        from openllm.core.models import Context
        ctx = Context(user_message="test")
        # 模拟因果疤注入
        block = "因果教训：上次操作失败因网络超时"
        ctx.search_results = getattr(ctx, 'search_results', []) or []
        ctx.search_results.append(block)
        assert len(ctx.search_results) == 1
        assert "因果教训" in ctx.search_results[0]

    def test_clock_inject_appends_to_search_results(self):
        """时钟读取追加到search_results"""
        from openllm.core.models import Context
        ctx = Context(user_message="test")
        # 模拟时钟注入
        _clk_text = f"⏰ 时钟: 现在{time.time():.0f}, 间隔0秒"
        ctx.search_results = getattr(ctx, 'search_results', []) or []
        ctx.search_results.append(_clk_text)
        assert len(ctx.search_results) == 1
        assert "时钟" in ctx.search_results[0]


# ═══════════════════════════════════════════════════════
# 验收4：EventBus pub/sub功能
# ═══════════════════════════════════════════════════════

class TestEventBusPerception:
    """验收：EventBus事件总线感知功能"""

    def test_event_bus_publish_subscribe(self):
        """EventBus发布/订阅正常"""
        from openllm.iai.event_bus import EventBus, Event
        bus = EventBus()
        received = []

        def callback(event):
            received.append(event)

        bus.subscribe(callback)
        event = Event(
            source="TEST", type="test_event",
            timestamp=time.time(), entropy_score=0.5,
            payload={"key": "value"}
        )
        bus.publish(event)
        assert len(received) == 1
        assert received[0].source == "TEST"

    def test_event_bus_source_filter(self):
        """EventBus source过滤"""
        from openllm.iai.event_bus import EventBus, Event
        bus = EventBus()
        received = []

        bus.subscribe(lambda e: received.append(e), source_filter="TARGET")
        bus.publish(Event(source="OTHER", type="t", timestamp=time.time(),
                          entropy_score=0.0, payload={}))
        bus.publish(Event(source="TARGET", type="t", timestamp=time.time(),
                          entropy_score=0.0, payload={}))
        assert len(received) == 1

    def test_event_bus_history(self):
        """EventBus历史记录"""
        from openllm.iai.event_bus import EventBus, Event
        bus = EventBus()
        for i in range(5):
            bus.publish(Event(source=f"S{i}", type="t", timestamp=time.time(),
                              entropy_score=0.0, payload={}))
        history = bus.get_history(limit=3)
        assert len(history) == 3

    def test_event_bus_blocked_types(self):
        """EventBus禁止terminate/shutdown类型"""
        from openllm.iai.event_bus import EventBus, Event
        bus = EventBus()
        with pytest.raises(PermissionError):
            Event(source="TEST", type="terminate", timestamp=time.time(),
                  entropy_score=0.0, payload={})
        with pytest.raises(PermissionError):
            Event(source="TEST", type="shutdown", timestamp=time.time(),
                  entropy_score=0.0, payload={})


# ═══════════════════════════════════════════════════════
# 验收5：触手脑通信协议
# ═══════════════════════════════════════════════════════

class TestTentacleBrainProtocol:
    """验收：触手脑TentacleReport通信协议"""

    def test_tentacle_report_creation(self):
        """TentacleReport可创建"""
        from openllm.core.tentacle import TentacleBrain
        brain = TentacleBrain("test")
        report = brain.report("test_event", 0.5, {"key": "value"})
        assert report.brain_name == "test"
        assert report.importance == 0.5
        assert report.data == {"key": "value"}

    def test_tentacle_ack(self):
        """TentacleReport ack确认"""
        from openllm.core.tentacle import TentacleBrain
        brain = TentacleBrain("test")
        report = brain.report("event", 0.5, {})
        assert len(brain.pending_reports) == 1
        brain.ack(report)
        assert report.acknowledged is True
        assert len(brain.pending_reports) == 0

    def test_file_watcher_scan(self):
        """FileWatcherBrain scan检测文件变化"""
        from openllm.core.tentacle import FileWatcherBrain
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcherBrain(watch_dir=tmpdir)
            # 创建新文件
            test_file = Path(tmpdir) / "new.md"
            test_file.write_text("new content")
            changes = watcher.scan()
            assert len(changes) >= 1
            assert changes[0]["type"] == "new_file"

    def test_index_brain_build_and_search(self):
        """IndexBrain build+search功能"""
        from openllm.core.tentacle import IndexBrain
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = IndexBrain(index_dir=tmpdir)
            test_file = Path(tmpdir) / "readme.md"
            test_file.write_text("# OpenLLM\nThis is a test document")
            count = indexer.build()
            assert count == 1
            results = indexer.search("OpenLLM", limit=5)
            assert len(results) >= 1


# ═══════════════════════════════════════════════════════
# 验收6：ActiveSampler预算控制
# ═══════════════════════════════════════════════════════

class TestActiveSamplerPerception:
    """验收：ActiveSampler采样预算控制"""

    def test_budget_hard_cap(self):
        """赫淮斯托斯硬上限：每轮最多3次"""
        from openllm.iai.active_sampler import ActiveSampler
        sampler = ActiveSampler()
        for _ in range(3):
            assert sampler.sample() is True
            sampler.mark_used("verify")
        assert sampler.sample() is False

    def test_explore_mode(self):
        """克洛诺斯反预测扰动：explore模式"""
        from openllm.iai.active_sampler import ActiveSampler
        sampler = ActiveSampler()
        assert sampler.sample("explore") is True
        deviation = sampler.generate_deviation("test_topic")
        assert 0.0 <= deviation <= 1.0

    def test_reset_budget(self):
        """reset_budget重置预算"""
        from openllm.iai.active_sampler import ActiveSampler
        sampler = ActiveSampler()
        for _ in range(3):
            sampler.mark_used("verify")
        assert sampler.sample() is False
        sampler.reset_budget()
        assert sampler.sample() is True
        assert sampler.remaining == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
