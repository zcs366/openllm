"""
tests/test_brain_isolation.py — 多Agent头脑隔离测试（PAL T-B-3）

验证目标：
  1. Event 带 brain_id 字段（默认 "default"）
  2. _emit 注入 BrainActivator.current() 的活跃头脑
  3. EventBus.subscribe 支持 brain_id 过滤
  4. 向后兼容：无头脑激活时 brain_id="default"
  5. 协作监听：brain_id=None 的订阅者收到所有头脑的事件
"""
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from openllm.iai.event_bus import Event, EventBus
from openllm.iai.brain import BrainActivator, BrainRegistry, BRAIN_VOCAB


# ── fixtures ──────────────────────────────────────────────

@pytest.fixture
def bus(tmp_path: Path) -> EventBus:
    """独立 EventBus，JSONL 写到 tmp_path。"""
    return EventBus(log_dir=tmp_path / "events")


@pytest.fixture
def activator(tmp_path: Path) -> BrainActivator:
    """独立 BrainActivator，状态文件写到 tmp_path。"""
    return BrainActivator(path=tmp_path / "active_brain.json")


@pytest.fixture
def registry(tmp_path: Path) -> BrainRegistry:
    """独立 BrainRegistry。"""
    return BrainRegistry(path=tmp_path / "brain_registry.json")


# ═══════════════════════════════════════════════════════════
# 1. Event 字段：默认 brain_id
# ═══════════════════════════════════════════════════════════

class TestEventBrainIdField:
    def test_default_brain_id(self):
        """Event 默认 brain_id='default'（向后兼容）。"""
        evt = Event(source="test", type="unit.test", timestamp=time.time(),
                    entropy_score=0.0, payload={"k": "v"})
        assert evt.brain_id == "default"

    def test_custom_brain_id(self):
        """Event 可以设置自定义 brain_id。"""
        evt = Event(source="test", type="unit.test", timestamp=time.time(),
                    entropy_score=0.0, payload={}, brain_id="brain-junshi")
        assert evt.brain_id == "brain-junshi"

    def test_to_dict_includes_brain_id(self):
        """to_dict 输出包含 brain_id。"""
        evt = Event(source="test", type="unit.test", timestamp=time.time(),
                    entropy_score=0.0, payload={}, brain_id="brain-zi-gong")
        d = evt.to_dict()
        assert "brain_id" in d
        assert d["brain_id"] == "brain-zi-gong"

    def test_from_dict_restores_brain_id(self):
        """from_dict 恢复 brain_id（旧数据无 brain_id 则默认 'default'）。"""
        d = {"source": "s", "type": "t", "timestamp": 1.0, "payload": {},
             "brain_id": "brain-bao"}
        evt = Event.from_dict(d)
        assert evt.brain_id == "brain-bao"

    def test_from_dict_legacy_no_brain_id(self):
        """from_dict 处理无 brain_id 的旧数据（默认 'default'）。"""
        d = {"source": "s", "type": "t", "timestamp": 1.0, "payload": {}}
        evt = Event.from_dict(d)
        assert evt.brain_id == "default"


# ═══════════════════════════════════════════════════════════
# 2. brain_id 注入：BrainActivator.current() 自动注入
# ═══════════════════════════════════════════════════════════

class TestBrainIdInjection:
    def test_no_active_brain_default(self, bus: EventBus, activator: BrainActivator):
        """无活跃头脑时 emit → brain_id='default'。"""
        from openllm.iai.core import IAI
        iai = IAI(log_dir=Path("/tmp/test_iai_brain_iso"))
        # 替换 bus 为我们的独立 bus
        iai.bus = bus
        # 确保 BrainActivator 读到我们的 activator（无活跃头脑）
        with patch("openllm.iai.core.BrainActivator", return_value=activator):
            iai.emit("test.event", {"msg": "hello"})
        history = bus.get_history()
        assert len(history) == 1
        assert history[0]["brain_id"] == "default"

    def test_active_brain_injected(self, bus: EventBus, activator: BrainActivator):
        """激活头脑A后 emit → brain_id='brain-A'。"""
        from openllm.iai.core import IAI
        activator.activate("brain-A")
        iai = IAI(log_dir=Path("/tmp/test_iai_brain_iso2"))
        iai.bus = bus
        with patch("openllm.iai.core.BrainActivator", return_value=activator):
            iai.emit("test.event", {"msg": "from A"})
        history = bus.get_history()
        assert len(history) == 1
        assert history[0]["brain_id"] == "brain-A"

    def test_switch_brain_changes_event_source(self, bus: EventBus, activator: BrainActivator):
        """切换头脑后 emit → brain_id 跟着变。"""
        from openllm.iai.core import IAI
        iai = IAI(log_dir=Path("/tmp/test_iai_brain_iso3"))
        iai.bus = bus

        activator.activate("brain-left")
        with patch("openllm.iai.core.BrainActivator", return_value=activator):
            iai.emit("step.1", {"n": 1})

        activator.activate("brain-right")
        with patch("openllm.iai.core.BrainActivator", return_value=activator):
            iai.emit("step.2", {"n": 2})

        history = bus.get_history()
        assert len(history) == 2
        assert history[0]["brain_id"] == "brain-left"
        assert history[0]["payload"]["n"] == 1
        assert history[1]["brain_id"] == "brain-right"
        assert history[1]["payload"]["n"] == 2


# ═══════════════════════════════════════════════════════════
# 3. EventBus.subscribe brain_id 过滤
# ═══════════════════════════════════════════════════════════

class TestSubscribeBrainIdFilter:
    def test_subscribe_specific_brain_only_receives_own(self, bus: EventBus):
        """订阅 brain_id='A' 只收到 A 的事件。"""
        received = []
        bus.subscribe(lambda e: received.append(e), brain_id="A")

        bus.publish(Event(source="s", type="t", timestamp=time.time(),
                          entropy_score=0.0, payload={}, brain_id="A"))
        bus.publish(Event(source="s", type="t", timestamp=time.time(),
                          entropy_score=0.0, payload={}, brain_id="B"))
        bus.publish(Event(source="s", type="t", timestamp=time.time(),
                          entropy_score=0.0, payload={}, brain_id="A"))

        assert len(received) == 2
        assert all(e.brain_id == "A" for e in received)

    def test_subscribe_none_receives_all(self, bus: EventBus):
        """订阅 brain_id=None（协作监听）收到所有头脑的事件。"""
        received = []
        bus.subscribe(lambda e: received.append(e), brain_id=None)

        bus.publish(Event(source="s", type="t", timestamp=time.time(),
                          entropy_score=0.0, payload={}, brain_id="A"))
        bus.publish(Event(source="s", type="t", timestamp=time.time(),
                          entropy_score=0.0, payload={}, brain_id="B"))
        bus.publish(Event(source="s", type="t", timestamp=time.time(),
                          entropy_score=0.0, payload={}, brain_id="default"))

        assert len(received) == 3

    def test_subscribe_default_brain_receives_only_default(self, bus: EventBus):
        """订阅 brain_id='default' 只收默认事件，不收特定头脑的。"""
        received = []
        bus.subscribe(lambda e: received.append(e), brain_id="default")

        bus.publish(Event(source="s", type="t", timestamp=time.time(),
                          entropy_score=0.0, payload={}, brain_id="default"))
        bus.publish(Event(source="s", type="t", timestamp=time.time(),
                          entropy_score=0.0, payload={}, brain_id="brain-X"))

        assert len(received) == 1
        assert received[0].brain_id == "default"

    def test_combined_source_type_brain_filter(self, bus: EventBus):
        """三维过滤：source + type + brain_id 同时匹配才分发。"""
        received = []
        bus.subscribe(lambda e: received.append(e),
                      source_filter="IAI", type_filter="reason",
                      brain_id="brain-strategy")

        # source 匹配但 brain_id 不匹配
        bus.publish(Event(source="IAI", type="reason", timestamp=time.time(),
                          entropy_score=0.0, payload={}, brain_id="brain-other"))
        # brain_id 匹配但 source 不匹配
        bus.publish(Event(source="IOS", type="reason", timestamp=time.time(),
                          entropy_score=0.0, payload={}, brain_id="brain-strategy"))
        # 全部匹配
        bus.publish(Event(source="IAI", type="reason", timestamp=time.time(),
                          entropy_score=0.0, payload={}, brain_id="brain-strategy"))

        assert len(received) == 1


# ═══════════════════════════════════════════════════════════
# 4. 多订阅者竞争：同一事件类型，不同 brain_id 过滤
# ═══════════════════════════════════════════════════════════

class TestMultiSubscriberIsolation:
    def test_two_brains_only_hear_own(self, bus: EventBus):
        """两个头脑各自订阅同一事件类型，只收到自己的事件。"""
        a_received = []
        b_received = []
        bus.subscribe(lambda e: a_received.append(e), brain_id="brain-A")
        bus.subscribe(lambda e: b_received.append(e), brain_id="brain-B")

        bus.publish(Event(source="s", type="tick", timestamp=time.time(),
                          entropy_score=0.0, payload={}, brain_id="brain-A"))
        bus.publish(Event(source="s", type="tick", timestamp=time.time(),
                          entropy_score=0.0, payload={}, brain_id="brain-B"))
        bus.publish(Event(source="s", type="tick", timestamp=time.time(),
                          entropy_score=0.0, payload={}, brain_id="brain-A"))

        assert len(a_received) == 2
        assert len(b_received) == 1
        assert all(e.brain_id == "brain-A" for e in a_received)
        assert all(e.brain_id == "brain-B" for e in b_received)

    def test_coordinator_sees_all(self, bus: EventBus):
        """协调者（brain_id=None）看到所有头脑的事件。"""
        coordinator_received = []
        bus.subscribe(lambda e: coordinator_received.append(e), brain_id=None)

        for bid in ["brain-A", "brain-B", "default", "brain-C"]:
            bus.publish(Event(source="s", type="tick", timestamp=time.time(),
                              entropy_score=0.0, payload={}, brain_id=bid))

        assert len(coordinator_received) == 4
        assert {e.brain_id for e in coordinator_received} == {
            "brain-A", "brain-B", "default", "brain-C"}


# ═══════════════════════════════════════════════════════════
# 5. JSONL 持久化：brain_id 落盘
# ═══════════════════════════════════════════════════════════

class TestJsonlBrainIdPersistence:
    def test_jsonl_contains_brain_id(self, bus: EventBus):
        """JSONL 日志中包含 brain_id 字段。"""
        bus.publish(Event(source="s", type="t", timestamp=time.time(),
                          entropy_score=0.0, payload={}, brain_id="brain-test"))
        log_file = bus._log_file
        assert log_file.exists()
        lines = log_file.read_text().strip().splitlines()
        assert len(lines) == 1
        import json
        entry = json.loads(lines[0])
        assert "brain_id" in entry
        assert entry["brain_id"] == "brain-test"


# ═══════════════════════════════════════════════════════════
# 6. 向后兼容：现有构造不破坏
# ═══════════════════════════════════════════════════════════

class TestBackwardCompatibility:
    def test_event_no_brain_id_kwarg(self):
        """旧代码不传 brain_id 也能构造 Event（默认 'default'）。"""
        evt = Event(source="x", type="y", timestamp=1.0, entropy_score=0.0, payload={})
        assert evt.brain_id == "default"

    def test_bus_publish_no_brain_id(self, bus: EventBus):
        """EventBus.publish 接受无 brain_id 的事件（默认 'default'）。"""
        count = bus.publish(Event(source="x", type="y", timestamp=time.time(),
                                  entropy_score=0.0, payload={}))
        assert count == 0  # 无订阅者
        history = bus.get_history()
        assert len(history) == 1
        assert history[0]["brain_id"] == "default"

    def test_subscribe_no_brain_id_kwarg(self, bus: EventBus):
        """subscribe 不传 brain_id 等同于 brain_id=None（听全部）。"""
        received = []
        bus.subscribe(lambda e: received.append(e))  # 无 brain_id 参数

        bus.publish(Event(source="s", type="t", timestamp=time.time(),
                          entropy_score=0.0, payload={}, brain_id="any-brain"))
        assert len(received) == 1
