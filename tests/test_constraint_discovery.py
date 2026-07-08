"""test_constraint_discovery.py — IOS 自涌现约束发现测试"""
import time
from unittest.mock import MagicMock

import pytest
from openllm.iai.event_bus import Event, EventBus
from openllm.ios.constraint_discovery import ConstraintDiscovery, _shannon_entropy


def _evt(source: str = "IAX", etype: str = "heartbeat.ping", ts: float = 0.0) -> Event:
    return Event(source=source, type=etype, timestamp=ts or time.time(), entropy_score=0.1, payload={})


# ── Shannon 熵 ──

class TestShannonEntropy:
    def test_empty(self):
        from collections import Counter
        assert _shannon_entropy(Counter()) == 0.0

    def test_uniform(self):
        from collections import Counter
        e = _shannon_entropy(Counter({"a": 50, "b": 50}))
        assert e == pytest.approx(1.0, abs=0.01)

    def test_concentrated(self):
        from collections import Counter
        e = _shannon_entropy(Counter({"a": 100, "b": 1}))
        assert e < 0.2


# ── ConstraintDiscovery 基本功能 ──

class TestBasicAPI:
    def test_observe_stores_event(self):
        cd = ConstraintDiscovery(window_size=10)
        e = _evt()
        cd.observe(e)
        assert len(cd._events) == 1

    def test_observe_sliding_window(self):
        cd = ConstraintDiscovery(window_size=5)
        for i in range(10):
            cd.observe(_evt(ts=float(i)))
        assert len(cd._events) == 5
        assert cd._events[0].timestamp == 5.0

    def test_list_proposals_empty(self):
        cd = ConstraintDiscovery()
        assert cd.list_proposals() == []

    def test_propose_constraint(self):
        cd = ConstraintDiscovery()
        anomaly = {"type": "frequency_storm", "description": "too many", "severity": "critical", "source_event": "evt-x"}
        p = cd.propose_constraint(anomaly)
        assert p["id"].startswith("cp-")
        assert p["status"] == "pending"
        assert len(cd.list_proposals()) == 1

    def test_approve_success(self):
        cd = ConstraintDiscovery()
        p = cd.propose_constraint({"type": "test", "description": "d", "source_event": "e"})
        assert cd.approve(p["id"], True) is True
        assert cd.list_proposals()[0]["status"] == "approved"

    def test_approve_reject(self):
        cd = ConstraintDiscovery()
        p = cd.propose_constraint({"type": "test", "description": "d", "source_event": "e"})
        cd.approve(p["id"], False)
        assert cd.list_proposals()[0]["status"] == "rejected"

    def test_approve_unknown(self):
        cd = ConstraintDiscovery()
        assert cd.approve("cp-nonexistent", True) is False


# ── 异常检测 ──

class TestAnomalyDetection:
    def test_no_anomaly_few_events(self):
        cd = ConstraintDiscovery()
        for i in range(3):
            cd.observe(_evt(ts=float(i)))
        assert cd.detect_anomaly() == []

    def test_frequency_storm(self):
        cd = ConstraintDiscovery(window_size=10, freq_threshold=0.6)
        for i in range(7):  # 7/10 = 0.7 >= 0.6
            cd.observe(_evt(ts=float(i)))
        anomalies = cd.detect_anomaly()
        assert any(a["type"] == "frequency_storm" for a in anomalies)

    def test_type_concentration(self):
        cd = ConstraintDiscovery(window_size=10, concentration_threshold=0.7, entropy_threshold=2.0)
        for i in range(9):
            cd.observe(_evt(etype="same.type", ts=float(i)))
        cd.observe(_evt(etype="other.type", ts=99.0))
        anomalies = cd.detect_anomaly()
        assert any(a["type"] == "type_concentration" for a in anomalies)

    def test_no_concentration_diverse(self):
        cd = ConstraintDiscovery(window_size=10, concentration_threshold=0.7)
        types = ["a", "b", "c", "d", "e"]
        for i in range(10):
            cd.observe(_evt(etype=types[i % 5], ts=float(i)))
        anomalies = cd.detect_anomaly()
        assert not any(a["type"] == "type_concentration" for a in anomalies)

    def test_source_monopoly(self):
        cd = ConstraintDiscovery(window_size=20)
        for i in range(15):  # 15/20 = 0.75 >= 0.3, single source
            cd.observe(_evt(source="ALONE", ts=float(i)))
        anomalies = cd.detect_anomaly()
        assert any(a["type"] == "source_monopoly" for a in anomalies)


# ── EventBus 集成 ──

class TestEventBusIntegration:
    def test_propose_publishes_event(self):
        bus = MagicMock(spec=EventBus)
        cd = ConstraintDiscovery(bus=bus)
        cd.propose_constraint({"type": "x", "description": "d", "source_event": "e"})
        bus.publish.assert_called_once()
        published_event = bus.publish.call_args[0][0]
        assert published_event.type == "constraint.proposed"
        assert published_event.source == "IOS.constraint_discovery"

    def test_attach_and_observe(self):
        bus = MagicMock(spec=EventBus)
        bus.subscribe.return_value = "sub-123"
        cd = ConstraintDiscovery(bus=bus)
        sid = cd.attach()
        assert sid == "sub-123"
        bus.subscribe.assert_called_once_with(cd.observe)

    def test_detach(self):
        bus = MagicMock(spec=EventBus)
        bus.subscribe.return_value = "sub-456"
        bus.unsubscribe.return_value = True
        cd = ConstraintDiscovery(bus=bus)
        cd.attach()
        assert cd.detach() is True
        bus.unsubscribe.assert_called_once_with("sub-456")

    def test_attach_without_bus_raises(self):
        cd = ConstraintDiscovery()
        with pytest.raises(RuntimeError):
            cd.attach()

    def test_observe_via_bus(self):
        bus = EventBus()
        cd = ConstraintDiscovery(bus=bus)
        cd.attach()
        e = _evt()
        bus.publish(e)
        assert len(cd._events) == 1
        cd.detach()

    def test_full_cycle_observe_detect_propose(self):
        bus = MagicMock(spec=EventBus)
        cd = ConstraintDiscovery(bus=bus, window_size=10, freq_threshold=0.5)
        for i in range(7):
            cd.observe(_evt(ts=float(i)))
        anomalies = cd.detect_anomaly()
        assert len(anomalies) > 0
        for a in anomalies:
            p = cd.propose_constraint(a)
            assert p["status"] == "pending"
        cd.approve(cd.list_proposals()[0]["id"], True)
        assert cd.list_proposals()[0]["status"] == "approved"
