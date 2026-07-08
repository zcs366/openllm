"""test_user_state.py — ika 用户状态感知模块测试"""
import time
import tempfile
from pathlib import Path

import pytest
from openllm.iai.event_bus import EventBus, Event
from openllm.iai.user_state import UserState, UserStateTracker, SILENCE_THRESHOLD


# ── UserState 数据类 ──

class TestUserState:
    def test_default_state(self):
        s = UserState()
        assert s.interaction_frequency == []
        assert s.correction_count == 0
        assert s.silence_duration == 0.0
        assert s.last_signal_time == 0.0
        assert s.total_interactions == 0


# ── UserStateTracker 核心功能 ──

class TestUserStateTracker:
    def test_record_interaction(self):
        tracker = UserStateTracker()
        state = tracker.record_interaction(timestamp=1000.0)
        assert state.total_interactions == 1
        assert state.last_signal_time == 1000.0
        assert state.interaction_frequency == []  # 第一次无间隔

    def test_record_multiple_interactions(self):
        tracker = UserStateTracker()
        tracker.record_interaction(timestamp=1000.0)
        tracker.record_interaction(timestamp=1010.0)
        tracker.record_interaction(timestamp=1025.0)
        state = tracker.get_state()
        assert state.total_interactions == 3
        assert state.interaction_frequency == [10.0, 15.0]
        assert state.last_signal_time == 1025.0

    def test_record_correction(self):
        tracker = UserStateTracker()
        state = tracker.record_correction(timestamp=2000.0)
        assert state.correction_count == 1
        assert state.total_interactions == 1  # 纠正也算一次交互

    def test_record_correction_increments_count(self):
        tracker = UserStateTracker()
        tracker.record_correction(timestamp=1000.0)
        tracker.record_correction(timestamp=1005.0)
        state = tracker.get_state()
        assert state.correction_count == 2
        assert state.total_interactions == 2


# ── 沉默告警 ──

class TestSilenceAlert:
    def test_no_interaction_no_alert(self):
        tracker = UserStateTracker()
        assert tracker.is_silence_alert() is False

    def test_recent_interaction_no_alert(self):
        tracker = UserStateTracker()
        tracker.record_interaction(timestamp=time.time())
        assert tracker.is_silence_alert() is False

    def test_long_silence_alerts(self):
        tracker = UserStateTracker(silence_threshold=5.0)
        tracker.record_interaction(timestamp=time.time() - 10.0)
        assert tracker.is_silence_alert() is True

    def test_custom_threshold(self):
        tracker = UserStateTracker(silence_threshold=60.0)
        tracker.record_interaction(timestamp=time.time() - 30.0)
        assert tracker.is_silence_alert() is False
        tracker2 = UserStateTracker(silence_threshold=1.0)
        tracker2.record_interaction(timestamp=time.time() - 5.0)
        assert tracker2.is_silence_alert() is True


# ── 参与度评分 ──

class TestEngagementScore:
    def test_no_interaction_score_zero(self):
        tracker = UserStateTracker()
        assert tracker.get_engagement_score() == 0.0

    def test_active_engagement(self):
        tracker = UserStateTracker()
        now = time.time()
        for i in range(5):
            tracker.record_interaction(timestamp=now + i * 2)  # 每2秒一次
        score = tracker.get_engagement_score()
        assert 0.5 <= score <= 1.0  # 活跃用户应有较高分

    def test_silence_kills_score(self):
        tracker = UserStateTracker(silence_threshold=5.0)
        tracker.record_interaction(timestamp=time.time() - 10.0)
        assert tracker.get_engagement_score() == 0.0

    def test_score_bounded(self):
        tracker = UserStateTracker()
        now = time.time()
        # 大量快速交互
        for i in range(50):
            tracker.record_interaction(timestamp=now + i * 0.1)
        score = tracker.get_engagement_score()
        assert 0.0 <= score <= 1.0

    def test_many_corrections_lower_score(self):
        tracker = UserStateTracker()
        now = time.time()
        # 10次交互，10次纠正 → 纠正率100% → 扣分
        for i in range(10):
            tracker.record_correction(timestamp=now + i * 5)
        score = tracker.get_engagement_score()
        # 对比：无纠正时
        tracker2 = UserStateTracker()
        for i in range(10):
            tracker2.record_interaction(timestamp=now + i * 5)
        score2 = tracker2.get_engagement_score()
        assert score < score2


# ── EventBus 集成 ──

class TestEventBusIntegration:
    def test_interaction_publishes_event(self):
        bus = EventBus(log_dir=Path(tempfile.mkdtemp()))
        tracker = UserStateTracker(bus=bus)
        tracker.record_interaction(timestamp=time.time())
        history = bus.get_history(source_filter="USER_STATE")
        assert len(history) >= 1
        assert history[-1]["type"] == "user.interaction"
        assert history[-1]["source"] == "USER_STATE"

    def test_correction_publishes_event(self):
        bus = EventBus(log_dir=Path(tempfile.mkdtemp()))
        tracker = UserStateTracker(bus=bus)
        tracker.record_correction(timestamp=time.time())
        history = bus.get_history(source_filter="USER_STATE")
        types = [e["type"] for e in history]
        assert "user.correction" in types

    def test_no_bus_still_works(self):
        tracker = UserStateTracker(bus=None)
        tracker.record_interaction()
        tracker.record_correction()
        state = tracker.get_state()
        assert state.total_interactions == 2
        assert state.correction_count == 1

    def test_event_subscriber_receives_notification(self):
        bus = EventBus(log_dir=Path(tempfile.mkdtemp()))
        received = []
        bus.subscribe(lambda e: received.append(e), source_filter="USER_STATE")
        tracker = UserStateTracker(bus=bus)
        tracker.record_interaction()
        assert len(received) == 1
        assert received[0].type == "user.interaction"


# ── 边界条件 ──

class TestEdgeCases:
    def test_intervals_capped(self):
        tracker = UserStateTracker()
        now = time.time()
        for i in range(25):  # 超过 MAX_INTERVALS=20
            tracker.record_interaction(timestamp=now + i * 10)
        state = tracker.get_state()
        assert len(state.interaction_frequency) <= 20

    def test_get_state_returns_snapshot(self):
        tracker = UserStateTracker()
        tracker.record_interaction(timestamp=1000.0)
        s1 = tracker.get_state()
        tracker.record_interaction(timestamp=1005.0)
        s2 = tracker.get_state()
        assert s1.total_interactions == 1
        assert s2.total_interactions == 2
        assert s1.last_signal_time != s2.last_signal_time
