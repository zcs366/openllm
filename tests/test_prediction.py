"""test_prediction.py — ISA 因果预测引擎测试"""
import time
import tempfile
from pathlib import Path

import pytest
from openllm.iai.event_bus import EventBus, Event
from openllm.iai.prediction import PredictionEngine


class TestEncode:
    def test_encode_returns_normalized_vector(self):
        vec = PredictionEngine._encode({"a": 1, "b": "hello", "c": [1, 2]})
        assert len(vec) == PredictionEngine.VECTOR_DIM
        norm = sum(x * x for x in vec) ** 0.5
        assert abs(norm - 1.0) < 1e-6

    def test_encode_bool(self):
        vec_t = PredictionEngine._encode({"flag": True})
        vec_f = PredictionEngine._encode({"flag": False})
        assert vec_t != vec_f

    def test_encode_empty_dict(self):
        vec = PredictionEngine._encode({})
        assert len(vec) == PredictionEngine.VECTOR_DIM
        assert all(v == 0.0 for v in vec)


class TestCosineDistance:
    def test_identical_vectors(self):
        a = [1.0, 0.0, 0.0]
        assert PredictionEngine._cosine_distance(a, a) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert PredictionEngine._cosine_distance(a, b) == pytest.approx(2.0)

    def test_perpendicular_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert PredictionEngine._cosine_distance(a, b) == pytest.approx(1.0)

    def test_range_0_to_2(self):
        for _ in range(20):
            import random
            a = [random.random() for _ in range(8)]
            b = [random.random() for _ in range(8)]
            d = PredictionEngine._cosine_distance(a, b)
            assert 0.0 <= d <= 2.0


class TestPredictNext:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.bus = EventBus(log_dir=Path(self.tmp), ttl_hours=24)
        self.engine = PredictionEngine(bus=self.bus)

    def test_returns_prediction_dict(self):
        result = self.engine.predict_next({"role": "user", "intent": "code"})
        assert "predicted_type" in result
        assert "confidence" in result
        assert "context_vector" in result
        assert "timestamp" in result

    def test_predicted_type_is_valid(self):
        result = self.engine.predict_next({"action": "write"})
        assert result["predicted_type"] in PredictionEngine._OUTPUT_TYPES

    def test_confidence_is_bounded(self):
        result = self.engine.predict_next({"data": 42})
        assert 0.0 <= result["confidence"] <= 1.0


class TestComputeError:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.bus = EventBus(log_dir=Path(self.tmp), ttl_hours=24)
        self.engine = PredictionEngine(bus=self.bus)

    def test_identical_prediction_zero_error(self):
        pred = self.engine.predict_next({"a": 1})
        actual = {"a": 1}  # 相同上下文
        error = self.engine.compute_error(pred, actual)
        # 相同上下文编码相同向量，误差应接近0
        assert error == pytest.approx(0.0, abs=1e-6)

    def test_different_contexts_positive_error(self):
        pred = self.engine.predict_next({"type": "text"})
        actual = {"completely": "different", "xyz": 999}
        error = self.engine.compute_error(pred, actual)
        assert error > 0.0

    def test_error_range(self):
        pred = self.engine.predict_next({"x": 1})
        actual = {"y": 2}
        error = self.engine.compute_error(pred, actual)
        assert 0.0 <= error <= 2.0


class TestShouldReroute:
    def test_below_threshold(self):
        engine = PredictionEngine()
        assert engine.should_reroute(0.1, threshold=0.3) is False
        assert engine.should_reroute(0.29, threshold=0.3) is False

    def test_above_threshold(self):
        engine = PredictionEngine()
        assert engine.should_reroute(0.5, threshold=0.3) is True
        assert engine.should_reroute(0.31, threshold=0.3) is True

    def test_at_threshold(self):
        engine = PredictionEngine()
        assert engine.should_reroute(0.3, threshold=0.3) is False  # > not >=

    def test_default_threshold(self):
        engine = PredictionEngine()
        assert engine.should_reroute(0.35) is True
        assert engine.should_reroute(0.25) is False


class TestUpdateExpectation:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.bus = EventBus(log_dir=Path(self.tmp), ttl_hours=24)
        self.engine = PredictionEngine(bus=self.bus, ema_alpha=0.5)

    def test_first_update_sets_expectation(self):
        self.engine.update_expectation({"x": 1})
        assert self.engine._count == 1
        # 第一次直接等于编码向量
        expected = PredictionEngine._encode({"x": 1})
        assert self.engine._expectation == expected

    def test_ema_convergence(self):
        """多次更新同一数据，期望向量应收敛。"""
        data = {"key": "code"}
        for _ in range(50):
            self.engine.update_expectation(data)
        final = self.engine._expectation
        target = PredictionEngine._encode(data)
        # 收敛后应非常接近
        dist = PredictionEngine._cosine_distance(final, target)
        assert dist < 0.1


class TestEventBusIntegration:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.bus = EventBus(log_dir=Path(self.tmp), ttl_hours=24)

    def test_reroute_event_published_on_high_error(self):
        received = []
        self.bus.subscribe(
            lambda e: received.append(e),
            source_filter="ISA",
            type_filter="reroute_needed",
        )
        engine = PredictionEngine(bus=self.bus, ema_alpha=0.15)
        # 先建立一些期望
        for _ in range(5):
            engine.update_expectation({"type": "code", "intent": "coding"})
        # 用完全不同的上下文触发高误差
        result = engine.predict_and_check(
            {"type": "code", "intent": "coding"},
            {"completely": "unrelated", "xyz": 999},
            threshold=0.1,
        )
        assert result["reroute"] is True
        assert len(received) == 1
        assert received[0].type == "reroute_needed"
        assert received[0].source == "ISA"
        assert "error" in received[0].payload

    def test_no_reroute_event_on_low_error(self):
        received = []
        self.bus.subscribe(
            lambda e: received.append(e),
            source_filter="ISA",
            type_filter="reroute_needed",
        )
        engine = PredictionEngine(bus=self.bus)
        ctx = {"type": "analysis"}
        engine.predict_and_check(ctx, ctx, threshold=0.5)
        assert len(received) == 0


class TestPredictAndCheck:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.bus = EventBus(log_dir=Path(self.tmp), ttl_hours=24)
        self.engine = PredictionEngine(bus=self.bus)

    def test_returns_complete_result(self):
        ctx = {"role": "user", "intent": "search"}
        actual = {"type": "search", "result": "found"}
        result = self.engine.predict_and_check(ctx, actual)
        assert "prediction" in result
        assert "error" in result
        assert "reroute" in result
        assert result["expectation_count"] == 1

    def test_count_increments(self):
        ctx = {"a": 1}
        self.engine.predict_and_check(ctx, ctx)
        self.engine.predict_and_check(ctx, ctx)
        assert self.engine._count == 2


class TestStandaloneEngine:
    def test_no_bus_creation(self):
        """无外部 bus 时自动创建内部 bus，不应崩溃。"""
        engine = PredictionEngine()
        result = engine.predict_and_check({"x": 1}, {"y": 2}, threshold=0.9)
        assert result["reroute"] is False or result["reroute"] is True
