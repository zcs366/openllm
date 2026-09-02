"""test_prediction_error.py — PAL T-F-6 预测偏差记录测试

测试矩阵：
1. match=True → 不记录
2. match=False → 记录+emit事件
3. source=user_feedback → 强制记录（独立验证直录）
4. append-only 验证
5. 订阅者收到 prediction.error 事件
"""
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from openllm.iai.event_bus import EventBus, Event
from openllm.iai.prediction import PredictionEngine


@pytest.fixture
def bus():
    """每个测试用独立 EventBus，避免污染。"""
    return EventBus(log_dir=Path(tempfile.mkdtemp()) / "events")


@pytest.fixture
def engine(bus):
    """创建 PredictionEngine。"""
    return PredictionEngine(bus=bus, ema_alpha=0.15)


@pytest.fixture
def errors_file(tmp_path):
    """每个测试用独立 JSONL 文件。"""
    return tmp_path / "prediction_errors.jsonl"


class TestRecordErrorBasic:
    """基础场景：match=True 不记录。"""

    def test_match_true_no_record(self, engine, errors_file):
        """match=True → recorded=False，不写文件。"""
        result = engine.record_error(
            prediction={"predicted_type": "text", "match": True, "confidence": 0.8},
            actual={"type": "text", "content": "hello"},
            source="compare",
            errors_file=errors_file,
        )
        assert result["recorded"] is False
        assert result["match"] is True
        assert not errors_file.exists()

    def test_match_false_record(self, engine, errors_file):
        """match=False → recorded=True，写入文件。"""
        result = engine.record_error(
            prediction={"predicted_type": "code", "match": False, "confidence": 0.5},
            actual={"type": "text", "content": "surprise"},
            source="compare",
            errors_file=errors_file,
        )
        assert result["recorded"] is True
        assert result["match"] is False
        assert result["error"] > 0
        assert errors_file.exists()


class TestUserFeedback:
    """source=user_feedback → 独立验证直录。"""

    def test_user_feedback_always_records(self, engine, errors_file):
        """user_feedback 不检查 match 字段，直接记录。"""
        result = engine.record_error(
            prediction={"predicted_type": "text", "confidence": 0.9},
            actual={"type": "text", "content": "correct"},
            source="user_feedback",
            errors_file=errors_file,
        )
        assert result["recorded"] is True
        assert result["match"] is False
        assert errors_file.exists()

    def test_user_feedback_content_in_file(self, engine, errors_file):
        """user_feedback 写入文件内容含 source='user_feedback'。"""
        engine.record_error(
            prediction={"predicted_type": "code", "confidence": 0.7},
            actual={"type": "diagram"},
            source="user_feedback",
            errors_file=errors_file,
        )
        with open(errors_file) as f:
            record = json.loads(f.readline())
        assert record["source"] == "user_feedback"
        assert record["match"] is False


class TestEventEmit:
    """订阅者收到 prediction.error 事件。"""

    def test_subscriber_receives_event(self, engine, errors_file):
        """订阅者能收到 prediction.error 事件。"""
        received = []
        engine._bus.subscribe(
            lambda e: received.append(e),
            type_filter="prediction.error",
        )
        engine.record_error(
            prediction={"predicted_type": "code", "match": False, "confidence": 0.6},
            actual={"type": "text"},
            source="compare",
            errors_file=errors_file,
        )
        assert len(received) == 1
        assert received[0].type == "prediction.error"
        assert received[0].payload["source"] == "compare"
        assert received[0].payload["match"] is False

    def test_no_event_on_match_true(self, engine, errors_file):
        """match=True 不发事件。"""
        received = []
        engine._bus.subscribe(
            lambda e: received.append(e),
            type_filter="prediction.error",
        )
        engine.record_error(
            prediction={"predicted_type": "text", "match": True, "confidence": 0.9},
            actual={"type": "text"},
            source="compare",
            errors_file=errors_file,
        )
        assert len(received) == 0


class TestAppendOnly:
    """append-only 验证。"""

    def test_multiple_writes_append(self, engine, errors_file):
        """多次记录追加写入，不覆盖。"""
        for i in range(3):
            engine.record_error(
                prediction={"predicted_type": "text", "match": False, "confidence": 0.5},
                actual={"type": "code"},
                source="compare",
                errors_file=errors_file,
            )
        with open(errors_file) as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 3

    def test_user_feedback_appends(self, engine, errors_file):
        """user_feedback 也追加不覆盖。"""
        engine.record_error(
            prediction={"predicted_type": "text", "confidence": 0.5},
            actual={"type": "code"},
            source="user_feedback",
            errors_file=errors_file,
        )
        engine.record_error(
            prediction={"predicted_type": "code", "confidence": 0.6},
            actual={"type": "diagram"},
            source="user_feedback",
            errors_file=errors_file,
        )
        with open(errors_file) as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 2


class TestFallbackMatch:
    """无 match 字段时自动计算。"""

    def test_no_match_field_compute(self, engine, errors_file):
        """prediction dict 无 match/prediction_match 字段时自行计算。"""
        # context_vector 差距大 → 高误差 → match=False
        result = engine.record_error(
            prediction={"predicted_type": "text", "confidence": 0.8,
                        "context_vector": [1.0, 0, 0, 0, 0, 0, 0, 0]},
            actual={"type": "code", "payload": "x" * 200},
            source="compare",
            errors_file=errors_file,
        )
        assert result["recorded"] is True
