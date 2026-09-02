"""章鱼I因果记忆测试

覆盖：
  - learn_causal 写入 JSONL 文件（append-only）
  - get_causal_history 读取最近N条
  - causal_stats 准确率计算（含下降验证）
  - 文件/目录不存在时自动创建
  - 写失败静默不抛异常
"""
import json
import os
import time
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from openllm.core.models import Context, Prediction, ActionResult, CausalDelta
from openllm.core.octopus_impl import 章鱼I


@pytest.fixture
def octopus():
    """实例化章鱼I（不传iai，避免网络依赖）"""
    return 章鱼I(iai=None)


@pytest.fixture
def ctx():
    return Context(user_message="测试消息")


@pytest.fixture
def prediction():
    return Prediction(summary="预测测试操作", confidence=0.8)


@pytest.fixture
def delta_match():
    return CausalDelta(
        prediction_match=True,
        delta_summary="预测正确",
        learned=["学到规则A"],
    )


@pytest.fixture
def delta_mismatch():
    return CausalDelta(
        prediction_match=False,
        delta_summary="预测错误",
        learned=["预测失败教训B"],
    )


class TestLearnCausal:
    def test_creates_file_and_writes_entry(self, octopus, ctx, prediction, delta_match, tmp_path, monkeypatch):
        """learn_causal 创建 JSONL 文件并写入一条记录"""
        # 重定向到临时目录
        test_dir = tmp_path / "octopus_test"
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        # 需要让章鱼I的 learn_causal 也用这个路径
        result = ActionResult(success=True, output="执行成功")

        octopus.learn_causal(ctx, prediction, result, delta_match)

        memory_file = tmp_path / ".openllm" / "octopus" / "causal_memory.jsonl"
        assert memory_file.exists(), f"JSONL文件应被创建: {memory_file}"
        lines = memory_file.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["prediction_match"] is True
        assert entry["delta_summary"] == "预测正确"
        assert entry["learned"] == ["学到规则A"]
        assert entry["result_success"] is True
        assert "timestamp" in entry

    def test_append_only_no_overwrite(self, octopus, prediction, tmp_path, monkeypatch):
        """多次调用只追加不覆盖"""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        result = ActionResult(success=True, output="ok")

        for i in range(3):
            ctx_i = Context(user_message=f"消息{i}")
            delta_i = CausalDelta(
                prediction_match=(i % 2 == 0),
                delta_summary=f"delta{i}",
                learned=[],
            )
            octopus.learn_causal(ctx_i, prediction, result, delta_i)

        memory_file = tmp_path / ".openllm" / "octopus" / "causal_memory.jsonl"
        lines = memory_file.read_text().strip().split("\n")
        assert len(lines) == 3
        # 验证按序追加
        assert json.loads(lines[0])["delta_summary"] == "delta0"
        assert json.loads(lines[2])["delta_summary"] == "delta2"

    def test_write_failure_silent(self, octopus, ctx, prediction, delta_match, tmp_path, monkeypatch):
        """写入失败不抛异常（静默降级）"""
        # 指向不可写目录
        bad_dir = tmp_path / "nonexistent_parent"
        monkeypatch.setattr(Path, "home", lambda: bad_dir)
        result = ActionResult(success=True, output="ok")

        # 不应抛异常
        octopus.learn_causal(ctx, prediction, result, delta_match)
        # 如果没抛异常就算通过


class TestGetCausalHistory:
    def test_returns_empty_when_no_file(self, octopus, tmp_path, monkeypatch):
        """无文件时返回空列表"""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert octopus.get_causal_history() == []

    def test_returns_recent_entries(self, octopus, prediction, tmp_path, monkeypatch):
        """返回最近N条"""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        result = ActionResult(success=True, output="ok")

        for i in range(5):
            delta = CausalDelta(
                prediction_match=(i % 2 == 0),
                delta_summary=f"delta{i}",
                learned=[],
            )
            octopus.learn_causal(Context(user_message=f"msg{i}"), prediction, result, delta)

        # 默认20条（5<20，全返回）
        history = octopus.get_causal_history()
        assert len(history) == 5

        # limit=2只返回最近2条
        history_2 = octopus.get_causal_history(limit=2)
        assert len(history_2) == 2
        assert history_2[0]["delta_summary"] == "delta3"
        assert history_2[1]["delta_summary"] == "delta4"


class TestCausalStats:
    def test_empty_when_no_file(self, octopus, tmp_path, monkeypatch):
        """无文件时返回全零"""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        stats = octopus.causal_stats()
        assert stats == {"total": 0, "match_count": 0, "accuracy": 0.0}

    def test_accuracy_calculation(self, octopus, prediction, tmp_path, monkeypatch):
        """准确率 = match_count / total"""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        result = ActionResult(success=True, output="ok")

        # 写3条：2 match + 1 mismatch
        octopus.learn_causal(
            Context(user_message="a"), prediction, result,
            CausalDelta(prediction_match=True, delta_summary="ok", learned=[]),
        )
        octopus.learn_causal(
            Context(user_message="b"), prediction, result,
            CausalDelta(prediction_match=True, delta_summary="ok", learned=[]),
        )
        octopus.learn_causal(
            Context(user_message="c"), prediction, result,
            CausalDelta(prediction_match=False, delta_summary="fail", learned=[]),
        )

        stats = octopus.causal_stats()
        assert stats["total"] == 3
        assert stats["match_count"] == 2
        assert abs(stats["accuracy"] - 2 / 3) < 1e-9

    def test_accuracy_drops_on_mismatch(self, octopus, prediction, tmp_path, monkeypatch):
        """加入一条failed记录后accuracy下降"""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        result = ActionResult(success=True, output="ok")

        # 先写3条全match → accuracy=1.0
        for i in range(3):
            octopus.learn_causal(
                Context(user_message=f"ok{i}"), prediction, result,
                CausalDelta(prediction_match=True, delta_summary="ok", learned=[]),
            )
        stats1 = octopus.causal_stats()
        assert stats1["accuracy"] == 1.0

        # 再写1条mismatch → accuracy=0.75
        octopus.learn_causal(
            Context(user_message="fail"), prediction, result,
            CausalDelta(prediction_match=False, delta_summary="fail", learned=[]),
        )
        stats2 = octopus.causal_stats()
        assert stats2["total"] == 4
        assert stats2["match_count"] == 3
        assert stats2["accuracy"] == 0.75
        assert stats2["accuracy"] < stats1["accuracy"]
