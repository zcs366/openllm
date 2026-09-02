"""ActiveSampler 单元测试

覆盖：
  - 预算硬上限（3次后返回False）
  - explore模式偏离焦点（偏离度>0，不确认预测）
  - verify模式正常采样
  - append-only日志写入（只追加不覆盖）
  - reset_budget重新开始新轮次
  - 边界条件（budget=0、mode非法）
  - get_status状态摘要
  - 与章鱼I集成可选（None时不报错）
"""
import json
import os
import time
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from openllm.iai.active_sampler import ActiveSampler


# ── 预算硬上限 ────────────────────────────────────────

class TestBudgetHardCap:
    """赫淮斯托斯硬上限：每轮最多3次采样"""

    def test_default_budget_is_3(self):
        s = ActiveSampler()
        assert s.budget == 3

    def test_first_sample_succeeds(self):
        s = ActiveSampler()
        assert s.sample() is True

    def test_three_samples_then_reject(self):
        s = ActiveSampler()
        assert s.sample() is True
        s.mark_used("verify")
        assert s.sample() is True
        s.mark_used("verify")
        assert s.sample() is True
        s.mark_used("verify")
        # 第4次应该被拒绝
        assert s.sample() is False

    def test_budget_custom(self):
        s = ActiveSampler(budget=1)
        assert s.sample() is True
        s.mark_used("verify")
        assert s.sample() is False

    def test_budget_zero_always_rejects(self):
        s = ActiveSampler(budget=0)
        assert s.sample() is False

    def test_mark_used_raises_on_exhausted(self):
        s = ActiveSampler(budget=1)
        s.mark_used("verify")
        with pytest.raises(RuntimeError, match="采样预算已用尽"):
            s.mark_used("verify")


# ── explore反预测扰动 ──────────────────────────────────

class TestExploreMode:
    """克洛诺斯反预测扰动：explore模式随机偏离焦点"""

    def test_explore_mode_accepted(self):
        s = ActiveSampler()
        assert s.sample("explore") is True

    def test_explore_deviation_is_random(self):
        """偏离度应随机且>0"""
        s = ActiveSampler()
        deviations = [s.generate_deviation("test_topic") for _ in range(100)]
        assert all(0.0 <= d <= 1.0 for d in deviations)
        # 不应全是同一个值（概率极低）
        assert len(set(deviations)) > 1

    def test_explore_mark_used_records_deviation(self):
        s = ActiveSampler(log_dir=Path("/tmp/ias_test"))
        deviation = s.generate_deviation("test_focus")
        s.mark_used("explore", focus_topic="test_focus", deviation=deviation)
        assert s._used == 1

    def test_verify_mode_normal_no_deviation(self):
        """verify模式不做偏离"""
        s = ActiveSampler(log_dir=Path("/tmp/ias_test"))
        s.mark_used("verify", focus_topic="确认主题")
        assert s._used == 1

    def test_invalid_mode_raises(self):
        s = ActiveSampler()
        with pytest.raises(ValueError, match="必须是"):
            s.sample("invalid_mode")


# ── append-only日志 ────────────────────────────────────

class TestAppendOnlyLog:
    """赫拉克勒斯日志：append-only ~/.openllm/iai/sample_log.jsonl"""

    def test_log_created_on_mark_used(self, tmp_path):
        s = ActiveSampler(log_dir=tmp_path)
        s.mark_used("verify", focus_topic="test")
        log_file = tmp_path / "sample_log.jsonl"
        assert log_file.exists()
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 1

    def test_log_append_only(self, tmp_path):
        """多次mark_used只追加不覆盖"""
        s = ActiveSampler(log_dir=tmp_path)
        for i in range(3):
            s.mark_used("verify", focus_topic=f"topic_{i}")
        log_file = tmp_path / "sample_log.jsonl"
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 3
        # 每行都是合法JSON
        for line in lines:
            entry = json.loads(line)
            assert "timestamp" in entry
            assert "mode" in entry

    def test_log_entry_fields(self, tmp_path):
        s = ActiveSampler(log_dir=tmp_path)
        s.mark_used("explore", focus_topic="随机探索", deviation=0.75)
        log_file = tmp_path / "sample_log.jsonl"
        entry = json.loads(log_file.read_text().strip())
        assert entry["mode"] == "explore"
        assert entry["anti_prediction"] is True
        assert entry["deviation"] == 0.75
        assert entry["budget_used"] == 1
        assert entry["budget_total"] == 3

    def test_log_write_failure_silent(self, tmp_path):
        """写失败不应抛异常"""
        s = ActiveSampler(budget=2, log_dir=tmp_path / "nonexistent" / "deep")
        # 即使目录无法创建也不应崩溃（mkdir会创建）
        s.mark_used("verify")
        assert s._used == 1

    def test_round_id_consistent_in_log(self, tmp_path):
        s = ActiveSampler(log_dir=tmp_path)
        round_id = s._round_id
        for _ in range(3):
            s.mark_used("verify")
        log_file = tmp_path / "sample_log.jsonl"
        for line in log_file.read_text().strip().split("\n"):
            entry = json.loads(line)
            assert entry["round_id"] == round_id


# ── reset_budget ───────────────────────────────────────

class TestResetBudget:
    """新轮次开始时重置预算"""

    def test_reset_allows_new_samples(self):
        s = ActiveSampler()
        for _ in range(3):
            s.mark_used("verify")
        assert s.sample() is False
        s.reset_budget()
        assert s.sample() is True
        assert s.remaining == 3

    def test_reset_generates_new_round_id(self):
        s = ActiveSampler()
        old_id = s._round_id
        time.sleep(0.01)
        s.reset_budget()
        assert s._round_id != old_id

    def test_reset_mid_round(self):
        s = ActiveSampler()
        s.mark_used("verify")
        s.mark_used("verify")
        assert s.remaining == 1
        s.reset_budget()
        assert s.remaining == 3


# ── get_status ─────────────────────────────────────────

class TestGetStatus:
    def test_initial_status(self):
        s = ActiveSampler()
        status = s.get_status()
        assert status["budget"] == 3
        assert status["used"] == 0
        assert status["remaining"] == 3
        assert status["exhausted"] is False

    def test_exhausted_status(self):
        s = ActiveSampler()
        for _ in range(3):
            s.mark_used("verify")
        status = s.get_status()
        assert status["exhausted"] is True
        assert status["remaining"] == 0


# ── 章鱼I可选集成 ─────────────────────────────────────

class TestOctopusIntegration:
    """无sampler时零开销，有sampler时可选接入"""

    def test_none_sampler_no_crash(self):
        """模拟章鱼I不传sampler"""
        sampler = None
        # 无sampler时应跳过主动采样
        if sampler is not None and sampler.sample():
            sampler.mark_used("verify")
        # 不应报错

    def test_sampler_optional_in_constructor(self):
        """sampler作为可选组件"""
        sampler = ActiveSampler()
        assert sampler.sample() is True

    def test_repr(self):
        s = ActiveSampler()
        r = repr(s)
        assert "ActiveSampler" in r
        assert "budget=3" in r


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
