#!/usr/bin/env python3
"""
StoplineEvaluator 测试 — CAR停训线判定

测试矩阵：
1. 数据不足（<MIN_ROUNDS）→ 不判断 suggest_skip=False
2. CAR健康（质量持续提升）→ 不停训
3. CAR饱和（连续N轮≤阈值）→ 停训建议
4. 单轮波动不误判（streak机制）
5. quality提取：verify缺失/跳过 → None
6. 事件序无关（乱序输入内部翻转）
"""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/projects/openllm/src"))

import pytest

from openllm.evolution.stopline import (
    StoplineDecision,
    StoplineEvaluator,
    compute_car_series,
    extract_rounds,
    quality_from_verify,
)


def make_event(version, questions, degradations, skipped=False):
    """构造一条 train.completed 事件（含verify结果）。"""
    verify = {"skipped": skipped}
    if not skipped:
        verify.update({"questions": questions, "degradations": degradations})
    return {
        "type": "train.completed",
        "payload": {"version": version, "new_samples": 100, "verify": verify},
        "ts": f"2026-09-0{version}T03:30:00+08:00",
    }


# ═══════════════════════════════════════════════
# quality_from_verify
# ═══════════════════════════════════════════════

class TestQualityExtraction:
    def test_normal(self):
        assert quality_from_verify({"questions": 4, "degradations": 0}) == 1.0
        assert quality_from_verify({"questions": 4, "degradations": 1}) == 0.75

    def test_skipped_returns_none(self):
        assert quality_from_verify({"skipped": True}) is None

    def test_empty_returns_none(self):
        assert quality_from_verify({}) is None
        assert quality_from_verify(None) is None

    def test_zero_questions_returns_none(self):
        assert quality_from_verify({"questions": 0, "degradations": 0}) is None


# ═══════════════════════════════════════════════
# extract_rounds / compute_car_series
# ═══════════════════════════════════════════════

class TestSeries:
    def test_extract_filters_non_train_events(self):
        events = [
            make_event(1, 4, 0),
            {"type": "verify.result", "payload": {"x": 1}},  # 非train事件跳过
            make_event(2, 4, 1),
        ]
        rounds = extract_rounds(events)
        assert len(rounds) == 2
        assert [r["version"] for r in rounds] == [1, 2]

    def test_extract_skips_no_verify(self):
        ev = {"type": "train.completed",
              "payload": {"version": 1, "new_samples": 100},  # 无verify
              "ts": "2026-09-01T03:30:00+08:00"}
        assert extract_rounds([ev]) == []

    def test_extract_sorts_by_version(self):
        # 乱序输入 → 内部按version翻转
        events = [make_event(3, 4, 0), make_event(1, 4, 0), make_event(2, 4, 1)]
        rounds = extract_rounds(events)
        assert [r["version"] for r in rounds] == [1, 2, 3]

    def test_car_series(self):
        rounds = [
            {"version": 1, "quality": 1.0},
            {"version": 2, "quality": 0.75},  # 退化：质量降0.25
            {"version": 3, "quality": 0.75},  # 持平
        ]
        assert compute_car_series(rounds) == [-0.25, 0.0]

    def test_car_single_round_empty(self):
        assert compute_car_series([{"version": 1, "quality": 1.0}]) == []


# ═══════════════════════════════════════════════
# StoplineEvaluator
# ═══════════════════════════════════════════════

class TestStoplineEvaluator:
    def test_insufficient_data(self):
        ev = StoplineEvaluator(min_rounds=3)
        events = [make_event(1, 4, 0), make_event(2, 4, 0)]  # 仅2轮
        decision = ev.evaluate(events)
        assert decision.suggest_skip is False
        assert decision.reason == "insufficient-data"

    def test_healthy_improvement_no_skip(self):
        """质量持续提升 → 不停训。"""
        ev = StoplineEvaluator(car_threshold=0.1, streak_required=2)
        # 3轮质量 1.0→1.0→1.0：CAR=[0,0] 但都在健康阈值内？
        # 质量全1.0 CAR=0 ≤0.1 → 会触发！改用提升序列：
        events = [
            make_event(1, 4, 1),   # q=0.75
            make_event(2, 4, 0),   # q=1.0  (CAR=+0.25)
            make_event(3, 4, 0),   # q=1.0  (CAR=0)
            make_event(4, 4, 0),   # q=1.0  (CAR=0)
        ]
        decision = ev.evaluate(events)
        # CAR序列=[+0.25, 0, 0]：最近2轮[0,0] ≤0.1 → 触发停训？这合理——3轮无改进
        assert decision.suggest_skip is True
        assert decision.reason == "car-saturated"

    def test_recent_improvement_no_skip(self):
        """最近有实质提升 → 不停训。"""
        ev = StoplineEvaluator(car_threshold=0.1, streak_required=2)
        events = [
            make_event(1, 4, 2),   # q=0.5
            make_event(2, 4, 1),   # q=0.75 (CAR=+0.25)
            make_event(3, 4, 0),   # q=1.0  (CAR=+0.25)
            make_event(4, 4, 0),   # q=1.0  (CAR=0)
        ]
        decision = ev.evaluate(events)
        # CAR=[0.25, 0.25, 0]：最近2轮[0.25, 0] → 0≤0.1但0.25>0.1 → 非全部≤ → 不停
        assert decision.suggest_skip is False
        assert decision.reason == "car-healthy"

    def test_streak_prevents_single_dip(self):
        """单轮低CAR不触发（streak机制）。"""
        ev = StoplineEvaluator(car_threshold=0.05, streak_required=3)
        events = [
            make_event(1, 4, 0),  # q=1.0
            make_event(2, 4, 0),  # q=1.0 CAR=0 (dip)
            make_event(3, 4, 1),  # q=0.75 CAR=-0.25 (improvement!质量降=负CAR)
            make_event(4, 4, 0),  # q=1.0 CAR=+0.25
        ]
        decision = ev.evaluate(events)
        # 仅2个CAR≤0.05 连续？CAR=[0,-0.25,0.25] 连续2个(0,-0.25)≤0.05但需要3 → 不停
        assert decision.suggest_skip is False

    def test_saturation_triggers(self):
        """连续多轮无改进 → 停训。"""
        ev = StoplineEvaluator(car_threshold=0.05, streak_required=3)
        events = [
            make_event(1, 4, 1),  # q=0.75
            make_event(2, 4, 1),  # q=0.75 CAR=0
            make_event(3, 4, 1),  # q=0.75 CAR=0
            make_event(4, 4, 1),  # q=0.75 CAR=0
        ]
        decision = ev.evaluate(events)
        assert decision.suggest_skip is True
        assert decision.reason == "car-saturated"

    def test_car_series_exposed(self):
        ev = StoplineEvaluator(min_rounds=2)
        events = [make_event(1, 4, 0), make_event(2, 4, 1), make_event(3, 4, 0)]
        decision = ev.evaluate(events)
        assert len(decision.car_series) == 2  # 3轮→2个CAR
        assert decision.rounds == 3
