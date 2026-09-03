"""
CAR 停训线判定 — 进化事件总线的第一个真实消费者（P0-B）
==========================================================

克洛诺斯轴（七神终裁 2026-09-04）："每晚都训"隐含"训练永远有益"假设被否定。
SAHOO CAR 曲线：大部分有价值改进发生在前5-7轮，之后每单位改进的漂移成本急剧上升。
学会停比学会训更根本。

机制:
  CAR (Capability Alignment Ratio) = 质量提升 / 漂移代价
  连续 N 轮 CAR 低于阈值 → 停训建议（材料够也不再训，等基线变动/新领域语料）

输入: 总线事件（train.completed 的 verify 结果序列）
输出: stopline.evaluated 建议（suggest_skip=True/False + 依据）
      停训建议是"建议"非"命令"——非强制（材料极少时忽略），决策权在调用方。

用法:
    from openllm.evolution.stopline import StoplineEvaluator, StoplineDecision
    ev = StoplineEvaluator(bus)
    decision = ev.evaluate(history_events)   # → StoplineDecision(suggest_skip, reason, car_series)

铁律:
  - 事件无 consumer 不阻断（停训线读历史，消费是主动 query 非推送）
  - 数据不足（< MIN_ROUNDS）→ 不判断（suggest_skip=False, reason='insufficient-data'）
  - CAR 用质量代理（verify 退化数反比）——无 GDI 真值前用退化率做保守代理
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── 默认参数（可覆盖；阈值须用真实数据校准——工程法典"magic number须实测"）
MIN_ROUNDS = 3            # 至少3轮才可判（数据不足不判断）
CAR_THRESHOLD = 0.15      # 连续N轮CAR低于此 → 停训建议（初始值，待真实数据校准）
STREAK_REQUIRED = 2       # 连续2轮低于阈值 → 触发（防单轮波动误判）
VERIFY_WEIGHT = 1.0       # verify退化率在质量代理中的权重
NEW_SAMPLE_WEIGHT = 0.0   # v0.1：新材料量不直接进CAR（等有质量指标后启用）

# 质量代理定义：
#   单轮质量 Q = 1 - degradations/questions（verify健康度，0-1）
#   漂移代价 D = 代理（v0.1无GDI真值 → 用常数1.0，即CAR=ΔQ）
#   CAR_t = Q_t - Q_{t-1}（质量提升量；无GDI时CAR简化为边际质量变化）


@dataclass
class StoplineDecision:
    """停训线判定结果。"""

    suggest_skip: bool
    reason: str
    car_series: List[float] = field(default_factory=list)
    quality_series: List[float] = field(default_factory=list)
    rounds: int = 0
    detail: str = ""


def quality_from_verify(verify: Optional[Dict[str, Any]]) -> Optional[float]:
    """从 verify 负载提取质量代理（0-1）。无 verify 数据返回 None。"""
    if not verify or verify.get("skipped") is True:
        return None
    questions = verify.get("questions", 0)
    degradations = verify.get("degradations", 0)
    if questions <= 0:
        return None
    return round(1.0 - degradations / questions, 4)


def extract_rounds(history_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """从总线事件序列提取有序训练轮次（新→旧 → 翻转为旧→新）。"""
    rounds = []
    for ev in history_events:
        payload = ev.get("payload", {})
        verify = payload.get("verify")
        if payload.get("version") is None or not verify:
            continue
        q = quality_from_verify(verify)
        if q is None:
            continue
        rounds.append({
            "version": payload["version"],
            "ts": ev.get("ts", ""),
            "quality": q,
            "new_samples": payload.get("new_samples", 0),
        })
    rounds.sort(key=lambda r: str(r["version"]))  # 版本号单调 → 时间序
    return rounds


def compute_car_series(rounds: List[Dict[str, Any]]) -> List[float]:
    """CAR序列 = 相邻轮质量差。首轮无CAR（无前基线）。"""
    car = []
    for i in range(1, len(rounds)):
        car.append(round(rounds[i]["quality"] - rounds[i - 1]["quality"], 4))
    return car


class StoplineEvaluator:
    """CAR停训线判定器。

    消费总线中 train.completed 事件序列，算 CAR 趋势，
    连续 STREAK_REQUIRED 轮低于 CAR_THRESHOLD → 建议停训。
    """

    def __init__(
        self,
        min_rounds: int = MIN_ROUNDS,
        car_threshold: float = CAR_THRESHOLD,
        streak_required: int = STREAK_REQUIRED,
    ) -> None:
        self.min_rounds = min_rounds
        self.car_threshold = car_threshold
        self.streak_required = streak_required

    def evaluate(self, history_events: List[Dict[str, Any]]) -> StoplineDecision:
        """对事件序列做停训判定。事件序不限（内部翻转）。"""
        rounds = extract_rounds(history_events)
        car_series = compute_car_series(rounds)
        qualities = [r["quality"] for r in rounds]

        if len(rounds) < self.min_rounds:
            return StoplineDecision(
                suggest_skip=False, reason="insufficient-data",
                car_series=car_series, quality_series=qualities,
                rounds=len(rounds),
                detail=f"仅{len(rounds)}轮 < {self.min_rounds}轮，数据不足不判断",
            )

        # 连续 streak_required 轮 CAR ≤ 阈值 → 停训
        if len(car_series) >= self.streak_required:
            recent = car_series[-self.streak_required:]
            if all(c <= self.car_threshold for c in recent):
                return StoplineDecision(
                    suggest_skip=True, reason="car-saturated",
                    car_series=car_series, quality_series=qualities,
                    rounds=len(rounds),
                    detail=(
                        f"连续{self.streak_required}轮CAR≤{self.car_threshold}"
                        f"（{recent}）——改进饱和，建议停训等新领域语料"
                    ),
                )

        return StoplineDecision(
            suggest_skip=False, reason="car-healthy",
            car_series=car_series, quality_series=qualities,
            rounds=len(rounds),
            detail=f"CAR未饱和（最近: {car_series[-self.streak_required:] if car_series else []}）——可继续训",
        )


# ═══ CLI（供 nightly_train launch 检测段调用）═══

def evaluate_from_bus(
    store_dir: Optional[Path] = None,
    days: int = 30,
    **kwargs: Any,
) -> StoplineDecision:
    """从真实总线读 train.completed 事件并判定。"""
    import sys as _sys
    _sys.path.insert(0, "/home/zcs/projects/openllm/src")
    from openllm.evolution.bus import EvolutionBus

    bus = EvolutionBus(store_dir=store_dir)
    events = bus.query(event_type="train.completed", days=days, limit=200)
    evaluator = StoplineEvaluator(**kwargs)
    return evaluator.evaluate(events)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="CAR停训线判定")
    ap.add_argument("--min-rounds", type=int, default=MIN_ROUNDS)
    ap.add_argument("--car-threshold", type=float, default=CAR_THRESHOLD)
    ap.add_argument("--streak", type=int, default=STREAK_REQUIRED)
    ap.add_argument("--days", type=int, default=30)
    args = ap.parse_args()

    decision = evaluate_from_bus(
        min_rounds=args.min_rounds,
        car_threshold=args.car_threshold,
        streak_required=args.streak,
        days=args.days,
    )
    print(f"停训线判定: suggest_skip={decision.suggest_skip} reason={decision.reason}")
    print(f"  轮数: {decision.rounds} | CAR序列: {decision.car_series}")
    print(f"  质量序列: {decision.quality_series}")
    print(f"  依据: {decision.detail}")
