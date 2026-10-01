"""遗忘合一钉子（决策记录动作5 · 成市拍板 A）。

规则：**判定生死只有一个源** = isa/temperature_engine（should_evict/EVICT_THRESHOLD）；
      iai/forgetting 只做「排序与提示」（提议），**不得自带衰减律**。
钉子验三件：① 衰减律确实在唯一实现处；② forgetting 真的在调它（不是自己算）；
            ③ forgetting 手里没有生死权。
"""
import re
from pathlib import Path

import pytest

from openllm.isa import temperature_engine as te
from openllm.iai import forgetting as fg


# ── ① 唯一实现：衰减律只在 temperature_engine 里 ──

def test_canonical_decay_law_values():
    hl = 86400.0
    assert te.decay_factor(0, hl) == 1.0
    assert te.decay_factor(hl, hl) == pytest.approx(0.5)      # 半衰期处得一半
    assert te.decay_factor(2 * hl, hl) == pytest.approx(0.25)
    assert te.decay_factor(hl, 0) == 0.0                       # 非法半衰期不炸


def test_forgetting_has_no_local_decay_law():
    """forgetting.py 源码里不许再出现自己算的衰减公式（红线可静态判）。"""
    src = Path(fg.__file__).read_text(encoding="utf-8")
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert not re.search(r"2\.0\s*\*\*|2\s*\*\*", code), "又自带了一份衰减律"


# ── ② 真在调唯一源（不是抄了一遍公式）──

def test_score_delegates_to_temperature_engine(monkeypatch):
    """把唯一源的实现换成哨兵：score 必须跟着变——证明它真在调，不是各算各的。"""
    sentinel = 0.4242
    monkeypatch.setattr(fg, "_DECAY_FN", lambda age, hl: sentinel)
    fc = fg.ForgettingCurve()
    assert fc.score("任意主题", age_seconds=12345.0) == sentinel


def test_score_matches_canonical_law_exactly():
    fc = fg.ForgettingCurve(half_life_s=86400.0)
    for age in (0.0, 3600.0, 86400.0, 3 * 86400.0):
        expected = max(fg.MIN_SCORE, te.decay_factor(age, 86400.0))
        assert fc.score("t", age_seconds=age) == pytest.approx(round(expected, 6))


# ── ③ forgetting 手里没有生死权 ──

def test_forgetting_cannot_judge_death():
    fc = fg.ForgettingCurve()
    for name in ("should_evict", "evict", "delete", "remove", "kill", "archive"):
        assert not hasattr(fc, name), f"forgetting 不该有 {name}——生死判定归 temperature_engine"


def test_apply_only_sorts_and_never_writes():
    """apply 只产排序（提议），不改输入、不落盘。"""
    fc = fg.ForgettingCurve()
    topics = ["a", "b", "c"]
    snapshot = list(topics)
    out = fc.apply(topics)
    assert topics == snapshot, "提议不许改写输入"
    assert isinstance(out, list) and all(isinstance(t, tuple) and len(t) == 2 for t in out)
    scores = [s for _, s in out]
    assert scores == sorted(scores, reverse=True), "输出必须是排好序的提议"
