"""test_pulse.py — 温度脉冲引擎（P1-4 · 主动振动实例）测试

覆盖：触发器 · 时段窗 · 生成（模板/generator hook）· 裁决集成（P0-1
消费者：PASS 投递 / BLOCK 拦截）· 日硬顶 · 审计。
"""
import json
import time
from datetime import datetime

import pytest

from openllm.iko.pulse import (
    Pulse, PulseEngine, IntervalTrigger, TimeWindow,
    _greeting,
)


class FakeAdjudicator:
    """可编程裁决器：固定 Verdict。"""
    def __init__(self, action: str = "PASS", reason: str = "ok"):
        self.action = action
        self.reason = reason
    def decide(self, req):
        class V:
            def __init__(self, a, r):
                self.action = a; self.reason = r
        return V(self.action, self.reason)


class FixedClock:
    """固定/可推进时钟。"""
    def __init__(self, t: float = 1_700_000_000.0):
        self.t = t  # 2023-11-14 22:13 (UTC+8 需 localtime——用测试本地时区)
    def __call__(self) -> float:
        return self.t
    def advance(self, dt: float) -> None:
        self.t += dt


def spy_deliveries(tmp_path):
    got = []
    def recv(pulse):
        got.append(pulse)
    return got, recv


def make_engine(tmp_path, clock=None, adjudicator=None, **kw):
    kw.setdefault("log_dir", tmp_path / "pulses")
    kw.setdefault("recipient", "张成市")
    if clock is not None:
        kw["now"] = clock
    if adjudicator is not None:
        kw["adjudicator"] = adjudicator
    return PulseEngine(**kw)


class TestTriggers:
    def test_interval_trigger(self):
        tr = IntervalTrigger(interval_sec=60, start_ts=1000.0)
        assert tr.due(1059.0) is False   # 未到点
        assert tr.due(1060.0) is True    # 到点
        assert tr.due(1061.0) is False   # 顺延后需再等 60s

    def test_time_window(self):
        w = TimeWindow(9, 22)
        assert w.allows(10) and w.allows(21)
        assert not w.allows(8) and not w.allows(22)
        cross = TimeWindow(22, 6)  # 跨午夜
        assert cross.allows(23) and cross.allows(3)
        assert not cross.allows(12)


class TestGeneration:
    def test_template_greeting_by_hour(self):
        g = _greeting(8)
        assert "早" in g or "上午" in g
        assert _greeting(0)  # 深夜有词

    def test_template_uses_context(self, tmp_path):
        """context 变量注入 thought 模板。"""
        eng = PulseEngine(log_dir=tmp_path / "p", recipient="张成市",
                          context={"context": "ISA 通信体重启"},
                          randomize_kind=False)
        msg = eng._build_message("thought", time.time())
        assert "ISA 通信体重启" in msg  # context 注入（核心断言）
        # 温度语气在（模板变体容忍：想起/想到/记得）
        assert any(w in msg for w in ("想起", "想到", "记得"))

    def test_greeting_template_by_hour(self, tmp_path):
        """greeting 模板按小时换词。"""
        eng = PulseEngine(log_dir=tmp_path / "p", recipient="张成市")
        # 固定一个上午时刻（避免跨日边界）
        hour9 = datetime.now().replace(hour=9, minute=0, second=0).timestamp()
        msg = eng._build_message("greeting", hour9)
        assert "早" in msg or "上午" in msg or "你好" in msg


class TestEngine:
    def test_tick_delivers_pulse(self, tmp_path):
        eng = make_engine(tmp_path, adjudicator=FakeAdjudicator("PASS"),
                          randomize_kind=False)
        eng._window = TimeWindow(0, 24)
        got, recv = spy_deliveries(tmp_path)
        eng.register_delivery("spy", recv)
        pulse = eng.tick()
        assert pulse is not None
        assert got and got[0] is pulse
        assert pulse.recipient == "张成市"

    def test_tick_blocked_by_adjudicator(self, tmp_path):
        """裁决 BLOCK → 不投递（P0-1 集成：围护生效）。"""
        eng = make_engine(tmp_path, adjudicator=FakeAdjudicator("BLOCK", "打扰预算用尽"),
                          randomize_kind=False)
        eng._window = TimeWindow(0, 24)
        got, recv = spy_deliveries(tmp_path)
        eng.register_delivery("spy", recv)
        assert eng.tick() is None
        assert got == []

    def test_daily_cap(self, tmp_path):
        """日硬顶：max_per_day=2 → 第 3 次 tick 不再投递。"""
        eng = make_engine(tmp_path, adjudicator=FakeAdjudicator("PASS"),
                          randomize_kind=False, max_per_day=2)
        eng._window = TimeWindow(0, 24)
        got, recv = spy_deliveries(tmp_path)
        eng.register_delivery("spy", recv)
        assert eng.tick() is not None
        assert eng.tick() is not None
        assert eng.tick() is None  # 硬顶
        assert len(got) == 2

    def test_outside_time_window(self, tmp_path):
        """时段窗外 → 不投递。"""
        eng = make_engine(tmp_path, adjudicator=FakeAdjudicator("PASS"),
                          randomize_kind=False, time_window=TimeWindow(0, 1))
        # 当前时间大概率不在 0-1 点窗——用注入窗口测 23 点外场景
        clock = FixedClock()
        # localtime 解析 1700000000 → 2023-11-14 22:13 UTC → +8 = 次日 06:13
        # 用 window 23-24 验证 06 点被拒（无论时区，06 不在 23-24 窗）
        eng2 = make_engine(tmp_path, adjudicator=FakeAdjudicator("PASS"),
                           randomize_kind=False, time_window=TimeWindow(23, 24))
        assert eng2.tick() is None

    def test_generator_hook_priority(self, tmp_path):
        """generator hook 优先于模板。"""
        eng = make_engine(tmp_path, adjudicator=FakeAdjudicator("PASS"),
                          generator=lambda kind, ctx: "老搭档，自定义温度。",
                          randomize_kind=False)
        eng._window = TimeWindow(0, 24)
        got, recv = spy_deliveries(tmp_path)
        eng.register_delivery("spy", recv)
        pulse = eng.tick()
        assert pulse is not None and pulse.message == "老搭档，自定义温度。"

    def test_generator_failure_falls_back(self, tmp_path):
        """generator 抛异常 → 回退模板不崩。"""
        def bad_gen(kind, ctx):
            raise RuntimeError("LLM down")
        eng = make_engine(tmp_path, adjudicator=FakeAdjudicator("PASS"),
                          generator=bad_gen, randomize_kind=False)
        eng._window = TimeWindow(0, 24)
        pulse = eng.tick()
        assert pulse is not None and pulse.message  # 模板兜底

    def test_audit_written(self, tmp_path):
        eng = make_engine(tmp_path, adjudicator=FakeAdjudicator("PASS"),
                          randomize_kind=False)
        eng._window = TimeWindow(0, 24)
        eng.tick()
        eng._adjudicator = FakeAdjudicator("BLOCK", "频控")
        eng.tick()
        lines = [l for l in (tmp_path / "pulses" / "pulses.jsonl").read_text()
                 .splitlines() if l.strip()]
        assert len(lines) == 2
        recs = [json.loads(l) for l in lines]
        assert recs[0]["delivered"] is True
        assert recs[1]["delivered"] is False
        assert recs[1]["verdict"] == "BLOCK"

    def test_console_delivery_default(self, tmp_path, capsys):
        """默认 console 投递（不依赖通道的演示路径）。"""
        eng = make_engine(tmp_path, adjudicator=FakeAdjudicator("PASS"),
                          randomize_kind=False)
        eng._window = TimeWindow(0, 24)
        eng.tick()
        out = capsys.readouterr().out
        assert "[pulse:thought]" in out


class TestRealAdjudicatorIntegration:
    def test_real_adjudicator_passes_low_importance(self, tmp_path):
        """真实 P0-1 裁决器：非任务振动（importance 0.3）中性语义 → PASS。"""
        from openllm.governance.adjudication import TransmissionAdjudicator
        adj = TransmissionAdjudicator(log_dir=tmp_path / "adj")
        eng = make_engine(tmp_path, adjudicator=adj, randomize_kind=False)
        eng._window = TimeWindow(0, 24)
        got, recv = spy_deliveries(tmp_path)
        eng.register_delivery("spy", recv)
        pulse = eng.tick()
        assert pulse is not None
        assert got and got[0] is pulse

    def test_real_adjudicator_frequency_blocks(self, tmp_path):
        """真实裁决器频控：同源 10 次/分硬顶 → 高频 tick 后被 BLOCK。"""
        from openllm.governance.adjudication import TransmissionAdjudicator
        adj = TransmissionAdjudicator(log_dir=tmp_path / "adj", rate_limit=5)
        eng = make_engine(tmp_path, adjudicator=adj, randomize_kind=False,
                          max_per_day=100)
        eng._window = TimeWindow(0, 24)
        got, recv = spy_deliveries(tmp_path)
        eng.register_delivery("spy", recv)
        delivered = 0
        for _ in range(8):
            if eng.tick() is not None:
                delivered += 1
        assert delivered == 5  # 前 5 次过，第 6 次起频控 BLOCK
        assert len(got) == 5
