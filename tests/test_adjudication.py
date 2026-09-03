"""test_adjudication.py — 传输裁决模块（P0-1 · 该不该传/醒）测试

覆盖：L0 硬规则（红线/凭据/黑名单/频控/自定义）一票否决 ·
L1 软评分（importance/semantic/冷却）· 失败安全（信号异常中性化 /
裁决器异常内部放行外部拒绝）· 审计落盘 · 便捷入口。
"""
import json
from typing import Optional

import pytest

from openllm.governance.adjudication import (
    TransmissionAdjudicator, TransmissionRequest, Verdict, PolicyRule,
    should_transmit, make_request,
)


class FakeSemantic:
    """可编程语义匹配器：固定相似度或抛异常。"""
    def __init__(self, sim: float = 0.5, raise_error: bool = False):
        self.sim = sim
        self.raise_error = raise_error
    def similarity(self, interest: str, body: str):
        if self.raise_error:
            raise RuntimeError("embedding 故障")
        return self.sim


class Clock:
    """可推进时钟（冷却/频控测试）。"""
    def __init__(self, t: float = 1000.0):
        self.t = t
    def __call__(self) -> float:
        return self.t
    def advance(self, dt: float) -> None:
        self.t += dt


def make_req(from_="A", to_: Optional[str] = "B", msg_type="message",
             body="普通协作消息",
             importance=0.5, channel="internal", **kw):
    return TransmissionRequest(from_agent=from_, to_agent=to_, msg_type=msg_type,
                               body=body, importance=importance, channel=channel, **kw)


def make_adjudicator(tmp_path, clock=None, **kw):
    kw.setdefault("log_dir", tmp_path / "adj")
    if clock is not None:
        kw["now"] = clock
    return TransmissionAdjudicator(**kw)


# ── L0 硬规则：一票否决 ──

class TestHardRules:
    def test_redline_type_blocked(self, tmp_path):
        adj = make_adjudicator(tmp_path)
        v = adj.decide(make_req(msg_type="terminate"))
        assert v.action == "BLOCK"
        assert "红线" in v.reason

    def test_redline_case_insensitive(self, tmp_path):
        adj = make_adjudicator(tmp_path)
        assert adj.decide(make_req(msg_type="KILL")).action == "BLOCK"
        assert adj.decide(make_req(msg_type="ShutDown")).action == "BLOCK"

    def test_secret_pattern_blocked(self, tmp_path):
        adj = make_adjudicator(tmp_path)
        v = adj.decide(make_req(body="我的key是 sk-abc1234567890abcdefghijkl，请查收"))
        assert v.action == "BLOCK"
        assert "凭据" in v.reason

    def test_secret_key_value_pattern(self, tmp_path):
        adj = make_adjudicator(tmp_path)
        assert adj.decide(make_req(body="配置项 api_key: super_secret_value_123")).action == "BLOCK"

    def test_plain_text_not_blocked(self, tmp_path):
        adj = make_adjudicator(tmp_path)
        assert adj.decide(make_req(body="今天完成了语义路由升级，测试全绿")).action == "PASS"

    def test_blacklist_pair(self, tmp_path):
        adj = make_adjudicator(tmp_path)
        adj.block_pair("A", "B")
        assert adj.decide(make_req(from_="A", to_="B")).action == "BLOCK"
        assert adj.decide(make_req(from_="A", to_="C")).action == "PASS"

    def test_blacklist_wildcard(self, tmp_path):
        adj = make_adjudicator(tmp_path)
        adj.block_pair("A", "*")
        assert adj.decide(make_req(from_="A", to_="B")).action == "BLOCK"
        assert adj.decide(make_req(from_="C", to_="A")).action == "PASS"  # 反向不受限

    def test_rate_limit_hard_cap(self, tmp_path):
        clock = Clock()
        adj = make_adjudicator(tmp_path, clock=clock, rate_limit=3)
        for _ in range(3):
            assert adj.decide(make_req()).action == "PASS"  # 前 3 次在限内
        v = adj.decide(make_req())  # 第 4 次
        assert v.action == "BLOCK"
        assert "频控" in v.reason

    def test_rate_limit_window_expiry(self, tmp_path):
        clock = Clock()
        adj = make_adjudicator(tmp_path, clock=clock, rate_limit=3, rate_window_sec=60)
        for _ in range(3):
            adj.decide(make_req())
        assert adj.decide(make_req()).action == "BLOCK"
        clock.advance(61)  # 窗口滑过 → 计数重置
        assert adj.decide(make_req()).action == "PASS"

    def test_custom_rule_registered(self, tmp_path):
        adj = make_adjudicator(tmp_path)
        adj.register_rule(PolicyRule("no-night", lambda req: "夜间禁传" if "夜间" in req.body else None))
        assert adj.decide(make_req(body="夜间任务汇报")).action == "BLOCK"
        assert adj.decide(make_req(body="白天正常协作")).action == "PASS"


# ── L1 信号层：软评分 ──

class TestSignalScoring:
    def test_default_neutral_passes(self, tmp_path):
        """中性基线（imp 0.5 / sem 0.5 / rec 1.0）→ score=0.65 ≥ 阈值 → PASS。"""
        adj = make_adjudicator(tmp_path)
        v = adj.decide(make_req())
        assert v.action == "PASS"
        assert v.score == pytest.approx(0.65)

    def test_semantic_relevance_raises_score(self, tmp_path):
        adj = make_adjudicator(tmp_path, semantic=FakeSemantic(0.9),
                               interests={"B": "模型训练"})
        v = adj.decide(make_req(body="LoRA 微调 7B 模型损失下降"))
        assert v.action == "PASS"
        assert v.score == pytest.approx(0.85, abs=1e-6)  # (0.1+0.45+0.3)/1.0

    def test_low_signal_blocked(self, tmp_path):
        """semantic 不相关 + importance 低 → score=0.37 < 0.5 → BLOCK。"""
        adj = make_adjudicator(tmp_path, semantic=FakeSemantic(0.1),
                               interests={"B": "模型训练"})
        v = adj.decide(make_req(body="晚饭吃了回锅肉", importance=0.1))
        assert v.action == "BLOCK"
        assert "信号不足" in v.reason

    def test_semantic_mismatch_high_importance_still_passes(self, tmp_path):
        """semantic 不相关但 importance 高 → (0.2*1+0.5*0.1+0.3)/1=0.55 → PASS。"""
        adj = make_adjudicator(tmp_path, semantic=FakeSemantic(0.1),
                               interests={"B": "模型训练"})
        v = adj.decide(make_req(body="无关内容但很重要", importance=1.0))
        assert v.action == "PASS"

    def test_recency_cooldown_blocks_then_expires(self, tmp_path):
        clock = Clock()
        adj = make_adjudicator(tmp_path, clock=clock, min_interval_sec=60)
        assert adj.decide(make_req()).action == "PASS"
        # 冷却期内第二次：rec=0 → (0.1+0.25+0)/1=0.35 < 0.5 → BLOCK
        v = adj.decide(make_req())
        assert v.action == "BLOCK"
        clock.advance(61)
        assert adj.decide(make_req()).action == "PASS"

    def test_interests_missing_neutral(self, tmp_path):
        """接收方无兴趣配置 → semantic 中性（不偏不倚）。"""
        adj = make_adjudicator(tmp_path, semantic=FakeSemantic(0.1))  # 无 interests
        assert adj.decide(make_req(importance=0.5)).action == "PASS"

    def test_no_body_no_semantic(self, tmp_path):
        adj = make_adjudicator(tmp_path, semantic=FakeSemantic(0.9),
                               interests={"B": "模型训练"})
        assert adj.decide(make_req(body="")).action == "PASS"  # 空 body → 中性

    def test_broadcast_to_none(self, tmp_path):
        """广播（to_agent=None）→ 无兴趣匹配 → 中性 PASS。"""
        adj = make_adjudicator(tmp_path, semantic=FakeSemantic(0.9),
                               interests={"B": "模型训练"})
        assert adj.decide(make_req(to_=None)).action == "PASS"


# ── 失败安全 ──

class TestFailSafe:
    def test_semantic_exception_neutralized(self, tmp_path):
        """semantic 信号抛异常 → 中性化，裁决不崩，回中性基线 PASS。"""
        adj = make_adjudicator(tmp_path, semantic=FakeSemantic(0.1, raise_error=True),
                               interests={"B": "模型训练"})
        v = adj.decide(make_req())
        assert v.action == "PASS"  # sem 中性 0.5 → 中性基线 0.65
        assert v.score == pytest.approx(0.65)

    def test_adjudicator_exception_internal_pass(self, tmp_path):
        """自定义规则抛异常 → 裁决器异常 → internal 放行（感知不阻断）。"""
        def boom(req):
            raise RuntimeError("规则引擎炸了")
        adj = make_adjudicator(tmp_path)
        adj.register_rule(PolicyRule("boom", boom))
        v = adj.decide(make_req())  # channel=internal
        assert v.action == "PASS"
        assert "异常降级" in v.reason

    def test_adjudicator_exception_external_block(self, tmp_path):
        """同场景 channel=im（对外）→ BLOCK（防失控）。"""
        def boom(req):
            raise RuntimeError("规则引擎炸了")
        adj = make_adjudicator(tmp_path)
        adj.register_rule(PolicyRule("boom", boom))
        v = adj.decide(make_req(channel="im"))
        assert v.action == "BLOCK"
        assert "异常降级" in v.reason


# ── L2 打扰治理（v1：预算 + 振荡护栏）──

class TestDisruptionGovernance:
    """打扰预算 + 振荡护栏——只作用于打扰型 wake/pulse，正常 comm 不误伤。"""

    def _wake(self, from_="A", to_="B", body="想到你"):
        return make_req(from_=from_, to_=to_, msg_type="wake", body=body)

    def test_daily_budget_enforced(self, tmp_path):
        """B 日预算 2：跨发送方合计，第 3 次打扰型 BLOCK。"""
        adj = make_adjudicator(tmp_path)
        adj.set_daily_budget("B", 2)
        assert adj.decide(self._wake(from_="A")).action == "PASS"
        assert adj.decide(self._wake(from_="C")).action == "PASS"  # 另一发送方也占 B 预算
        v = adj.decide(self._wake(from_="A"))
        assert v.action == "BLOCK"
        assert "打扰预算" in v.reason

    def test_budget_resets_next_day(self, tmp_path):
        """跨日重置：推进一天后预算恢复。"""
        clock = Clock()
        adj = make_adjudicator(tmp_path, clock=clock)
        adj.set_daily_budget("B", 1)
        assert adj.decide(self._wake()).action == "PASS"
        assert adj.decide(self._wake()).action == "BLOCK"
        clock.advance(86400)  # 次日
        assert adj.decide(self._wake()).action == "PASS"

    def test_non_disruptive_not_counted(self, tmp_path):
        """正常 comm.ask 不占打扰预算（预算=1 时 ask 可多次）。"""
        adj = make_adjudicator(tmp_path)
        adj.set_daily_budget("B", 1)
        for _ in range(3):
            assert adj.decide(make_req(msg_type="comm.ask")).action == "PASS"

    def test_oscillation_guard(self, tmp_path):
        """打扰型互唤：A→B、B→A 成对后，第 3 次方向 BLOCK（振荡冷却）。"""
        clock = Clock()
        adj = make_adjudicator(tmp_path, clock=clock)
        assert adj.decide(self._wake(from_="A", to_="B")).action == "PASS"
        assert adj.decide(self._wake(from_="B", to_="A")).action == "PASS"  # 成对互唤完成
        v = adj.decide(self._wake(from_="A", to_="B"))  # 再唤醒 → 振荡
        assert v.action == "BLOCK"
        assert "振荡" in v.reason

    def test_oscillation_not_apply_to_comm(self, tmp_path):
        """正常 comm.ask/reply 双向通信不受振荡护栏影响。"""
        adj = make_adjudicator(tmp_path)
        assert adj.decide(make_req(from_="A", to_="B", msg_type="comm.ask")).action == "PASS"
        assert adj.decide(make_req(from_="B", to_="A", msg_type="comm.reply")).action == "PASS"
        assert adj.decide(make_req(from_="A", to_="B", msg_type="comm.ask")).action == "PASS"

    def test_oscillation_window_expiry(self, tmp_path):
        """窗口滑过后互唤记录过期 → 振荡解除。"""
        clock = Clock()
        adj = make_adjudicator(tmp_path, clock=clock, osc_window_sec=60)
        assert adj.decide(self._wake(from_="A", to_="B")).action == "PASS"
        assert adj.decide(self._wake(from_="B", to_="A")).action == "PASS"
        assert adj.decide(self._wake(from_="A", to_="B")).action == "BLOCK"
        clock.advance(61)
        assert adj.decide(self._wake(from_="A", to_="B")).action == "PASS"


# ── 审计 ──

class TestAudit:
    def test_audit_written(self, tmp_path):
        adj = make_adjudicator(tmp_path)
        adj.decide(make_req())
        adj.decide(make_req(msg_type="terminate"))
        log_file = tmp_path / "adj" / "adjudication.jsonl"
        assert log_file.exists()
        lines = [json.loads(l) for l in log_file.read_text().splitlines() if l.strip()]
        assert len(lines) == 2
        assert lines[0]["action"] == "PASS"
        assert lines[1]["action"] == "BLOCK"
        assert lines[0]["from"] == "A" and lines[1]["reason"]

    def test_verdict_fields(self, tmp_path):
        adj = make_adjudicator(tmp_path)
        v = adj.decide(make_req())
        assert isinstance(v, Verdict)
        assert v.adjudicator_version == "0.1.0"
        assert v.policy_hits == []  # L0 未命中


# ── 便捷入口 ──

class TestConvenience:
    def test_make_request(self):
        req = make_request(from_agent="军师", to_agent="副军师", body="hello")
        assert isinstance(req, TransmissionRequest)
        assert req.channel == "internal"
        assert req.importance == 0.5

    def test_should_transmit_singleton(self, tmp_path, monkeypatch):
        """便捷入口：单例裁决器返回 Verdict（注入 tmp log_dir 防污染家目录）。"""
        import openllm.governance.adjudication as adj_mod
        fresh = TransmissionAdjudicator(log_dir=tmp_path / "sgl")
        monkeypatch.setattr(adj_mod, "_default_adjudicator", fresh)
        v = should_transmit(make_req())
        assert isinstance(v, Verdict)
        assert v.action in ("PASS", "BLOCK")
