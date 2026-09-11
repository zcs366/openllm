"""
rule_change_probe 测试 — ≥8条含德墨忒尔反例
=============================================
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from openllm.evolution.bus import EvolutionBus, DigestEvent, KNOWN_TYPES
from openllm.evolution.rule_change_probe import (
    RuleChangeProbe,
    RuleChangeReport,
    _crc32,
)


@pytest.fixture
def tmp_bus(tmp_path):
    """创建临时 bus（隔离真实数据）。"""
    return EvolutionBus(store_dir=tmp_path)


def _make_train_event(version, questions=10, degradations=2, producer="nightly_train"):
    """构造 train.completed 事件。"""
    return DigestEvent(
        type="train.completed",
        producer=producer,
        payload={
            "version": version,
            "new_samples": 150,
            "epochs": 3,
            "verify": {
                "questions": questions,
                "degradations": degradations,
                "skipped": False,
            },
        },
    )


# ── RC1: 模块可 import ──
def test_import():
    """RuleChangeProbe / RuleChangeReport 存在"""
    assert RuleChangeProbe is not None
    assert RuleChangeReport is not None


# ── RC4: 德墨忒尔反例——空转（quality恒定→behavior_diff=0）→ rsi-idle ──
def test_demeter_idle(tmp_bus):
    """
    构造"每天微调一条无用规则"事件序列：
    3轮训练但质量完全相同（Q=0.8恒定）→ 每轮behavior_diff=0 → verdict='rsi-idle'。
    这是RC4核心验收：不能把空转变更当成有效改进。
    """
    for v in ("v1", "v2", "v3"):
        tmp_bus.append(_make_train_event(v, questions=10, degradations=2))  # Q=0.8 恒定

    probe = RuleChangeProbe(bus=tmp_bus)
    probe.register_as_consumer()
    report = probe.probe()

    # 关键断言：count=3但verdict必须是rsi-idle
    assert report.verdict == "rsi-idle", (
        f"德墨忒尔反例失败：空转变更被判为有效！"
        f"verdict={report.verdict}, effective={report.effective_changes}, "
        f"idle={report.idle_changes}"
    )
    assert report.effective_changes == 0
    assert report.idle_changes == 3
    # 首轮无前基线(diff=0) + 第2/3轮质量不变(diff=0)
    for r in report.rule_diff:
        assert r["behavior_diff"] == 0


# ── RC5-1: 有效变更（质量提升）→ rsi-effective ──
def test_effective_change(tmp_bus):
    """train.completed 质量递增 → behavior_diff≠0 → rsi-effective"""
    tmp_bus.append(_make_train_event("v1", questions=10, degradations=5))  # Q=0.5
    tmp_bus.append(_make_train_event("v2", questions=10, degradations=2))  # Q=0.8, diff=0.3

    probe = RuleChangeProbe(bus=tmp_bus)
    probe.register_as_consumer()
    report = probe.probe()
    assert report.verdict == "rsi-effective"
    assert report.effective_changes == 1  # 第2轮有delta
    # 第1轮：首轮无基线 diff=0; 第2轮：diff=0.3≠0
    assert report.rule_diff[0]["behavior_diff"] == 0  # 首轮
    assert report.rule_diff[1]["behavior_diff"] != 0  # 第2轮有提升


# ── RC5-2: 零事件 → no-change ──
def test_no_change(tmp_bus):
    """总线无事件 → no-change"""
    probe = RuleChangeProbe(bus=tmp_bus)
    report = probe.probe()
    assert report.verdict == "no-change"
    assert report.effective_changes == 0
    assert report.idle_changes == 0
    assert len(report.rule_diff) == 0


# ── RC5-3: 消费者注册 ──
def test_register_as_consumer(tmp_bus):
    """register_as_consumer 真调 bus.registry.register"""
    probe = RuleChangeProbe(bus=tmp_bus)
    probe.register_as_consumer()
    for etype in RuleChangeProbe.CONSUMED_TYPES:
        consumers = tmp_bus.registry.consumers_for(etype)
        assert "rule_change_probe" in consumers, f"未注册到 {etype}"


# ── RC5-4: emit 写回总线 ──
def test_emit_report(tmp_bus):
    """emit_report 写 type='rule.probed' 事件到总线"""
    tmp_bus.append(_make_train_event("v1", questions=10, degradations=2))
    probe = RuleChangeProbe(bus=tmp_bus)
    probe.register_as_consumer()
    result = probe.emit_report()
    assert result["event_type"] == "rule.probed"
    events = tmp_bus.query(event_type="rule.probed")
    assert len(events) == 1
    assert events[0]["producer"] == "rule_change_probe"
    assert events[0]["payload"]["verdict"] == "rsi-idle"  # 首轮无基线，diff=0，属空转


# ── RC5-5: before_hash/after_hash 确定性 ──
def test_hash_deterministic():
    """同输入同 hash（zlib.crc32 确定性）"""
    h1 = _crc32("test_value")
    h2 = _crc32("test_value")
    assert h1 == h2
    assert h1 != _crc32("other_value")
    assert len(h1) == 8  # 8位十六进制


# ── RC5-6: gate 事件也算变更 ──
def test_gate_event_is_change(tmp_bus):
    """gate.passed → 行为变更（behavior_diff=+1）"""
    tmp_bus.append(DigestEvent(
        type="gate.passed",
        producer="nightly_train",
        payload={"verdict": "passed", "hash": "xyz789"},
    ))
    probe = RuleChangeProbe(bus=tmp_bus)
    probe.register_as_consumer()
    report = probe.probe()
    assert report.verdict == "rsi-effective"
    assert report.rule_diff[0]["behavior_diff"] == 1
    assert report.rule_diff[0]["ids"].startswith("gate_")


# ── RC5-7: skill 事件也算变更 ──
def test_skill_event_is_change(tmp_bus):
    """skill.created → 行为变更（behavior_diff=+1）"""
    tmp_bus.append(DigestEvent(
        type="skill.created",
        producer="isn",
        payload={"skill_id": "new_skill", "name": "test"},
    ))
    probe = RuleChangeProbe(bus=tmp_bus)
    probe.register_as_consumer()
    report = probe.probe()
    assert report.verdict == "rsi-effective"
    assert report.rule_diff[0]["behavior_diff"] == 1
    assert "new_skill" in report.rule_diff[0]["ids"]


# ── RC3: rule_diff 全部六字段 ──
def test_rule_diff_six_fields(tmp_bus):
    """rule_diff 每条含 {count, ids, before_hash, after_hash, trigger, behavior_diff}"""
    tmp_bus.append(_make_train_event("v1", questions=10, degradations=2))
    probe = RuleChangeProbe(bus=tmp_bus)
    probe.register_as_consumer()
    report = probe.probe()
    assert len(report.rule_diff) == 1
    r = report.rule_diff[0]
    required_keys = {"count", "ids", "before_hash", "after_hash", "trigger", "behavior_diff"}
    assert required_keys.issubset(set(r.keys())), f"缺少字段: {required_keys - set(r.keys())}"


# ── RC2: emit_report 真调 bus.append ──
def test_emit_calls_bus_append(tmp_bus):
    """emit_report 事件被写入总线文件"""
    probe = RuleChangeProbe(bus=tmp_bus)
    probe.register_as_consumer()
    probe.emit_report()
    all_events = tmp_bus.query()
    probed = [e for e in all_events if e["type"] == "rule.probed"]
    assert len(probed) == 1
    assert probed[0]["producer"] == "rule_change_probe"


# ── 额外：混合事件场景 ──
def test_mixed_events(tmp_bus):
    """多种事件类型混合：train + gate + skill → 正确计数"""
    tmp_bus.append(_make_train_event("v1", questions=10, degradations=0))  # Q=1.0
    tmp_bus.append(DigestEvent(type="gate.passed", producer="gate", payload={"hash": "a"}))
    tmp_bus.append(DigestEvent(type="skill.created", producer="isn", payload={"skill_id": "s1"}))
    tmp_bus.append(DigestEvent(type="skill.retired", producer="isn", payload={"skill_id": "old"}))

    probe = RuleChangeProbe(bus=tmp_bus)
    probe.register_as_consumer()
    report = probe.probe()
    # train首轮diff=0(无基线), gate=+1, skill.created=+1, skill.retired=-1
    assert report.effective_changes == 3
    assert report.idle_changes == 1  # 首轮train
    assert report.verdict == "rsi-effective"


# ── 额外：质量下降也视为有效变更 ──
def test_quality_regression_is_effective(tmp_bus):
    """质量下降 → behavior_diff<0 ≠0 → 有效变更（退化也是信号）"""
    tmp_bus.append(_make_train_event("v1", questions=10, degradations=2))  # Q=0.8
    tmp_bus.append(_make_train_event("v2", questions=10, degradations=5))  # Q=0.5, diff=-0.3

    probe = RuleChangeProbe(bus=tmp_bus)
    probe.register_as_consumer()
    report = probe.probe()
    assert report.verdict == "rsi-effective"
    assert report.rule_diff[1]["behavior_diff"] < 0
