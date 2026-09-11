"""
规则变更探针 — R度量：每晚训练是否空转
==========================================

三元矛盾B方案v2的第一块可测指标（七神终裁 2026-09-06）：
证明"每晚训练非空转"。count≥1会被"定时脚本每天微调一条无用规则"骗过，
必须区分 **有效变更**（behavior_diff≠0）vs **空转变更**（count有但behavior_diff=0）。

判据: behavior_diff≠0 即有效（零阈值纪律：⑥-j死阈陷阱——不拍脑袋定阈值）。

输入: 进化事件总线（train.completed / gate.passed / gate.rejected / skill.created / skill.retired）
输出: RuleChangeReport（rule_diff六字段 + effective/idle计数 + verdict）

消费者模式（同 stopline.py 族）：query → 提取 → 判定 → dataclass结果

用法:
    from openllm.evolution.rule_change_probe import RuleChangeProbe
    probe = RuleChangeProbe(bus=evolution_bus)
    report = probe.probe(days=7)
    print(report.verdict)  # 'rsi-effective' / 'rsi-idle' / 'no-change'

铁律:
  - behavior_diff 用非零判定（≠0即有效），不引入 magic number
  - before_hash/after_hash 用 zlib.crc32 确定性哈希（bus.py L82 复用）
  - emit_report 写回总线 type='rule.probed'（供后续三角制衡消费）
  - train.completed 的 behavior_diff = 质量增量（本轮 - 上轮），首轮无前基线则 diff=0
"""

from __future__ import annotations

import zlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _crc32(text: str) -> str:
    """确定性哈希（zlib.crc32，复用 bus.py L82 模式，非内置 hash）。"""
    return f"{zlib.crc32(text.encode('utf-8')):08x}"


@dataclass
class RuleChangeReport:
    """规则变更报告——R度量。"""
    rule_diff: List[Dict[str, Any]] = field(default_factory=list)
    effective_changes: int = 0
    idle_changes: int = 0
    verdict: str = "no-change"
    detail: str = ""


class RuleChangeProbe:
    """
    规则变更探针——总线消费者。

    query 总线事件 → 提取行为变更 → 判定有效/空转 → 生成报告。
    消费者模式：与 StoplineEvaluator 同族。

    train.completed 的行为变更 = 相邻轮质量差（delta quality）。
    首轮无前基线 → diff=0（不计为有效，因为无法证明变化）。
    gate/skill 事件各自独立计为变更。
    """

    # 消费哪些事件类型（注册用）
    CONSUMED_TYPES = [
        "train.completed",
        "gate.passed",
        "gate.rejected",
        "skill.created",
        "skill.retired",
    ]

    def __init__(self, bus: Any = None, store_dir: Optional[Path] = None) -> None:
        self._bus = bus
        self._store_dir = store_dir

    def _get_bus(self) -> Any:
        """惰性获取 bus 实例。"""
        if self._bus is not None:
            return self._bus
        from openllm.evolution.bus import EvolutionBus
        self._bus = EvolutionBus(store_dir=self._store_dir)
        return self._bus

    def register_as_consumer(self) -> None:
        """注册为 bus 消费者（反断头管）。"""
        bus = self._get_bus()
        for etype in self.CONSUMED_TYPES:
            bus.registry.register(etype, "rule_change_probe")

    def probe(self, days: int = 7) -> RuleChangeReport:
        """
        从总线读事件，计算R度量。

        train.completed 的 behavior_diff = 相邻轮质量增量（after - before）。
        首轮无前基线 → behavior_diff=0（空转，无法证明变化）。
        gate/skill 事件各自独立计为变更。

        Returns:
            RuleChangeReport: rule_diff六字段 + effective/idle计数 + verdict
        """
        bus = self._get_bus()

        # query 所有相关事件类型
        all_events: List[Dict[str, Any]] = []
        for etype in self.CONSUMED_TYPES:
            all_events.extend(bus.query(event_type=etype, days=days, limit=500))

        # 按 ts 排序（旧→新），相同ts用索引保序
        for i, ev in enumerate(all_events):
            ev["_idx"] = i
        all_events.sort(key=lambda e: (e.get("ts", ""), e["_idx"]))

        rule_diff: List[Dict[str, Any]] = []

        # ── train.completed：按版本排序，逐轮处理，behavior_diff = 质量增量 ──
        train_events = [e for e in all_events if e.get("type") == "train.completed"]
        # 按版本号排序（旧→新）——ts可能相同（同秒append），版本号保证正确序
        train_events.sort(key=lambda e: str(e.get("payload", {}).get("version", "")))
        prev_quality: Optional[float] = None
        for ev in train_events:
            payload = ev.get("payload", {})
            verify = payload.get("verify", {})
            questions = verify.get("questions", 0)
            degradations = verify.get("degradations", 0)
            if questions <= 0:
                continue
            quality = round(1.0 - degradations / questions, 6)
            version = payload.get("version", "?")

            before_val = prev_quality if prev_quality is not None else None
            after_val = quality

            if before_val is not None:
                behavior_diff = round(after_val - before_val, 6)
            else:
                behavior_diff = 0  # 首轮无前基线，不计有效

            before_hash = _crc32(str(before_val)) if before_val is not None else ""
            after_hash = _crc32(str(after_val))

            rule_diff.append({
                "count": 1,
                "ids": f"v{version}",
                "before_hash": before_hash,
                "after_hash": after_hash,
                "trigger": ev.get("producer", ""),
                "behavior_diff": behavior_diff,
            })
            prev_quality = quality

        # ── gate / skill 事件：各自独立计为变更 ──
        for ev in all_events:
            etype = ev.get("type", "")
            payload = ev.get("payload", {})
            ids = ""
            before_val: Any = None
            after_val: Any = None
            behavior_diff: Any = 0

            if etype == "gate.passed":
                ids = f"gate_{payload.get('hash', '?')}"
                before_val = 0
                after_val = 1
                behavior_diff = 1
            elif etype == "gate.rejected":
                ids = f"gate_{payload.get('hash', '?')}"
                before_val = 0
                after_val = -1
                behavior_diff = -1
            elif etype == "skill.created":
                skill_id = payload.get("skill_id", payload.get("id", "?"))
                ids = f"skill_{skill_id}"
                after_val = skill_id
                behavior_diff = 1
            elif etype == "skill.retired":
                skill_id = payload.get("skill_id", payload.get("id", "?"))
                ids = f"skill_{skill_id}"
                before_val = skill_id
                behavior_diff = -1
            else:
                continue

            before_hash = _crc32(str(before_val)) if before_val is not None else ""
            after_hash = _crc32(str(after_val)) if after_val is not None else ""

            rule_diff.append({
                "count": 1,
                "ids": ids,
                "before_hash": before_hash,
                "after_hash": after_hash,
                "trigger": ev.get("producer", ""),
                "behavior_diff": behavior_diff,
            })

        effective = sum(1 for r in rule_diff if r["behavior_diff"] != 0)
        idle = len(rule_diff) - effective

        if len(rule_diff) == 0:
            verdict = "no-change"
            detail = "查询范围内无规则变更事件"
        elif effective >= 1:
            verdict = "rsi-effective"
            detail = f"{effective}条有效变更（behavior_diff≠0），{idle}条空转"
        else:
            verdict = "rsi-idle"
            detail = f"全部{idle}条变更为空转（behavior_diff=0），训练无行为变化"

        return RuleChangeReport(
            rule_diff=rule_diff,
            effective_changes=effective,
            idle_changes=idle,
            verdict=verdict,
            detail=detail,
        )

    def emit_report(self) -> Dict[str, Any]:
        """
        把 RuleChangeReport 作为新事件 type='rule.probed' 写回总线。
        供后续三角制衡耦合探针消费。
        """
        from openllm.evolution.bus import DigestEvent
        bus = self._get_bus()
        report = self.probe()
        event = DigestEvent(
            type="rule.probed",
            producer="rule_change_probe",
            payload={
                "effective_changes": report.effective_changes,
                "idle_changes": report.idle_changes,
                "verdict": report.verdict,
                "detail": report.detail,
                "rule_diff_count": len(report.rule_diff),
            },
        )
        bus.append(event)
        return {"event_type": "rule.probed", "verdict": report.verdict}


# ═══ CLI ═══

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="规则变更探针 R度量")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--emit", action="store_true", help="emit报告写回总线")
    args = ap.parse_args()

    probe = RuleChangeProbe()
    probe.register_as_consumer()
    report = probe.probe(days=args.days)

    print(f"R度量报告: verdict={report.verdict}")
    print(f"  有效变更: {report.effective_changes} | 空转: {report.idle_changes}")
    print(f"  依据: {report.detail}")
    print(f"  rule_diff条数: {len(report.rule_diff)}")
    for i, r in enumerate(report.rule_diff[:5]):
        print(f"  [{i}] {r['ids']} trigger={r['trigger']} diff={r['behavior_diff']}")

    if args.emit:
        result = probe.emit_report()
        print(f"  emit: {result}")


if __name__ == "__main__":
    main()
