"""
三角制衡耦合探针 — T4：三元矛盾B方案v2最终合成器
==================================================

消费 rule.probed（R度量）和 knowledge.probed（K度量）两条事件流，
计算K-R耦合系数，输出三角制衡的生命体征。

设计原则（与 rule_change_probe.py / knowledge_monotonic.py 同族）：
  - append-only 消费者模式：query → 提取 → 对齐 → 判定 → dataclass
  - 零LLM调用，纯确定性IO
  - 中文docstring
  - 阈值纪律：从数据分布长，不拍脑袋定 magic number

核心指标：
  - k_r_coupling: K有变化的日子中，R也有响应的比例（三角制衡有无生命证据）
  - k_diversity: K verdict 的多样性（活的涨落 vs 死水一潭）
  - r_health: R 侧有效变更占比（effective/(effective+idle)）

Verdict 四态：
  - insufficient-data: 有效事件对 <2 天（诚实优先）
  - decoupled: K动R死不动（三模块摆设——B方案最怕的死状）
  - coupled: K动R也动（三角制衡有生命证据）
  - dormant: K全stagnant且R全no-change（系统静默，非死亡但休眠）

用法：
    from openllm.evolution.triad_coupling_probe import TriadCouplingProbe
    probe = TriadCouplingProbe(bus=evolution_bus)
    report = probe.probe(days=30)
    print(report.verdict)

铁律：
  - 耦合系数用非零判定（R有效=effective_changes>0 或 verdict变化）
  - 德墨忒尔反向测试：K动R死→decoupled，K动R也动→coupled，全静→dormant
  - emit_report 写回总线 type='triad.probed'
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TriadReport:
    """三角制衡探针报告——T4。"""
    days_observed: int = 0               # 实际观察到的有效天数（R与K都有数据的天）
    k_r_coupling: float = 0.0            # K变化日中R也有响应的比例 [0.0, 1.0]
    k_diversity: float = 0.0             # K verdict 多样性（Shannon熵归一化）
    r_health: float = 0.0                # R 有效变更占比
    verdict: str = "insufficient-data"   # 四态判定
    detail: str = ""                     # 人类可读解释


class TriadCouplingProbe:
    """
    三角制衡耦合探针——总线消费者。

    消费 rule.probed + knowledge.probed → 时间对齐 → 计算耦合指标 → 判定。
    消费者模式：与 RuleChangeProbe / KnowledgeMonotonicProbe 同族。
    """

    # 消费哪些事件类型
    CONSUMED_TYPES = ["rule.probed", "knowledge.probed"]

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
        """注册为 bus 消费者（反断头管，幂等）。"""
        bus = self._get_bus()
        for etype in self.CONSUMED_TYPES:
            bus.registry.register(etype, "triad_coupling_probe")

    def probe(self, days: int = 30) -> TriadReport:
        """
        从总线读 rule.probed 和 knowledge.probed 事件序列，
        时间对齐后计算耦合指标并输出判定。

        Returns:
            TriadReport: days_observed / k_r_coupling / k_diversity / r_health / verdict
        """
        bus = self._get_bus()

        # ① query 两条事件流
        r_events = bus.query(event_type="rule.probed", days=days, limit=500)
        k_events = bus.query(event_type="knowledge.probed", days=days, limit=500)

        # ② 按日期分组（ts 截取 YYYY-MM-DD 作为日键）
        r_by_day: Dict[str, Dict[str, Any]] = {}
        for ev in r_events:
            day = ev.get("ts", "")[:10]  # "2026-09-06T..." → "2026-09-06"
            if not day or len(day) < 10:
                continue
            day = day[:10]
            payload = ev.get("payload", {})
            r_by_day[day] = {
                "verdict": payload.get("verdict", "no-change"),
                "effective_changes": payload.get("effective_changes", 0),
                "idle_changes": payload.get("idle_changes", 0),
            }

        k_by_day: Dict[str, Dict[str, Any]] = {}
        for ev in k_events:
            day = ev.get("ts", "")[:10]
            if not day or len(day) < 10:
                continue
            day = day[:10]
            payload = ev.get("payload", {})
            k_by_day[day] = {
                "verdict": payload.get("verdict", "stagnant"),
                "capacity_delta": payload.get("capacity_delta", {}),
            }

        # ③ 时间对齐：取 R 与 K 都有数据的日期
        common_days = sorted(set(r_by_day.keys()) & set(k_by_day.keys()))

        # ④ 判定 insufficient-data（诚实优先）
        if len(common_days) < 2:
            total_events = len(r_events) + len(k_events)
            detail = (
                f"有效事件对天数不足：{len(common_days)}天（需≥2天）。"
                f"rule.probed={len(r_events)}条, knowledge.probed={len(k_events)}条, "
                f"共{total_events}条事件"
            )
            return TriadReport(
                days_observed=len(common_days),
                k_r_coupling=0.0,
                k_diversity=0.0,
                r_health=0.0,
                verdict="insufficient-data",
                detail=detail,
            )

        # ⑤ 计算耦合指标
        # k_r_coupling: K有变化的日子中，R也有响应的比例
        # K有变化 = verdict不是stagnant也不是insufficient_data
        # R有响应 = effective_changes > 0 或 verdict包含'effective'
        k_change_days = 0
        k_r_coupled_days = 0
        r_effective_total = 0
        r_idle_total = 0

        for day in common_days:
            k_info = k_by_day[day]
            r_info = r_by_day[day]

            k_verdict = k_info["verdict"]
            r_effective = r_info["effective_changes"]
            r_idle = r_info["idle_changes"]
            r_verdict = r_info["verdict"]

            # K有变化：verdict 不是 stagnant / insufficient_data
            k_has_change = k_verdict not in ("stagnant", "insufficient_data")
            if k_has_change:
                k_change_days += 1
                # R有响应：effective > 0 或 verdict = rsi-effective
                r_responded = (r_effective > 0) or (r_verdict == "rsi-effective")
                if r_responded:
                    k_r_coupled_days += 1

            r_effective_total += r_effective
            r_idle_total += r_idle

        # 耦合系数
        if k_change_days > 0:
            k_r_coupling = k_r_coupled_days / k_change_days
        else:
            k_r_coupling = 0.0

        # k_diversity: K verdict 多样性（Shannon 熵归一化）
        k_verdict_counts: Dict[str, int] = defaultdict(int)
        for day in common_days:
            kv = k_by_day[day]["verdict"]
            k_verdict_counts[kv] += 1
        k_diversity = _normalized_shannon(k_verdict_counts)

        # r_health
        r_total = r_effective_total + r_idle_total
        r_health = r_effective_total / r_total if r_total > 0 else 0.0

        # ⑥ verdict 判定（德墨忒尔：K-R耦合是三角制衡的生命证据）
        r_strict = _is_r_dead(r_by_day, common_days)
        k_is_alive = k_change_days > 0

        if k_is_alive and not r_strict:
            # K在动且R也有响应 → 三角制衡有生命
            verdict = "coupled"
            detail = (
                f"三角制衡活跃：K侧{k_change_days}天变化中"
                f"{k_r_coupled_days}天R有响应"
                f"（k_r_coupling={k_r_coupling:.3f}，"
                f"r_health={r_health:.3f}）"
            )
        elif k_is_alive and r_strict:
            # K在动但R死不动 → 三模块摆设
            verdict = "decoupled"
            detail = (
                f"K-R解耦：K侧{k_change_days}天有变化，但R侧有效变更=0。"
                f"三模块耦合断裂（k_r_coupling={k_r_coupling:.3f}）"
            )
        else:
            # K不活 且 R也不活 → 休眠
            verdict = "dormant"
            detail = (
                f"系统静默：K侧全部stagnant，R侧全部no-change。"
                f"非死亡但休眠（{len(common_days)}天观察期）"
            )

        return TriadReport(
            days_observed=len(common_days),
            k_r_coupling=round(k_r_coupling, 4),
            k_diversity=round(k_diversity, 4),
            r_health=round(r_health, 4),
            verdict=verdict,
            detail=detail,
        )

    def emit_report(self) -> Dict[str, Any]:
        """
        把 TriadReport 作为新事件 type='triad.probed' 写回总线。
        """
        from openllm.evolution.bus import DigestEvent
        bus = self._get_bus()
        report = self.probe()
        event = DigestEvent(
            type="triad.probed",
            producer="triad_coupling_probe",
            payload={
                "days_observed": report.days_observed,
                "k_r_coupling": report.k_r_coupling,
                "k_diversity": report.k_diversity,
                "r_health": report.r_health,
                "verdict": report.verdict,
                "detail": report.detail,
            },
        )
        bus.append(event)
        return {"event_type": "triad.probed", "verdict": report.verdict}


# ═══ 辅助函数 ═══

def _is_r_dead(r_by_day: Dict[str, Dict[str, Any]], days: List[str]) -> bool:
    """R是否'严格死'：所有天的 effective_changes == 0 且 verdict != rsi-effective。"""
    for day in days:
        info = r_by_day.get(day, {})
        if info.get("effective_changes", 0) > 0:
            return False
        if info.get("verdict") == "rsi-effective":
            return False
    return True


def _normalized_shannon(counts: Dict[str, int]) -> float:
    """Shannon 熵归一化：H / log2(|categories|)，结果 [0, 1]。

    - 全部相同 verdict → 0.0（无多样性）
    - 均匀分布 → 1.0（最大多样性）
    """
    import math
    total = sum(counts.values())
    if total == 0 or len(counts) <= 1:
        return 0.0
    entropy = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    max_entropy = math.log2(len(counts))
    return entropy / max_entropy if max_entropy > 0 else 0.0


# ═══ CLI ═══

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="三角制衡耦合探针 T4")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--emit", action="store_true", help="emit报告写回总线")
    args = ap.parse_args()

    probe = TriadCouplingProbe()
    probe.register_as_consumer()
    report = probe.probe(days=args.days)

    print(f"三角制衡报告: verdict={report.verdict}")
    print(f"  days_observed={report.days_observed}")
    print(f"  k_r_coupling={report.k_r_coupling}")
    print(f"  k_diversity={report.k_diversity}")
    print(f"  r_health={report.r_health}")
    print(f"  依据: {report.detail}")

    if args.emit:
        result = probe.emit_report()
        print(f"  emit: {result}")


if __name__ == "__main__":
    main()
