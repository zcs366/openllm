"""
OpenLLM Evolution Bus — 进化事件总线 v0.1
=========================================

七神启示终裁产物（2026-09-04）：进化仪表的本质是进化事件总线，不是仪表盘。
总线必须先于仪表——任何测量不自带消费者闭环 = 断头管（viability_logger血泪）。

链路: 生产者(训练/验证/技能事件) → append → DigestEvent → 消费者注册表 → 决策回路

核心概念:
  - DigestEvent: 统一事件格式 (type/novelty/hash/ts/producer/payload/consumer_status)
  - ConsumerRegistry: 声明谁消费什么 type（反断头管保障）
  - 测量铁律: 每条测量必须回答"谁读它、读完触发什么"——
    无consumer的type上线 = 告警（断头管血泪的第四次重演预防）

用法:
    from openllm.evolution.bus import EvolutionBus, DigestEvent
    bus = EvolutionBus(store_dir="~/.openllm/evolution")
    bus.register_consumer("train.completed", "stopline")
    bus.append(DigestEvent(type="train.completed", producer="nightly_train",
                           payload={"version": "v3", "new_samples": 150}))
    events = bus.query(type="train.completed")
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_STORE_DIR = Path.home() / ".openllm" / "evolution"

# 事件类型枚举（宽松：消费者需要什么type就注册什么type，不预先画全家福）
KNOWN_TYPES = {
    "train.completed",     # nightly_train 完成（含new_samples/epochs/verify结果）
    "train.skipped",       # nightly_train 跳过（材料不足/GPU忙/停训线）
    "verify.result",       # merged 冒烟验证结果（退化/健康）
    "viability.update",    # V值采样（viability_logger 并入总线）
    "skill.retired",       # ISN技能退役
    "skill.created",       # ISN技能创建
    "stopline.evaluated",  # 停训线判定（第一个真实消费者产出）
    "drift.sample",        # 漂移仪表采样（P2-G shadow）
    "metabolic.sample",    # 代谢变量采样（P1-E）
}


@dataclass
class DigestEvent:
    """进化事件统一格式（赫尔墨斯 schema：type/novelty/hash/ts 核心）。"""

    type: str
    producer: str
    payload: Dict[str, Any] = field(default_factory=dict)
    ts: str = ""                 # ISO时间戳，空则append时填
    novelty: float = 0.0         # vs近N日同类事件的embedding距离（新信息量，v0.1可缺省0）
    hash: str = ""               # 内容指纹（去重），空则append时算
    consumer_status: str = "pending"  # 已被哪个消费者处理（防重复消费）

    def __post_init__(self) -> None:
        if not self.type:
            raise ValueError("DigestEvent.type 必填")
        if not self.producer:
            raise ValueError("DigestEvent.producer 必填")

    def compute_hash(self) -> str:
        """内容指纹：type+producer+payload 的确定性哈希（zlib.crc32，非内置hash）。"""
        canonical = json.dumps(
            {"type": self.type, "producer": self.producer, "payload": self.payload},
            ensure_ascii=False, sort_keys=True, default=str,
        )
        return f"evt_{zlib_crc32(canonical):08x}"


def zlib_crc32(text: str) -> int:
    """确定性哈希（工程法典PITFALL 55：内置hash进程级随机，禁用）。"""
    import zlib
    return zlib.crc32(text.encode("utf-8"))


class ConsumerRegistry:
    """消费者注册表：type → 消费者名单。

    反断头管保障：每类事件必须有≥1注册消费者，否则append时告警。
    回答赫淮斯托斯测量铁律的前半问："谁读它"。
    """

    def __init__(self, registry_path: Optional[Path] = None) -> None:
        self.registry_path = registry_path or (DEFAULT_STORE_DIR / "consumers.json")
        self._consumers: Dict[str, List[str]] = {}
        self._load()

    def _load(self) -> None:
        if self.registry_path and self.registry_path.exists():
            try:
                data = json.loads(self.registry_path.read_text(encoding="utf-8"))
                self._consumers = data.get("consumers", {})
            except (json.JSONDecodeError, OSError):
                self._consumers = {}

    def _save(self) -> None:
        if not self.registry_path:
            return
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(
            json.dumps({"consumers": self._consumers}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def register(self, event_type: str, consumer: str) -> None:
        """声明消费者。幂等。"""
        self._consumers.setdefault(event_type, [])
        if consumer not in self._consumers[event_type]:
            self._consumers[event_type].append(consumer)
        self._save()

    def unregister(self, event_type: str, consumer: str) -> None:
        if event_type in self._consumers and consumer in self._consumers[event_type]:
            self._consumers[event_type].remove(consumer)
            self._save()

    def consumers_for(self, event_type: str) -> List[str]:
        return list(self._consumers.get(event_type, []))

    def has_consumer(self, event_type: str) -> bool:
        return bool(self._consumers.get(event_type))

    def unregistered_types(self, known_types: Optional[set] = None) -> List[str]:
        """已知但无人消费的type（审计用）。"""
        known = known_types or KNOWN_TYPES
        return sorted(t for t in known if not self.has_consumer(t))


class EvolutionBus:
    """进化事件总线：append → 文件；query → 过滤读取。

    存储: append-only JSONL（evolution_YYYYMMDD.jsonl 按日轮转）
    反断头管: append 时检查 consumer 注册表，无人读 → 告警（不阻塞写入）
    消费: 消费者主动 query（v0.1 不做推送/订阅——先跑通数据流）
    """

    def __init__(self, store_dir: Optional[Path] = None) -> None:
        self.store_dir = Path(store_dir) if store_dir else DEFAULT_STORE_DIR
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.registry = ConsumerRegistry(self.store_dir / "consumers.json")

    # ── 写入 ──

    def append(self, event: DigestEvent, warn_unconsumed: bool = True) -> DigestEvent:
        """追加一条事件。返回补齐 ts/hash 后的事件。

        反断头管：type 无注册消费者且 warn_unconsumed=True → 打印告警。
        告警不阻塞写入（总线是增强不是门禁——降级铁律）。
        """
        if not event.ts:
            event.ts = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        if not event.hash:
            event.hash = event.compute_hash()

        if warn_unconsumed and not self.registry.has_consumer(event.type):
            print(
                f"⚠️ [evolution-bus] type='{event.type}' 无注册消费者——"
                f"断头管风险。谁读它？读完触发什么？(producer={event.producer})",
                flush=True,
            )

        today = datetime.now().strftime("%Y%m%d")
        log_path = self.store_dir / f"evolution_{today}.jsonl"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
        return event

    # ── 读取 ──

    def query(
        self,
        event_type: Optional[str] = None,
        producer: Optional[str] = None,
        since: Optional[str] = None,       # ISO时间戳，>=since
        days: int = 7,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """查询事件。返回 dict 列表（新→旧）。"""
        results: List[Dict[str, Any]] = []
        paths = sorted(self.store_dir.glob("evolution_*.jsonl"), reverse=True)
        read_seq = 0
        for path in paths[:days]:
            try:
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if event_type and rec.get("type") != event_type:
                            continue
                        if producer and rec.get("producer") != producer:
                            continue
                        if since and rec.get("ts", "") < since:
                            continue
                        read_seq += 1
                        rec["_read_seq"] = read_seq
                        results.append(rec)
            except OSError:
                continue
        # 新→旧：主键ts降序；同ts（同秒append）→ read_seq降序（后append的在前）
        results.sort(
            key=lambda r: (str(r.get("ts", "")), r.get("_read_seq", 0)),
            reverse=True,
        )
        for r in results:
            r.pop("_read_seq", None)
        return results[:limit]

    def count(self, event_type: Optional[str] = None) -> int:
        return len(self.query(event_type=event_type, limit=10_000))

    def health(self) -> Dict[str, Any]:
        """健康检查：存储可写 + 已知type的消费覆盖。

        可写性用临时文件探测（不append探针——append-only事件流不容污染）。
        """
        ok = True
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(dir=self.store_dir, delete=True) as tf:
                tf.write(b"probe")
        except OSError:
            ok = False
        unregistered = self.registry.unregistered_types()
        return {
            "ok": ok,
            "store_dir": str(self.store_dir),
            "known_types": len(KNOWN_TYPES),
            "unregistered_types": unregistered,
            "unregistered_count": len(unregistered),
        }


# ═══ CLI ═══

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="进化事件总线 CLI")
    parser.add_argument("--append", action="store_true", help="追加一条测试事件")
    parser.add_argument("--type", default="train.completed")
    parser.add_argument("--producer", default="cli")
    parser.add_argument("--query", action="store_true", help="查询最近事件")
    parser.add_argument("--type-filter", default="", help="query的type过滤")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--register-consumer", nargs=2, metavar=("TYPE", "CONSUMER"),
                        help="注册消费者")
    args = parser.parse_args()

    bus = EvolutionBus()

    if args.register_consumer:
        bus.registry.register(args.register_consumer[0], args.register_consumer[1])
        print(f"已注册消费者: {args.register_consumer[1]} → {args.register_consumer[0]}")
        return

    if args.append:
        ev = DigestEvent(type=args.type, producer=args.producer,
                         payload={"note": "CLI manual append"})
        bus.append(ev)
        print(f"已追加: {ev.type} hash={ev.hash} ts={ev.ts}")
        return

    if args.query:
        recs = bus.query(event_type=args.type_filter or None, days=args.days)
        print(f"最近 {len(recs)} 条事件:")
        for r in recs:
            print(json.dumps(r, ensure_ascii=False))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
