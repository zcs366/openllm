"""
knowledge_monotonic.py — K度量：知识单调性探针（三元矛盾B方案v2第二块）
======================================================================

对标MemBench(2506.21605)四指标：capacity_delta / recall / accuracy / verdict。

四容器载体：
  1. capsule — TextCapsule(v06) + DeltaCapsule(v07) + checkpoint 文件
  2. jiak    — ~/.hermes/jiak/cards/*.json 卡片
  3. growth  — ~/.openllm/identity/growth.jsonl 成长记录
  4. ledger  — ~/.openllm/consolidation_state.json 固化账本

设计原则（与 rule_change_probe.py 同族）：
  - append-only JSONL 快照（每行一条，永不修改/删除）
  - 零LLM调用，纯确定性IO
  - 中文docstring

用法：
    probe = KnowledgeMonotonicProbe()
    probe.take_snapshot()
    report = probe.check_monotonic()
    probe.emit_report()

上下文格言：学而不辍，记而不失。
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── 四容器路径（与各自模块一致）───────────────────────────

# capsule.py: CAPSULE_DIR = Path(__file__).parent.parent.parent.parent / "caps"
_CAPSULE_DIR = Path(__file__).parent.parent.parent.parent / "caps"

# jiak_lifecycle.py: CARDS_DIR = Path.home() / ".hermes" / "jiak" / "cards"
_JIAK_CARDS_DIR = Path.home() / ".hermes" / "jiak" / "cards"

# soul_growth.py 默认路径
_GROWTH_PATH = Path.home() / ".openllm" / "identity" / "growth.jsonl"

# consolidation_score.py 默认路径
_CONSOLIDATION_STATE = Path.home() / ".openllm" / "consolidation_state.json"


@dataclass
class KnowledgeReport:
    """K度量探针报告。"""
    capacity_delta: Dict[str, int]   # 各容器计数增量（正=增长，负=丢失）
    recall: float                     # 旧内容保有率 0.0~1.0（1.0=全部保有）
    accuracy_sample: float            # 新增条目格式完整性抽样 0.0~1.0
    verdict: str                      # monotonic/regression/stagnant/insufficient_data
    detail: str                       # 人类可读解释


def _md5(text: str) -> str:
    """确定性内容指纹（MD5，用于快照间recall比对）。"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _count_capsule_files() -> tuple[int, list[str]]:
    """统计capsule容器：v06+v07+checkpoint文件总数，返回(计数, [内容哈希列表])。"""
    if not _CAPSULE_DIR.exists():
        return 0, []
    hashes = []
    for p in _CAPSULE_DIR.glob("*.json"):
        # 读取文件内容计算哈希（recall用）
        try:
            content = p.read_text(encoding="utf-8")
            hashes.append(_md5(content))
        except OSError:
            pass
    return len(hashes), hashes


def _count_jiak_cards() -> tuple[int, list[str]]:
    """统计jiak卡片容器。"""
    if not _JIAK_CARDS_DIR.exists():
        return 0, []
    hashes = []
    for p in _JIAK_CARDS_DIR.glob("*.json"):
        try:
            content = p.read_text(encoding="utf-8")
            hashes.append(_md5(content))
        except OSError:
            pass
    return len(hashes), hashes


def _count_growth_records() -> tuple[int, list[str]]:
    """统计soul_growth容器（JSONL行数）。"""
    if not _GROWTH_PATH.exists():
        return 0, []
    hashes = []
    try:
        lines = _GROWTH_PATH.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if line.strip():
                hashes.append(_md5(line.strip()))
    except OSError:
        pass
    return len(hashes), hashes


def _count_ledger_entries() -> tuple[int, list[str]]:
    """统计consolidation账本中的固化条目数。"""
    if not _CONSOLIDATION_STATE.exists():
        return 0, []
    try:
        data = json.loads(_CONSOLIDATION_STATE.read_text(encoding="utf-8"))
        ledger = data.get("ledger", [])
        hashes = [_md5(json.dumps(e, ensure_ascii=False, sort_keys=True, default=str))
                  for e in ledger]
        return len(ledger), hashes
    except (OSError, json.JSONDecodeError):
        return 0, []


# 容器名 → 采集函数
_CONTAINERS = {
    "capsule": _count_capsule_files,
    "jiak":    _count_jiak_cards,
    "growth":  _count_growth_records,
    "ledger":  _count_ledger_entries,
}


class KnowledgeMonotonicProbe:
    """K度量探针：四容器快照diff + recall计算 + 灾难性遗忘检测。

    每次 take_snapshot() 在 snapshot_dir 下追加一行JSONL快照。
    check_monotonic() 比对最近两份快照，输出 KnowledgeReport。
    """

    def __init__(self, snapshot_dir: Optional[Path] = None) -> None:
        """
        Args:
            snapshot_dir: 快照存储目录，默认 ~/.openllm/knowledge_snapshots/
        """
        if snapshot_dir is None:
            snapshot_dir = Path.home() / ".openllm" / "knowledge_snapshots"
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._snapshot_file = self.snapshot_dir / "snapshots.jsonl"

    # ── 快照采集 ──────────────────────────────────────────

    def take_snapshot(self) -> dict:
        """对当前四容器做快照：计数 + 内容哈希列表。

        Returns:
            快照字典（也追加到 snapshots.jsonl）
        """
        ts = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        snapshot: Dict[str, Any] = {"ts": ts}

        counts = {}
        all_hashes: Dict[str, List[str]] = {}
        for name, fn in _CONTAINERS.items():
            count, hashes = fn()
            counts[name] = count
            all_hashes[name] = hashes

        snapshot["counts"] = counts
        snapshot["content_hashes"] = all_hashes  # recall比对用

        # 内容摘要digest（全库哈希拼接的截断）
        combined = "|".join(
            h for hs in all_hashes.values() for h in hs
        )
        snapshot["content_digest"] = _md5(combined) if combined else "empty"

        # append-only追加
        with open(self._snapshot_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")

        return snapshot

    # ── 快照读取 ──────────────────────────────────────────

    def _load_snapshots(self) -> list[dict]:
        """读取所有快照（按时间升序）。"""
        if not self._snapshot_file.exists():
            return []
        snapshots = []
        for line in self._snapshot_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    snapshots.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return snapshots

    # ── 单调性检查 ────────────────────────────────────────

    def check_monotonic(self) -> KnowledgeReport:
        """比对最近两份快照，输出K度量报告。

        Returns:
            KnowledgeReport
        """
        snapshots = self._load_snapshots()
        if len(snapshots) < 2:
            return KnowledgeReport(
                capacity_delta={},
                recall=1.0,
                accuracy_sample=1.0,
                verdict="insufficient_data",
                detail=f"快照数不足：{len(snapshots)}份（需≥2份）",
            )

        old = snapshots[-2]
        new = snapshots[-1]

        # 1. 容量增量
        capacity_delta: Dict[str, int] = {}
        for name in _CONTAINERS:
            old_count = old.get("counts", {}).get(name, 0)
            new_count = new.get("counts", {}).get(name, 0)
            capacity_delta[name] = new_count - old_count

        total_delta = sum(capacity_delta.values())

        # 2. Recall：旧内容哈希在新快照中的保有率
        old_hashes_all = set()
        for hs in old.get("content_hashes", {}).values():
            old_hashes_all.update(hs)
        new_hashes_all = set()
        for hs in new.get("content_hashes", {}).values():
            new_hashes_all.update(hs)

        if old_hashes_all:
            recall = len(old_hashes_all & new_hashes_all) / len(old_hashes_all)
        else:
            recall = 1.0  # 旧快照为空，视作全保有

        # 3. Accuracy：新增条目格式完整性抽样
        #    抽样策略：检查新快照的counts值均为非负整数（基础完整性）
        new_counts = new.get("counts", {})
        valid = sum(1 for v in new_counts.values()
                    if isinstance(v, (int, float)) and v >= 0)
        accuracy_sample = valid / max(len(new_counts), 1)

        # 4. 裁决
        if recall < 1.0:
            verdict = "regression"
        elif total_delta > 0:
            verdict = "monotonic"
        elif total_delta == 0:
            verdict = "stagnant"
        else:
            verdict = "regression"  # 容量负增长

        # 5. 细节
        if verdict == "regression":
            lost = len(old_hashes_all - new_hashes_all)
            detail = (
                f"灾难性遗忘检测：丢失{lost}条旧内容 "
                f"（recall={recall:.3f}）；容量变化={capacity_delta}"
            )
        elif verdict == "monotonic":
            detail = f"知识单调增长：新增{total_delta}条；容量变化={capacity_delta}"
        elif verdict == "stagnant":
            detail = f"知识停滞：零增长；各容器计数={new_counts}"
        else:
            detail = f"数据不足：{verdict}"

        return KnowledgeReport(
            capacity_delta=capacity_delta,
            recall=round(recall, 4),
            accuracy_sample=round(accuracy_sample, 4),
            verdict=verdict,
            detail=detail,
        )

    # ── 总线发射 ──────────────────────────────────────────

    def emit_report(self) -> Optional[dict]:
        """将K度量报告写回evolution bus（type='knowledge.probed'）。

        Returns:
            发射的事件字典，或None（bus不可用时）
        """
        try:
            from openllm.evolution.bus import EvolutionBus, DigestEvent
        except ImportError:
            return None

        report = self.check_monotonic()
        bus = EvolutionBus()
        event = DigestEvent(
            type="knowledge.probed",
            producer="knowledge_monotonic_probe",
            payload=asdict(report),
        )
        bus.append(event)
        return {"type": event.type, "ts": event.ts, "hash": event.hash}
