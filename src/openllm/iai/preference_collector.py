"""preference_collector.py — critique→偏好对收集器。

订阅 EventBus 的 reasoning.proposed / reasoning.critiqued 事件，
按顺序配对，verdict=reject 存 rejected 样本，verdict=approve 存 chosen 候选，
append-only 写 ~/.openllm/iai/preference_pairs.jsonl。

IAI 可选挂载：collector = PreferenceCollector(bus)，需显式 start()。
"""
import json
import logging
import time
from pathlib import Path
from typing import Optional

from openllm.iai.event_bus import EventBus, Event

logger = logging.getLogger("openllm.iai.preference_collector")

PREFERENCE_FILE = Path.home() / ".openllm" / "iai" / "preference_pairs.jsonl"


class PreferenceCollector:
    """critique→偏好对收集器。

    订阅 reasoning.proposed + reasoning.critiqued，
    按到达顺序配对，写 JSONL，提供统计。
    """

    def __init__(self, bus: EventBus, path: Optional[Path] = None):
        self._bus = bus
        self._path = path or PREFERENCE_FILE
        self._sid_proposed: Optional[str] = None
        self._sid_critiqued: Optional[str] = None
        self._started = False

        # 配对缓冲：最近一次 proposed
        self._pending_proposal: Optional[dict] = None

        # 统计
        self._stats = {"total": 0, "rejected": 0, "approved": 0}

    def start(self) -> None:
        """启动订阅（显式调用）。重复调用安全。"""
        if self._started:
            return
        self._sid_proposed = self._bus.subscribe(
            self._on_proposed, type_filter="reasoning.proposed")
        self._sid_critiqued = self._bus.subscribe(
            self._on_critiqued, type_filter="reasoning.critiqued")
        self._started = True
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def stop(self) -> None:
        """停止订阅。"""
        if not self._started:
            return
        if self._sid_proposed:
            self._bus.unsubscribe(self._sid_proposed)
            self._sid_proposed = None
        if self._sid_critiqued:
            self._bus.unsubscribe(self._sid_critiqued)
            self._sid_critiqued = None
        self._started = False

    def _on_proposed(self, event: Event) -> None:
        """reasoning.proposed 到达 → 缓存 proposal。"""
        try:
            payload = event.payload or {}
            self._pending_proposal = {
                "content": payload.get("content", ""),
                "confidence": payload.get("confidence", 0.0),
                "timestamp": event.timestamp,
            }
        except Exception as e:
            logger.warning("[pref_collector] proposed 处理失败: %s", e)

    def _on_critiqued(self, event: Event) -> None:
        """reasoning.critiqued 到达 → 与缓存 proposal 配对，写 JSONL。"""
        try:
            critique_payload = event.payload or {}
            verdict = critique_payload.get("verdict", "")
            proposal = self._pending_proposal

            if not proposal:
                # 无配对 proposal，跳过
                return

            # 配对消费（单次）
            self._pending_proposal = None

            # user_message 暂留空（context.built 事件由另一订阅者消费）
            record = {
                "type": "rejected" if verdict == "reject" else "chosen",
                "proposal_content": proposal.get("content", ""),
                "proposal_confidence": proposal.get("confidence", 0.0),
                "critique_verdict": verdict,
                "concerns_count": critique_payload.get("concerns", 0),
                "user_message": "",
                "proposal_timestamp": proposal.get("timestamp", 0.0),
                "critique_timestamp": event.timestamp,
                "collector_timestamp": time.time(),
                "source_ref": f"tick:{event.event_id}",
            }

            self._append_jsonl(record)
            self._stats["total"] += 1
            if verdict == "reject":
                self._stats["rejected"] += 1
            elif verdict == "approve":
                self._stats["approved"] += 1
        except Exception as e:
            logger.warning("[pref_collector] critiqued 处理失败: %s", e)

    def _append_jsonl(self, record: dict) -> None:
        """append-only 写 JSONL（失败不抛异常）。"""
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("[pref_collector] JSONL 写入失败: %s", e)

    def preference_stats(self) -> dict:
        """返回统计快照。"""
        return {
            "total": self._stats["total"],
            "rejected": self._stats["rejected"],
            "approved": self._stats["approved"],
            "file": str(self._path),
            "file_exists": self._path.exists(),
        }
