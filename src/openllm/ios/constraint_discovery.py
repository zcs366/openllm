"""constraint_discovery.py — IOS 自涌现约束原型 (≤150行)

从运行时事件流中自动发现异常模式，生成约束提案。
不调用LLM——纯统计规则（滑动窗口频率 / 类型分布熵 / 来源集中度）。
设计哲学：约束管"绝对不可越界的红线"，不管执行细节。
"""
import math
import time
import uuid
import logging
from collections import Counter
from typing import Any, Optional

from openllm.iai.event_bus import Event, EventBus

logger = logging.getLogger("openllm.ios.constraint_discovery")


def _shannon_entropy(counts: Counter) -> float:
    """Shannon 熵 (bit)。"""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)


class ConstraintDiscovery:
    """IOS 自涌现约束发现器。

    滑动窗口观察事件流 → 统计异常检测 → 约束提案生成 → 人工审批。
    """

    def __init__(
        self,
        bus: Optional[EventBus] = None,
        window_size: int = 50,
        freq_threshold: float = 0.6,
        entropy_threshold: float = 0.8,
        concentration_threshold: float = 0.7,
    ):
        self._bus = bus
        self._window_size = window_size
        self._freq_threshold = freq_threshold
        self._entropy_threshold = entropy_threshold
        self._concentration_threshold = concentration_threshold
        self._events: list[Event] = []
        self._proposals: dict[str, dict] = {}
        self._sub_id: Optional[str] = None

    # ── EventBus 集成 ──
    def attach(self) -> str:
        """订阅 EventBus，自动 observe 每个事件。"""
        if self._bus is None:
            raise RuntimeError("未绑定 EventBus，无法 attach")
        self._sub_id = self._bus.subscribe(self.observe)
        return self._sub_id

    def detach(self) -> bool:
        if self._bus and self._sub_id:
            return self._bus.unsubscribe(self._sub_id)
        return False

    # ── 核心 API ──

    def observe(self, event: Event) -> None:
        """接收事件并推入滑动窗口。"""
        self._events.append(event)
        if len(self._events) > self._window_size:
            self._events = self._events[-self._window_size:]

    def detect_anomaly(self) -> list[dict]:
        """检测异常模式。返回异常列表。"""
        if len(self._events) < 5:
            return []
        anomalies: list[dict] = []
        anomalies.extend(self._check_frequency())
        anomalies.extend(self._check_type_concentration())
        anomalies.extend(self._check_source_concentration())
        return anomalies

    def propose_constraint(self, anomaly: dict) -> dict:
        """从异常生成约束提案。"""
        pid = f"cp-{uuid.uuid4().hex[:8]}"
        proposal = {
            "id": pid,
            "type": anomaly["type"],
            "description": anomaly["description"],
            "severity": anomaly.get("severity", "warning"),
            "source_event": anomaly.get("source_event", ""),
            "created_at": time.time(),
            "status": "pending",
        }
        self._proposals[pid] = proposal
        # 发布 constraint.proposed 事件
        if self._bus:
            self._bus.publish(Event(
                source="IOS.constraint_discovery",
                type="constraint.proposed",
                timestamp=time.time(),
                entropy_score=0.0,
                payload={"proposal_id": pid, "description": proposal["description"]},
            ))
        logger.info(f"[constraint_discovery] 提案生成: {pid} — {proposal['description'][:60]}")
        return proposal

    def list_proposals(self) -> list[dict]:
        """列出所有提案。"""
        return list(self._proposals.values())

    def approve(self, proposal_id: str, approved: bool) -> bool:
        """审批提案。返回是否成功。"""
        if proposal_id not in self._proposals:
            return False
        self._proposals[proposal_id]["status"] = "approved" if approved else "rejected"
        return True

    # ── 异常检测规则（纯统计，无 LLM） ──

    def _check_frequency(self) -> list[dict]:
        """频率异常：窗口内事件数接近满载，说明可能风暴。"""
        if len(self._events) >= self._window_size * self._freq_threshold:
            return [{"type": "frequency_storm",
                     "description": f"滑动窗口内事件数 {len(self._events)}/{self._window_size}，疑似事件风暴",
                     "severity": "critical", "source_event": self._events[-1].event_id}]
        return []

    def _check_type_concentration(self) -> list[dict]:
        """类型集中度：某类型占比过高 → 单调异常。"""
        type_counts = Counter(e.type for e in self._events)
        if not type_counts:
            return []
        dominant_type, dominant_count = type_counts.most_common(1)[0]
        ratio = dominant_count / len(self._events)
        if ratio >= self._concentration_threshold:
            entropy = _shannon_entropy(type_counts)
            if entropy < self._entropy_threshold:
                return [{"type": "type_concentration",
                         "description": f"类型 '{dominant_type}' 占比 {ratio:.0%}，熵 {entropy:.2f}bit，疑似单调循环",
                         "severity": "warning", "source_event": dominant_type}]
        return []

    def _check_source_concentration(self) -> list[dict]:
        """来源集中度：单一来源主导事件流。"""
        src_counts = Counter(e.source for e in self._events)
        if len(src_counts) <= 1 and len(self._events) >= self._window_size * 0.3:
            dominant = src_counts.most_common(1)[0]
            return [{"type": "source_monopoly",
                     "description": f"来源 '{dominant[0]}' 独占 {dominant[1]} 个事件，无多源交互",
                     "severity": "warning", "source_event": dominant[0]}]
        return []
