"""
heartbeat.py — IAX纯状态机心跳。不调用LLM，不终止对话。
"""
import time
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional
logger = logging.getLogger("openllm.heartbeat")

class HeartbeatState(Enum):
    IDLE = auto(); CHECKING = auto(); DISPATCHING = auto()
    MONITORING = auto(); STOPPED = auto()

@dataclass(frozen=True)
class HeartbeatEvent:
    event_type: str
    source: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

# 合法状态转换表
_TRANSITIONS = {
    HeartbeatState.IDLE: {HeartbeatState.CHECKING, HeartbeatState.STOPPED},
    HeartbeatState.CHECKING: {HeartbeatState.DISPATCHING},
    HeartbeatState.DISPATCHING: {HeartbeatState.MONITORING},
    HeartbeatState.MONITORING: {HeartbeatState.IDLE},
    HeartbeatState.STOPPED: set(),
}

class Heartbeat:
    """纯状态机心跳：状态检查、事件分发、超时监控。
    红线：无终止对话权限。"""
    def __init__(self, interval_ms: int = 100, timeout_ms: int = 5000):
        if interval_ms <= 0:
            raise ValueError("interval_ms must be positive")
        self._interval = interval_ms / 1000.0
        self._timeout = timeout_ms / 1000.0
        self._state = HeartbeatState.IDLE
        self._tick_count = 0
        self._start_time: Optional[float] = None
        self._last_tick_time: Optional[float] = None
        self._event_queue: List[HeartbeatEvent] = []
        self._handlers: Dict[str, Callable[[HeartbeatEvent], None]] = {}
        self._body_watchdog: Dict[str, float] = {}
        self._timeouts_detected: List[str] = []

    @property
    def state(self) -> HeartbeatState: return self._state
    @property
    def tick_count(self) -> int: return self._tick_count
    @property
    def uptime(self) -> Optional[float]:
        return (time.time() - self._start_time) if self._start_time else None
    @property
    def is_running(self) -> bool: return self._state != HeartbeatState.STOPPED

    def _transition(self, target: HeartbeatState) -> None:
        if target not in _TRANSITIONS.get(self._state, set()):
            raise ValueError(f"Invalid transition: {self._state.name} → {target.name}")
        self._state = target

    def start(self) -> None:
        if self._state is HeartbeatState.STOPPED:
            raise RuntimeError("Heartbeat stopped; create a new instance.")
        if self._start_time is None:
            self._start_time = time.time()
            logger.info("[heartbeat] 启动 interval=%.0fms timeout=%.0fms",
                        self._interval * 1000, self._timeout * 1000)

    def stop(self) -> None:
        """停止心跳。安全操作，不终止任何对话。"""
        if self._state is not HeartbeatState.STOPPED:
            self._state = HeartbeatState.STOPPED
            logger.info("[heartbeat] 停止 total_ticks=%d", self._tick_count)

    def register_handler(self, event_type: str, handler: Callable[[HeartbeatEvent], None]) -> None:
        self._handlers[event_type] = handler

    def enqueue(self, event: HeartbeatEvent) -> None:
        """IAI慢路径分析结果入队。"""
        self._event_queue.append(event)

    def register_body(self, body_name: str) -> None:
        self._body_watchdog[body_name] = time.time()

    def heartbeat_from(self, body_name: str) -> None:
        """来自体的心跳，刷新看门狗。"""
        if body_name in self._body_watchdog:
            self._body_watchdog[body_name] = time.time()

    def tick(self) -> Dict[str, Any]:
        """单次心跳tick。返回 {state, tick, events_dispatched, timeouts, elapsed_ms}"""
        if self._state is HeartbeatState.STOPPED:
            return {"state": "STOPPED", "tick": self._tick_count,
                    "events_dispatched": 0, "timeouts": [], "elapsed_ms": 0}
        t0 = time.time()
        # IDLE → CHECKING
        self._transition(HeartbeatState.CHECKING)
        self._tick_count += 1
        timeouts = self._check_timeouts()
        # CHECKING → DISPATCHING
        self._transition(HeartbeatState.DISPATCHING)
        dispatched = self._dispatch_events()
        # DISPATCHING → MONITORING → IDLE
        self._transition(HeartbeatState.MONITORING)
        self._last_tick_time = time.time()
        self._transition(HeartbeatState.IDLE)
        elapsed_ms = (time.time() - t0) * 1000
        return {"state": self._state.name, "tick": self._tick_count,
                "events_dispatched": dispatched, "timeouts": timeouts,
                "elapsed_ms": round(elapsed_ms, 2)}

    def _check_timeouts(self) -> List[str]:
        """检查超时的体。仅上报，不终止。"""
        now = time.time()
        self._timeouts_detected.clear()
        for body, last_seen in self._body_watchdog.items():
            if now - last_seen > self._timeout:
                self._timeouts_detected.append(body)
                logger.warning("[heartbeat] 超时 body=%s elapsed=%.1fs", body, now - last_seen)
        return list(self._timeouts_detected)

    def _dispatch_events(self) -> int:
        """分发队列中的事件。"""
        dispatched = 0
        while self._event_queue:
            event = self._event_queue.pop(0)
            handler = self._handlers.get(event.event_type)
            if handler:
                try:
                    handler(event)
                    dispatched += 1
                except Exception as exc:
                    logger.error("[heartbeat] handler错误 type=%s err=%s", event.event_type, exc)
            else:
                logger.debug("[heartbeat] 无处理器 type=%s", event.event_type)
        return dispatched
