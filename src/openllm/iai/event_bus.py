"""event_bus.py — IAI 事件总线 (≤200行)

内存 pub/sub + JSONL 持久化，让五体状态变化被连续感知。
事件 Schema: {source, type, timestamp, entropy_score, payload}
赫尔墨斯红线：任何体不得有终止对话的权限。
"""
import json, logging, time, threading, uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("openllm.iai.event_bus")
EVENT_LOG_DIR = Path.home() / ".openllm" / "events"
EVENT_TTL_HOURS = 24

@dataclass
class Event:
    """事件实体。红线：禁止 terminate/shutdown 类型。"""
    source: str; type: str; timestamp: float; entropy_score: float
    payload: dict[str, Any]
    event_id: str = field(default_factory=lambda: f"evt-{uuid.uuid4().hex[:12]}")
    _BLOCKED = frozenset({"terminate", "shutdown", "kill", "abort"})
    def __post_init__(self):
        if self.type.lower() in self._BLOCKED:
            raise PermissionError(f"赫尔墨斯红线：'{self.type}' 被禁止——任何体不得终止对话。")
    def to_dict(self) -> dict:
        return {"event_id": self.event_id, "source": self.source, "type": self.type,
                "timestamp": self.timestamp, "entropy_score": self.entropy_score,
                "payload": self.payload}
    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        return cls(source=d["source"], type=d["type"], timestamp=d["timestamp"],
                   entropy_score=d.get("entropy_score", 0.0), payload=d.get("payload", {}),
                   event_id=d.get("event_id", f"evt-{uuid.uuid4().hex[:12]}"))

@dataclass
class Subscriber:
    """订阅者，支持 source/type 双维过滤。"""
    subscriber_id: str; callback: Callable[[Event], None]
    source_filter: Optional[str] = None; type_filter: Optional[str] = None

class EventBus:
    """发布/订阅事件总线。内存分发 + JSONL 持久化。"""
    def __init__(self, log_dir: Optional[Path] = None, ttl_hours: int = EVENT_TTL_HOURS):
        self._subs: dict[str, Subscriber] = {}
        self._history: list[dict] = []
        self._lock = threading.Lock()
        self._log_dir = log_dir or EVENT_LOG_DIR
        self._ttl_sec = ttl_hours * 3600
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_dir / "events.jsonl"

    def subscribe(self, cb: Callable[[Event], None],
                  source_filter: Optional[str] = None,
                  type_filter: Optional[str] = None) -> str:
        sid = f"sub-{uuid.uuid4().hex[:8]}"
        with self._lock:
            self._subs[sid] = Subscriber(sid, cb, source_filter, type_filter)
        return sid

    def unsubscribe(self, sid: str) -> bool:
        with self._lock:
            return self._subs.pop(sid, None) is not None

    def publish(self, event: Event) -> int:
        """发布事件，分发给匹配订阅者并持久化。返回通知数量。"""
        notified = 0
        with self._lock:
            for sub in self._subs.values():
                if self._match(sub, event):
                    try:
                        sub.callback(event); notified += 1
                    except Exception as e:
                        logger.warning(f"[event_bus] 回调异常: {e}")
            self._history.append(event.to_dict())
        self._append_jsonl(event)
        return notified

    def _match(self, sub: Subscriber, event: Event) -> bool:
        if sub.source_filter and sub.source_filter != event.source: return False
        if sub.type_filter and sub.type_filter != event.type: return False
        return True

    def get_history(self, source_filter: Optional[str] = None,
                    type_filter: Optional[str] = None, limit: int = 100) -> list[dict]:
        with self._lock:
            r = self._history
            if source_filter: r = [e for e in r if e["source"] == source_filter]
            if type_filter: r = [e for e in r if e["type"] == type_filter]
            return r[-limit:]

    def subscriber_count(self) -> int:
        with self._lock: return len(self._subs)

    def _append_jsonl(self, event: Event):
        try:
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        except IOError as e:
            logger.warning(f"[event_bus] JSONL 写入失败: {e}")

    def cleanup_expired(self) -> int:
        """清理过期事件(TTL)。返回清理数量。"""
        cutoff = time.time() - self._ttl_sec
        cleaned = 0
        with self._lock:
            before = len(self._history)
            self._history = [e for e in self._history if e["timestamp"] > cutoff]
            cleaned += before - len(self._history)
        if self._log_file.exists():
            try:
                lines = self._log_file.read_text(encoding="utf-8").splitlines()
                kept = [l for l in lines if l.strip() and
                        json.loads(l).get("timestamp", 0) > cutoff]
                cleaned += len(lines) - len(kept)
                if cleaned:
                    self._log_file.write_text(
                        "\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
            except (IOError, json.JSONDecodeError) as e:
                logger.warning(f"[event_bus] JSONL 清理失败: {e}")
        return cleaned

    def load_from_jsonl(self) -> int:
        if not self._log_file.exists(): return 0
        cutoff = time.time() - self._ttl_sec
        loaded = 0
        with self._lock:
            for line in self._log_file.read_text(encoding="utf-8").splitlines():
                if not line.strip(): continue
                try:
                    d = json.loads(line)
                    if d.get("timestamp", 0) > cutoff:
                        self._history.append(d); loaded += 1
                except (json.JSONDecodeError, KeyError): continue
        return loaded

class BaseEventEmitter:
    """五体事件发射器基类。每个体继承并实现 emit_event()。"""
    BODY_NAME: str = "UNKNOWN"
    def __init__(self, bus: EventBus): self._bus = bus
    def emit_event(self, event_type: str, payload: Optional[dict] = None,
                   entropy_score: float = 0.0) -> Event:
        event = Event(source=self.BODY_NAME, type=event_type,
                      timestamp=time.time(), entropy_score=entropy_score,
                      payload=payload or {})
        self._bus.publish(event)
        return event

class SignalBridge:
    """将 core/signal.py 的文件系统信号桥接到 EventBus。"""
    def __init__(self, bus: EventBus, poll_interval: float = 5.0):
        self._bus = bus; self._interval = poll_interval
        self._seen: set[str] = set(); self._running = False
    def _poll(self):
        try:
            from openllm.core.signal import signal_recv
            for sig in signal_recv(caller_pid="event_bus_bridge", limit=50):
                sid = sig.get("signal_id", "")
                if sid in self._seen: continue
                self._seen.add(sid)
                self._bus.publish(Event(
                    source="SIGNAL", type=sig.get("type", "signal.unknown"),
                    timestamp=time.time(), entropy_score=0.0,
                    payload={"signal_id": sid, "from": sig.get("from"),
                             "to": sig.get("to"), "body": sig.get("payload")}))
        except ImportError: pass
        except Exception as e:
            logger.warning(f"[event_bus] Signal 桥接异常: {e}")
    def start(self):
        self._running = True
        def loop():
            while self._running: self._poll(); time.sleep(self._interval)
        threading.Thread(target=loop, daemon=True, name="signal-bridge").start()
    def stop(self): self._running = False
