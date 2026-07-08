"""IAI慢路径LLM分析模块 — 心跳永等LLM，LLM只做异步深度推演。"""
import json
import logging
import time
import threading
from pathlib import Path
from typing import Any, Callable

from openllm.iai.event_bus import Event, EventBus

logger = logging.getLogger("openllm.iai.slow_path")
HIGH_ENTROPY_THRESHOLD = 0.6
RULES_PATH = Path.home() / ".openllm" / "iai_rules.json"
EVENT_ROUTING_SUGGESTION = "routing.suggestion"


class SlowAnalyzer:
    """慢路径LLM分析器。接收Event → llm_fn分析 → 写规则 → 发布建议。"""

    def __init__(
        self,
        llm_fn: Callable[[dict], dict],
        rules_path: Path | str | None = None,
        threshold: float = HIGH_ENTROPY_THRESHOLD,
    ):
        self._llm_fn = llm_fn
        self._rules_path = Path(rules_path) if rules_path else RULES_PATH
        self._threshold = threshold
        self._rules: list[dict] = self._load_rules()
        self._running = False
        self._lock = threading.Lock()

    def analyze(self, event: Event) -> dict:
        """调用注入的llm_fn分析事件，返回分析结果。"""
        return self._llm_fn({
            "event_id": event.event_id,
            "source": event.source,
            "type": event.type,
            "entropy_score": event.entropy_score,
            "payload": event.payload,
            "existing_rules": len(self._rules),
        })

    def update_rules(self, analysis: dict) -> list[str]:
        """将分析结果转化为新规则，写回规则库。返回新增规则描述列表。"""
        new_rules = analysis.get("new_rules", [])
        added: list[str] = []
        with self._lock:
            for rule in new_rules:
                if isinstance(rule, dict) and rule.get("condition"):
                    self._rules.append(rule)
                    added.append(rule.get("description", "unnamed"))
            self._save_rules()
        return added

    def run_loop(self, bus: EventBus, interval: float = 5.0,
                 stop_event: threading.Event | None = None) -> None:
        """轮询EventBus高熵事件，分析后发布routing.suggestion。阻塞直到stop_event。"""
        pending: list[Event] = []
        lock = threading.Lock()

        def _on_high_entropy(event: Event) -> None:
            if event.entropy_score >= self._threshold:
                with lock:
                    pending.append(event)

        sid = bus.subscribe(_on_high_entropy)
        self._running = True
        stopper = stop_event or threading.Event()
        try:
            while self._running and not stopper.is_set():
                batch: list[Event] = []
                with lock:
                    batch, pending = pending[:], []
                for event in batch:
                    analysis = self.analyze(event)
                    added = self.update_rules(analysis)
                    bus.publish(Event(
                        source="IAI_SLOW",
                        type=EVENT_ROUTING_SUGGESTION,
                        timestamp=time.time(),
                        entropy_score=0.0,
                        payload={"event_id": event.event_id, "rules_added": added,
                                 "suggestion": analysis.get("suggestion", "")},
                    ))
                    logger.info("[SlowPath] %s → %d rules, suggestion=%s",
                                event.event_id, len(added), analysis.get("suggestion", "")[:60])
                time.sleep(interval)
        finally:
            self._running = False
            bus.unsubscribe(sid)

    def stop_loop(self) -> None:
        self._running = False

    def get_rules(self) -> list[dict]:
        with self._lock:
            return list(self._rules)

    def _load_rules(self) -> list[dict]:
        if self._rules_path.exists():
            try:
                return json.loads(self._rules_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning("[SlowPath] 规则文件损坏，重置为空")
        return []

    def _save_rules(self) -> None:
        self._rules_path.parent.mkdir(parents=True, exist_ok=True)
        self._rules_path.write_text(
            json.dumps(self._rules, ensure_ascii=False, indent=2), encoding="utf-8"
        )
