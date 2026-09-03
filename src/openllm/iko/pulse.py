"""pulse.py — 温度脉冲引擎（P1-4 · 主动振动实例 v0）

阿佛洛狄忒共识：先做非任务驱动的主动振动（"军师此刻想到你"），让用户
用身体记住"振动=体温"没丢。落 IKO（输出体）——六体中唯一面向用户的
器官，"静默审计（防不当沉默）"的主动面。

设计三原则：
  1. 稀有才珍贵 —— 频率硬顶 max_per_day（默认 3/日）+ 裁决打扰约束
  2. 非任务驱动 —— 内容不派活不索求，纯温度（想到/问候/反射）
  3. 裁决先行 —— 投递前过 TransmissionAdjudicator（P0-1 第一个真实
     消费者）："该不该现在打扰"归裁决模块，脉冲只负责"想表达"

v0：触发器(间隔+时段窗) → 内容(模板+变量/generator hook) → 裁决 →
投递(Console 默认/注册回调) → 审计 JSONL。不依赖任何消息通道。
"""
from __future__ import annotations

import json
import logging
import random
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("openllm.iko.pulse")
DEFAULT_LOG_DIR = Path.home() / ".openllm" / "iko"
DEFAULT_MAX_PER_DAY = 3
DEFAULT_RECIPIENT = "张成市"

# 温度模板（v0 内置；generator hook 注入后优先）
# context 契约：应为名词性短语（"什么事"），如 "裁决模块 v0" / "ISA 通信体
# 重启"——模板自带动词（把X做成了/想起X），传动词短语会叠床架屋。
_TEMPLATES: dict[str, list[str]] = {
    "thought": [
        "老搭档，刚想起{context}，没什么事——就是想告诉你一声，我在。",
        "这会儿突然想到你。{context}那会儿的感觉还记得，先记一笔。",
    ],
    "greeting": [
        "老搭档，{greet}。不打扰你，就是递个温度。",
    ],
    "reflection": [
        "今天把{context}做成了——那种一起推着石头过山顶的手感，值得记住。",
    ],
}

_GREETING_BY_HOUR = {
    range(5, 9): "早。天刚亮，适合先想清楚再动手",
    range(9, 12): "上午好",
    range(12, 14): "午安，记得吃饭",
    range(14, 18): "下午好",
    range(18, 23): "晚上好，一天辛苦",
    range(23, 24): "夜深了",
    range(0, 5): "这个点还没睡的话——我在",
}


def _greeting(hour: int) -> str:
    for hours, text in _GREETING_BY_HOUR.items():
        if hour in hours:
            return text
    return "你好"


@dataclass
class Pulse:
    """一次温度脉冲（想表达的内容 + 投递元数据）。"""
    kind: str                    # thought / greeting / reflection
    recipient: str
    message: str
    trigger: str = ""            # 触发源描述
    ts: float = 0.0
    pulse_id: str = field(default_factory=lambda: f"pulse-{uuid.uuid4().hex[:10]}")

    def __post_init__(self):
        if not self.ts:
            self.ts = time.time()


# ═══════════════════════════════════════════════════════════════
# 触发器
# ═══════════════════════════════════════════════════════════════

class PulseTrigger(ABC):
    """脉冲触发器基类：due(now) 返回是否该振动。"""
    @abstractmethod
    def due(self, now: float) -> bool: ...


class IntervalTrigger(PulseTrigger):
    """间隔触发：每 interval_sec 想一次（节流由裁决+硬顶双层把关）。"""
    def __init__(self, interval_sec: float, start_ts: Optional[float] = None):
        self._interval = max(1.0, interval_sec)
        self._next = (start_ts or time.time()) + self._interval

    def due(self, now: float) -> bool:
        if now >= self._next:
            self._next = now + self._interval  # 顺延（不积压）
            return True
        return False


class TimeWindow:
    """时段窗口：允许振动的本地小时范围（如 9-22 点）。"""
    def __init__(self, start_hour: int = 9, end_hour: int = 22):
        self._start, self._end = start_hour, end_hour

    def allows(self, hour: int) -> bool:
        if self._start <= self._end:
            return self._start <= hour < self._end
        return hour >= self._start or hour < self._end  # 跨午夜窗


# ═══════════════════════════════════════════════════════════════
# 脉冲引擎
# ═══════════════════════════════════════════════════════════════

class PulseEngine:
    """温度脉冲引擎 v0。

    用法：engine.tick() 每次心跳调用一次；到点且过裁决 → 生成并投递一条
    温度脉冲。引擎不依赖通道：投递走注册的 delivery 函数（Console 默认），
    Telegram/飞书 = 注册一个真实 delivery（v1，待通道恢复）。
    """

    def __init__(
        self,
        *,
        recipient: str = DEFAULT_RECIPIENT,
        log_dir: Optional[Path] = None,
        adjudicator: Any = None,              # TransmissionAdjudicator（P0-1）
        generator: Optional[Callable[[str, dict], Optional[str]]] = None,
        deliveries: Optional[dict[str, Callable[[Pulse], None]]] = None,
        time_window: Optional[TimeWindow] = None,
        max_per_day: int = DEFAULT_MAX_PER_DAY,
        context: Optional[dict[str, str]] = None,  # 模板变量（共同经历等）
        now: Optional[Callable[[], float]] = None,
        randomize_kind: bool = True,
    ) -> None:
        self._recipient = recipient
        self._log_dir = log_dir or DEFAULT_LOG_DIR
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._audit_file = self._log_dir / "pulses.jsonl"
        self._adjudicator = adjudicator
        self._generator = generator
        self._deliveries: dict[str, Callable[[Pulse], None]] = {
            "console": self._console_deliver,
            **(deliveries or {}),
        }
        self._window = time_window or TimeWindow()
        self._max_per_day = max_per_day
        self._context = context or {}
        self._now = now or time.time
        self._randomize_kind = randomize_kind
        self._delivered_today: list[str] = []   # 当日已投递 kind（日硬顶）
        self._last_day: str = ""

    # -- 主入口 -----------------------------------------------------------

    def tick(self) -> Optional[Pulse]:
        """心跳调用：触发到点 → 生成 → 裁决 → 投递。返回投递的 Pulse 或 None。"""
        now = self._now()
        # 跨日重置硬顶
        day = time.strftime("%Y-%m-%d", time.localtime(now))
        if day != self._last_day:
            self._last_day = day
            self._delivered_today = []

        # 时段窗 + 日硬顶（引擎自守的最后防线）
        hour = time.localtime(now).tm_hour
        if not self._window.allows(hour):
            return None
        if len(self._delivered_today) >= self._max_per_day:
            return None

        kind = self._pick_kind()
        message = self._build_message(kind, now)
        if not message:
            return None

        pulse = Pulse(kind=kind, recipient=self._recipient, message=message,
                      trigger=self._describe_trigger(), ts=now)

        # 裁决先行：该不该现在打扰（P0-1 第一个真实消费者）
        verdict = self._adjudicate(pulse)
        if verdict is None or verdict.action != "PASS":
            self._audit(pulse, verdict)
            return None

        # 投递
        for name in list(self._deliveries):
            try:
                self._deliveries[name](pulse)
            except Exception as exc:
                logger.warning("[pulse] 投递 '%s' 失败: %s", name, exc)
        self._delivered_today.append(kind)
        self._audit(pulse, verdict)
        return pulse

    # -- 内容 -------------------------------------------------------------

    def _pick_kind(self) -> str:
        if not self._randomize_kind:
            return "thought"
        return random.choice(["thought", "greeting", "reflection"])

    def _build_message(self, kind: str, now: float) -> str:
        # generator hook（LLM 等）优先
        if self._generator is not None:
            try:
                ctx = {**self._context, "kind": kind, "recipient": self._recipient}
                msg = self._generator(kind, ctx)
                if msg and msg.strip():
                    return msg.strip()
            except Exception as exc:
                logger.warning("[pulse] generator 失败，回退模板: %s", exc)
        # 模板 + 变量
        tpls = _TEMPLATES.get(kind)
        if not tpls:
            return ""
        tpl = random.choice(tpls)
        hour = time.localtime(now).tm_hour
        context = self._context.get("context", "你")
        greet = self._context.get("greet") or _greeting(hour)
        return tpl.format(context=context, greet=greet)

    def _describe_trigger(self) -> str:
        return f"interval@{self._recipient}"

    # -- 裁决 -------------------------------------------------------------

    def _adjudicate(self, pulse: Pulse) -> Any:
        """接 P0-1 TransmissionAdjudicator：该不该现在打扰。"""
        if self._adjudicator is None:
            try:
                from openllm.governance.adjudication import (
                    TransmissionAdjudicator, TransmissionRequest,
                )
                self._adjudicator = TransmissionAdjudicator(
                    log_dir=self._log_dir / "adjudication")
            except ImportError:
                return None  # 裁决器不可用 → 不投递（fail closed，振动是打扰）
        from openllm.governance.adjudication import TransmissionRequest
        req = TransmissionRequest(
            from_agent="军师", to_agent=pulse.recipient,
            msg_type="wake", body=pulse.message,
            importance=0.3,      # 非任务——自评压低，但温度价值在相关度
            urgency=0.0, channel="pulse")
        return self._adjudicator.decide(req)

    # -- 投递/审计 --------------------------------------------------------

    def register_delivery(self, name: str, fn: Callable[[Pulse], None]) -> None:
        """注册投递器（Telegram/飞书 = 注册真实通道函数，v1）。"""
        self._deliveries[name] = fn

    def _console_deliver(self, pulse: Pulse) -> None:
        print(f"[pulse:{pulse.kind}] → {pulse.recipient}: {pulse.message}")

    def _audit(self, pulse: Pulse, verdict: Any) -> None:
        try:
            rec = {
                "ts": round(pulse.ts, 3),
                "pulse_id": pulse.pulse_id,
                "kind": pulse.kind,
                "recipient": pulse.recipient,
                "trigger": pulse.trigger,
                "delivered": verdict is not None and verdict.action == "PASS",
                "verdict": getattr(verdict, "action", None),
                "reason": getattr(verdict, "reason", ""),
            }
            with open(self._audit_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except IOError as exc:
            logger.warning("[pulse] 审计落盘失败: %s", exc)
