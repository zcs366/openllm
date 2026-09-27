"""
CLI 实时活动流观察器 — 旁路轮询 turn.phase_metrics，不碰 stdout 封印。

设计约束：
- 纯 stdlib，不 import core（可单测）
- StringIO 封印语义不变，活动流从 phase_metrics 结构化数据旁路渲染
- daemon 线程轮询，run_once 同步执行 + GIL，append 线程安全
"""
import threading
import time
from typing import Any, Callable, Optional

# ── 阶段标签：phase → (体, 中文标签) ──
PHASE_LABELS: dict[str, tuple[str, str]] = {
    "context":           ("ISA",   "建境"),
    "awakening_inject":  ("IAI",   "苏醒注入"),
    "causal_inject":     ("ISA",   "因果疤"),
    "clock_read":        ("IAX",   "读钟"),
    "search":            ("章鱼I", "索引搜索"),
    "replay":            ("ISA",   "证据回放"),
    "predict":           ("IAI",   "因果预测"),
    "drift":             ("IAX",   "漂移检测"),
    "risk":              ("IOS",   "风险门"),
    "reason":            ("章鱼I", "左右脑对弈"),
    "decide":            ("IOS",   "仲裁"),
    "execute":           ("ISN",   "执行"),
    "validate":          ("IOS",   "结果验证"),
    "validate_retry":    ("IOS",   "验证重试"),
    "guard":             ("IOS",   "护栏"),
    "guard_rate_limit":  ("IOS",   "护栏限速"),
    "guard_forbidden":   ("IOS",   "护栏禁区"),
    "guard_recorded":    ("IOS",   "护栏留痕"),
    "learn":             ("ISA",   "学习"),
    "evolve":            ("IOS",   "进化"),
    "output":            ("IKO",   "输出"),
    "feedback":          ("IAX",   "反馈"),
    "checkpoint":        ("IAX",   "存档"),
    # P0.5(20260925) 心跳综合回路上线后补标签（未登记阶段本可回退原名，此为归位）
    "synthesize":        ("IAX",   "综合消化"),
}

# ── 状态标记 ──
STATUS_MARK: dict[str, str] = {
    "ok":     "✓",
    "skip":   "·",
    "warn":   "⚠",
    "error":  "✗",
    "denied": "✗",
}


def format_phase(entry: dict) -> str:
    """把一条 phase_metrics entry 格式化为一行活动流。

    格式: ``  {体} {标签} {mark} {dur}  {detail}``
    - duration_ms > 0 才显示耗时（>999ms 换算为秒如 "1.2s"）
    - detail 去换行截 40 字符
    - 纯文本无色码
    - 缺键用 .get 不抛
    """
    phase = entry.get("phase", "?")
    body, label = PHASE_LABELS.get(phase, ("?", phase))
    status = entry.get("status", "?")
    mark = STATUS_MARK.get(status, status)
    dur_ms = entry.get("duration_ms", 0) or 0
    detail = (entry.get("detail") or "").replace("\n", " ")[:40]

    parts = [f"  {body} {label} {mark}"]
    if dur_ms > 0:
        if dur_ms > 999:
            parts.append(f"{dur_ms / 1000:.1f}s")
        else:
            parts.append(f"{dur_ms:.0f}ms")
    if detail:
        parts.append(detail)
    return " ".join(parts)


class PhaseWatcher:
    """daemon 线程轮询 turn.phase_metrics，新条目逐条回调 on_entry。

    Parameters
    ----------
    get_turn : callable
        返回当前 Turn 对象或 None。通常为
        ``lambda: getattr(self.agent.session, "active_turn", None)``。
    on_entry : callable
        每条新 phase_metrics entry 的回调 ``on_entry(entry_dict)``。
    interval : float
        轮询间隔秒数，默认 0.5。
    """

    def __init__(
        self,
        get_turn: Callable[[], Any],
        on_entry: Callable[[dict], None],
        interval: float = 0.5,
    ):
        self._get_turn = get_turn
        self._on_entry = on_entry
        self._interval = interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # 追踪状态
        self._turn_id: Optional[str] = None
        self._seen_count: int = 0
        # 最后一次非 None turn 的引用，用于 stop 前追平残余
        self._last_turn_ref: Any = None
        # 公开账本：本 watcher 见过的最后一个 turn（stop 后不清空）。
        # 2026-09-25 P0 验收补刀：main.py 轮末 turn_account 依赖它——
        # 若只在 on_entry 回调瞬间抓 active_turn，残余条目在 stop 时 drain
        # 补印、那时 active_turn 已为 None，小账会永远丢（竞态非时序运气）。
        self.last_turn: Any = None

    def start(self) -> None:
        """启动 daemon 轮询线程。"""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="PhaseWatcher"
        )
        self._thread.start()

    def stop(self) -> None:
        """停止轮询，追平残余条目，join 等待线程结束。"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    # ── 内部 ──

    def _poll_loop(self) -> None:
        """轮询主循环，异常时线程静默退出。"""
        try:
            while not self._stop_event.is_set():
                self._poll_once()
                self._stop_event.wait(timeout=self._interval)
            # stop 后再追平一次残余
            self._drain_remaining()
        except Exception:
            pass  # 线程静默退出

    def _poll_once(self) -> None:
        """单次轮询：检测 turn 变化 + 新增条目。"""
        turn = self._get_turn()
        if turn is None:
            # turn 为 None 时，对最后引用的 turn 追平残余
            if self._last_turn_ref is not None:
                self._drain_from(self._last_turn_ref)
                self._last_turn_ref = None
                self._turn_id = None
                self._seen_count = 0
            return

        self.last_turn = turn
        turn_id = getattr(turn, "id", None) or id(turn)
        metrics = getattr(turn, "phase_metrics", [])

        if turn_id != self._turn_id:
            # New turn started — reset count to capture all entries from now
            self._turn_id = turn_id
            self._seen_count = 0  # Don't skip pre-existing entries
            self._last_turn_ref = turn
        elif len(metrics) > self._seen_count:
            # 同一 turn 内新条目
            new_entries = metrics[self._seen_count:]
            self._seen_count = len(metrics)
            for entry in new_entries:
                self._on_entry(entry)

        self._last_turn_ref = turn

    def _drain_remaining(self) -> None:
        """stop 前对最后引用的 turn 追平残余（去重）。"""
        turn = self._last_turn_ref
        if turn is not None:
            self._drain_from(turn)
            self._last_turn_ref = None

    def _drain_from(self, turn: Any) -> None:
        """从指定 turn 追平_seen_count之后的残余条目（去重）。"""
        metrics = getattr(turn, "phase_metrics", [])
        if len(metrics) > self._seen_count:
            new_entries = metrics[self._seen_count:]
            self._seen_count = len(metrics)
            for entry in new_entries:
                self._on_entry(entry)


def turn_account(turn: Any) -> str:
    """轮末一行小账。例: ``心跳 9 阶段 · 风险low · 23.4s``。

    turn None 返回 ""。
    """
    if turn is None:
        return ""
    metrics = getattr(turn, "phase_metrics", [])
    n = len(metrics)
    risk = getattr(turn, "risk_level", "low")
    if hasattr(turn, "duration_ms"):
        dur = turn.duration_ms / 1000
        return f"心跳 {n} 阶段 · 风险{risk} · {dur:.1f}s"
    return f"心跳 {n} 阶段 · 风险{risk}"
