"""CLI 活性指示器（spinner）— DR-20260927-04。

病灶：模型思考/工具执行期间 CLI 完全静默（黑咕咚盲等），
老搭档体感"无声无息、不友好"。

修法：daemon 线程在当前行刷 braille 转轮 + 已耗时秒数。
- 写入指定流（fast path 封印期传 old_out，旁路 StringIO）
- pause()/resume()：与活动流/进度行共屏——先擦行→别人打印→再续刷
- stop()：擦行收尾，行还给正文渲染
- 纯 stdlib；终端无 ANSI（非 tty）时自动哑火，零副作用
"""
from __future__ import annotations

import sys
import threading
import time
from typing import Optional, TextIO

__all__ = ["Spinner", "ansi_ok"]

_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_CLEAR = "\r\033[K"  # 回行首 + 清到行尾


def ansi_ok(stream: Optional[TextIO] = None) -> bool:
    """当前流是否支持 ANSI 覆写（非 tty 的管道/测试不刷）。"""
    s = stream if stream is not None else sys.stdout
    try:
        return bool(s.isatty())
    except Exception:
        return False


class Spinner:
    """行内转轮： ``⠹ 思考中 · 12s``。

    用法：
        sp = Spinner("思考中", stream=old_out).start()
        try:
            ... 干活 ...
        finally:
            sp.stop()
    与他人共屏：
        sp.pause(); print("一行别的"); sp.resume()
    """

    def __init__(self, text: str = "思考中", stream: Optional[TextIO] = None,
                 interval: float = 0.12):
        self._text = text
        self._stream = stream if stream is not None else sys.stdout
        self._interval = interval
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0 = 0.0
        self._enabled = ansi_ok(self._stream)

    # ── 生命周期 ──

    def start(self) -> "Spinner":
        if not self._enabled:
            return self  # 非 tty：哑火，不刷不写
        self._t0 = time.monotonic()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="openllm-spinner")
        self._thread.start()
        return self

    def stop(self) -> None:
        self._erase()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None

    def pause(self) -> None:
        """擦掉本行让给别人打印（轮子继续计时）。"""
        self._paused.set()
        self._erase()

    def resume(self) -> None:
        self._paused.clear()

    def update(self, text: str) -> None:
        """换提示词（如 思考中→工具中）。"""
        self._text = text

    # ── 内部 ──

    def _erase(self) -> None:
        if not self._enabled:
            return
        try:
            self._stream.write(_CLEAR)
            self._stream.flush()
        except Exception:
            pass

    def _loop(self) -> None:
        i = 0
        while not self._stop.wait(self._interval):
            if self._paused.is_set():
                continue
            elapsed = int(time.monotonic() - self._t0)
            line = f"\r{_FRAMES[i % len(_FRAMES)]} {self._text} · {elapsed}s"
            try:
                self._stream.write(line)
                self._stream.flush()
            except Exception:
                return  # 流没了：线程静默退出
            i += 1
