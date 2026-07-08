"""
openLLM Graceful Shutdown — Drain→Dormant→Waking 状态机。

从 Hermes v0.18.0 的 graceful shutdown 学来。
个人版精简：三阶段停机，信号处理器，状态快照。

状态机：
  RUNNING   → DRAINING → DORMANT → WAKING → RUNNING
  (运行中)    (排空)     (休眠)    (唤醒)    (恢复)

停机流程：
  1. 收到 SIGTERM/SIGINT → begin_drain()
  2. 拒绝新请求，等待当前工具调用完成
  3. flush 所有待写入数据
  4. 保存最终状态快照 → DORMANT
"""

import signal
import time
import logging
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger("openllm.shutdown")


class ShutdownState(Enum):
    """停机状态机。"""
    RUNNING = "running"      # 正常运行，接受请求
    DRAINING = "draining"    # 排空中：拒绝新请求，等待进行中任务完成
    DORMANT = "dormant"      # 已休眠：数据已flush，可恢复
    WAKING = "waking"        # 唤醒中：从DORMANT恢复到RUNNING


class GracefulShutdown:
    """
    优雅停机控制器。

    用法：
        shutdown = GracefulShutdown()
        shutdown.register_signal_handlers()

        # 在chat()入口检查
        if not shutdown.can_accept:
            return "Agent正在停机中，请稍候..."

        # 显式停机
        shutdown.begin_drain()
        # ... 等待完成 ...
        shutdown.begin_suspend()  # → DORMANT

        # 唤醒
        shutdown.begin_wake()
        # ... 恢复状态 ...
        shutdown.complete_wake()  # → RUNNING
    """

    DRAIN_TIMEOUT = 10.0  # 最长排空等待秒数

    def __init__(self):
        self.state = ShutdownState.RUNNING
        self._on_drain: Optional[Callable] = None
        self._on_flush: Optional[Callable] = None
        self._on_suspend: Optional[Callable] = None
        self._drain_start: float = 0.0
        self._suspend_time: float = 0.0
        self._registered = False
    def register_signal_handlers(self):
        """注册SIGTERM/SIGINT处理器。幂等。"""
        if self._registered:
            return
        try:
            signal.signal(signal.SIGTERM, self._handle_signal)
            signal.signal(signal.SIGINT, self._handle_signal)
            self._registered = True
            logger.info("优雅停机信号处理器已注册 (SIGTERM, SIGINT)")
        except (OSError, ValueError) as e:
            logger.warning(f"信号处理器注册失败(可能非主线程): {e}")

    def _handle_signal(self, signum, frame):
        """信号处理器。"""
        sig_name = signal.Signals(signum).name
        logger.info(f"收到 {sig_name}，开始优雅停机...")
        self.begin_drain()

    # ── 状态转换 ────────────────────────────────────

    def begin_drain(self):
        """开始排空：RUNNING → DRAINING。"""
        if self.state != ShutdownState.RUNNING:
            logger.debug(f"begin_drain忽略: 当前状态={self.state.value}")
            return
        self.state = ShutdownState.DRAINING
        self._drain_start = time.time()
        logger.info("状态: RUNNING → DRAINING")
        if self._on_drain:
            self._on_drain()

    def is_drained(self) -> bool:
        """排空是否完成：等待时间超限 或 手动标记。"""
        if self.state != ShutdownState.DRAINING:
            return True
        elapsed = time.time() - self._drain_start
        return elapsed >= self.DRAIN_TIMEOUT

    def begin_suspend(self):
        """挂起：→ DORMANT。数据已flush。

        接受DRAINING（正常排空后）和RUNNING（显式sleep调用，无需排空）。
        个人版无异步任务，sleep()里drain→suspend背靠背是正确行为。
        """
        if self.state not in (ShutdownState.DRAINING, ShutdownState.RUNNING):
            logger.debug(f"begin_suspend忽略: 当前状态={self.state.value}")
            return
        self.state = ShutdownState.DORMANT
        self._suspend_time = time.time()
        logger.info("状态: → DORMANT")
        if self._on_suspend:
            self._on_suspend()

    def begin_wake(self):
        """唤醒开始：DORMANT → WAKING。"""
        if self.state != ShutdownState.DORMANT:
            logger.debug(f"begin_wake忽略: 当前状态={self.state.value}")
            return
        self.state = ShutdownState.WAKING
        logger.info("状态: DORMANT → WAKING")

    def complete_wake(self):
        """唤醒完成：WAKING → RUNNING。"""
        if self.state != ShutdownState.WAKING:
            logger.debug(f"complete_wake忽略: 当前状态={self.state.value}")
            return
        self.state = ShutdownState.RUNNING
        self._suspend_time = 0.0
        logger.info("状态: WAKING → RUNNING")

    # ── 属性 ────────────────────────────────────────

    @property
    def can_accept(self) -> bool:
        """是否能接受新请求。"""
        return self.state == ShutdownState.RUNNING

    @property
    def is_dormant(self) -> bool:
        """是否已休眠。"""
        return self.state == ShutdownState.DORMANT

    @property
    def uptime_since_suspend(self) -> float:
        """DORMANT状态持续时间（秒）。"""
        if self._suspend_time <= 0:
            return 0.0
        return time.time() - self._suspend_time

    # ── 回调注册 ────────────────────────────────────

    def on_drain(self, fn: Callable):
        """注册排空回调。"""
        self._on_drain = fn

    def on_flush(self, fn: Callable):
        """注册flush回调。"""
        self._on_flush = fn

    def on_suspend(self, fn: Callable):
        """注册挂起回调。"""
        self._on_suspend = fn

    def status(self) -> dict:
        """状态快照。"""
        return {
            "state": self.state.value,
            "can_accept": self.can_accept,
            "drain_elapsed": time.time() - self._drain_start if self._drain_start else 0,
            "dormant_duration": self.uptime_since_suspend,
        }
