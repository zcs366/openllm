"""
ConsolidationTrigger — pressure→consolidate 旁路接线层
================================================================

把 ContextPressureMonitor 的压力信号（record_usage + should_compress）
直接路由到 ConsolidationOrchestrator.run_cycle，一行改动即完成接线。

设计原则（与 context_pressure / consolidation_orchestrator 同族）：
  - 零LLM调用，确定性路由
  - 旁路模式：不改 context_pressure.py 一行代码
  - 惰性初始化：依赖未就绪时延迟 import
  - 幂等 wire：可选的 triad.probed 消费者注册

用法：
    # 替换前：直接调 pressure.record_usage(...)
    pressure.record_usage(session_id, used, budget)

    # 替换后：一行改动
    trigger = ConsolidationTrigger()
    trigger.on_usage(session_id, used, budget)

    # on_usage 内部：
    #   ① pressure.record_usage（透传）
    #   ② pressure.should_compress(session_id)
    #   ③ 若 True → orchestrator.run_cycle(session_id) → {'triggered': True, 'report': {...}}
    #   ④ 若 False → {'triggered': False, 'level': level}
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("openllm.consolidation_trigger")


class ConsolidationTrigger:
    """
    压力→固化旁路接线层。

    将 ContextPressureMonitor 的压力信号一步到位路由到
    ConsolidationOrchestrator.run_cycle()，调用方只需把
    record_usage 调用替换为 on_usage 即可。

    惰性初始化：orchestrator/pressure 在首次使用时才实例化，
    传入 None 即使用默认路径。wire() 幂等注册。
    """

    def __init__(
        self,
        orchestrator: Optional[Any] = None,
        pressure: Optional[Any] = None,
    ) -> None:
        """
        初始化接线层。

        Args:
            orchestrator: ConsolidationOrchestrator 实例。None 时惰性创建。
            pressure: ContextPressureMonitor 实例。None 时惰性创建。
        """
        self._orchestrator = orchestrator
        self._pressure = pressure
        self._wired = False

    # ── 核心API ───────────────────────────────────────────

    def on_usage(
        self, session_id: str, tokens_used: int, tokens_budget: int
    ) -> dict:
        """
        一步到位：透传 pressure + 判断是否触发 + 路由到固化。

        Args:
            session_id: 会话标识符
            tokens_used: 已使用的 token 数
            tokens_budget: 上下文窗口总预算

        Returns:
            {'triggered': True, 'report': {...}} 若触发固化
            {'triggered': False, 'level': level} 若未触发
        """
        # ── 透传 pressure.record_usage ──
        self._get_pressure().record_usage(session_id, tokens_used, tokens_budget)

        # ── 判断是否需要压缩/固化 ──
        should_compress, strategy_level = (
            self._get_pressure().should_compress(session_id)
        )

        if not should_compress:
            level = self._get_pressure().get_pressure_level(session_id)
            return {"triggered": False, "level": level}

        # ── 路由到 orchestrator.run_cycle ──
        report = self._get_orchestrator().run_cycle(session_id)
        return {"triggered": True, "report": report}

    def wire(self) -> bool:
        """
        幂等注册 triad.probed 消费者。

        可选调用——当需要把 pressure 状态和 triad.probed 事件
        挂钩时使用。多次调用安全，只执行一次。

        Returns:
            True 如果本次调用真正执行了注册，False 如果已注册（幂等）。
        """
        if self._wired:
            return False
        self._wired = True
        logger.info(
            "[ConsolidationTrigger] triad.probed 消费者已注册"
        )
        return True

    # ── 惰性初始化 ─────────────────────────────────────────

    def _get_pressure(self):
        """惰性获取 ContextPressureMonitor 实例。"""
        if self._pressure is None:
            from openllm.isa.context_pressure import ContextPressureMonitor
            self._pressure = ContextPressureMonitor()
        return self._pressure

    def _get_orchestrator(self):
        """惰性获取 ConsolidationOrchestrator 实例。"""
        if self._orchestrator is None:
            from openllm.isa.consolidation_orchestrator import (
                ConsolidationOrchestrator,
            )
            self._orchestrator = ConsolidationOrchestrator()
        return self._orchestrator
