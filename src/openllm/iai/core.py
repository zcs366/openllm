"""iai/core.py — IAI 容器：章鱼I的运行环境（IAI融合·2026-09-02）。

章鱼I是IAI的核心器官（推理心脏），IAI是章鱼I的头骨/神经/血管
（感知、通信、学习基础设施）。

IAI = EventBus(通信神经) + PredictionEngine(因果感知·共享实例) + emit(事件发布)

设计依据：D-0901-IAI-融合决策记录（先接线后搬家·Step 1）
- 推理链路事件：context.built → reasoning.proposed → reasoning.critiqued
  → decision.made → action.executed（自进化管道的事件源）
- PredictionEngine 共享实例：EMA期望向量跨调用累积（此前每次new丢状态）
"""
import logging
import time
from pathlib import Path
from typing import Callable, Optional

from openllm.iai.event_bus import Event, EventBus
from openllm.iai.prediction import PredictionEngine
from openllm.iai.brain import BrainActivator

logger = logging.getLogger("openllm.iai.core")

PREDICTOR_STATE_FILE = Path.home() / ".openllm" / "iai" / "predictor_state.json"


class IAI:
    """IAI — 感知/通信/学习基础设施（章鱼I的运行环境）。"""

    def __init__(self, log_dir: Optional[Path] = None):
        self.bus = EventBus(log_dir=log_dir)
        self.predictor = PredictionEngine(bus=self.bus)
        self._subscriber_ids: list[str] = []
        # 跨重启恢复预测器状态（尽力而为，失败静默）
        try:
            if self.predictor.load(PREDICTOR_STATE_FILE):
                logger.info("[iai] PredictionEngine 状态已恢复 (count=%d)",
                            self.predictor._count)
        except Exception:
            pass

    def emit(self, event_type: str, payload: Optional[dict] = None,
             source: str = "iai") -> int:
        """发布事件到总线（失败不阻塞主流程）。返回通知数量。

        brain_id 自动从 BrainActivator.current() 注入——当前活跃头脑。
        """
        try:
            brain_id = BrainActivator().current() or "default"
            return self.bus.publish(Event(
                source=source, type=event_type, timestamp=time.time(),
                entropy_score=0.0, payload=payload or {}, brain_id=brain_id))
        except Exception as e:
            logger.warning("[iai] emit %s 失败: %s", event_type, e)
            return 0

    def subscribe(self, cb: Callable[[Event], None],
                  source_filter: Optional[str] = None,
                  type_filter: Optional[str] = None,
                  brain_id: Optional[str] = None) -> str:
        """注册订阅者（记录sid以便管理）。brain_id=None=听全部。"""
        sid = self.bus.subscribe(cb, source_filter, type_filter, brain_id)
        self._subscriber_ids.append(sid)
        return sid

    def gate(self, interruption_type: str, target: str,
             source_brain: Optional[str] = None, blocked: bool = True) -> int:
        """连续性门控：发布SALIENCE_GATE事件（十律⑥·蜻蜓门控）。

        当主任务在进行中（heartbeat tick间），收到突发信号/用户消息时
        发布 SALIENCE_GATE(blocked=True) 标记主任务轨迹保持（不打断）；
        主任务完成→发布 SALIENCE_GATE(blocked=False) 允许切换。

        interruption_type: 'task_switch' | 'user_message' | 'signal'
        target: 主任务标识
        source_brain: 发起门控的大脑（可选）
        blocked: True=阻断干扰保持主任务 / False=允许切换
        """
        payload = {
            "interruption_type": interruption_type,
            "target": target,
            "blocked": blocked,
            "timestamp": time.time(),
        }
        if source_brain:
            payload["source_brain"] = source_brain
        return self.emit("salience.gate", payload, source="iai.gate")

    def event_stats(self) -> dict:
        """事件总线快照（IKO仪表/健康检查用）。"""
        return {
            "subscribers": self.bus.subscriber_count(),
            "history": len(self.bus.get_history(limit=10000)),
        }

    def persist_predictor(self) -> bool:
        """持久化预测器状态（训练/停机前调用）。"""
        try:
            self.predictor.save(PREDICTOR_STATE_FILE)
            return True
        except Exception as e:
            logger.warning("[iai] 预测器状态保存失败: %s", e)
            return False
