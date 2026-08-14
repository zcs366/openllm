"""iai — 事件总线 + 因果预测。

合并说明：IAI原有6个文件，4个已deprecated（router/slow_path/user_state/topology）。
路由功能已迁移至octopus.reason()，心跳不再调用IAI路由器。
保留：event_bus（核心基础设施）、prediction（因果预测引擎）。
"""
from openllm.iai.event_bus import (
    EventBus, Event, Subscriber, BaseEventEmitter, SignalBridge,
)
from openllm.iai.prediction import PredictionEngine

__all__ = [
    "EventBus", "Event", "Subscriber", "BaseEventEmitter", "SignalBridge",
    "PredictionEngine",
]
