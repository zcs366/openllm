"""
openLLM 感知层 — ISA记忆 + IAI触手脑 + 章鱼I搜索预测

三体架构第一层：感知层 = "我知道什么"
- ISA：记忆召回 + 上下文构建
- IAI：触手脑搜索 + 事件总线 + 预测引擎
- 章鱼I：左右脑推理 + 因果预测
"""
from .isa_impl import ISA
from .octopus_impl import 章鱼I, _LeftBrain, _RightBrain

# IAI触手脑（可选导入）
try:
    from ..iai.prediction import PredictionEngine as IAIPrediction
    from ..iai.event_bus import EventBus as IAIEventBus
    _HAS_IAI = True
except ImportError:
    _HAS_IAI = False

__all__ = ["ISA", "章鱼I", "_LeftBrain", "_RightBrain"]
if _HAS_IAI:
    __all__.extend(["IAIPrediction", "IAIEventBus"])
