"""iax — 心跳引擎 + 可逆效应。

六体之一：IAX 是主循环的心跳引擎，5阶段(PERCEIVE→DECIDE→EXECUTE→LEARN→FEEDBACK)。

归位说明（2026-09-12）：从 core/ 归拢至此——
  - heartbeat.py            纯状态机心跳（红线：无终止对话权）
  - agent_heartbeat.py      5阶段精简心跳（从12阶段精简）
  - hemispheres.py          左右脑对弈
  - hemispheres_enhanced.py 增强仲裁
  - clock.py                时钟账本（纪元节律）
  - awakening.py            苏醒协议
  - cordis.py               可逆效应代数（IAX-Cordis设计草案 T-D-1/T-D-2）
"""
from openllm.iax.heartbeat import HeartbeatState, HeartbeatEvent
from openllm.iax.agent_heartbeat import execute_tick
from openllm.iax.hemispheres import HemispherePair, HemisphereState
from openllm.iax.hemispheres_enhanced import EnhancedHemispherePair
from openllm.iax.clock import Clock
from openllm.iax.awakening import AwakeningProtocol
from openllm.iax.cordis import Effect, EffectContext, CoeffectContext, Provider

__all__ = [
    "HeartbeatState", "HeartbeatEvent",
    "execute_tick",
    "HemispherePair", "HemisphereState",
    "EnhancedHemispherePair",
    "Clock",
    "AwakeningProtocol",
    "Effect", "EffectContext", "CoeffectContext", "Provider",
]
