"""Cordis — 可逆效应代数 + 反应式共效应运行时。

DSH-Cordis 形式模型最小 Python 实现。
核心保证：副作用自带逆 → 卸载 = 逆序吞后悔药 → 环境零残渣。
"""

from openllm.cordis.runtime import (
    Effect,
    EffectContext,
    CoeffectContext,
    Provider,
)

__all__ = ["Effect", "EffectContext", "CoeffectContext", "Provider"]
