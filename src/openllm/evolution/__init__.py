"""
OpenLLM Evolution — 进化事件总线 + 停训线（七神终裁产物 2026-09-04）。

总线先于仪表：任何测量不自带消费者闭环 = 断头管。
停训线 = 总线的第一个真实消费者（CAR曲线：学会停比学会训更根本）。
"""
from .bus import (
    KNOWN_TYPES,
    ConsumerRegistry,
    DigestEvent,
    EvolutionBus,
    zlib_crc32,
)
from .stopline import (
    StoplineDecision,
    StoplineEvaluator,
    compute_car_series,
    extract_rounds,
    quality_from_verify,
)

__all__ = [
    "KNOWN_TYPES",
    "ConsumerRegistry",
    "DigestEvent",
    "EvolutionBus",
    "zlib_crc32",
    "StoplineDecision",
    "StoplineEvaluator",
    "compute_car_series",
    "extract_rounds",
    "quality_from_verify",
]

