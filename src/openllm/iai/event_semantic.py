"""event_semantic.py — event_bus 语义路由引擎（P0-2，ISA通信体战役）

订阅者可声明语义兴趣（自由文本），事件到达时用 dense embedding 相似度
判定相关性。与 event_bus 的精确字段过滤互补：
  - 精确过滤器（source/type/brain_id）= 硬门槛，有则须匹配
  - semantic_filter（兴趣文本）        = 相关性层，有兴趣则须语义命中

失败降级铁律：embedding 初始化/encode 抛任何异常（模型未下载/依赖缺失/
网络）→ similarity() 返回 None（放弃语义判定）。语义路由是增强不是门禁，
绝不因 embedding 故障瘫痪事件总线。

依赖纪律（对齐 embedding.py DR-20260828-01）：
  - 顶层只 import stdlib + numpy（轻）
  - sentence_transformers 只在 _ensure_engine() 内惰性 import（重依赖链）
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger("openllm.iai.event_semantic")

# payload 中优先提取文本的键（按序取第一个非空字符串值）
TEXT_KEYS = ("text", "body", "content", "message")


def event_payload_to_text(payload: dict[str, Any]) -> str:
    """从事件 payload 提取可编码文本：优先语义字段，兜底 JSON 序列化。"""
    for key in TEXT_KEYS:
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val
    try:
        return json.dumps(payload, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(payload)


class SemanticMatcher:
    """兴趣文本 × 事件文本 的 embedding 相似度判定。

    similarity() 返回 None = 降级（embedding 不可用），调用方不得视为命中。
    """

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5") -> None:
        self._model_name = model_name
        self._engine: Any = None  # 惰性加载；Any=可注入替身（测试/未来换模型）

    def _ensure_engine(self):
        if self._engine is None:
            # 惰性 import：sentence_transformers 重依赖链不阻塞模块导入
            from openllm.embedding import EmbeddingEngine
            self._engine = EmbeddingEngine(self._model_name)
        return self._engine

    def similarity(self, interest: str, event_text: str) -> Optional[float]:
        """返回 cosine 相似度 ∈ [0,1]；embedding 不可用时返回 None（降级）。"""
        try:
            import numpy as np

            engine = self._ensure_engine()
            vec_a = np.asarray(engine.encode(interest), dtype=np.float64)
            vec_b = np.asarray(engine.encode(event_text), dtype=np.float64)
            denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
            if denom == 0.0:
                return 0.0
            return float(np.dot(vec_a, vec_b) / denom)
        except Exception as exc:  # 模型未下载/依赖缺失/网络——一律降级不阻断
            logger.warning(
                "[event_semantic] embedding 不可用，语义判定降级: %s", exc
            )
            return None

    def is_match(self, interest: str, event_text: str, threshold: float) -> bool:
        """语义命中判定。降级（None）视为不命中——语义维度不擅自放行。"""
        sim = self.similarity(interest, event_text)
        if sim is None:
            return False
        return sim >= threshold
