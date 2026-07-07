"""
CausalProvider — 因果记忆provider

包装现有的CausalMemoryStore，暴露MemoryProvider接口。
因果教训链：prediction → actual → delta → lesson → pattern

检索策略：
1. 从query提取context_features
2. 按features搜索CausalMemoryStore
3. 按confidence×frequency排序
"""

import time
import logging
from typing import Any, Dict, List, Optional

from ..memory_bus import (
    MemoryProvider, MemoryRecord, WriteRequest, WriteResult, Query
)

logger = logging.getLogger("openllm.providers.causal")


class CausalProvider:
    """因果记忆provider"""

    def __init__(self, causal_store=None):
        """
        Args:
            causal_store: CausalMemoryStore实例。为None时延迟初始化。
        """
        self._store = causal_store

    @property
    def name(self) -> str:
        return "causal"

    @property
    def priority(self) -> int:
        return 20

    def _ensure_store(self):
        """延迟初始化CausalMemoryStore"""
        if self._store is None:
            from ..causal_memory import CausalMemoryStore
            from pathlib import Path
            store_dir = Path.home() / ".openllm" / "memory" / "causal"
            self._store = CausalMemoryStore(store_dir=store_dir)

    def search(self, query: Query) -> List[MemoryRecord]:
        """从因果记忆中检索"""
        self._ensure_store()

        # 从query文本提取context_features
        features = query.text.lower().split()
        # 过滤停用词
        features = [f for f in features if len(f) > 1]

        if not features:
            return []

        try:
            causal_results = self._store.search(
                context_features=features,
                max_results=query.top_k * 2,
            )
        except Exception as e:
            logger.error(f"CausalMemory search error: {e}")
            return []

        records = []
        for cr in causal_results:
            # 构建内容：lesson是核心
            content = getattr(cr, "lesson", "")
            if not content:
                content = getattr(cr, "delta", "")

            if not content:
                continue

            # score = confidence × (1 + log(frequency))
            confidence = getattr(cr, "prediction_confidence", 0.5)
            frequency = getattr(cr, "frequency", 1) if hasattr(cr, "frequency") else 1
            import math
            score = confidence * (1 + math.log(max(frequency, 1)))

            record = MemoryRecord(
                record_id=getattr(cr, "memory_id", f"causal:{hash(content)}"),
                content=content,
                source="causal",
                record_type="lesson",
                importance=confidence,
                temperature=0.8,  # 因果记忆温度衰减慢
                trust_level=getattr(cr, "trust_level", "internal").value
                    if hasattr(getattr(cr, "trust_level", ""), "value")
                    else str(getattr(cr, "trust_level", "internal")),
                tags=getattr(cr, "context_features", []),
                timestamp=getattr(cr, "created_at", time.time()),
                context={
                    "action_signature": getattr(cr, "action_signature", ""),
                    "prediction": getattr(cr, "prediction", ""),
                    "delta": getattr(cr, "delta", ""),
                    "pattern_id": getattr(cr, "pattern_id", ""),
                },
                score=score,
                provider="causal",
            )

            # 过滤条件
            if query.min_importance > 0 and record.importance < query.min_importance:
                continue

            records.append(record)

        records.sort(key=lambda r: r.score, reverse=True)
        return records[:query.top_k]

    def store(self, request: WriteRequest) -> WriteResult:
        """
        因果记忆写入——将insight转为因果链存储。
        """
        self._ensure_store()

        try:
            cr = self._store.store(
                action_signature=request.metadata.get("action_signature", request.source),
                context_features=request.tags or request.content.split()[:5],
                prediction=request.metadata.get("prediction", ""),
                prediction_confidence=request.metadata.get("confidence", 0.5),
                actual_result=request.metadata.get("actual", ""),
                actual_success=request.metadata.get("success", True),
                delta=request.metadata.get("delta", ""),
                delta_magnitude=request.metadata.get("delta_mag", 0.0),
                lesson=request.content,
                source=request.source,
                session_id=request.session_id,
            )
            return WriteResult(
                success=True,
                record_id=getattr(cr, "memory_id", ""),
                provider="causal",
            )
        except Exception as e:
            return WriteResult(
                success=False,
                reason=f"causal store error: {e}",
            )

    def count(self) -> int:
        self._ensure_store()
        try:
            stats = self._store.stats()
            return stats.get("total", 0)
        except Exception:
            return 0

    def health(self) -> Dict[str, Any]:
        self._ensure_store()
        try:
            stats = self._store.stats()
            return {
                "status": "ok",
                "total": stats.get("total", 0),
                "patterns": stats.get("patterns", 0),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
