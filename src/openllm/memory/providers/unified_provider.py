"""
UnifiedProvider — 温度记忆provider

包装现有的UnifiedMemory（hot/warm/cold三层），暴露MemoryProvider接口。
作为MemoryBus的兜底provider，接收其他provider不接受的写入。

检索策略：
1. 对query做关键词匹配
2. 按importance×temperature排序
3. 支持layer过滤
"""

import time
import logging
from typing import Any, Dict, List, Optional

from ..memory_bus import (
    MemoryProvider, MemoryRecord, WriteRequest, WriteResult, Query
)

logger = logging.getLogger("openllm.providers.unified")


class UnifiedProvider:
    """温度记忆provider（兜底）"""

    def __init__(self, unified_memory=None):
        """
        Args:
            unified_memory: UnifiedMemory实例。为None时延迟初始化。
        """
        self._memory = unified_memory

    @property
    def name(self) -> str:
        return "unified"

    @property
    def priority(self) -> int:
        return 40  # 最低优先级（兜底）

    def _ensure_memory(self):
        """延迟初始化UnifiedMemory"""
        if self._memory is None:
            from ..unified_memory import UnifiedMemory
            from pathlib import Path
            base = Path.home() / ".openllm" / "memory"
            self._memory = UnifiedMemory(memory_dir=base)

    def search(self, query: Query) -> List[MemoryRecord]:
        """从温度记忆中检索"""
        self._ensure_memory()

        try:
            results = self._memory.retrieve(query.text, top_n=query.top_k * 2)
        except Exception as e:
            logger.error(f"UnifiedMemory retrieve error: {e}")
            return []

        records = []
        for entry in results:
            # entry可能是MemoryEntry或dict
            if hasattr(entry, "key"):
                key = entry.key
                value = entry.value
                importance = entry.importance
                heat = entry.heat
                tags = entry.tags
                created = entry.created_at
                layer = entry.layer
            elif isinstance(entry, dict):
                key = entry.get("key", "")
                value = entry.get("value", {})
                importance = entry.get("importance", 0.5)
                heat = entry.get("heat", 0.0)
                tags = entry.get("tags", [])
                created = entry.get("created_at", 0)
                layer = entry.get("layer", "warm")
            else:
                continue

            content = str(value) if value else key

            record = MemoryRecord(
                record_id=f"unified:{key}",
                content=content[:500],
                source="unified",
                record_type="insight",
                importance=importance,
                temperature=min(1.0, importance + heat * 0.3),
                trust_level="internal",
                tags=tags,
                timestamp=created,
                context={"layer": layer, "key": key},
                score=importance * (1 + heat * 0.1),
                provider="unified",
            )

            # 过滤条件
            if query.min_importance > 0 and record.importance < query.min_importance:
                continue
            if query.record_types and record.record_type not in query.record_types:
                continue

            records.append(record)

        records.sort(key=lambda r: r.score, reverse=True)
        return records[:query.top_k]

    def store(self, request: WriteRequest) -> WriteResult:
        """温度记忆写入"""
        self._ensure_memory()

        try:
            entry = self._memory.store(
                key=f"{request.source}:{int(time.time())}",
                value={"content": request.content, "source": request.source},
                importance=request.importance,
                layer=request.metadata.get("layer", "warm"),
                tags=request.tags,
            )
            return WriteResult(
                success=True,
                record_id=f"unified:{entry.key}" if hasattr(entry, "key") else "",
                provider="unified",
            )
        except Exception as e:
            return WriteResult(
                success=False,
                reason=f"unified store error: {e}",
            )

    def count(self) -> int:
        self._ensure_memory()
        try:
            stats = self._memory.get_stats()
            return stats.get("total_entries", 0)
        except Exception:
            return 0

    def health(self) -> Dict[str, Any]:
        self._ensure_memory()
        try:
            stats = self._memory.get_stats()
            return {
                "status": "ok",
                "total": stats.get("total_entries", 0),
                "layers": stats.get("by_layer", {}),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
