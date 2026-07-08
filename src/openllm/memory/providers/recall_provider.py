"""
RecallProvider — RECALL时间线日志provider

读取 ~/.hermes/jiak/RECALL.jsonl 中的时间线记录。
支持：FTS5关键词搜索、时间范围过滤、类型过滤。

检索策略：
1. 加载RECALL.jsonl（最近N天）
2. 对query做关键词匹配（简单文本搜索）
3. 按时间+相关性排序
"""

import json
import time
import math
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..memory_bus import (
    MemoryProvider, MemoryRecord, WriteRequest, WriteResult, Query, tokenize
)

logger = logging.getLogger("openllm.providers.recall")

JIAK_DIR = Path.home() / ".hermes" / "jiak"
RECALL_PATH = JIAK_DIR / "RECALL.jsonl"

# 模块级import recall_append（避免sys.path堆积）
_recall_append = None
_recall_append_loaded = False

def _get_recall_append():
    global _recall_append, _recall_append_loaded
    if not _recall_append_loaded:
        _recall_append_loaded = True
        try:
            import importlib.util as _ilu
            _spec = _ilu.spec_from_file_location("recall_append", JIAK_DIR / "recall_append.py")
            if _spec and _spec.loader:
                _mod = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                _recall_append = _mod.validate_and_append
        except ImportError:
            pass
    return _recall_append


class RecallProvider:
    """RECALL时间线日志provider"""

    def __init__(self, recall_path: Optional[Path] = None):
        self._path = recall_path or RECALL_PATH
        self._records: Optional[List[Dict]] = None
        self._loaded_at: float = 0

    @property
    def name(self) -> str:
        return "recall"

    @property
    def priority(self) -> int:
        return 10

    def search(self, query: Query) -> List[MemoryRecord]:
        """从RECALL记录中检索"""
        records = self._load_records()
        if not records:
            return []

        query_lower = query.text.lower()
        query_words = tokenize(query.text)
        matched = []

        for rec in records:
            content = rec.get("content", rec.get("text", ""))
            if not content:
                continue

            content_lower = content.lower()
            # 简单关键词匹配
            word_hits = sum(1 for w in query_words if w in content_lower)
            if word_hits == 0 and query_lower not in content_lower:
                continue

            score = word_hits / max(len(query_words), 1)
            # 时间衰减：越新分数越高
            ts = rec.get("timestamp", rec.get("_written_at", 0))
            if ts > 0:
                age_days = (time.time() - ts) / 86400
                import math
                time_factor = math.exp(-0.01 * age_days)
                score *= (0.7 + 0.3 * time_factor)

            record = MemoryRecord(
                record_id=f"recall:{hash(content) % 10**8}",
                content=content[:500],  # 截断过长内容
                source="recall",
                record_type=rec.get("type", "event"),
                importance=rec.get("importance", 0.5),
                temperature=self._compute_temperature(ts),
                trust_level=rec.get("trust_level", "internal"),
                tags=rec.get("tags", []),
                timestamp=ts,
                context={
                    "session_id": rec.get("session_id", ""),
                    "role": rec.get("role", ""),
                    "written_by": rec.get("_written_by", ""),
                },
                score=score,
                provider="recall",
            )

            # 过滤条件
            if query.min_importance > 0 and record.importance < query.min_importance:
                continue
            if query.record_types and record.record_type not in query.record_types:
                continue

            matched.append(record)

        matched.sort(key=lambda r: r.score, reverse=True)
        return matched[:query.top_k]

    def store(self, request: WriteRequest) -> WriteResult:
        """
        RECALL写入——通过recall_append.py写入。
        MemoryBus不直接写jsonl，调用现有的recall_append接口。
        """
        validate_and_append = _get_recall_append()
        if validate_and_append is None:
            return WriteResult(
                success=False,
                reason="recall_append.py not available",
            )

        try:
            record = {
                "content": request.content,
                "type": request.record_type,
                "source": request.source,
                "tags": request.tags,
                "importance": request.importance,
                "session_id": request.session_id,
                "timestamp": time.time(),
            }

            result = validate_and_append(record)
            if result.get("success"):
                # 清除缓存，下次检索重新加载
                self._records = None
                return WriteResult(
                    success=True,
                    record_id=result.get("id", ""),
                    provider="recall",
                )
            else:
                return WriteResult(
                    success=False,
                    reason=result.get("error", "recall_append failed"),
                )
        except Exception as e:
            return WriteResult(
                success=False,
                reason=f"recall write error: {e}",
            )

    def count(self) -> int:
        records = self._load_records()
        return len(records) if records else 0

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok" if self._path.exists() else "degraded",
            "path": str(self._path),
            "exists": self._path.exists(),
            "record_count": self.count(),
        }

    def _load_records(self, max_age_days: int = 30) -> List[Dict]:
        """加载RECALL记录（带缓存，10秒过期）"""
        now = time.time()
        if self._records is not None and now - self._loaded_at < 10:
            return self._records

        if not self._path.exists():
            return []

        records = []
        cutoff = now - (max_age_days * 86400)

        try:
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        ts = rec.get("timestamp", rec.get("_written_at", 0))
                        if ts >= cutoff:
                            records.append(rec)
                    except json.JSONDecodeError:
                        continue
        except OSError as e:
            logger.error(f"Failed to load RECALL: {e}")
            return []

        self._records = records
        self._loaded_at = now
        return records

    def _compute_temperature(self, timestamp: float) -> float:
        if timestamp == 0:
            return 0.5
        age_hours = (time.time() - timestamp) / 3600
        return math.exp(-0.029 * age_hours)
