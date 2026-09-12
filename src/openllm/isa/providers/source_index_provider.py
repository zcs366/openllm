"""
SourceIndexProvider — 来源索引provider

让openLLM可以通过MemoryBus访问来源索引（论文、博客、新闻等）。
底层委托给 ~/.hermes/scripts/source_index.py 的 SourceIndex 类。

功能：
  - store: 写入来源（URL去重+content_hash去重）
  - search: 搜索来源（title/url模糊匹配+类型/质量过滤）
  - enrich: 补充来源元数据（引用数、撤回状态、质量评分）
"""

import sys
import time
import logging
from pathlib import Path
from typing import Any, Dict, List

# 确保 source_index 可导入
_SCRIPTS = str(Path.home() / ".hermes" / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from source_index import SourceIndex

from ..memory_bus import (
    MemoryProvider, MemoryRecord, WriteRequest, WriteResult, Query, tokenize
)

logger = logging.getLogger("openllm.providers.source_index")

DEFAULT_DB = str(Path.home() / ".hermes" / "data" / "source_index.db")


class SourceIndexProvider:
    """来源索引provider — 让openLLM可以通过MemoryBus访问来源索引"""

    def __init__(self, db_path: str = ""):
        self._db_path = db_path or DEFAULT_DB
        self._index = SourceIndex(db_path=self._db_path)

    # ── MemoryProvider协议 ──

    @property
    def name(self) -> str:
        return "source_index"

    @property
    def priority(self) -> int:
        return 3  # 低于jiak(0)/recall(1)/causal(2)，来源索引是辅助层

    @property
    def input_schema(self) -> set:
        """search()使用的Query字段"""
        return {"text", "top_k", "sources", "token_budget"}

    @property
    def output_schema(self) -> set:
        """search()产出的MemoryRecord字段"""
        return {"record_id", "content", "source", "record_type",
                "importance", "temperature", "trust_level",
                "tags", "timestamp", "context", "score", "provider"}

    # ── 核心操作 ──

    def search(self, query) -> List[MemoryRecord]:
        """搜索来源索引，返回MemoryRecord列表。支持Query对象或str。"""
        # 支持standalone调用：search("keyword")
        if isinstance(query, str):
            query_text = query
            source_type = None
            top_k = 10
            min_importance = 0.0
            min_temperature = 0.0
            record_types = None
        else:
            query_text = query.text
            top_k = query.top_k
            min_importance = query.min_importance
            min_temperature = query.min_temperature
            record_types = query.record_types
            # 解析source_type过滤（从Query.sources中提取已知类型）
            source_type = None
            if query.sources:
                known = {"paper", "blog", "news", "social", "code"}
                for s in query.sources:
                    if s in known:
                        source_type = s
                        break

        try:
            rows = self._index.search(
                query=query_text,
                source_type=source_type,
                limit=top_k * 2,
            )
        except Exception as e:
            logger.error(f"SourceIndex搜索异常: {e}")
            return []

        records = []
        for row in rows:
            # 构建content：优先title，其次url截断
            title = row.get("title", "") or ""
            url = row.get("url", "")
            content = f"{title}\n{url}" if title else url
            if not content:
                continue

            # 关键词匹配评分
            query_kw = tokenize(query_text)
            title_kw = tokenize(title) if title else set()
            url_kw = tokenize(url)
            combined_kw = title_kw | url_kw
            overlap = query_kw & combined_kw
            keyword_score = len(overlap) / max(len(query_kw), 1) if query_kw else 0.3

            # 质量分和使用频次加权
            quality = row.get("quality_score", 0.0) or 0.0
            use_count = row.get("use_count", 0) or 0
            freq_bonus = min(use_count / 10.0, 0.3)
            final_score = keyword_score * 0.5 + quality * 0.3 + freq_bonus

            # 标签
            tags = []
            stype = row.get("source_type", "unknown")
            if stype:
                tags.append(stype)
            if quality >= 0.7:
                tags.append("high_quality")

            # 时间戳
            first_seen = row.get("first_seen", "")
            try:
                from datetime import datetime
                ts = datetime.fromisoformat(first_seen).timestamp() if first_seen else time.time()
            except (ValueError, TypeError):
                ts = time.time()

            # 重要性映射：quality_score → importance
            importance = min(max(quality, 0.0), 1.0) if quality else 0.5

            record = MemoryRecord(
                record_id=f"source:{row.get('id', 0)}",
                content=content,
                source="source_index",
                record_type="insight",
                importance=importance,
                temperature=0.8,  # 来源索引内容衰减慢
                trust_level="internal",
                tags=tags,
                timestamp=ts,
                context={
                    "source_id": row.get("id"),
                    "url": url,
                    "source_type": stype,
                    "quality_score": quality,
                    "use_count": use_count,
                },
                score=final_score,
                provider="source_index",
            )

            # 过滤
            if min_importance > 0 and record.importance < min_importance:
                continue
            if min_temperature > 0 and record.temperature < min_temperature:
                continue
            if record_types and record.record_type not in record_types:
                continue

            records.append(record)

        # 按score排序，取top_k
        records.sort(key=lambda r: r.score, reverse=True)
        return records[:top_k]

    def store(self, request) -> WriteResult:
        """写入来源到索引。支持WriteRequest或raw dict。"""
        if isinstance(request, WriteRequest):
            metadata = request.metadata
        elif isinstance(request, dict):
            metadata = request
        else:
            return WriteResult(success=False, reason=f"不支持的请求类型: {type(request)}")

        url = metadata.get("url", "")
        title = metadata.get("title", "")
        content = metadata.get("content", "")
        source_type = metadata.get("source_type", "unknown")

        # 必须有URL
        if not url:
            return WriteResult(
                success=False,
                blocked=True,
                reason="source_index需要URL字段",
            )

        try:
            source_id = self._index.add(
                url=url,
                title=title,
                content=content,
                source_type=source_type,
                metadata=metadata,
            )
            return WriteResult(
                success=True,
                record_id=f"source:{source_id}",
                provider="source_index",
            )
        except Exception as e:
            logger.error(f"SourceIndex写入异常: {e}")
            return WriteResult(
                success=False,
                reason=f"写入异常: {e}",
            )

    def count(self) -> int:
        """统计来源总数"""
        try:
            stats = self._index.stats()
            return stats.get("total_sources", 0)
        except Exception:
            return 0

    def health(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            stats = self._index.stats()
            return {
                "status": "ok",
                "total_sources": stats.get("total_sources", 0),
                "by_type": stats.get("by_type", {}),
                "avg_quality": stats.get("avg_quality", 0.0),
                "db_path": self._db_path,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ── 扩展操作 ──

    def enrich(self, source_id: int, **kwargs) -> dict:
        """补充来源元数据"""
        try:
            self._index.enrich(source_id, **kwargs)
            return self._index.get(source_id=source_id) or {}
        except Exception as e:
            logger.error(f"SourceIndex enrich异常: {e}")
            return {}

    def record_usage(self, source_id: int, research_id: str, citation_text: str = ''):
        """记录来源被某研究引用"""
        try:
            self._index.record_usage(source_id, research_id, citation_text)
        except Exception as e:
            logger.error(f"SourceIndex record_usage异常: {e}")


def register(bus) -> SourceIndexProvider:
    """注册到MemoryBus"""
    provider = SourceIndexProvider()
    bus.register(provider)
    return provider
