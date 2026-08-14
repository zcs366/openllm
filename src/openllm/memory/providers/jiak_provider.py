"""
JiakProvider — jiak卡片记忆provider

读取 ~/.hermes/jiak/ 下的结构化知识卡片。
支持：卡片检索、意见查询、条级注入。

检索策略：
1. 从index.json加载关键词→card_id映射
2. 对query做关键词匹配
3. 加载匹配card的alive opinions
4. 按relevance×importance排序
"""

import json
import time
import math
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..memory_bus import (
    MemoryProvider, MemoryRecord, WriteRequest, WriteResult, Query, tokenize
)

logger = logging.getLogger("openllm.providers.jiak")

JIAK_DIR = Path.home() / ".hermes" / "jiak"


class JiakProvider:
    """jiak卡片记忆provider"""

    def __init__(self, jiak_dir: Optional[Path] = None):
        self._dir = jiak_dir or JIAK_DIR
        self._index: Optional[Dict] = None
        self._index_loaded_at: float = 0

    @property
    def name(self) -> str:
        return "jiak"

    @property
    def priority(self) -> int:
        return 0  # 最优先

    @property
    def input_schema(self) -> set:
        """search()使用的Query字段"""
        return {"text", "top_k", "min_importance", "min_temperature",
                "record_types", "sources", "token_budget"}

    @property
    def output_schema(self) -> set:
        """search()产出的MemoryRecord字段"""
        return {"record_id", "content", "source", "record_type",
                "importance", "temperature", "trust_level",
                "tags", "timestamp", "context", "score", "provider"}

    def search(self, query: Query) -> List[MemoryRecord]:
        """从jiak卡片中检索"""
        index = self._load_index()
        if not index:
            return []

        # 关键词匹配（jieba中文分词）
        query_keywords = tokenize(query.text)
        matched_cards = []

        for card_id, card_meta in index.get("cards", {}).items():
            card_keywords = set(kw.lower() for kw in card_meta.get("keywords", []))
            overlap = query_keywords & card_keywords
            if overlap:
                score = len(overlap) / max(len(query_keywords), 1)
                matched_cards.append((card_id, score, card_meta))

        # 按score排序
        matched_cards.sort(key=lambda x: x[1], reverse=True)

        records = []
        for card_id, score, card_meta in matched_cards[:query.top_k * 2]:
            card_path = self._dir / "cards" / f"{card_id}.json"
            if not card_path.exists():
                continue

            try:
                card_data = json.loads(card_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            # 提取alive opinions作为条级注入
            opinions = card_data.get("opinions", [])
            for op in opinions:
                if op.get("status") != "alive":
                    continue

                content = op.get("text", op.get("content", ""))
                if not content:
                    continue

                record = MemoryRecord(
                    record_id=f"jiak:{card_id}:{op.get('id', '')}",
                    content=content,
                    source="jiak",
                    record_type="opinion",
                    importance=op.get("importance", card_meta.get("importance", 0.5)),
                    temperature=self._compute_temperature(op),
                    trust_level="internal",
                    tags=card_meta.get("keywords", []),
                    timestamp=op.get("created_at", card_data.get("created_at", 0)),
                    context={
                        "card_id": card_id,
                        "opinion_id": op.get("id", ""),
                        "topic": card_meta.get("topic", ""),
                    },
                    score=score,
                    provider="jiak",
                )

                # 过滤条件
                if query.min_importance > 0 and record.importance < query.min_importance:
                    continue
                if query.min_temperature > 0 and record.temperature < query.min_temperature:
                    continue
                if query.record_types and record.record_type not in query.record_types:
                    continue
                if query.sources and record.source not in query.sources:
                    continue

                records.append(record)

        return records[:query.top_k]

    def store(self, request: WriteRequest) -> WriteResult:
        """
        jiak写入——当前为只读provider。
        jiak卡片的写入由jiak_api.py管理，MemoryBus不直接写入。
        """
        return WriteResult(
            success=False,
            reason="jiak是只读provider，写入由jiak_api管理",
        )

    def count(self) -> int:
        """统计jiak卡片数量"""
        index = self._load_index()
        if not index:
            return 0
        return len(index.get("cards", {}))

    def health(self) -> Dict[str, Any]:
        """健康检查"""
        index_path = self._dir / "index.json"
        cards_dir = self._dir / "cards"
        return {
            "status": "ok" if index_path.exists() else "degraded",
            "index_exists": index_path.exists(),
            "cards_dir_exists": cards_dir.exists(),
            "card_count": self.count(),
        }

    def _load_index(self) -> Optional[Dict]:
        """加载index.json（带缓存，5秒过期）"""
        now = time.time()
        if self._index is not None and now - self._index_loaded_at < 5:
            return self._index

        index_path = self._dir / "index.json"
        if not index_path.exists():
            return None

        try:
            self._index = json.loads(index_path.read_text(encoding="utf-8"))
            self._index_loaded_at = now
            return self._index
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load jiak index: {e}")
            return None

    def _compute_temperature(self, opinion: Dict) -> float:
        """计算意见温度（简化版）"""
        created = opinion.get("created_at", 0)
        if created == 0:
            return 0.5
        age_hours = (time.time() - created) / 3600
        # 指数衰减，24h半衰期
        return math.exp(-0.029 * age_hours)
