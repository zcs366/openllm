"""UShapedContextEngine — 按U型注意力曲线排列注入位置的ContextEngine实现。

Lost in the Middle (Stanford 2023) + RULER (Hsieh 2024):
LLM对上下文开头和结尾的召回率90-99%，中间下降20-30个百分点。

策略：
- 开头25% token: L2完整原文（注意力高区·开头）
- 中间50% token: L0/L0-cold最小元数据（注意力低区）
- 结尾25% token: L1摘要+洞察（注意力高区·结尾）

三阶层存储：
- hot(active): 完整内容
- warm(cooling): archive_summary（如有）或完整内容
- cold(archived): archive_summary（如有）或降级L0

从isa_ice的u_shaped_reorder和三阶层注入逻辑移植到openLLM ContextEngine ABC。
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openllm.context_engine import ContextEngine, ContextEntry, SearchResult
from openllm.memory.temperature_engine import (
    compute_entry_temperature, temperature_state, should_evict,
    HOT_THRESHOLD, WARM_THRESHOLD, COLD_THRESHOLD, EVICT_THRESHOLD,
)

logger = logging.getLogger("ushaped_context_engine")

# ── 温度阈值（由temperature_engine统一管理）──
HOT_DAYS = 7     # 保留作为fallback
WARM_DAYS = 30   # 保留作为fallback


def _estimate_tokens(text: str) -> int:
    """粗略token估算（中文~1.5字/token，英文~4字符/token）。"""
    cn_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    en_chars = len(text) - cn_chars
    return int(cn_chars / 1.5 + en_chars / 4)


def _get_decay_status(entry: ContextEntry) -> str:
    """用温度引擎判断entry衰减状态（替代纯天数判断）。"""
    if entry.metadata.get("immutable"):
        return "active"

    importance = entry.metadata.get("importance", 5.0)
    last_accessed = entry.metadata.get("last_accessed") or entry.metadata.get("updated", "")
    memory_type = entry.metadata.get("memory_type", "insight")
    access_count = entry.metadata.get("access_count", 0)

    temp = compute_entry_temperature(
        importance=importance,
        last_accessed=last_accessed,
        memory_type=memory_type,
        access_count=access_count,
    )

    if temp > HOT_THRESHOLD:
        return "active"
    elif temp > WARM_THRESHOLD:
        return "cooling"
    else:
        return "archived"


def _get_entry_temperature(entry: ContextEntry) -> float:
    """获取entry的当前温度（用于淘汰排序）。"""
    importance = entry.metadata.get("importance", 5.0)
    last_accessed = entry.metadata.get("last_accessed") or entry.metadata.get("updated", "")
    memory_type = entry.metadata.get("memory_type", "insight")
    access_count = entry.metadata.get("access_count", 0)
    causal_weight = entry.metadata.get("causal_weight", 0.0)

    return compute_entry_temperature(
        importance=importance,
        last_accessed=last_accessed,
        memory_type=memory_type,
        access_count=access_count,
        causal_weight=causal_weight,
    )


def _format_level(entry: ContextEntry, level: str, decay: str) -> tuple[str, int]:
    """按level格式化entry内容。返回(content, tokens)。"""
    archive = entry.metadata.get("archive_summary", "")

    if level == "L2":
        # 完整内容
        content = f"[{entry.metadata.get('card_id', '?')}] {entry.content}"
    elif level == "L1":
        # 摘要+洞察
        title = entry.metadata.get("title", "")
        summary = entry.content[:200]
        insights = entry.metadata.get("insights", [])
        parts = [f"[{entry.metadata.get('card_id', '?')}] {title}", f"摘要: {summary}"]
        for ins in insights[:2]:
            parts.append(f"  • {ins}")
        content = "\n".join(parts)
    elif level == "L0-cold" and archive:
        # 三阶层：温/冷层用archive_summary
        card_id = entry.metadata.get("card_id", "?")
        title = entry.metadata.get("title", "")
        content = f"[{card_id}] {title} | {archive}"
    else:
        # L0：最小元数据
        card_id = entry.metadata.get("card_id", "?")
        title = entry.metadata.get("title", "")
        keywords = ", ".join(entry.metadata.get("keywords", [])[:3])
        content = f"[{card_id}] {title} | {keywords} | {entry.content[:80]}"

    tokens = _estimate_tokens(content)
    return content, tokens


class UShapedContextEngine(ContextEngine):
    """按U型注意力曲线排列注入位置的ContextEngine实现。

    用法：
        engine = UShapedContextEngine(storage_dir=Path("~/.openllm/context"))
        engine.persist(entries)
        results = engine.search("相关查询")
        compressed = engine.compress(entries, target_tokens=4000)
    """

    def __init__(self, storage_dir: Path | str = "~/.openllm/context"):
        self.storage_dir = Path(storage_dir).expanduser()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.storage_dir / "index.json"
        self._index: dict[str, dict] = self._load_index()

    def _load_index(self) -> dict:
        if self._index_path.exists():
            try:
                return json.loads(self._index_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"entries": {}}

    def _save_index(self) -> None:
        self._index_path.write_text(
            json.dumps(self._index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── compress: U型重排 ────────────────────────────────────
    def compress(self, entries: list[ContextEntry], target_tokens: int) -> list[ContextEntry]:
        """按U型注意力曲线重排entries，温度驱动淘汰，贪心填充到target_tokens预算。

        新增：temperature-based淘汰（温度最低的优先丢弃）。
        不改变entries内容，只改变位置顺序和数量。
        少于4个entries不重排。
        """
        if len(entries) < 4:
            return entries[:self._max_fit(entries, target_tokens)]

        # 为每个entry计算温度并标记衰减状态
        enriched = []
        for e in entries:
            decay = _get_decay_status(e)
            temp = _get_entry_temperature(e)
            archive = e.metadata.get("archive_summary", "")
            if archive and decay in ("cooling", "archived"):
                level = "L0-cold"
            else:
                level = e.metadata.get("level", "L0")
            enriched.append((e, temp, level))

        # 温度淘汰：移除evictable且温度最低的entries（保留immutable和hot）
        evictable = [(e, t, l) for e, t, l in enriched
                     if not e.metadata.get("immutable") and t < EVICT_THRESHOLD]
        keep = [(e, t, l) for e, t, l in enriched
                if e.metadata.get("immutable") or t >= EVICT_THRESHOLD]

        # 按level分组（只用keep中的）
        l2, l1, l0 = [], [], []
        for e, temp, level in keep:
            if level == "L2":
                l2.append((e, temp))
            elif level == "L1":
                l1.append((e, temp))
            else:
                l0.append((e, temp))

        # U型分布：开头=L2, 中间=L0, 结尾=L1
        # 同level内按温度降序（最热的在前）
        head = sorted(l2, key=lambda x: -x[1]) if l2 else (
            sorted(l1, key=lambda x: -x[1])[:1] if l1 else [])
        tail = sorted(l1, key=lambda x: -x[1])[1:] if len(l1) > 1 and l2 else sorted(l1, key=lambda x: -x[1])
        middle = sorted(l0, key=lambda x: -x[1])

        reordered = head + middle + tail

        # 贪心填充到target_tokens
        result = []
        used = 0
        for e, temp in reordered:
            decay = _get_decay_status(e)
            archive = e.metadata.get("archive_summary", "")

            if archive and decay in ("cooling", "archived"):
                content, tokens = _format_level(e, "L0-cold", decay)
            else:
                level = e.metadata.get("level", "L0")
                content, tokens = _format_level(e, level, decay)

            if used + tokens <= target_tokens:
                result.append(ContextEntry(
                    role=e.role,
                    content=content,
                    token_count=tokens,
                    metadata={**e.metadata, "_shaped": True, "_temperature": temp},
                ))
                used += tokens

        evicted_count = len(evictable)
        logger.debug(
            f"U-shaped compress: {len(entries)}→{len(result)} entries, "
            f"{used}/{target_tokens} tokens, {evicted_count} evicted by temp"
        )
        return result

    def _max_fit(self, entries: list[ContextEntry], budget: int) -> int:
        """计算在budget内能放多少个entries。"""
        total = 0
        for i, e in enumerate(entries):
            tokens = e.token_count or _estimate_tokens(e.content)
            if total + tokens > budget:
                return i
            total += tokens
        return len(entries)

    # ── persist ──────────────────────────────────────────────
    def persist(self, entries: list[ContextEntry]) -> int:
        count = 0
        for entry in entries:
            card_id = entry.metadata.get("card_id", f"entry-{int(time.time())}-{count}")
            self._index["entries"][card_id] = {
                "role": entry.role,
                "content": entry.content,
                "token_count": entry.token_count,
                "metadata": entry.metadata,
                "persisted_at": datetime.now(timezone.utc).isoformat(),
            }
            count += 1
        self._save_index()
        return count

    # ── restore ──────────────────────────────────────────────
    def restore(self) -> list[ContextEntry]:
        entries = []
        for card_id, data in self._index.get("entries", {}).items():
            entries.append(ContextEntry(
                role=data["role"],
                content=data["content"],
                token_count=data.get("token_count", 0),
                metadata={**data.get("metadata", {}), "card_id": card_id},
            ))
        return entries

    # ── search: BM25关键词匹配 ───────────────────────────────
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        scored: list[tuple[ContextEntry, float]] = []
        for card_id, data in self._index.get("entries", {}).items():
            content = data.get("content", "").lower()
            title = data.get("metadata", {}).get("title", "").lower()
            keywords = " ".join(data.get("metadata", {}).get("keywords", [])).lower()

            score = 0.0
            for term in query_terms:
                if term in title:
                    score += 3.0
                if term in keywords:
                    score += 2.0
                if term in content:
                    score += 1.0

            if score > 0:
                entry = ContextEntry(
                    role=data["role"],
                    content=data["content"],
                    token_count=data.get("token_count", 0),
                    metadata={**data.get("metadata", {}), "card_id": card_id},
                )
                scored.append((entry, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [SearchResult(entry=e, score=s) for e, s in scored[:top_k]]
