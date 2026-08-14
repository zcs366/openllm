"""
DeltaCapsuleProvider — Δ胶囊记忆provider

读取 ~/.openllm/capsules/ 下的 v06（TextCapsule）/v07（DeltaCapsule）文件。
v06 = 可读胶囊（decisions/insights/outputs/unresolved），v07 = 语义Δ向量。

检索策略（匠石法则：纯关键词匹配，不依赖embedding，零额外依赖）：
1. 按mtime倒序扫描最近N个v06文件（新胶囊优先，1222个全扫太慢）
2. jieba分词提取query关键词（memory_bus.tokenize，jieba不可用时降级split）
3. 对每个胶囊的 insights + decisions.summary 文本做关键词重叠计分
4. 按score降序，返回top_k条MemoryRecord

写入策略：
- store() 通过 openllm-memory 的 MemoryOS.write() 写入 TextCapsule
  → 自动生成 v06 + v07 + 共振检测（MemoryOS内部完成）
"""

import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..memory_bus import (
    MemoryProvider, MemoryRecord, WriteRequest, WriteResult, Query, tokenize
)

logger = logging.getLogger("openllm.providers.delta_capsule")

CAPSULE_DIR = Path.home() / ".openllm" / "capsules"

# openllm-memory MemoryOS 所在仓库（任务指定路径）
# 注意：pip安装的openllm_memory(v1.1.1, /mnt/i)没有MemoryOS/core.py，
# 且其路径优先于~/projects——因此需要显式路径回退加载。
_OPENLLM_MEMORY_SRC = Path.home() / "projects" / "openllm-memory" / "src"


def _import_memory_core():
    """导入openllm_memory.core（含MemoryOS/TextCapsule）。

    优先正常import；失败则把~/projects/openllm-memory/src加入sys.path重试。
    返回core模块；不可用时抛ImportError。
    """
    try:
        from openllm_memory.core import MemoryOS, TextCapsule  # noqa: F401
        import openllm_memory.core as _core
        return _core
    except ImportError:
        pass
    # 回退：从~/projects/openllm-memory/src直接按文件加载core.py
    # （pip版openllm_memory已缓存在sys.modules且__path__指向/mnt/i，
    #  单纯改sys.path无效——必须用spec_from_file_location绕过包缓存）
    import importlib.util
    core_path = _OPENLLM_MEMORY_SRC / "openllm_memory" / "core.py"
    if not core_path.exists():
        raise ImportError(f"openllm_memory.core不可用且回退路径不存在: {core_path}")
    spec = importlib.util.spec_from_file_location("openllm_memory_core_fallback", core_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载openllm_memory.core回退模块: {core_path}")
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    return core

# 扫描窗口：按mtime倒序只看最近N个v06文件
# 通电Phase 1b(2026-08-14)：从200提升到2000，覆盖全量1586个胶囊
_SCAN_RECENT_N = 2000

# 每条content最大长度（防止超长胶囊撑爆token预算）
_MAX_CONTENT_LEN = 300


class DeltaCapsuleProvider:
    """Δ胶囊记忆provider（openllm-memory MemoryOS的MemoryBus适配器）"""

    def __init__(self, capsule_dir: Optional[Path] = None):
        self._dir = Path(capsule_dir) if capsule_dir else CAPSULE_DIR
        self._os = None  # MemoryOS懒加载（写入时才需要）

    @property
    def name(self) -> str:
        return "delta_capsule"

    @property
    def priority(self) -> int:
        return 15  # 在 causal(20) 之前、recall(10) 之后；unified(40)兜底之前

    @property
    def input_schema(self) -> set:
        """search()使用的Query字段"""
        return {"text", "top_k", "min_importance", "record_types", "sources", "token_budget"}

    @property
    def output_schema(self) -> set:
        """search()产出的MemoryRecord字段"""
        return {"record_id", "content", "source", "record_type",
                "importance", "temperature", "trust_level",
                "tags", "timestamp", "context", "score", "provider"}

    # ═══ 检索 ═══

    def search(self, query: Query) -> List[MemoryRecord]:
        """从Δ胶囊中检索——关键词匹配 insights/decisions"""
        if not self._dir.exists():
            return []

        query_keywords = tokenize(query.text)
        if not query_keywords:
            return []

        candidates: List[tuple] = []  # (score, snippet, capsule_meta)

        # 按mtime倒序取最近N个v06文件
        v06_files = sorted(
            self._dir.glob("v06_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:_SCAN_RECENT_N]

        for path in v06_files:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            session_id = data.get("session_id", path.stem)
            timestamp = data.get("timestamp", path.stat().st_mtime)
            provenance = data.get("provenance") or {}

            # 收集可匹配片段：(text, record_type)
            snippets: List[tuple] = []
            for ins in data.get("insights", []):
                text = ins if isinstance(ins, str) else str(ins)
                if text.strip():
                    snippets.append((text, "insight"))
            for dec in data.get("decisions", []):
                if isinstance(dec, dict):
                    text = dec.get("summary", "") or json.dumps(dec, ensure_ascii=False)[:200]
                else:
                    text = str(dec)
                if text.strip():
                    snippets.append((text, "decision"))

            for text, record_type in snippets:
                text_lower = text.lower()
                text_tokens = tokenize(text)
                overlap = query_keywords & text_tokens
                # 子串兜底：中文长词jieba切分不一致时，直接子串命中也算
                # 修复（T-ISA-5）：overlap必然substr命中，用集合去重避免双重计数
                # （旧实现 len(overlap)+substr_hits 可致score>1.0虚高）
                matched = overlap | {kw for kw in query_keywords if kw in text_lower}
                hit = len(matched)
                if hit == 0:
                    continue
                score = hit / max(len(query_keywords), 1)
                candidates.append((score, text, {
                    "session_id": session_id,
                    "record_type": record_type,
                    "timestamp": timestamp,
                    "provenance": provenance,
                }))

        if not candidates:
            return []

        candidates.sort(key=lambda x: x[0], reverse=True)

        # 多样性控制（T-ISA-5）：同一session最多2条，防止单文件snippets刷屏
        # （同一capsule文件的多个insights/decisions分数相同，会挤占其他provider位置）
        session_counts: Dict[str, int] = {}
        records = []
        for score, text, meta in candidates[:query.top_k * 3]:
            session_id = meta.get("session_id", "?")
            if session_counts.get(session_id, 0) >= 2:
                continue
            session_counts[session_id] = session_counts.get(session_id, 0) + 1

            record_type = meta["record_type"]
            if query.record_types and record_type not in query.record_types:
                session_counts[session_id] -= 1  # 过滤不算占用
                continue

            record = MemoryRecord(
                record_id=f"capsule:{meta['session_id']}:{hash(text) & 0xffffff:x}",
                content=text[:_MAX_CONTENT_LEN],
                source="capsule",
                record_type=record_type,
                importance=min(0.5 + score * 0.5, 1.0),
                temperature=0.6,  # Δ胶囊=沉积岩，恒定温度（不随时间衰减）
                trust_level="internal",
                tags=[meta["session_id"]],
                timestamp=meta["timestamp"],
                context={
                    "session_id": meta["session_id"],
                    "provenance": meta["provenance"],
                    "capsule_dir": str(self._dir),
                },
                score=round(score, 4),
                provider="delta_capsule",
            )

            if query.min_importance > 0 and record.importance < query.min_importance:
                continue
            if query.sources and record.source not in query.sources:
                continue

            records.append(record)

        return records[:query.top_k]

    # ═══ 写入 ═══

    def _ensure_memory_os(self):
        """懒加载MemoryOS（openllm-memory包的core.py）"""
        if self._os is None:
            core = _import_memory_core()
            self._os = core.MemoryOS(capsule_dir=str(self._dir))
        return self._os

    def store(self, request: WriteRequest) -> WriteResult:
        """
        Δ胶囊写入——构造TextCapsule，通过MemoryOS.write()落盘。

        MemoryOS.write()自动完成：v06落盘 + v07语义向量 + 共振检测。
        metadata可传入 insights/decisions/outputs/unresolved/provenance。
        """
        try:
            core = _import_memory_core()
            TextCapsule = core.TextCapsule
        except ImportError as e:
            return WriteResult(success=False, reason=f"openllm-memory不可用: {e}")

        try:
            md = request.metadata or {}
            session_id = request.session_id or f"bus-{int(time.time())}"

            # content映射：优先用metadata显式字段，否则content作为单条insight
            insights = md.get("insights") or ([request.content] if request.content else [])
            decisions = md.get("decisions") or []
            if request.record_type == "decision" and request.content:
                decisions = decisions + [{"summary": request.content,
                                          "agent": request.source}]

            capsule = TextCapsule(
                session_id=session_id,
                decisions=decisions,
                outputs=md.get("outputs", []),
                insights=insights,
                unresolved=md.get("unresolved", []),
                provenance=md.get("provenance", {
                    "soul_id": md.get("soul_id", ""),
                    "agent_id": request.source,
                    "session_id": session_id,
                }),
            )

            mem_os = self._ensure_memory_os()
            v06_path = mem_os.write(capsule)
            resonance = len(getattr(mem_os, "last_resonance_events", []) or [])

            return WriteResult(
                success=True,
                record_id=f"capsule:{session_id}",
                provider="delta_capsule",
                reason=f"resonance_events={resonance}" if resonance else "",
            )
        except Exception as e:
            logger.error(f"DeltaCapsule store error: {e}")
            return WriteResult(success=False, reason=f"delta_capsule store error: {e}")

    # ═══ 统计/健康 ═══

    def count(self) -> int:
        """胶囊数量（以v06可读胶囊为准）"""
        if not self._dir.exists():
            return 0
        return sum(1 for _ in self._dir.glob("v06_*.json"))

    def health(self) -> Dict[str, Any]:
        """健康检查"""
        if not self._dir.exists():
            return {"status": "degraded", "capsule_dir": str(self._dir),
                    "capsules_v06": 0, "capsules_v07": 0}
        v06 = sum(1 for _ in self._dir.glob("v06_*.json"))
        v07 = sum(1 for _ in self._dir.glob("v07_*.json"))
        try:
            import openllm_memory  # noqa: F401
            mem_available = True
        except ImportError:
            mem_available = False
        return {
            "status": "ok",
            "capsule_dir": str(self._dir),
            "capsules_v06": v06,
            "capsules_v07": v07,
            "openllm_memory_available": mem_available,
        }
