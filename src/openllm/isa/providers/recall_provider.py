"""
RecallProvider — RECALL时间线日志provider

读取 ~/.hermes/jiak/RECALL.jsonl 中的时间线记录。
支持：FTS5关键词搜索、时间范围过滤、类型过滤。

检索策略：
1. 加载RECALL.jsonl（最近N天）
2. 对query做关键词匹配（简单文本搜索）
3. 按时间+相关性排序

时间戳纪律（2026-10-01 军师拍板，勿回退）：
- 取值链 `timestamp → ts → _timestamp → _written_at`，每个候选过 `_coerce_ts`
  （数值 / 数值字符串 / ISO8601 含 'Z' 与纯日期），规整后就地写回 epoch float。
- **只有"时间已知且早于 cutoff"才过滤**；时间缺失/不可解析的记录保留，
  计入 `last_load_stats['missing_ts']`。旧写法"缺失→0→被窗口丢掉"曾让
  177 条真实记录里的 173 条隐形（且 `ts` 字段整条不读），是本次 P0 的病根。
- 单条脏数据只影响本条：坏 JSON / 类型异常逐条跳过并计数，绝不击穿整条通道。
- 统计只在实例首次加载时 warning 一次（`_load_records` 每 10s 缓存过期会重读，
  逐次 warning 会变成新噪声），其后降 debug。
"""

import json
import time
import math
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..memory_bus import (
    MemoryProvider, MemoryRecord, WriteRequest, WriteResult, Query, tokenize
)
# 时间戳单一判定源（审计 P1-5，2026-10-01）：实现收敛到 openllm.isa.timeutil，
# 本模块保留旧名兼容别名——"增强不替代"纪律（钉子测试 import 旧名必须继续绿）。
from ..timeutil import TS_FIELDS as _TS_FIELDS, coerce_ts, pick_ts

_coerce_ts = coerce_ts
_pick_ts = pick_ts

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
        self.last_load_stats: Dict[str, int] = {}
        self._warned_stats: bool = False

    @property
    def name(self) -> str:
        return "recall"

    @property
    def priority(self) -> int:
        return 10

    @property
    def input_schema(self) -> set:
        """search()使用的Query字段"""
        return {"text", "top_k", "min_importance", "record_types", "token_budget"}

    @property
    def output_schema(self) -> set:
        """search()产出的MemoryRecord字段"""
        return {"record_id", "content", "source", "record_type",
                "importance", "temperature", "trust_level",
                "tags", "timestamp", "context", "score", "provider"}

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
            # 时间衰减：越新分数越高（取值链与 _load_records 同源，2026-10-01）
            ts = _pick_ts(rec)[0]
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
                # 写侧规范字段（成市拍板 2026-10-01）：ts = epoch float。
                # 本路径已走门房 validate_and_append；字段名统一，门房对
                # ts/timestamp 均豁免于内容哈希（_EXCLUDE_FIELDS）。
                "ts": time.time(),
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
            "load_stats": dict(self.last_load_stats),
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
        n_ok = 0          # 时间已知且解析成功（无论是否保留）
        n_bad = 0         # 单条解析失败（坏JSON/类型异常）→ 跳过
        n_missing = 0     # 时间字段缺失/不可解析 → 保留并计入 missing_ts

        try:
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if not isinstance(rec, dict):
                            n_bad += 1
                            continue
                        ts, ts_field = _pick_ts(rec)
                        ts_known = ts_field is not None and ts > 0
                        if ts_known:
                            n_ok += 1
                            # 就地规整成 epoch float：下游（search/_compute_temperature）
                            # 不再碰到原始字符串/异常类型
                            rec[ts_field] = ts
                        else:
                            n_missing += 1
                            if ts_field is not None:
                                # 不可解析的脏值也规整为0：缺失≠丢弃，但下游
                                # (search 的 `ts > 0`) 不得再拿字符串去比数字
                                rec[ts_field] = 0.0
                        # 语义（军师拍板 2026-10-01）：只有"时间已知且早于
                        # cutoff"才过滤；缺失/不可解析不得静默丢弃
                        if (not ts_known) or ts >= cutoff:
                            records.append(rec)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        n_bad += 1
                        continue
        except OSError as e:
            logger.error(f"Failed to load RECALL: {e}")
            return []

        # 单次汇总（不逐条刷日志）：脏数据只影响本条，不击穿整条通道
        self.last_load_stats = {
            "total_lines": n_ok + n_missing + n_bad,
            "ok": n_ok,
            "skipped_bad": n_bad,
            "missing_ts": n_missing,
            "kept": len(records),
        }
        if n_bad or n_missing:
            # 只在首次加载时告警一次（2026-10-01 军师：_load_records 每 10 秒缓存过期
            # 就会重读，逐次 warning 会变成新的日志噪声），其后降为 debug。
            log = logger.warning if not self._warned_stats else logger.debug
            self._warned_stats = True
            log(
                "RECALL %s: 保留%d条（时间ok: %d, 解析失败跳过: %d, "
                "缺失/不可解析时间戳保留: %d）",
                self._path, len(records), n_ok, n_bad, n_missing)

        self._records = records
        self._loaded_at = now
        return records

    def _compute_temperature(self, timestamp: float) -> float:
        if timestamp == 0:
            return 0.5
        age_hours = (time.time() - timestamp) / 3600
        return math.exp(-0.029 * age_hours)
