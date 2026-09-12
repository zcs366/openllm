"""
context_audit_logger.py — 上下文审计日志器

记录每次 SA→子agent 委派的核心上下文状态，生成可检索的 append-only JSONL 数据集。
用于测上下文策展质量（prefix稳定/压缩时机/注入顺序）时的数据支撑。

与 execution_recorder 的区别：
  execution_recorder 记录的是单次工具调用（tool_name + args + status），
  context_audit_logger 记录的是 SA→子agent 委派的上下文全貌（注入内容/目标/结果/token用量）。
  前者是执行层，后者是策展层。

用法:
    from openllm.isa.context_audit_logger import ContextAuditLogger
    logger = ContextAuditLogger()
    record = logger.log_delegation(
        target="subagent-A",
        objective="搜索arXiv论文",
        context_summary="用户要求搜索attention mechanics相关论文",
        injected="system_prompt + 论文检索指令",
        result_summary="找到5篇相关论文",
        started_at=1000.0,
        finished_at=1005.0,
        tokens=1200,
        tags=["research", "arxiv"],
    )
    recent = logger.load(recent_n=10)
    stats = logger.stats()
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("openllm.context_audit_logger")

# ── 路径常量 ──────────────────────────────────────────────────

DEFAULT_LOG_DIR = Path.home() / ".openllm"
DEFAULT_LOG_FILE = "context_audit.jsonl"


class ContextAuditLogger:
    """上下文审计日志器 —— SA→子agent 委派的 append-only JSONL 记录。

    与 execution_recorder 的区别：execution_recorder 记录单次工具调用
    （tool_name + args + status），本记录器记录委派的上下文全貌
    （注入内容/目标/结果/token/耗时），是策展层而非执行层。
    """

    def __init__(self, log_path: Optional[Path] = None) -> None:
        """初始化日志器。

        Args:
            log_path: 日志文件路径。为 None 时用 ~/.openllm/context_audit.jsonl。
        """
        self._log_path = log_path or (DEFAULT_LOG_DIR / DEFAULT_LOG_FILE)

    def log_delegation(
        self,
        *,
        target: str,
        objective: str,
        context_summary: str,
        injected: str,
        result_summary: str,
        started_at: float,
        finished_at: float,
        tokens: int = 0,
        tags: Optional[list] = None,
    ) -> dict:
        """记录一次 SA→子agent 委派。

        追加写入一条 JSONL 记录，原子单条写入，不覆盖历史。

        Args:
            target: 子agent 名称/标识
            objective: 委派目标
            context_summary: 上下文摘要（发给子agent的上下文概述）
            injected: 注入内容描述（实际注入到子agent prompt 的内容）
            result_summary: 结果摘要（子agent返回的结果概述）
            started_at: 开始时间戳（time.time()）
            finished_at: 结束时间戳（time.time()）
            tokens: token 消耗量
            tags: 可选标签列表

        Returns:
            写入的完整记录 dict。
        """
        duration_s = round(finished_at - started_at, 3)
        record: Dict[str, Any] = {
            "ts": time.time(),
            "target": target,
            "objective": objective,
            "context_summary": context_summary,
            "injected": injected,
            "result_summary": result_summary,
            "duration_s": duration_s,
            "tokens": tokens,
            "tags": tags or [],
        }
        self._append_record(record)
        return record

    def load(self, recent_n: int = 50) -> List[dict]:
        """读取最近 N 条审计记录。

        Args:
            recent_n: 返回记录数上限。

        Returns:
            最近 N 条有效记录列表（文件不存在返回空列表，损坏行跳过）。
        """
        if not self._log_path.exists():
            return []
        records: List[dict] = []
        try:
            with open(self._log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception:
            return []
        return records[-recent_n:]

    def stats(self) -> dict:
        """统计信息：总数、按 target 分组、平均耗时、token 总和。

        Returns:
            包含 total/targets/avg_duration_s/total_tokens 的 dict。
        """
        all_records = self._load_all()
        if not all_records:
            return {
                "total": 0,
                "targets": {},
                "avg_duration_s": 0.0,
                "total_tokens": 0,
            }

        target_count: Dict[str, int] = defaultdict(int)
        target_duration: Dict[str, float] = defaultdict(float)
        total_duration = 0.0
        total_tokens = 0

        for rec in all_records:
            t = rec.get("target", "unknown")
            d = rec.get("duration_s", 0.0)
            target_count[t] += 1
            target_duration[t] += d
            total_duration += d
            total_tokens += rec.get("tokens", 0)

        targets = {}
        for t in target_count:
            cnt = target_count[t]
            targets[t] = {
                "count": cnt,
                "avg_duration_s": round(target_duration[t] / cnt, 3),
            }

        return {
            "total": len(all_records),
            "targets": targets,
            "avg_duration_s": round(total_duration / len(all_records), 3),
            "total_tokens": total_tokens,
        }

    # ── 内部方法 ────────────────────────────────────────────────

    def _append_record(self, record: dict) -> None:
        """追加一条记录到 JSONL 文件。自动创建目录。"""
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _load_all(self) -> List[dict]:
        """加载全部记录（内部用，stats 统计需要全量）。"""
        if not self._log_path.exists():
            return []
        records: List[dict] = []
        try:
            with open(self._log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception:
            return []
        return records
