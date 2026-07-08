"""
MistakeLedger — Agent的犯错记录本

M2Note范式的openLLM落地实现：
- JSONL追加写入（原子操作，不覆盖）
- 按关键词/时间范围查询
- 统计分析（按severity/agent分布）

用户面术语："经验本"/"犯错记录"
禁止暴露：Ledger/Schema/append等技术词

铁律：
- ①不搬VLM错误定义（Agent特有错误：tool_call失败/用户纠正/IO-S扣分）
- ②无数据不校准（Cobb-Douglas远期）
- ③不暴露技术概念给用户
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import fcntl as _fcntl
    _HAS_FCNTL = True
except ImportError:
    _fcntl = None  # type: ignore
    _HAS_FCNTL = False  # Windows: 退化为无锁


class MistakeLedger:
    """Agent的犯错记录本——存储、查询、统计错误记录"""

    def __init__(self, path: str = "~/.hermes/openllm/mistakes.jsonl"):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        what: str,
        why: str,
        agent: str = "",
        severity: str = "medium",
        session_id: str = "",
        tags: Optional[list[str]] = None,
    ) -> dict:
        """
        追加一条错误记录。

        Args:
            what: 发生了什么错误（一句话描述）
            why: 为什么是错误（期望与实际的差距）
            agent: 哪个Agent/模块犯的错
            severity: 错误严重程度 (low/medium/high/critical)
            session_id: 发生错误的会话ID
            tags: 可选标签列表

        Returns:
            完整的错误记录（含自动生成的timestamp和id）
        """
        if not what or not what.strip():
            raise ValueError("what不能为空")
        if not why or not why.strip():
            raise ValueError("why不能为空")
        if severity not in ("low", "medium", "high", "critical"):
            raise ValueError(f"severity必须是low/medium/high/critical，收到: {severity}")

        record = {
            "id": f"mistake-{int(time.time() * 1000)}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "what": what.strip(),
            "why": why.strip(),
            "agent": agent.strip(),
            "severity": severity,
            "session_id": session_id.strip(),
            "tags": tags or [],
        }

        # JSONL追加写入（文件锁保护多进程安全）
        with open(self.path, "a", encoding="utf-8") as f:
            if _HAS_FCNTL and _fcntl:
                _fcntl.flock(f, _fcntl.LOCK_EX)  # 排他锁
            try:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
            finally:
                if _HAS_FCNTL and _fcntl:
                    _fcntl.flock(f, _fcntl.LOCK_UN)  # 释放锁

        return record

    def query(
        self,
        keyword: str = "",
        since: str = "",
        severity: str = "",
        agent: str = "",
        limit: int = 10,
    ) -> list[dict]:
        """
        查询错误记录。

        Args:
            keyword: 关键词匹配（在what和why中搜索）
            since: ISO8601时间字符串，只返回此时间之后的记录
            severity: 按severity过滤
            agent: 按agent过滤
            limit: 最大返回条数

        Returns:
            匹配的记录列表（按时间倒序）
        """
        if not self.path.exists():
            return []

        records = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # 关键词过滤
                if keyword:
                    kw = keyword.lower()
                    if kw not in record.get("what", "").lower() and \
                       kw not in record.get("why", "").lower():
                        continue

                # 时间过滤
                if since:
                    if record.get("timestamp", "") < since:
                        continue

                # severity过滤
                if severity and record.get("severity") != severity:
                    continue

                # agent过滤
                if agent and record.get("agent") != agent:
                    continue

                records.append(record)

        # 按时间倒序，取limit条
        records.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
        return records[:limit]

    def get_stats(self, since: str = "") -> dict:
        """
        返回错误统计。

        Args:
            since: ISO8601时间字符串，只统计此时间之后的记录

        Returns:
            统计字典：总数/按severity分布/按agent分布/最近N条
        """
        if not self.path.exists():
            return {"total": 0, "by_severity": {}, "by_agent": {}, "recent": []}

        records = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if since and record.get("timestamp", "") < since:
                    continue
                records.append(record)

        by_severity = {}
        by_agent = {}
        for r in records:
            sev = r.get("severity", "unknown")
            by_severity[sev] = by_severity.get(sev, 0) + 1

            ag = r.get("agent", "unknown") or "unknown"
            by_agent[ag] = by_agent.get(ag, 0) + 1

        # 最近5条
        records.sort(key=lambda r: r.get("timestamp", ""), reverse=True)

        return {
            "total": len(records),
            "by_severity": by_severity,
            "by_agent": by_agent,
            "recent": records[:5],
        }

    def count(self) -> int:
        """返回总记录数"""
        if not self.path.exists():
            return 0
        count = 0
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count
