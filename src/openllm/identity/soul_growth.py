"""
SoulGrowthLedger — SOUL层可增长成长容器
================================================================

回答的问题：固化产物写入哪里？
  Soul（soul.py）是只读宪法层——身份锚点，无save方法。
  SoulGrowth是可增长的事后沉淀——固化成长产物的承载容器。
  两者分开：Soul锚定不变身份，SoulGrowth记录成长轨迹。

设计原则（与 soul.py / auto_causal_writer.py 同族）：
  - append-only JSONL（每条一行JSON，只追加不修改）
  - 零LLM调用，纯确定性IO
  - 与identity层风格一致（dataclass/中文docstring）
  - 定位：承接 ConsolidationScorer 固化成功的bullet，
    作为SOUL层的慢层沉淀

用法：
    ledger = SoulGrowthLedger()
    ledger.record(bullet_id="b1", content="WSL中I盘=/mnt/i/",
                  origin="consolidation", score=29.0)
    recent = ledger.load(recent_n=50)
    info = ledger.stats()

上下文格言：
  固化非终点，成长无止境。
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("openllm.soul_growth")


@dataclass
class GrowthRecord:
    """单条成长记录——固化产物在SOUL层的落盘结构。"""
    bullet_id: str
    content: str
    origin: str = "consolidation"        # 来源：consolidation / manual / review
    score: float = 0.0                   # 固化时的Score值
    provenance_trust: str = ""           # 来源信任级别
    ts: float = field(default_factory=time.time)
    rollback_reason: str = ""            # 非空=已回滚，记录回滚原因

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SoulGrowthLedger:
    """SOUL层成长库——append-only JSONL，承载固化成长产物。

    与 Soul（只读锚点）分开：SoulGrowth是可增长的事后沉淀。

    append-only纪律：
      - 打开文件时用 'a' 追加模式
      - 每条记录原子写入（单条JSON + 换行）
      - load() 时从文件末尾读取最近N条
      - 永不修改/删除已有记录（回滚=标记rollback_reason，不删行）
    """

    def __init__(self, growth_path: Optional[Path] = None) -> None:
        """
        Args:
            growth_path: JSONL文件路径，默认 ~/.openllm/identity/growth.jsonl
        """
        if growth_path is None:
            growth_path = Path.home() / ".openllm" / "identity" / "growth.jsonl"
        self._path = Path(growth_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        bullet_id: str,
        content: str,
        origin: str = "consolidation",
        score: float = 0.0,
        provenance_trust: str = "",
        ts: Optional[float] = None,
    ) -> dict:
        """追加一条成长记录（append-only JSONL）。

        Args:
            bullet_id: 固化条目ID（关联 ConsolidationScorer）
            content: 固化内容
            origin: 来源标识（consolidation/manual/review）
            score: 固化时的Score值
            provenance_trust: 来源信任级别
            ts: 时间戳，默认当前时间

        Returns:
            写入的记录字典
        """
        rec = GrowthRecord(
            bullet_id=bullet_id,
            content=content,
            origin=origin,
            score=score,
            provenance_trust=provenance_trust,
            ts=ts or time.time(),
        )
        line = json.dumps(rec.to_dict(), ensure_ascii=False)
        # append-only: 用 'a' 模式，每条原子写入
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        logger.info(
            "SoulGrowth: bullet_id=%s origin=%s score=%.1f",
            bullet_id, origin, score,
        )
        return rec.to_dict()

    def load(self, recent_n: int = 50) -> list:
        """读取最近N条成长记录。

        Args:
            recent_n: 返回最近N条，0=全部

        Returns:
            成长记录列表（按时间升序）
        """
        if not self._path.exists():
            return []
        records: list = []
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("SoulGrowth: 跳过损坏行")
        except OSError as e:
            logger.warning("SoulGrowth: 读取失败: %s", e)
            return []
        if recent_n > 0:
            records = records[-recent_n:]
        return records

    def mark_rollback(self, bullet_id: str, reason: str) -> bool:
        """标记回滚：追加一条带rollback_reason的记录（append-only）。

        不修改/删除原记录，追加新行表示回滚。

        Returns:
            True if a record was found and rollback appended.
        """
        records = self.load(recent_n=0)
        target = None
        for r in records:
            if r.get("bullet_id") == bullet_id and not r.get("rollback_reason"):
                target = r
        if target is None:
            return False
        target["rollback_reason"] = reason
        target["ts"] = time.time()
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(target, ensure_ascii=False) + "\n")
        logger.info("SoulGrowth: rollback bullet_id=%s reason=%s", bullet_id, reason)
        return True

    def stats(self) -> dict:
        """返回成长库统计信息。"""
        records = self.load(recent_n=0)
        consolidation_n = sum(
            1 for r in records if r.get("origin") == "consolidation"
        )
        latest_content = records[-1]["content"][:100] if records else ""
        return {
            "total_records": len(records),
            "consolidation_records": consolidation_n,
            "latest_content_preview": latest_content,
            "growth_path": str(self._path),
        }
