"""bias_monitor.py — 自动化偏差监控

阿瑞斯天启：验证器自身可能有偏差（自动化偏差累积）。
核心思路：记录每次验证的判决，按来源统计偏差率。

偏差定义：
  - 偏差率 = |通过率 - 0.5| × 2.0
  - 0.0 = 完全平衡（50/50）
  - 1.0 = 完全偏向（100%一方）

设计原则：
  - 独立模块，不修改engine.py
  - JSON持久化到 ~/.openllm/bias_state.json
  - 线程安全（threading.Lock）
"""

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("openllm.bias_monitor")

# 持久化路径
_BIAS_STATE_DIR = Path.home() / ".openllm"
_BIAS_STATE_FILE = _BIAS_STATE_DIR / "bias_state.json"


@dataclass
class VerificationRecord:
    """单次验证记录。"""
    claim: str
    verdict: bool
    source: str
    timestamp: float = field(default_factory=time.time)


class BiasMonitor:
    """自动化偏差监控器。

    用法：
        monitor = BiasMonitor()
        monitor.record("地球是圆的", True, "rule_validator")
        monitor.record("太阳是方的", False, "rule_validator")
        bias = monitor.get_bias("rule_validator")
        print(f"偏差率: {bias:.3f}")
    """

    def __init__(self, state_path: Optional[Path] = None):
        """初始化偏差监控器。

        Args:
            state_path: 状态文件路径，默认 ~/.openllm/bias_state.json
        """
        self._state_path = state_path or _BIAS_STATE_FILE
        self._lock = threading.Lock()
        self._records: list[VerificationRecord] = []
        self._load()

    def record(self, claim: str, verdict: bool, source: str) -> None:
        """记录一次验证判决。

        Args:
            claim: 被验证的声明
            verdict: 验证结果（True=通过，False=拒绝）
            source: 验证来源标识（如 'rule_validator', 'llm_validator'）
        """
        rec = VerificationRecord(claim=claim, verdict=verdict, source=source)
        with self._lock:
            self._records.append(rec)
            self._save()

    def get_bias(self, source: str) -> float:
        """计算某来源的偏差率。

        偏差率 = |通过率 - 0.5| × 2.0
        0.0 = 完全平衡, 1.0 = 完全偏向一方

        Args:
            source: 验证来源标识

        Returns:
            偏差率（0.0 ~ 1.0），无记录时返回 0.0
        """
        with self._lock:
            source_records = [r for r in self._records if r.source == source]
        if not source_records:
            return 0.0

        true_count = sum(1 for r in source_records if r.verdict)
        total = len(source_records)
        approval_rate = true_count / total
        return abs(approval_rate - 0.5) * 2.0

    def is_biased(self, source: str, threshold: float = 0.3) -> bool:
        """判断某来源是否有显著偏差。

        Args:
            source: 验证来源标识
            threshold: 偏差阈值（默认0.3，超过则认为有显著偏差）

        Returns:
            True 表示有显著偏差
        """
        return self.get_bias(source) > threshold

    def summary(self) -> dict:
        """偏差统计摘要。

        Returns:
            包含各来源统计和全局统计的字典
        """
        with self._lock:
            records = list(self._records)

        if not records:
            return {
                "total_records": 0,
                "sources": {},
            }

        sources: dict[str, dict] = {}
        for rec in records:
            s = rec.source
            if s not in sources:
                sources[s] = {"total": 0, "true_count": 0, "false_count": 0}
            sources[s]["total"] += 1
            if rec.verdict:
                sources[s]["true_count"] += 1
            else:
                sources[s]["false_count"] += 1

        # 计算各来源偏差率
        for s, stats in sources.items():
            approval_rate = stats["true_count"] / stats["total"] if stats["total"] > 0 else 0.0
            stats["approval_rate"] = round(approval_rate, 4)
            stats["bias_rate"] = round(abs(approval_rate - 0.5) * 2.0, 4)
            stats["is_biased"] = stats["bias_rate"] > 0.3

        # 全局统计
        total = len(records)
        true_count = sum(1 for r in records if r.verdict)
        global_approval = true_count / total if total > 0 else 0.0

        return {
            "total_records": total,
            "global_approval_rate": round(global_approval, 4),
            "global_bias_rate": round(abs(global_approval - 0.5) * 2.0, 4),
            "sources": sources,
        }

    def clear(self) -> None:
        """清空所有记录并删除持久化文件。"""
        with self._lock:
            self._records.clear()
            if self._state_path.exists():
                self._state_path.unlink()
        logger.info("BiasMonitor: all records cleared")

    def _save(self) -> None:
        """持久化到JSON文件。"""
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            data = [asdict(r) for r in self._records]
            self._state_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("BiasMonitor: 保存状态失败: %s", e)

    def _load(self) -> None:
        """从JSON文件加载状态。"""
        if not self._state_path.exists():
            return
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            self._records = [VerificationRecord(**r) for r in data]
        except Exception as e:
            logger.warning("BiasMonitor: 加载状态失败，从空状态开始: %s", e)
            self._records = []
