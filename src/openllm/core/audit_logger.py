"""audit_logger.py — 验证日志可审计

阿瑞斯天启：每次验证的工具输出+LLM判断过程必须落盘，人类可追溯。

日志格式：JSONL追加写入 ~/.openllm/audit_log.jsonl
每条记录包含：timestamp, claim, method, result, logger_id
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Optional


class AuditLogger:
    """验证日志记录器——每次验证过程落盘，人类可追溯。"""

    DEFAULT_LOG_PATH = os.path.expanduser("~/.openllm/audit_log.jsonl")
    LOGGER_ID = "openllm.audit_logger"

    def __init__(self, log_path: Optional[str] = None):
        """初始化日志记录器。

        Args:
            log_path: 日志文件路径，默认 ~/.openllm/audit_log.jsonl
        """
        self.log_path = log_path or self.DEFAULT_LOG_PATH
        # 确保目录存在
        log_dir = os.path.dirname(self.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def log_verification(
        self,
        claim: str,
        method: str,
        result: dict,
        timestamp: Optional[str] = None,
    ) -> None:
        """记录一次验证事件。

        Args:
            claim: 被验证的声明
            method: 验证方法（如 'rule', 'llm', 'combined'）
            result: 验证结果字典（必须包含 'passed' 键）
            timestamp: ISO格式时间戳，默认自动生成
        """
        record = {
            "timestamp": timestamp or self._now_iso(),
            "claim": claim,
            "method": method,
            "result": result,
            "logger_id": self.LOGGER_ID,
        }
        self._append(record)

    def get_logs(self, limit: int = 100) -> list[dict]:
        """获取最近N条日志。

        Args:
            limit: 返回的最大记录数，默认100

        Returns:
            最近的日志记录列表（按时间正序）
        """
        if not os.path.exists(self.log_path):
            return []
        records = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        # 返回最后N条，保持正序
        return records[-limit:]

    def export(self, path: str) -> None:
        """导出日志为JSON文件。

        Args:
            path: 导出文件路径
        """
        logs = self.get_logs(limit=999999)
        export_dir = os.path.dirname(path)
        if export_dir:
            os.makedirs(export_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)

    # ── 内部方法 ──────────────────────────────────────

    def _append(self, record: dict) -> None:
        """追加一条记录到JSONL文件。"""
        line = json.dumps(record, ensure_ascii=False)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    @staticmethod
    def _now_iso() -> str:
        """返回当前UTC时间的ISO格式字符串。"""
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
