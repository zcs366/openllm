"""checkpoint.py — TaskCheckpoint: 最小任务存档模块。

保存/恢复任务执行状态，包含：
  - history:     对话历史 (list[dict])
  - tool_calls:  工具调用序列 (list[dict])
  - results:     中间结果 (list[dict])
  - timestamp:   存档时间 (ISO 8601)

设计约束：
  - 独立模块，不修改 engine.py
  - JSON 序列化，支持 datetime
  - 线程安全（单文件原子写入）
"""

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("openllm.checkpoint")


def _json_serial(obj: Any) -> str:
    """JSON serializer for objects not serializable by default json code."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def _default_state() -> dict:
    """Return an empty checkpoint state with all required keys."""
    return {
        "history": [],
        "tool_calls": [],
        "results": [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


class TaskCheckpoint:
    """最小任务存档——保存/恢复任务执行状态到 JSON 文件。

    Usage:
        ckpt = TaskCheckpoint()
        ckpt.save(state, "/tmp/task_001.json")
        restored = ckpt.load("/tmp/task_001.json")
    """

    def __init__(self, backup: bool = True):
        """
        Args:
            backup: save() 时是否保留备份文件（原文件更名为 .bak）。
        """
        self.backup = backup

    def save(self, state: dict, path: str) -> None:
        """序列化 state 到 JSON 文件（原子写入）。

        Args:
            state: 必须包含 history/tool_calls/results/timestamp 字段，
                   缺失字段自动补默认值。
            path:  目标文件路径。
        """
        # 补默认值
        merged = _default_state()
        merged.update(state)

        # 确保 timestamp 为 ISO 字符串
        ts = merged.get("timestamp")
        if isinstance(ts, datetime):
            merged["timestamp"] = ts.isoformat()
        elif ts is None:
            merged["timestamp"] = datetime.now(timezone.utc).isoformat()

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        # 原子写入：先写临时文件再 rename
        fd, tmp_path = tempfile.mkstemp(
            dir=str(target.parent), suffix=".tmp", prefix=".ckpt_"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(
                    merged, f, ensure_ascii=False, indent=2, default=_json_serial
                )
                f.write("\n")

            # 可选备份
            if self.backup and target.exists():
                bak_path = target.with_suffix(target.suffix + ".bak")
                os.replace(str(target), str(bak_path))

            os.replace(tmp_path, str(target))
            logger.info(f"checkpoint saved: {target} "
                        f"({len(merged.get('history', []))} msgs)")
        except Exception:
            # 清理临时文件
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def load(self, path: str) -> dict:
        """从 JSON 文件恢复 state。

        Args:
            path: checkpoint 文件路径。

        Returns:
            state dict (history/tool_calls/results/timestamp)。

        Raises:
            FileNotFoundError: 文件不存在。
            json.JSONDecodeError: 文件内容非法。
        """
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"checkpoint not found: {target}")

        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 确保四个关键字段存在
        merged = _default_state()
        merged.update(data)

        logger.info(f"checkpoint loaded: {target} "
                     f"({len(merged.get('history', []))} msgs)")
        return merged
