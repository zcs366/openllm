"""
execution_recorder.py — 工具执行记录器（CCL执行节点）

把每次工具执行写入RECALL.jsonl，闭合构成性因果环（CCL）的执行节点。
通过subprocess调用recall_append.py写入，失败绝不阻塞工具执行。

用法：
    from openllm.memory.execution_recorder import record_execution
    record_execution("read_file", {"path": "/tmp/test"}, status="ok",
                     duration_ms=12.5, result_summary="文件内容...")
"""

import atexit
import json
import logging
import os
import subprocess
import threading
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("openllm.execution_recorder")

RECALL_APPEND_SCRIPT = os.path.expanduser(
    "~/.hermes/jiak/scripts/recall_append.py"
)

# ── 后台写入队列 ──────────────────────────────────────────────

_record_queue: list[dict] = []
_queue_lock = threading.Lock()
_flush_registered = False


def _flush_queue():
    """进程退出前把队列里剩余的记录刷写到RECALL。"""
    global _record_queue
    with _queue_lock:
        pending = list(_record_queue)
        _record_queue.clear()

    for record in pending:
        _write_one_sync(record)


def _ensure_atexit():
    """注册atexit钩子（只注册一次）。"""
    global _flush_registered
    if not _flush_registered:
        atexit.register(_flush_queue)
        _flush_registered = True


def _background_writer():
    """后台线程：从队列取记录，逐条写入RECALL。"""
    while True:
        record = None
        with _queue_lock:
            if _record_queue:
                record = _record_queue.pop(0)
        if record is None:
            break
        _write_one_sync(record)


def _write_one_sync(record: dict):
    """同步写入一条记录到RECALL（调用recall_append.py）。"""
    try:
        json_line = json.dumps(record, ensure_ascii=False)
        subprocess.run(
            ["python3", RECALL_APPEND_SCRIPT, json_line],
            capture_output=True,
            timeout=10,
        )
    except Exception:
        logger.debug("recall_append写入失败", exc_info=True)


# ── 公开API ────────────────────────────────────────────────────

def record_execution(
    tool_name: str,
    args_summary: Optional[dict] = None,
    status: str = "ok",
    duration_ms: float = 0.0,
    result_summary: str = "",
):
    """记录一次工具执行到RECALL.jsonl。

    异步写入（daemon线程），失败只logger.debug，主流程零影响。

    Args:
        tool_name: 工具名称
        args_summary: 参数摘要（JSON-serializable dict）
        status: "ok" | "error" | "security_block"
        duration_ms: 执行耗时（毫秒）
        result_summary: 结果摘要（前200字符截断）
    """
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": "tool_call",
        "tool": tool_name,
        "status": status,
        "duration_ms": round(duration_ms, 1),
        "summary": str(result_summary)[:200],
        "source": "openllm",
    }
    if args_summary:
        record["args_summary"] = {
            k: str(v)[:100] for k, v in args_summary.items()
        }

    try:
        with _queue_lock:
            _record_queue.append(record)

        _ensure_atexit()

        t = threading.Thread(target=_background_writer, daemon=True)
        t.start()
    except Exception:
        logger.debug("record_execution入队失败", exc_info=True)
