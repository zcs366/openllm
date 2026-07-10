"""
openLLM 降级追踪 — 让静默失败可见

每个try/except降级点加一行trace，记录到日志文件。
"""
import time
import json
from pathlib import Path

LOG_PATH = Path.home() / ".openllm" / "output" / "degradation_log.jsonl"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def trace_degradation(body: str, phase: str, error: Exception, detail: str = ""):
    """记录一次降级事件"""
    record = {
        "timestamp": time.time(),
        "body": body,
        "phase": phase,
        "error": type(error).__name__,
        "message": str(error)[:200],
        "detail": detail[:100],
    }
    try:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass  # 日志写入失败不阻塞主流程
