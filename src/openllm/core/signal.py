"""signal.py — 文件系统总线IPC

来自老IO-S syscall/signal.py：
  - 文件系统就是总线——不需要网络、MCP、A2A
  - 信号生命周期: signal_send → cap_check → 写文件 + audit → signal_recv
  - TTL自动清理（24小时）
  - 标准信封格式
"""

import json
import logging
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("openllm.signal")

OPENLLM_HOME = Path.home() / ".openllm"
SIGNAL_DIR = OPENLLM_HOME / "signals"
SIGNAL_TTL_HOURS = 24

_ts = lambda: datetime.now(timezone.utc).isoformat()


def _cleanup_old_signals(ttl_hours: int = SIGNAL_TTL_HOURS) -> int:
    if not SIGNAL_DIR.exists():
        return 0
    cutoff = time.time() - (ttl_hours * 3600)
    cleaned = 0
    for sf in list(SIGNAL_DIR.glob("*.json")):
        try:
            if sf.stat().st_mtime < cutoff:
                sf.unlink()
                cleaned += 1
        except (OSError, PermissionError):
            continue
    if cleaned:
        logger.info(f"[signal] 清理 {cleaned} 个过期信号 (>{ttl_hours}h)")
    return cleaned


def signal_send(caller_pid: str, dest: Optional[str] = None,
                body: Optional[dict] = None, priority: int = 1,
                trace_id: Optional[str] = None,
                signal: Optional[dict] = None) -> dict:
    """格式化信号payload，写入信号文件。

    支持两种调用格式：
    1. 标准信封: signal_send(signal={type, from, to, payload, timestamp})
    2. 旧格式:   signal_send(dest, body, priority)
    """
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    ts = _ts()

    if signal:
        envelope_type = signal.get("type", "unknown")
        envelope_from = signal.get("from", caller_pid)
        envelope_to = signal.get("to", "broadcast")
        envelope_payload = signal.get("payload", {})
        envelope_ts = signal.get("timestamp", ts)
    else:
        dest = dest or "broadcast"
        envelope_type = "message"
        envelope_from = caller_pid
        envelope_to = dest
        envelope_payload = body or {}
        envelope_ts = ts

    signal_id = f"sig-{envelope_to}-{int(time.time()*1000)}"

    payload = {
        "signal_id": signal_id,
        "type": envelope_type,
        "from": envelope_from,
        "to": envelope_to,
        "payload": envelope_payload,
        "timestamp": envelope_ts,
        "caller_pid": caller_pid,
        "sent_at": ts,
        "priority": priority,
        "status": "sent",
        "ack": None,
        "audit": {
            "trace_id": trace_id or f"sig-{signal_id}",
            "route_log": [{
                "hop": "kernel", "at": ts,
                "from": caller_pid, "to": envelope_to,
            }],
            "cap_check": "passed_at_kernel",
            "recorded_at": ts,
        },
    }

    signal_file = SIGNAL_DIR / f"{signal_id}.json"
    signal_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2))

    _cleanup_old_signals()

    return {
        "signal_id": signal_id, "dest": envelope_to,
        "type": envelope_type, "audit": payload["audit"],
    }


def signal_recv(caller_pid: str, dest: Optional[str] = None,
                signal_type: Optional[str] = None,
                limit: int = 10) -> list[dict]:
    """接收信号（读信号文件）。

    按时间倒序返回最新的limit条信号。
    可按dest和signal_type过滤。
    """
    if not SIGNAL_DIR.exists():
        return []

    signals = []
    for sf in sorted(SIGNAL_DIR.glob("*.json"),
                     key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(sf.read_text())
            # 过滤
            if dest and data.get("to") != dest:
                continue
            if signal_type and data.get("type") != signal_type:
                continue
            signals.append(data)
            if len(signals) >= limit:
                break
        except (json.JSONDecodeError, IOError):
            continue

    return signals


def signal_ack(signal_id: str, caller_pid: str,
               status: str = "received") -> dict:
    """确认收到信号。"""
    signal_file = SIGNAL_DIR / f"{signal_id}.json"
    if not signal_file.exists():
        return {"error": f"信号不存在: {signal_id}"}

    try:
        data = json.loads(signal_file.read_text())
        data["ack"] = {
            "by": caller_pid,
            "at": _ts(),
            "status": status,
        }
        signal_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2))
        return {"signal_id": signal_id, "ack": data["ack"]}
    except (json.JSONDecodeError, IOError) as e:
        return {"error": str(e)}
