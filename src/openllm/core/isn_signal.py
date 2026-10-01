"""
openLLM ISN信号协议 — 从isn_signal.py提取

ISN↔ISA↔IOS三系统信号收发。
标准信封: {type, from, to, payload, timestamp}
"""
import json
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

# 常量
SIGNAL_DIR = os.path.expanduser("~/.io-s/signals")
RECALL_PATH = os.path.expanduser("~/.hermes/jiak/RECALL.jsonl")
SKILLS_DIR = os.path.expanduser("~/.hermes/skills")

TZ = timezone(timedelta(hours=8))


def make_envelope(signal_type: str, sender: str, target: str, payload: dict) -> dict:
    """构造标准信号信封"""
    return {
        "type": signal_type,
        "from": sender,
        "to": target,
        "payload": payload,
        "timestamp": datetime.now(TZ).isoformat()
    }


def send_signal(signal_type: str, sender: str, target: str, payload: dict) -> bool:
    """发送信号到IO-S syscall"""
    try:
        import requests
        envelope = make_envelope(signal_type, sender, target, payload)
        resp = requests.post(
            "http://127.0.0.1:8770/syscall",
            json=envelope,
            timeout=5
        )
        return resp.status_code == 200
    except Exception:
        return False


def skill_created(skill_name: str, skill_dir: str) -> bool:
    """发送skill_created信号到ISA"""
    return send_signal(
        signal_type="skill_created",
        sender="ISN",
        target="ISA",
        payload={"name": skill_name, "dir": skill_dir}
    )


def poll_dreams() -> list[dict]:
    """轮询dream_insight信号"""
    try:
        dream_dir = Path(SIGNAL_DIR) / "dreams"
        if not dream_dir.exists():
            return []
        
        signals = []
        for f in sorted(dream_dir.glob("*.json")):
            try:
                with open(f) as fp:
                    signal = json.load(fp)
                    if signal.get("to") == "ISN":
                        signals.append(signal)
                        f.unlink()  # 消费后删除
            except Exception:
                pass
        return signals
    except Exception:
        return []


def record_to_recall(signal: dict):
    """记录信号到RECALL。

    写侧时间戳规范（成市拍板 2026-10-01）：canonical 字段 `ts`（epoch
    float）。优先走门房 scripts/recall_append.py（与 execution_recorder
    同款模式）；门房不可用时回退直写，但仍落规范字段，不再写 `timestamp`。
    """
    record = {
        "type": "isn_signal",
        "signal_type": signal.get("type"),
        "from": signal.get("from"),
        "payload": signal.get("payload"),
        "ts": time.time(),
    }
    gate = os.path.join(os.path.dirname(RECALL_PATH), "scripts", "recall_append.py")
    if os.path.exists(gate) and os.path.exists(RECALL_PATH):
        try:
            import subprocess
            import sys
            subprocess.run(
                [sys.executable, gate, json.dumps(record, ensure_ascii=False)],
                capture_output=True, timeout=10,
            )
            return
        except Exception:
            pass  # 门房失败不阻塞主流程，回退直写
    try:
        with open(RECALL_PATH, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass
