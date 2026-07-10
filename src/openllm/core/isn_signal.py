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
    """记录信号到RECALL"""
    try:
        with open(RECALL_PATH, "a") as f:
            f.write(json.dumps({
                "type": "isn_signal",
                "signal_type": signal.get("type"),
                "from": signal.get("from"),
                "payload": signal.get("payload"),
                "timestamp": time.time()
            }, ensure_ascii=False) + "\n")
    except Exception:
        pass
