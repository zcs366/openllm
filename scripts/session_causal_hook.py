#!/usr/bin/env python3
"""
session_causal_hook.py — Session因果记忆自动提取脚本
===================================================

独立脚本，可被 Hermes cron 或 post-session hook 调用。
读取最近一个 session 的最后 N 条消息，提取 user 意图和 assistant 结果，
调用 AutoCausalWriter.record() 写入因果记忆。

用法:
    python scripts/session_causal_hook.py [--last-n 5] [--session-dir DIR]

设计约束:
    - 不修改 engine.py 或 main_loop.py（不侵入主循环）
    - 独立脚本，可被外部调用
    - 写入失败不抛异常，只 log.warning
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from openllm.memory.auto_causal_writer import AutoCausalWriter

logger = logging.getLogger("session_causal_hook")


def find_latest_session(session_dir: Optional[Path] = None) -> Optional[Path]:
    """找到最近修改的 session 目录。"""
    base = session_dir or Path.home() / ".hermes" / "sessions"
    if not base.exists():
        return None
    
    sessions = sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for s in sessions:
        if s.is_dir():
            return s
    return None


def extract_messages_from_session(session_path: Path, last_n: int = 5) -> List[Dict]:
    """
    从 session 目录提取最近 N 条 user/assistant 消息。
    
    尝试读取 session 的 JSON 消息文件。
    """
    messages = []
    
    # 尝试多种消息存储格式
    for msg_file in ["messages.jsonl", "messages.json", "conversation.jsonl"]:
        fpath = session_path / msg_file
        if fpath.exists():
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    if msg_file.endswith(".jsonl"):
                        all_msgs = [json.loads(line) for line in f if line.strip()]
                    else:
                        all_msgs = json.load(f)
                
                # 筛选 user/assistant 消息
                filtered = [
                    m for m in all_msgs
                    if m.get("role") in ("user", "assistant")
                ]
                messages = filtered[-last_n * 2:]  # 取最近的 user+assistant 对
                break
            except Exception as e:
                logger.warning(f"Failed to read {fpath}: {e}")
                continue
    
    return messages


def extract_causal_pairs(messages: List[Dict]) -> List[Dict]:
    """
    从消息列表中提取 user→assistant 因果对。
    
    每个 user 消息 + 紧随的 assistant 消息构成一个因果对。
    """
    pairs = []
    i = 0
    while i < len(messages):
        if messages[i].get("role") == "user":
            user_text = messages[i].get("content", "")
            # 找下一个 assistant 消息
            if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
                assistant_text = messages[i + 1].get("content", "")
                if user_text and assistant_text:
                    pairs.append({
                        "action": user_text[:200],  # user 意图
                        "actual": assistant_text[:200],  # assistant 结果
                        "prediction": "",  # 无显式预测
                        "success": True,  # 默认成功（assistant 正常回复）
                    })
                i += 2
            else:
                i += 1
        else:
            i += 1
    
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Extract causal memories from session")
    parser.add_argument("--last-n", type=int, default=5, help="Number of message pairs to extract")
    parser.add_argument("--session-dir", type=str, default=None, help="Session directory path")
    args = parser.parse_args()

    session_dir = Path(args.session_dir) if args.session_dir else None
    session = find_latest_session(session_dir)
    
    if not session:
        print("No session found.")
        return
    
    logger.info(f"Processing session: {session.name}")
    
    messages = extract_messages_from_session(session, args.last_n)
    if not messages:
        print("No messages found in session.")
        return
    
    pairs = extract_causal_pairs(messages)
    if not pairs:
        print("No causal pairs extracted.")
        return
    
    writer = AutoCausalWriter()
    count = 0
    for pair in pairs:
        try:
            writer.record(
                action=pair["action"],
                prediction=pair["prediction"],
                actual=pair["actual"],
                success=pair["success"],
                context=f"session:{session.name}",
            )
            count += 1
        except Exception as e:
            logger.warning(f"Failed to record causal pair: {e}")
    
    print(f"Recorded {count} causal memories from session {session.name}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
