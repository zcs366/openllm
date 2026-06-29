"""OpenLLM 章鱼搜索工具 — 连接章鱼记忆系统"""
import os
import json
import subprocess
from pathlib import Path

OCTOPUS_DIR = Path.home() / ".hermes" / "octopus"
JIAK_DIR = Path.home() / ".hermes" / "jiak"
RECALL_PATH = JIAK_DIR / "RECALL.jsonl"
CARDS_DIR = JIAK_DIR / "cards"


def tool_octopus_search(query: str, limit: int = 5) -> str:
    """搜索章鱼记忆系统（RECALL + jiak cards）。
    
    Args:
        query: 搜索关键词
        limit: 返回结果数量
    
    Returns:
        搜索结果文本
    """
    results = []
    
    # 1. 搜索 RECALL.jsonl
    if RECALL_PATH.exists():
        with open(RECALL_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    content = record.get("content", "")
                    if query.lower() in content.lower():
                        results.append({
                            "source": "RECALL",
                            "type": record.get("type", "?"),
                            "content": content[:200],
                            "ts": record.get("ts", ""),
                        })
                except json.JSONDecodeError:
                    continue
    
    # 2. 搜索 jiak cards
    if CARDS_DIR.exists():
        for card_file in CARDS_DIR.glob("*.json"):
            try:
                with open(card_file, encoding="utf-8") as f:
                    card = json.load(f)
                card_text = json.dumps(card, ensure_ascii=False)
                if query.lower() in card_text.lower():
                    results.append({
                        "source": "jiak_card",
                        "card_id": card.get("id", card_file.stem),
                        "summary": card.get("summary", "")[:200],
                    })
            except (json.JSONDecodeError, Exception):
                continue
    
    # 3. 格式化输出
    if not results:
        return f"未找到与 '{query}' 相关的记忆"
    
    output = f"🐙 章鱼搜索 '{query}' — 找到 {len(results)} 条结果：\n\n"
    for i, r in enumerate(results[:limit], 1):
        if r["source"] == "RECALL":
            output += f"{i}. [{r['type']}] {r['content'][:150]}...\n"
        else:
            output += f"{i}. [card:{r['card_id']}] {r['summary'][:150]}...\n"
    
    return output


def tool_octopus_self_model() -> str:
    """查看章鱼自省状态。"""
    try:
        import sys
        sys.path.insert(0, str(OCTOPUS_DIR))
        from scripts.self_model import self_model
        model = self_model()
        return json.dumps(model, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        return f"章鱼自省失败: {e}"


if __name__ == "__main__":
    # 测试
    print(tool_octopus_search("OpenLLM"))
    print("---")
    print(tool_octopus_self_model()[:500])
