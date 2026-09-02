"""
jiak_writer.py — 清洗后数据写入jiak卡片
铁律：走jiak_api.write_card()协议，不自己json.dump
"""
import os
import sys
import hashlib
from typing import List, Optional
from ..base import ILMDocument

# jiak_api在~/.hermes/jiak/目录下
JIAK_DIR = os.path.expanduser("~/.hermes/jiak")


def _generate_card_id(doc: ILMDocument) -> str:
    """
    生成jiak卡片ID
    格式: ilm-{source}-{content_hash[:8]}-{timestamp整数}
    """
    ts_int = int(doc.timestamp) if doc.timestamp else 0
    return f"ilm-{doc.source}-{doc.content_hash[:8]}-{ts_int}"


def _build_card(doc: ILMDocument) -> dict:
    """
    将ILMDocument转换为jiak卡片格式
    永久保留source_ref（Paper2铁律：压缩不丢来源）
    """
    card = {
        "card_id": _generate_card_id(doc),
        "type": f"ilm_{doc.doc_type}",
        "title": doc.content[:50] + ("..." if len(doc.content) > 50 else ""),
        "summary": doc.content[:200],
        "source_ref": doc.source_ref,  # 永久字段：可追溯到原始来源
        "doc_type": doc.doc_type,
        "quality_score": doc.quality_score,
        "domain": doc.domain,
        "content_hash": doc.content_hash,
        "tags": [doc.source, doc.doc_type],
        "insights": [doc.content],
        "decisions": [],
        "created_from": "ilm_pipeline",
    }
    
    # 添加来源特定元数据
    if doc.metadata:
        card["ilm_metadata"] = doc.metadata
    
    return card


def write_to_jiak(documents: List[ILMDocument],
                  type_whitelist: List[str] = None) -> dict:
    """
    批量写入jiak卡片
    
    铁律：
    - 走jiak_api.write_card()协议（不自己json.dump）
    - source_ref永远存在
    - type在白名单中（新卡片）
    
    Returns:
        {"written": int, "skipped": int, "errors": int}
    """
    try:
        import jiak_api
    except ImportError:
        print("[jiak_writer] jiak_api not found, falling back to direct write")
        return _write_fallback(documents)
    
    stats = {"written": 0, "skipped": 0, "errors": 0}
    
    for doc in documents:
        card = _build_card(doc)
        card_id = card["card_id"]
        
        # 类型白名单检查
        if type_whitelist and card["type"] not in type_whitelist:
            stats["skipped"] += 1
            continue
        
        try:
            jiak_api.write_card(card_id, card)
            stats["written"] += 1
        except Exception as e:
            stats["errors"] += 1
            print(f"[jiak_writer] Error writing {card_id}: {e}")
    
    print(f"[jiak_writer] Written: {stats['written']}, Skipped: {stats['skipped']}, Errors: {stats['errors']}")
    return stats


def _write_fallback(documents: List[ILMDocument]) -> dict:
    """
    降级写入：直接写JSON文件（仅在jiak_api不可用时）
    """
    import json
    import time
    
    cards_dir = os.path.join(JIAK_DIR, "cards")
    os.makedirs(cards_dir, exist_ok=True)
    
    stats = {"written": 0, "skipped": 0, "errors": 0}
    
    for doc in documents:
        card = _build_card(doc)
        card_id = card["card_id"]
        card_path = os.path.join(cards_dir, f"{card_id}.json")
        
        try:
            card["updated"] = time.time()
            with open(card_path, 'w', encoding='utf-8') as f:
                json.dump(card, f, ensure_ascii=False, indent=2)
            stats["written"] += 1
        except Exception as e:
            stats["errors"] += 1
            print(f"[jiak_writer] Fallback error: {e}")
    
    print(f"[jiak_writer] Fallback written: {stats['written']}")
    return stats


if __name__ == "__main__":
    # 测试
    test_docs = [
        ILMDocument(
            source="session",
            source_path="test.db",
            content="用户说不可以讨好我，评估要真实严厉。这是关系信号。",
            doc_type="relation",
            timestamp=1000.0,
            metadata={"session_id": "test", "message_id": 1},
            source_ref="session:test:msg:1",
        ),
    ]
    
    stats = write_to_jiak(test_docs)
    print(f"Test result: {stats}")
