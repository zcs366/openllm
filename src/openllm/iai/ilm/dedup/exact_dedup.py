"""
exact_dedup.py — SHA-256精确去重（NeMo-Curator双层方案的第一层）
铁律：精确去重在前，模糊去重(MinHash)在后
"""
import hashlib
from typing import List, Dict, Set
from ..base import ILMDocument


def content_sha256(text: str) -> str:
    """计算内容SHA-256"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def exact_dedup(documents: List[ILMDocument]) -> List[ILMDocument]:
    """
    SHA-256精确去重
    完全相同的内容只保留第一条
    
    Returns:
        去重后的文档列表
    """
    seen: Set[str] = set()
    result = []
    dup_count = 0

    for doc in documents:
        h = content_sha256(doc.content)
        if h not in seen:
            seen.add(h)
            doc.content_hash = h  # 确保content_hash是SHA-256
            result.append(doc)
        else:
            dup_count += 1

    print(f"[exact_dedup] {len(documents)} → {len(result)} ({dup_count} exact duplicates removed)")
    return result


if __name__ == "__main__":
    docs = [
        ILMDocument("test", "a.py", "这是完全一样的内容", "technical", 0),
        ILMDocument("test", "b.py", "这是完全一样的内容", "technical", 0),
        ILMDocument("test", "c.py", "这是不同的内容", "relation", 0),
    ]
    result = exact_dedup(docs)
    assert len(result) == 2
    print(f"✅ {len(docs)} → {len(result)}")
