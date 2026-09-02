"""
domain_mixer.py — 领域配比控制
铁律：关系40% / 技术40% / 工程20%
"""
import random
from typing import List, Dict
from ..base import ILMDocument, DocType


# 默认配比
DEFAULT_RATIOS = {
    DocType.RELATION.value: 0.40,    # 关系信号40%
    DocType.TECHNICAL.value: 0.40,   # 技术insight40%
    DocType.ENGINEERING.value: 0.20, # 工程决策20%
}

# 未知类型不参与配比，但保留
KEEP_UNKNOWN = True


def mix_documents(documents: List[ILMDocument],
                  ratios: Dict[str, float] = None,
                  max_total: int = 0,
                  shuffle: bool = True) -> List[ILMDocument]:
    """
    按领域配比筛选文档
    
    Args:
        documents: 输入文档列表
        ratios: 领域配比（默认DEFAULT_RATIOS）
        max_total: 最大总文档数（0=不限制）
        shuffle: 是否打乱顺序
    
    Returns:
        配比后的文档列表
    """
    if ratios is None:
        ratios = DEFAULT_RATIOS.copy()
    
    # 按类型分组
    by_type = {}
    for doc in documents:
        dt = doc.doc_type
        if dt not in by_type:
            by_type[dt] = []
        by_type[dt].append(doc)
    
    # 计算各类型应保留的数量
    known_count = sum(len(docs) for dt, docs in by_type.items()
                      if dt in ratios)
    
    if max_total <= 0:
        # 不限制总数，按比例分配
        max_total = known_count
    
    # 按配比分配名额
    selected = []
    for doc_type, ratio in ratios.items():
        if doc_type not in by_type:
            continue
        
        quota = int(max_total * ratio)
        available = by_type[doc_type]
        
        # 按质量分数排序（如果有的话）
        available.sort(key=lambda d: d.quality_score, reverse=True)
        
        # 取配额内的
        selected.extend(available[:quota])
    
    # 保留未知类型（不占配额）
    if KEEP_UNKNOWN and DocType.UNKNOWN.value in by_type:
        # 未知类型最多保留已选数量的10%
        unknown_quota = max(1, len(selected) // 10)
        unknown_docs = by_type[DocType.UNKNOWN.value]
        selected.extend(unknown_docs[:unknown_quota])
    
    # 打乱顺序
    if shuffle:
        random.shuffle(selected)
    
    # 统计
    stats = {}
    for doc in selected:
        stats[doc.doc_type] = stats.get(doc.doc_type, 0) + 1
    
    print(f"[domain_mixer] {len(documents)} → {len(selected)} (ratio: {stats})")
    return selected


def get_domain_stats(documents: List[ILMDocument]) -> Dict[str, int]:
    """统计各领域的文档数"""
    stats = {}
    for doc in documents:
        stats[doc.doc_type] = stats.get(doc.doc_type, 0) + 1
    return stats


if __name__ == "__main__":
    # 测试配比
    docs = []
    for i in range(50):
        docs.append(ILMDocument("test", f"a{i}.py", f"关系信号内容{i}" * 5, "relation", 0))
    for i in range(50):
        docs.append(ILMDocument("test", f"b{i}.py", f"技术insight内容{i}" * 5, "technical", 0))
    for i in range(30):
        docs.append(ILMDocument("test", f"c{i}.py", f"工程决策内容{i}" * 5, "engineering", 0))
    for i in range(10):
        docs.append(ILMDocument("test", f"d{i}.py", f"未知内容{i}" * 5, "unknown", 0))
    
    print("Before mixing:", get_domain_stats(docs))
    mixed = mix_documents(docs, max_total=100)
    print("After mixing:", get_domain_stats(mixed))
