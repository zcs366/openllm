"""
混合检索模块：BM25 + Embedding语义匹配

提供 hybrid_score() 函数，用于MemoryBus的混合重排。
"""
import math
import re
from collections import Counter
from typing import List

# ═══ BM25评分 ═══

def bm25_score(
    query_tokens: List[str],
    doc_tokens: List[str],
    df: Counter,
    N: int,
    avg_dl: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """计算单个文档的BM25分数"""
    score = 0.0
    dl = len(doc_tokens)
    tf = Counter(doc_tokens)
    for q in query_tokens:
        if q not in tf:
            continue
        doc_freq = df.get(q, 0)
        if doc_freq == 0:
            continue
        idf = math.log((N - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
        tf_norm = (tf[q] * (k1 + 1)) / (tf[q] + k1 * (1 - b + b * dl / max(avg_dl, 1)))
        score += idf * tf_norm
    return score


def tokenize(text: str) -> List[str]:
    """简单分词：英文按空格，中文按字符对"""
    tokens = []
    # 英文
    tokens.extend(w.lower() for w in re.findall(r'[a-zA-Z_]{2,}', text.lower()))
    # 中文bigram
    ch = [c for c in text if '\u4e00' <= c <= '\u9fff']
    tokens.extend(ch[i] + ch[i + 1] for i in range(len(ch) - 1))
    return tokens


# ═══ Embedding相似度（延迟加载） ═══

_model = None
_model_path = "/home/zcs/models/bge-small-zh-v1.5"
_model_tried = False

def _get_model():
    """懒加载Embedding模型"""
    global _model, _model_tried
    if _model_tried:
        return _model
    _model_tried = True
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_model_path)
    except Exception:
        _model = None
    return _model


def embed_similarity(text_a: str, text_b: str) -> float:
    """计算两段文本的Embedding余弦相似度"""
    model = _get_model()
    if model is None:
        return 0.0
    try:
        import numpy as np
        embs = model.encode([text_a, text_b], normalize_embeddings=True)
        return float(np.dot(embs[0], embs[1]))
    except Exception:
        return 0.0


# ═══ 混合评分 ═══

def hybrid_score(
    query: str,
    document: str,
    bm25_weight: float = 0.6,
    embed_weight: float = 0.4,
) -> float:
    """
    混合评分：BM25关键词匹配 + Embedding语义匹配。
    
    Args:
        query: 查询文本
        document: 文档文本
        bm25_weight: BM25权重（默认0.6）
        embed_weight: Embedding权重（默认0.4）
        
    Returns:
        混合分数（0-1之间）
    """
    # BM25分数（归一化到0-1）
    q_tokens = tokenize(query)
    d_tokens = tokenize(document)
    # 单文档模式：df=1表示当前文档包含该词，N=1是退化情况
    # 用词频覆盖率替代标准BM25
    q_set = set(q_tokens)
    d_set = set(d_tokens)
    overlap = len(q_set & d_set)
    coverage = overlap / max(len(q_set), 1)
    bm25_norm = min(coverage, 1.0)
    
    # Embedding余弦相似度（已经是0-1）
    embed_sim = embed_similarity(query, document)
    
    # 混合
    return bm25_weight * bm25_norm + embed_weight * embed_sim
