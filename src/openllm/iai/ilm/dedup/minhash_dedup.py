"""
minhash_dedup.py — MinHash观点级去重
工程法典①匠石：去重是纯数学，不需要LLM。
"""
import hashlib
import re
from typing import List, Tuple, Dict
from ..base import ILMDocument


def _tokenize(text: str) -> List[str]:
    """中文2-gram + 英文分词（与hermes_search一致）"""
    tokens = []
    
    # 中文：逐字+2-gram
    chinese_chars = re.findall(r'[\u4e00-\u9fff]+', text)
    for seg in chinese_chars:
        tokens.extend(list(seg))  # 单字
        for i in range(len(seg) - 1):
            tokens.append(seg[i:i+2])  # 2-gram
    
    # 英文：按空格分词，取>=3字符的
    english_words = re.findall(r'[a-zA-Z]+', text)
    tokens.extend(w.lower() for w in english_words if len(w) >= 3)
    
    return tokens


def _minhash_signature(tokens: List[str], num_perm: int = 128) -> List[int]:
    """计算MinHash签名"""
    signature = [float('inf')] * num_perm
    
    for token in tokens:
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        for i in range(num_perm):
            perm_hash = (h * (i + 1) + i) % (2**31)
            if perm_hash < signature[i]:
                signature[i] = perm_hash
    
    return signature


def _jaccard_estimate(sig1: List[int], sig2: List[int]) -> float:
    """从MinHash签名估算Jaccard相似度"""
    if len(sig1) != len(sig2):
        return 0.0
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


class MinHashDeduplicator:
    """MinHash观点级去重器"""
    
    def __init__(self, threshold: float = 0.7, num_perm: int = 128):
        """
        Args:
            threshold: Jaccard相似度阈值（>此值视为重复）
            num_perm: MinHash排列数（越高越精确，越慢）
        """
        self.threshold = threshold
        self.num_perm = num_perm
        self.signatures = []  # [(doc_index, signature, doc)]
    
    def add(self, doc: ILMDocument) -> int:
        """添加文档，返回索引"""
        tokens = _tokenize(doc.content)
        sig = _minhash_signature(tokens, self.num_perm)
        idx = len(self.signatures)
        self.signatures.append((idx, sig, doc))
        return idx
    
    def find_duplicates(self) -> List[Tuple[int, int, float]]:
        """
        查找重复对
        
        Returns:
            [(idx1, idx2, similarity), ...]
        """
        duplicates = []
        n = len(self.signatures)
        
        for i in range(n):
            for j in range(i + 1, n):
                sim = _jaccard_estimate(
                    self.signatures[i][1],
                    self.signatures[j][1]
                )
                if sim >= self.threshold:
                    duplicates.append((i, j, sim))
        
        return duplicates
    
    def merge_duplicates(self, duplicates: List[Tuple[int, int, float]]) -> List[ILMDocument]:
        """
        合并重复文档组，保留信息量最大的版本
        
        铁律：去重必须区分重复vs多视角
        - 同类型+高相似 → 合并（真正的重复）
        - 不同类型+高相似 → 都保留（多视角，是洞察来源）
        
        Returns:
            去重后的文档列表
        """
        # 构建连通分量（Union-Find）
        parent = list(range(len(self.signatures)))
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry
        
        # 多视角保护：不同doc_type的不合并
        multi_perspective_pairs = set()
        for i, j, sim in duplicates:
            doc_i = self.signatures[i][2]
            doc_j = self.signatures[j][2]
            if doc_i.doc_type != doc_j.doc_type:
                # 不同类型 → 多视角，不union，都保留
                multi_perspective_pairs.add((i, j))
            else:
                # 同类型 → 真重复，union
                union(i, j)
        
        # 按组分组
        groups = {}
        for idx in range(len(self.signatures)):
            root = find(idx)
            if root not in groups:
                groups[root] = []
            groups[root].append(idx)
        
        # 每组保留信息量最大的
        result = []
        merged_count = 0
        for root, members in groups.items():
            best_idx = max(members, key=lambda i: len(self.signatures[i][2].content))
            best_doc = self.signatures[best_idx][2]
            best_doc.metadata["dedup_group_size"] = len(members)
            if len(members) > 1:
                best_doc.metadata["merged_from"] = len(members)
                merged_count += len(members) - 1
            result.append(best_doc)
        
        # 多视角文档也加入结果（它们没被union）
        for i, j in multi_perspective_pairs:
            # 确保不重复添加
            for idx in (i, j):
                if not any(d.doc_id == self.signatures[idx][2].doc_id for d in result):
                    perspective_doc = self.signatures[idx][2]
                    perspective_doc.metadata["multi_perspective"] = True
                    result.append(perspective_doc)
        
        print(f"[minhash_dedup] {len(self.signatures)} docs → {len(result)} "
              f"({merged_count} merged, {len(multi_perspective_pairs)} multi-perspective)")
        return result
    
    def dedup(self) -> List[ILMDocument]:
        """一步完成：添加所有文档+去重+合并"""
        duplicates = self.find_duplicates()
        print(f"[minhash_dedup] Found {len(duplicates)} duplicate pairs")
        return self.merge_duplicates(duplicates)


def dedup_documents(documents: List[ILMDocument], 
                    threshold: float = 0.7) -> List[ILMDocument]:
    """便捷函数：直接对文档列表去重"""
    deduper = MinHashDeduplicator(threshold=threshold)
    for doc in documents:
        deduper.add(doc)
    return deduper.dedup()


if __name__ == "__main__":
    # 测试去重
    docs = [
        ILMDocument("test", "a.py", "这是一个关于架构设计的讨论，涉及多个技术栈的选择和对比。", "technical", 0),
        ILMDocument("test", "b.py", "这是一个关于架构设计的讨论，涉及多个技术栈的选择和对比分析。", "technical", 0),  # 高度相似
        ILMDocument("test", "c.py", "用户说不可以讨好我，评估要真实严厉。这是关系信号。", "relation", 0),
        ILMDocument("test", "d.py", "用户说不可以讨好我，评估必须真实严厉。这是重要的关系信号。", "relation", 0),  # 高度相似
        ILMDocument("test", "e.py", "完全不同的内容，关于搜索工具的评估和测试结果。", "engineering", 0),
    ]
    
    result = dedup_documents(docs, threshold=0.6)
    print(f"\nOriginal: {len(docs)}, After dedup: {len(result)}")
    for doc in result:
        size = doc.metadata.get("dedup_group_size", 1)
        print(f"  [{doc.doc_type}] (group={size}) {doc.content[:60]}...")
