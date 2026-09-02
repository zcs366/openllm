"""
session_extractor.py — state.db → ILMDocument流
铁律：三类数据三套清洗规则（不可合并）
"""
import sqlite3
import os
import re
from typing import Generator
from ..base import ILMDocument, SourceType, DocType


# ═══════════════════════════════════════════════════════════════
# 三套独立规则（铁律：关系/技术/工程各一套，不可合并）
# ═══════════════════════════════════════════════════════════════

# 规则1：关系信号检测（用户纠正/偏好/决策风格/称呼）
RELATION_PATTERNS = [
    # 用户纠正（最强信号）
    r"不[可以许]", r"错了", r"不对", r"别[这样那样]", r"停",
    r"我说的不是", r"你理解错了", r"重新来", r"不要这样",
    # 偏好表达
    r"我喜欢", r"我讨厌", r"我习惯", r"我不喜欢", r"偏好",
    r"以后.{0,5}要", r"从今以后", r"记住这个",
    # 决策风格
    r"你觉得", r"你的判断", r"你来分析", r"你来决定",
    r"先.{0,10}后", r"优先", r"不重要", r"必须",
    # 关系称呼（强关系信号）
    r"老搭档", r"军师", r"老铁",
]

# 规则2：技术insight检测（架构/原理/发现/方法论）
TECHNICAL_PATTERNS = [
    # 架构与设计
    r"架构", r"设计模式", r"方案.{0,5}选择", r"原理", r"机制",
    # 算法与模型
    r"算法", r"模型.{0,5}(训练|推理|优化)", r"参数", r"权重",
    # 发现与验证
    r"发现", r"验证.{0,5}(了|结果)", r"实验.{0,5}(结果|证明)",
    r"结论.{0,5}(是|表明)", r"论文.{0,5}(说|表明|提出)",
    # 技术术语（高频）
    r"(attention|transformer|embedding|minhash|faiss|bge|sqlite|ollama)",
    r"(D0|ICA|ISA|IO-S|ISN|IKO)",
]

# 规则3：工程决策检测（任务/优先级/交付/审计）
ENGINEERING_PATTERNS = [
    # 优先级与任务
    r"P[0-2]", r"里程碑", r"交付", r"部署", r"上线",
    # 审计与质量
    r"测试.{0,5}(通过|完成)", r"审计", r"修复", r"重构",
    # 流程与规范
    r"PAL", r"任务书", r"执行报告", r"三碑",
    r"优先级", r"工期", r"截止",
    # 完成状态
    r"完成", r"通过", r"验收", r"落地",
]


def _match_patterns(text: str, patterns: list) -> int:
    """统计文本匹配模式的数量"""
    hits = 0
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            hits += 1
    return hits


def _classify_content(text: str) -> str:
    """
    三套独立规则分类（铁律：不合并为一套）
    每类独立计分，取最高分
    """
    relation_score = _match_patterns(text, RELATION_PATTERNS)
    technical_score = _match_patterns(text, TECHNICAL_PATTERNS)
    engineering_score = _match_patterns(text, ENGINEERING_PATTERNS)
    
    scores = {
        DocType.RELATION.value: relation_score,
        DocType.TECHNICAL.value: technical_score,
        DocType.ENGINEERING.value: engineering_score,
    }
    
    max_score = max(scores.values())
    if max_score == 0:
        return DocType.UNKNOWN.value
    
    # 返回最高分的类型
    for doc_type, score in scores.items():
        if score == max_score:
            return doc_type
    
    return DocType.UNKNOWN.value


def _build_source_ref(session_id: str, message_id: int) -> str:
    """
    永久字段：构建source_ref
    格式: "session:{session_id}:msg:{message_id}"
    压缩后仍可追溯到原始来源
    """
    return f"session:{session_id}:msg:{message_id}"


def extract_sessions(db_path: str, 
                     min_content_len: int = 10,
                     roles: tuple = ("user", "assistant"),
                     max_docs: int = 0,
                     session_id: str = None) -> Generator[ILMDocument, None, None]:
    """
    从state.db提取session消息
    
    铁律：
    - source_ref永远填充（不丢来源）
    - 三类数据用三套规则分类（不合并）
    - 增量模式：只处理新增，不做全量重建
    - session_id过滤：只处理指定session（session接入模式）
    """
    if not os.path.exists(db_path):
        print(f"[session_extractor] DB not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    placeholders = ",".join("?" * len(roles))
    query = f"""
        SELECT id, session_id, role, content, timestamp, active, compacted
        FROM messages
        WHERE role IN ({placeholders})
          AND active = 1
          AND compacted = 0
          AND LENGTH(content) >= ?
          {"AND session_id = ?" if session_id else ""}
        ORDER BY timestamp ASC
    """
    
    params = list(roles) + [min_content_len]
    if session_id:
        params.append(session_id)
    cursor.execute(query, params)
    
    count = 0
    while True:
        rows = cursor.fetchmany(1000)
        if not rows:
            break
        
        for row in rows:
            content = row["content"].strip()
            if not content:
                continue
            
            doc = ILMDocument(
                source=SourceType.SESSION.value,
                source_path=db_path,
                content=content,
                doc_type=_classify_content(content),
                timestamp=row["timestamp"] or 0.0,
                metadata={
                    "session_id": row["session_id"],
                    "role": row["role"],
                    "message_id": row["id"],
                },
                source_ref=_build_source_ref(row["session_id"], row["id"]),
            )
            yield doc
            count += 1
            
            if max_docs > 0 and count >= max_docs:
                conn.close()
                return
    
    conn.close()
    print(f"[session_extractor] Extracted {count} documents from {db_path}")


def get_session_stats(db_path: str) -> dict:
    """获取session统计信息"""
    if not os.path.exists(db_path):
        return {"error": "DB not found"}
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    stats = {}
    cursor.execute("SELECT COUNT(*) FROM messages")
    stats["total_messages"] = cursor.fetchone()[0]
    
    cursor.execute("SELECT role, COUNT(*) FROM messages GROUP BY role")
    stats["by_role"] = dict(cursor.fetchall())
    
    cursor.execute("SELECT COUNT(DISTINCT session_id) FROM messages")
    stats["total_sessions"] = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM messages WHERE LENGTH(content) >= 10")
    stats["messages_ge_10chars"] = cursor.fetchone()[0]
    
    conn.close()
    return stats


if __name__ == "__main__":
    import sys
    db = os.path.expanduser("~/.hermes/state.db")
    if len(sys.argv) > 1:
        db = sys.argv[1]
    
    print("=== Session Stats ===")
    stats = get_session_stats(db)
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    print("\n=== Sample Extraction (first 10) ===")
    for i, doc in enumerate(extract_sessions(db, max_docs=10)):
        print(f"[{i+1}] {doc.doc_type} | ref={doc.source_ref} | {doc.content[:60]}...")
