"""
recall_extractor.py — RECALL.jsonl → ILMDocument流
铁律：source_ref永久填充，三类规则独立分类
"""
import json
import os
import re
import time
from datetime import datetime
from typing import Generator
from ..base import ILMDocument, SourceType, DocType


def _safe_float(val) -> float:
    """安全转float"""
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _parse_timestamp(val) -> float:
    """兼容float和ISO格式时间戳"""
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            try:
                dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
                return dt.timestamp()
            except Exception:
                return 0.0
    return 0.0


# 关系/技术/工程关键词（与session_extractor共享，但针对RECALL格式调整）
RELATION_KEYWORDS = [
    "纠正", "偏好", "不喜欢", "记住", "以后", "风格",
    "老搭档", "军师", "关系", "信任", "你的判断",
]
TECHNICAL_KEYWORDS = [
    "架构", "发现", "验证", "实验", "论文", "算法",
    "模型", "attention", "transformer", "embedding",
    "D0", "ISA", "IO-S", "ISN", "IKO",
]
ENGINEERING_KEYWORDS = [
    "P0", "P1", "交付", "完成", "测试", "审计",
    "PAL", "里程碑", "部署", "修复", "重构",
]


def _classify_text(text: str) -> str:
    """三套独立规则分类"""
    r = sum(1 for kw in RELATION_KEYWORDS if kw in text)
    t = sum(1 for kw in TECHNICAL_KEYWORDS if kw in text)
    e = sum(1 for kw in ENGINEERING_KEYWORDS if kw in text)
    mx = max(r, t, e)
    if mx == 0:
        return DocType.UNKNOWN.value
    if r == mx:
        return DocType.RELATION.value
    if t == mx:
        return DocType.TECHNICAL.value
    return DocType.ENGINEERING.value


def _extract_text(entry: dict) -> str:
    """从RECALL条目提取可清洗的文本内容"""
    parts = []
    # 优先取content字段
    if entry.get("content"):
        parts.append(str(entry["content"]))
    # 取summary
    if entry.get("summary"):
        parts.append(str(entry["summary"]))
    # 取event
    if entry.get("event"):
        parts.append(str(entry["event"]))
    # 取topic
    if entry.get("topic"):
        parts.append(str(entry["topic"]))
    return " ".join(parts).strip()


def _build_source_ref(entry: dict) -> str:
    """
    永久字段：构建source_ref
    格式: "recall:{type}:{timestamp}"
    """
    entry_type = entry.get("type", "unknown")
    return f"recall:{entry_type}:{int(_parse_timestamp(entry.get('timestamp') or entry.get('ts')))}"


def extract_recall(jsonl_path: str,
                   min_content_len: int = 10,
                   max_docs: int = 0) -> Generator[ILMDocument, None, None]:
    """
    从RECALL.jsonl提取经验条目
    
    铁律：
    - source_ref永远填充
    - 三类数据三套规则
    - 跳过无内容的条目（如纯inject类型无文本）
    """
    if not os.path.exists(jsonl_path):
        print(f"[recall_extractor] File not found: {jsonl_path}")
        return
    
    count = 0
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            content = _extract_text(entry)
            if len(content) < min_content_len:
                continue
            
            doc = ILMDocument(
                source=SourceType.RECALL.value,
                source_path=jsonl_path,
                content=content,
                doc_type=_classify_text(content),
                timestamp=_parse_timestamp(entry.get("timestamp") or entry.get("ts")),
                metadata={
                    "recall_type": entry.get("type", ""),
                    "importance": entry.get("importance", 0),
                    "session_id": entry.get("session_id", ""),
                    "line_num": line_num,
                },
                source_ref=_build_source_ref(entry),
                quality_score=_safe_float(entry.get("importance", 0)) / 10.0,
            )
            yield doc
            count += 1
            
            if max_docs > 0 and count >= max_docs:
                return
    
    print(f"[recall_extractor] Extracted {count} documents from {jsonl_path}")


if __name__ == "__main__":
    import sys
    path = os.path.expanduser("~/.hermes/jiak/RECALL.jsonl")
    if len(sys.argv) > 1:
        path = sys.argv[1]
    
    for i, doc in enumerate(extract_recall(path, max_docs=10)):
        print(f"[{i+1}] {doc.doc_type} | ref={doc.source_ref} | q={doc.quality_score:.2f} | {doc.content[:60]}...")
