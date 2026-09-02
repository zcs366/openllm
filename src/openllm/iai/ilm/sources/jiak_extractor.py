"""
jiak_extractor.py — jiak卡片 → ILMDocument流
铁律：source_ref永久填充，三类规则独立分类
"""
import json
import os
import glob
import re
from typing import Generator
from ..base import ILMDocument, SourceType, DocType


def _safe_float(val) -> float:
    """安全转float，字符串如'high'返回0"""
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _parse_timestamp(val) -> float:
    """兼容float和ISO格式时间戳"""
    from datetime import datetime
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


RELATION_KEYWORDS = [
    "纠正", "偏好", "记住", "以后", "风格", "关系", "信任",
]
TECHNICAL_KEYWORDS = [
    "架构", "发现", "验证", "实验", "论文", "算法",
    "模型", "D0", "attention", "ISA", "IO-S", "ISN",
]
ENGINEERING_KEYWORDS = [
    "P0", "P1", "交付", "完成", "测试", "审计",
    "PAL", "里程碑", "部署",
]


def _classify_text(text: str) -> str:
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


def _extract_text(card: dict) -> str:
    """从jiak卡片提取可清洗的文本内容"""
    parts = []
    if card.get("title"):
        parts.append(str(card["title"]))
    if card.get("summary"):
        parts.append(str(card["summary"]))
    # insights数组
    for insight in card.get("insights", []):
        if isinstance(insight, str):
            parts.append(insight)
        elif isinstance(insight, dict):
            parts.append(str(insight.get("content", "")))
    # decisions数组
    for dec in card.get("decisions", []):
        if isinstance(dec, str):
            parts.append(dec)
        elif isinstance(dec, dict):
            parts.append(str(dec.get("content", "")))
    # consciousness_notes
    if card.get("consciousness_notes"):
        parts.append(str(card["consciousness_notes"]))
    return " ".join(parts).strip()


def _build_source_ref(card: dict) -> str:
    """永久字段：source_ref = "jiak:{card_id}" """
    card_id = card.get("card_id", "unknown")
    return f"jiak:{card_id}"


def extract_jiak(cards_dir: str,
                 min_content_len: int = 10,
                 max_docs: int = 0) -> Generator[ILMDocument, None, None]:
    """
    从jiak卡片目录提取结构化记忆
    
    铁律：
    - source_ref永远填充
    - 三类数据三套规则
    - 跳过空内容卡片
    """
    if not os.path.exists(cards_dir):
        print(f"[jiak_extractor] Dir not found: {cards_dir}")
        return
    
    card_files = glob.glob(os.path.join(cards_dir, "*.json"))
    count = 0
    
    for card_path in sorted(card_files):
        try:
            with open(card_path, 'r', encoding='utf-8') as f:
                card = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue
        
        content = _extract_text(card)
        if len(content) < min_content_len:
            continue
        
        doc = ILMDocument(
            source=SourceType.JIAK.value,
            source_path=card_path,
            content=content,
            doc_type=_classify_text(content),
            timestamp=_parse_timestamp(card.get("created") or card.get("updated")),
            metadata={
                "card_id": card.get("card_id", ""),
                "card_type": card.get("type", ""),
                "importance": card.get("importance", 0),
                "keywords": card.get("keywords", []),
            },
            source_ref=_build_source_ref(card),
            quality_score=_safe_float(card.get("importance", 0)) / 10.0,
        )
        yield doc
        count += 1
        
        if max_docs > 0 and count >= max_docs:
            return
    
    print(f"[jiak_extractor] Extracted {count} documents from {cards_dir}")


if __name__ == "__main__":
    import sys
    path = os.path.expanduser("~/.hermes/jiak/cards")
    if len(sys.argv) > 1:
        path = sys.argv[1]
    
    for i, doc in enumerate(extract_jiak(path, max_docs=10)):
        print(f"[{i+1}] {doc.doc_type} | ref={doc.source_ref} | q={doc.quality_score:.2f} | {doc.content[:60]}...")
