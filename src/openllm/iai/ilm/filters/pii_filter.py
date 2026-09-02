"""
pii_filter.py — PII去除（NeMo-Curator第5步）
铁律：PII必须在清洗管线中去除，不可跳过
"""
import re
from ..base import ILMDocument


# PII检测模式
PII_PATTERNS = {
    "phone": re.compile(r'1[3-9]\d{9}'),  # 中国手机号
    "email": re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
    "id_card": re.compile(r'\d{17}[\dXx]'),  # 身份证号
    "ip_address": re.compile(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'),
    "bank_card": re.compile(r'\d{16,19}'),  # 银行卡号（16-19位纯数字）
}

# 替换文本
REPLACEMENTS = {
    "phone": "[手机号已脱敏]",
    "email": "[邮箱已脱敏]",
    "id_card": "[身份证已脱敏]",
    "ip_address": "[IP已脱敏]",
    "bank_card": "[银行卡已脱敏]",
}


def mask_pii(text: str) -> tuple:
    """
    检测并脱敏PII
    
    Returns:
        (masked_text, pii_found_dict)
    """
    pii_found = {}
    masked = text

    for pii_type, pattern in PII_PATTERNS.items():
        matches = pattern.findall(masked)
        if matches:
            pii_found[pii_type] = len(matches)
            masked = pattern.sub(REPLACEMENTS[pii_type], masked)

    return masked, pii_found


def filter_pii(documents: list) -> tuple:
    """
    批量PII过滤
    
    Returns:
        (filtered_docs, stats)
    """
    stats = {"total": 0, "pii_masked": 0, "pii_types": {}}

    for doc in documents:
        stats["total"] += 1
        masked, pii_found = mask_pii(doc.content)

        if pii_found:
            doc.content = masked
            doc.metadata["pii_masked"] = True
            doc.metadata["pii_types"] = pii_found
            stats["pii_masked"] += 1
            for k, v in pii_found.items():
                stats["pii_types"][k] = stats["pii_types"].get(k, 0) + v

    print(f"[pii_filter] {stats['total']} docs, {stats['pii_masked']} had PII masked: {stats['pii_types']}")
    return documents, stats


if __name__ == "__main__":
    tests = [
        "联系方式：13812345678，邮箱：test@example.com",
        "服务器IP：192.168.1.100，身份证：110101199001011234",
        "没有PII的正常技术讨论内容",
    ]
    for t in tests:
        masked, found = mask_pii(t)
        print(f"  PII: {found if found else 'none'}")
        print(f"  In:  {t}")
        print(f"  Out: {masked}")
        print()
