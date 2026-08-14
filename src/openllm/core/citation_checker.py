"""Citation Checker — 源引用输出解析。

检测模型输出中事实性声明的引用标注情况。
System prompt要求：每个事实声明标注 [来源: ...]，未确认标注 ⚠️未验证。
本模块在输出后检查是否有事实声明缺少来源标注。

用法:
    from openllm.core.citation_checker import check_citations
    report = check_citations(text)
    if report.citation_rate < 0.5:
        print(f"引用率过低: {report.citation_rate:.1%}")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class ClaimMatch:
    """单条事实声明匹配结果。"""
    text: str          # 匹配到的原文片段
    pattern: str       # 命中的模式名称
    has_citation: bool  # 是否有来源标注


@dataclass
class CitationReport:
    """引用检测报告。"""
    total_claims: int = 0       # 事实声明总数
    cited_claims: int = 0       # 有来源标注的声明数
    uncited_claims: int = 0     # 缺少来源标注的声明数
    citation_rate: float = 0.0  # 引用率 = cited / total
    details: List[ClaimMatch] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.total_claims > 0:
            self.citation_rate = self.cited_claims / self.total_claims


# ── 事实声明检测模式 ──────────────────────────────────────────────
# 每个模式: (名称, 正则, 说明)
# 覆盖: 数字/日期/URL/论文引用/人名/百分比/统计数据

_CLAIM_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    # URL
    ("url", re.compile(
        r'https?://[^\s<>\)\]\"]+',
        re.IGNORECASE,
    ), "URL地址"),

    # 论文引用: (Author et al., Year) 或 [Author et al., Year] 或 arXiv:XXXX
    ("paper", re.compile(
        r'(?:[A-Z][a-z]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][a-z]+))?,\s*\d{4})'
        r'|'
        r'arXiv:\s*\d{4}\.\d{4,5}(?:v\d+)?',
    ), "论文引用"),

    # 日期: YYYY年MM月DD日 / YYYY-MM-DD / MM/DD/YYYY / YYYY年MM月 / YYYY年
    ("date", re.compile(
        r'\d{4}年(?:\d{1,2}月(?:\d{1,2}日)?)?'
        r'|\d{4}-\d{2}-\d{2}'
        r'|\d{1,2}/\d{1,2}/\d{4}',
    ), "日期"),

    # 百分比/统计数据
    ("statistic", re.compile(
        r'\d+(?:\.\d+)?\s*[%％]'
        r'|(?:占|达到|增长|下降|超过|不足)\s*\d+(?:\.\d+)?'
        r'|(?:约|大约|近|超过)\s*\d+(?:\.\d+)?(?:\s*[万亿千百])?',
    ), "统计数据"),

    # 数字声明: 带逗号分隔的大数字 或 带中文单位的数字 或 纯数字≥4位
    # 不用\b开头，用非贪婪匹配避免中文干扰
    ("number", re.compile(
        r'(?:\d{1,3}(?:,\d{3})+)'          # 1,400,000
        r'|(?:\d{4,})'                       # 10000+
        r'|(?:\d+(?:\.\d+)?\s*[万亿千百百万十]+)'  # 10万, 1.5亿
        r'|(?:\d+(?:\.\d+)?\s*(?:million|billion|trillion))'
        r'|(?:\d+(?:\.\d+)?\s*[种位条个名项件次套])',  # 100种, 50名
    ), "数字声明"),

    # 人名（中文两字以上 + 英文大写开头两词）
    ("person", re.compile(
        r'(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)'  # 英文名
        r'|'
        r'(?:(?:^|(?<=\s|，|,))(?:[^\x00-\x7F]{2,4})(?=(?:表示|说|认为|指出|提出|发现|证实|创立)))',  # 中文人名
    ), "人名"),
    # 注意：中文引用格式只作为引用标记（在citation_pattern中），
    # 不作为claims——因为它们本身就是引用，不需要再找引用。
]


def check_citations(text: str) -> CitationReport:
    """检测文本中事实声明的引用标注情况。

    Args:
        text: 模型输出的完整文本。

    Returns:
        CitationReport: 包含统计数据和逐条明细的报告。
    """
    if not text or not text.strip():
        return CitationReport()

    # 1. 提取所有已有标注的区间 (start, end)
    #    模式: [来源: ...] 或 ⚠️未验证 或 [citation: ...] 或 [ref: ...]
    citation_pattern = re.compile(
        r'\[(?:来源|citation|ref)\s*:\s*[^\]]*\]'
        r'|⚠️\s*未验证'
        r'|⚠️\s*未(?:确认|验证)'
        r'|(?:据|根据|依据)\s*[^\s，。]{2,10}\s*(?:报道|显示|数据|统计|调查|报告|研究|分析)'
        r'|(?:数据来源|来源|出处|参考)\s*[:：]\s*[^\s，。]+',
        re.IGNORECASE,
    )
    cited_spans: list[tuple[int, int]] = [
        (m.start(), m.end()) for m in citation_pattern.finditer(text)
    ]

    def _is_near_citation(pos: int, window: int = 80) -> bool:
        """检查 pos 位置是否在同一句内有标注。"""
        # 找到pos所在句子的边界（中文句号、问号、叹号、换行）
        sentence_start = max(0, text.rfind('。', 0, pos)
                                  if text.rfind('。', 0, pos) >= 0
                                  else text.rfind('\n', 0, pos)
                                  if text.rfind('\n', 0, pos) >= 0
                                  else 0)
        # 如果在句首附近，往前看一句
        if sentence_start > 0 and pos - sentence_start < 5:
            prev_sep = max(text.rfind('。', 0, sentence_start - 1),
                          text.rfind('\n', 0, sentence_start - 1))
            sentence_start = prev_sep + 1 if prev_sep >= 0 else 0

        sentence_end = len(text)
        for sep in ('。', '！', '？', '\n'):
            idx = text.find(sep, pos)
            if idx >= 0 and idx < sentence_end:
                sentence_end = idx + 1

        for c_start, c_end in cited_spans:
            # 标注在声明所在句内
            if c_start >= sentence_start and c_start <= sentence_end:
                return True
        return False

    # 2. 检测事实声明
    details: list[ClaimMatch] = []
    seen: set[int] = set()  # 去重: 按起始位置

    for pattern_name, pattern, _desc in _CLAIM_PATTERNS:
        for m in pattern.finditer(text):
            start = m.start()
            if start in seen:
                continue
            seen.add(start)
            has_cite = _is_near_citation(m.end())
            details.append(ClaimMatch(
                text=m.group(0),
                pattern=pattern_name,
                has_citation=has_cite,
            ))

    total = len(details)
    cited = sum(1 for d in details if d.has_citation)
    uncited = total - cited

    return CitationReport(
        total_claims=total,
        cited_claims=cited,
        uncited_claims=uncited,
        citation_rate=cited / total if total > 0 else 0.0,
        details=details,
    )
