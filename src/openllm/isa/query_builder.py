"""query_builder — 检索 query 提炼层

病灶（2026-10-01 成市亲测）：用户消息整段原文直接当检索词，下游
recall_provider.search() 按"命中词数/总词数"打分——长文本里主题词被
自己的噪声词稀释，检索退化为掷骰子。

本模块把"整条消息"提炼成"主题词串"：
  ① 中英文都支持（中文 jieba + 词性过滤，英文小写 + 停用词）
  ② 去停用词与标点
  ③ 频次×词性权重取 top ≤max_terms 主题词
  ④ 保留上下文：短文本（≤SHORT_UNIT_LIMIT 个计量单元）原样返回；
     提炼为空/异常一律回退原文。**任何时候不返回空串**（空 query =
     检索零结果，比搜得差更恶劣）。

无新依赖：jieba 已在仓库内（memory_bus.py 同款懒加载模式）；jieba 不
可用时降级为 ASCII split + 中文单字过滤，提炼为空则回退原文。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Optional, Set

__all__ = ["build_query"]

# ── 短文本阈值：≤20 个计量单元（中文单字 / 英文单词各算 1）不提炼 ──
SHORT_UNIT_LIMIT = 20

# ── 中文词性白名单（实词：名词/动词/形容词/机构名等） ──
_ALLOWED_POS: Set[str] = {
    "n", "ns", "nt", "nz", "nl", "nj", "eng",      # 名词系 + 英文
    "v", "vn", "vd", "vg",                          # 动词系
    "a", "an", "ad", "ag",                          # 形容词系
    "i", "l",                                       # 成语/惯用语
}

# ── 停用词（中文虚词/口语/指代 + 英文高频词） ──
_STOP_WORDS: Set[str] = {
    # 中文
    "的", "了", "着", "过", "吗", "呢", "吧", "啊", "呀", "哦", "嗯", "哈", "嘛",
    "我", "你", "他", "她", "它", "们", "您", "咱们", "大家",
    "这", "那", "这个", "那个", "这些", "那些", "这里", "那里", "这样", "那样",
    "就", "都", "也", "还", "又", "才", "只", "更", "最", "太", "很", "挺",
    "和", "与", "或", "及", "而", "但", "并", "且", "所以", "因为", "如果",
    "虽然", "但是", "然后", "不过", "可是", "而且", "于是",
    "在", "于", "把", "被", "让", "给", "对", "从", "向", "由", "以", "以及",
    "不是", "没有", "一个", "一些", "什么", "怎么", "怎样", "为什么", "哪些",
    "多少", "可以", "可能", "应该", "需要", "觉得", "知道", "现在", "时候",
    "东西", "事情", "问题", "一下", "一直", "已经", "还是", "就是", "其实",
    "真的", "确实", "反正", "总之", "另外", "比如", "例如", "关于", "对于",
    "我们", "你们", "他们", "她们", "自己", "所有", "每", "各", "等等",
    # 英文
    "the", "a", "an", "and", "or", "but", "so", "if", "then", "than", "that",
    "this", "these", "those", "it", "its", "is", "are", "was", "were", "be",
    "been", "being", "do", "does", "did", "have", "has", "had", "will",
    "would", "could", "should", "can", "may", "might", "must", "shall",
    "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "our", "their", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "into", "about", "over", "under",
    "not", "no", "yes", "just", "very", "really", "some", "any", "all",
    "each", "other", "such", "too", "also", "there", "here", "when",
    "where", "what", "which", "who", "whom", "how", "why",
}

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_ASCII_WORD_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")


def _get_jieba_posseg():
    """懒加载 jieba.posseg；不可用返回 False（同款模式见 memory_bus._get_jieba）。"""
    try:
        import jieba.posseg as _pseg
        return _pseg
    except Exception:
        return False


def _count_units(text: str) -> int:
    """计量单元数：中文每字 1 个 + 英文/数字词每串 1 个。"""
    cjk = len(_CJK_RE.findall(text))
    ascii_words = len(_ASCII_WORD_RE.findall(_CJK_RE.sub(" ", text)))
    return cjk + ascii_words


def _fallback_tokens(text: str) -> List[str]:
    """jieba 不可用时的降级分词：ASCII 词 + 长度≥2 的非停用片段。
    （中文整句 split 不出 ≥2 字词 → 提炼为空 → 上层回退原文，宁粗不空。）"""
    tokens: List[str] = []
    for seg in re.split(r"[\s\W]+", text.lower(), flags=re.UNICODE):
        if not seg or seg in _STOP_WORDS:
            continue
        if _CJK_RE.search(seg):
            if len(seg) < 2:
                continue
        elif len(seg) < 2:
            continue
        tokens.append(seg)
    return tokens


def build_query(text: str, *, max_terms: int = 12) -> str:
    """从原文提炼检索 query（主题词串）。

    保证：
    - 短文本（≤20 计量单元）→ 原样返回（strip 后非空则不改写）；
    - 提炼结果为空 / 任何异常 → 回退原文；原文也是空/None → 返回 ""；
    - 同一输入两次调用结果一致（纯函数，无随机、无外部状态）。
    """
    if not isinstance(text, str) or not text.strip():
        return text.strip() if isinstance(text, str) else ""

    original = text.strip()

    # ① 短文本原样返回——12 个孤立词比一句短话更糟
    if _count_units(original) <= SHORT_UNIT_LIMIT:
        return original

    try:
        terms = _extract_terms(original, max_terms)
        if terms:
            return terms
    except Exception:
        pass  # 提炼失败 → 回退原文（比返回空串或崩溃都好）

    # ② 回退原文
    return original


def _extract_terms(text: str, max_terms: int) -> Optional[str]:
    """词性过滤 + 频次×稀有度加权取 top 主题词；失败/为空返回 None。"""
    pseg = _get_jieba_posseg()
    candidates: List[tuple] = []  # (word, pos, first_index)

    if pseg is not False:
        seen: Set[str] = set()
        for idx, pair in enumerate(pseg.cut(text.lower())):
            w = pair.word.strip()
            flag = pair.flag or "x"
            if not w or w in _STOP_WORDS:
                continue
            if _CJK_RE.search(w):
                if len(w) < 2:
                    continue
            else:
                if len(_ASCII_WORD_RE.findall(w)) == 0:
                    continue  # 纯标点/符号
                if w in seen:
                    continue  # 英文按词形去重（保留首次位置）
            head = flag[0]
            if head not in _ALLOWED_POS and flag not in _ALLOWED_POS:
                continue
            seen.add(w)
            candidates.append((w, flag, idx))
    else:
        for idx, w in enumerate(_fallback_tokens(text)):
            candidates.append((w, "n", idx))

    if not candidates:
        return None

    # 打分：频次 × 词性权重；长词（专有概念）小幅加权
    freq = Counter(w for w, _, _ in candidates)
    total = sum(freq.values()) or 1

    def pos_weight(flag: str) -> float:
        head = flag[0]
        if head in "nvra":
            return 1.0
        if head in "tnr":  # 时间/数词/人名 → 噪声偏高
            return 0.6
        return 0.8

    scored = []
    for word, flag, first in candidates:
        f = freq[word] / total
        # 稀有度：越长的词组越可能是专门术语，非饱和加权
        rarity = 1.0 + 0.25 * math.log(max(len(word), 1))
        score = (f ** 0.5) * pos_weight(flag) * rarity
        scored.append((score, -first, word))  # -first：同分时保留先出现的

    # 稳定排序：分数降序 → 位置升序 → 词本身（确定性兜底）
    scored.sort(key=lambda t: (-t[0], -t[1], t[2]))

    # 压缩保证：提炼串显著短于原文（≤40%），超了就砍最低分词
    limit = max(1, int(len(text) * 0.4))
    chosen: List[str] = []
    for _, _, w in scored:
        if w not in chosen:
            chosen.append(w)
        if len(chosen) >= max_terms:
            break
    query = " ".join(chosen)
    while len(chosen) > 2 and len(query) > limit:
        chosen.pop()
        query = " ".join(chosen)

    return query if query.strip() else None
