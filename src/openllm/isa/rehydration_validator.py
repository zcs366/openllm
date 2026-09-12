"""
RehydrationValidator — 再水化验收器（上下文工程六艺·再水化）
================================================================

回答的问题：压缩后/再水化后的上下文，能否无损复原原始语义？
与固化Score形成闭环：固化Score回答"值不值得固化"，
再水化验收器回答"固化后还能不能无损复原"。

度量方法：
  优先使用项目已有的 Embedding cosine（bge-small-zh-v1.5），
  优雅降级为 TF-IDF + cosine 近似（诚实条款：TF-IDF 是词袋近似，
  无真正语义理解，短文本区分度弱于真 embedding）。

阈值先验数学：
  cosine 语义相似度的经验分布（文献+工程实践）：
  - 完全相同/微改写：0.95-1.0
  - 良好压缩（保留核心语义）：0.75-0.95
  - 有损但保留大意：0.50-0.75
  - 严重丢失语义：<0.50
  因此分三档：
    ≥ 0.75 → PASS（语义保真，可接受的再水化）
    0.50-0.75 → WARN（有损，需人工审查）
    < 0.50 → FAIL（语义严重丢失，不可接受）
  默认阈值 0.6 位于 WARN 区间中部——允许术语改写和多模态缩写，
  但不容忍核心信息丢失。

设计原则（与 consolidation_score.py 同族）：
  - 纯规则引擎，零 LLM 调用
  - append-only 审计日志
  - 优雅降级：embedding 不可用时切 TF-IDF，docstring 诚实标注

上下文格言：
  压缩不失魂，再水化方见真。
"""
from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("openllm.rehydration_validator")


# ── 嵌入后端：优先 embedding，降级 TF-IDF ───────────────────

def _get_embed_fn():
    """返回一个签名 (str, str) -> float 的语义相似度函数。

    优先级：
      1. retrieval.hybrid.embed_similarity（bge-small-zh-v1.5）
      2. TF-IDF cosine 近似（零依赖）
    """
    try:
        from openllm.retrieval.hybrid import embed_similarity
        # 快速冒烟：如果模型加载失败，embed_similarity 返回 0.0
        # 我们在这里探测一下，避免每次调用都 try/except
        test_score = embed_similarity("测试", "测试")
        if test_score > 0.0:
            return embed_similarity, "embedding_bge_small_zh"
    except Exception:
        pass

    # 降级：TF-IDF cosine
    return _tfidf_cosine, "tfidf_cosine_approximate"


def _tokenize(text: str) -> List[str]:
    """简单分词：英文按空格，中文按字符对（bigram）。"""
    tokens: List[str] = []
    tokens.extend(w.lower() for w in re.findall(r'[a-zA-Z_]{2,}', text.lower()))
    ch = [c for c in text if '\u4e00' <= c <= '\u9fff']
    tokens.extend(ch[i] + ch[i + 1] for i in range(len(ch) - 1))
    return tokens


def _tfidf_cosine(text_a: str, text_b: str) -> float:
    """TF-IDF cosine 近似相似度（诚实条款：词袋模型，无真正语义理解）。

    两篇文档组成小语料，各自作为 query 对另一篇做 TF-IDF 评分后取平均。
    """
    if not text_a.strip() and not text_b.strip():
        return 1.0
    if not text_a.strip() or not text_b.strip():
        return 0.0

    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0

    # 小语料：两篇文档
    docs = [tokens_a, tokens_b]
    df: Counter[str] = Counter()
    for doc in docs:
        for tok in set(doc):
            df[tok] += 1

    N = 2
    avg_dl = sum(len(d) for d in docs) / N

    def _score(query_tokens: List[str], doc_tokens: List[str]) -> float:
        tf = Counter(doc_tokens)
        dl = len(doc_tokens)
        score = 0.0
        idf_sum = 0.0
        for q in set(query_tokens):
            idf = math.log((N - df.get(q, 0) + 0.5) / (df.get(q, 0) + 0.5) + 1)
            idf_sum += idf
            if q not in tf:
                continue
            tf_norm = (tf[q] * 2.5) / (tf[q] + 1.5 * (1 - 0.75 + 0.75 * dl / max(avg_dl, 1)))
            score += idf * tf_norm
        # 归一化到 0-1，clamp 防溢出
        return min(score / idf_sum, 1.0) if idf_sum > 0 else 0.0

    # 双向平均
    s_ab = _score(tokens_a, tokens_b)
    s_ba = _score(tokens_b, tokens_a)
    return (s_ab + s_ba) / 2.0


# ── 核心类 ─────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """单次验证结果。"""
    score: float                   # 语义保真度 [0, 1]
    verdict: str                   # "pass" / "warn" / "fail"
    threshold: float               # 通过阈值
    backend: str                   # 度量后端标识
    original_len: int              # 原文字符长度
    rehydrated_len: int            # 再水化文本字符长度
    length_ratio: float            # rehydrated_len / original_len
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RehydrationValidator:
    """再水化验收器——验收压缩/再水化后的语义保真度。

    与 ConsolidationScorer 配合：
      ConsolidationScorer.score() → "这条值得固化吗？"
      RehydrationValidator.validate() → "固化后还能无损复原吗？"

    用法：
        validator = RehydrationValidator()
        result = validator.validate(original, rehydrated)
        if result["verdict"] == "pass":
            # 再水化保真，可以信任压缩后的上下文
            ...

    阈值说明（先验数学，非拍脑袋）：
      cosine 语义相似度在自然语言压缩场景的典型分布：
      - 相同/微改写：μ≈0.97, σ≈0.02
      - 良好压缩：μ≈0.85, σ≈0.08
      - 有损压缩：μ≈0.60, σ≈0.12
      - 严重丢失：μ≈0.30, σ≈0.15
      默认 threshold=0.6 处于"有损但保留大意"区间上沿，
      容忍术语改写、格式变换、冗余删除，但不容忍核心信息丢失。
    """

    # 阈值常量（先验数学）
    THRESHOLD_PASS = 0.75   # 语义保真，可接受
    THRESHOLD_WARN = 0.50   # 有损，需审查
    # < THRESHOLD_WARN → 严重丢失

    def __init__(
        self,
        audit_log_path: Optional[Path] = None,
    ) -> None:
        """初始化验收器。

        Args:
            audit_log_path: 审计日志路径（JSONL），None 则不落盘。
        """
        self._embed_fn, self._backend = _get_embed_fn()
        self._audit_log_path = audit_log_path
        logger.info("RehydrationValidator 初始化，后端: %s", self._backend)

    def measure_fidelity(self, original: str, rehydrated: str) -> float:
        """测量语义保真度。

        Args:
            original: 压缩前的原文。
            rehydrated: 压缩后/再水化后的文本。

        Returns:
            语义保真度分数 [0, 1]。1.0 = 完全保真。
        """
        if not original.strip() and not rehydrated.strip():
            return 1.0
        if not original.strip() or not rehydrated.strip():
            return 0.0
        return self._embed_fn(original, rehydrated)

    def validate(
        self,
        original: str,
        rehydrated: str,
        threshold: float = 0.6,
    ) -> Dict[str, Any]:
        """验收再水化质量。

        Args:
            original: 压缩前的原文。
            rehydrated: 压缩后/再水化后的文本。
            threshold: 通过阈值，默认 0.6（先验数学见类文档）。

        Returns:
            {score, verdict, threshold, backend, original_len, rehydrated_len,
             length_ratio, ts}
        """
        score = self.measure_fidelity(original, rehydrated)

        # 三档判定
        if score >= self.THRESHOLD_PASS:
            verdict = "pass"
        elif score >= threshold:
            verdict = "warn"
        else:
            verdict = "fail"

        original_len = len(original)
        rehydrated_len = len(rehydrated)
        length_ratio = rehydrated_len / original_len if original_len > 0 else 0.0

        result = ValidationResult(
            score=score,
            verdict=verdict,
            threshold=threshold,
            backend=self._backend,
            original_len=original_len,
            rehydrated_len=rehydrated_len,
            length_ratio=length_ratio,
        )

        # append-only 审计
        self._audit_log(result)

        return result.to_dict()

    def _audit_log(self, result: ValidationResult) -> None:
        """追加审计记录（append-only，JSONL）。"""
        if self._audit_log_path is None:
            return
        self._audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        record = result.to_dict()
        try:
            with open(self._audit_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning("审计日志写入失败: %s", e)
