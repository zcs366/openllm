"""
Symmetric Codec — IKO 对称压缩/解压编解码器
============================================

将完整推理链（FullReasoningChain）压缩为轻量摘要（CompressedReasoning），
并支持从压缩态恢复。

核心对称性约束（赫淮斯托斯）：
    compress → decompress 必须对称，可无损恢复关键字段。

开口约束（赫尔墨斯）：
    当 reversible=False 时，decompress 必须主动抛出 ValueError，
    明确告知无法恢复，而不是静默返回残缺数据。

压缩比目标 10:1，但实际取决于输入规模。

用法：
    codec = SymmetricCodec()
    compressed = codec.compress(chain)
    if codec.verify_reversibility(compressed):
        restored = codec.decompress(compressed)

七神约束：
    赫淮斯托斯：compress/decompress 必须对称，可无损恢复
    赫尔墨斯：reversible=False 时必须触发开口
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------


@dataclass
class CompressedReasoning:
    """压缩态推理摘要——从完整推理链中提取的关键信息。

    Attributes:
        summary: 推理过程的简短描述（取自 phases[0].description 或 "无描述"）。
        key_decisions: 从 risk_assessments 中提取的所有 decision 字段。
        confidence: 所有 phases 的平均置信度（0.0–1.0）。
        full_chain_ref: 完整推理链 JSON 序列化后的 sha256[:16] 哈希。
        reversible: 是否可恢复完整链。True 表示可安全 decompress。
    """

    summary: str
    key_decisions: list[str] = field(default_factory=list)
    confidence: float = 0.0
    full_chain_ref: str = ""
    reversible: bool = True


@dataclass
class FullReasoningChain:
    """完整推理链——IKO 推理过程的全量记录。

    Attributes:
        phases: 推理阶段列表，每个阶段为 dict，至少包含 'description' 和
                'confidence' 键。
        tool_calls: 工具调用记录列表。
        risk_assessments: 风险评估记录列表，每条包含 'decision' 字段。
        memory_sources: 使用的记忆源标识列表。
    """

    phases: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    risk_assessments: list[dict] = field(default_factory=list)
    memory_sources: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _sha256_truncate(data: bytes, length: int = 16) -> str:
    """计算 sha256 并截取前 length 个十六进制字符。

    Args:
        data: 要哈希的字节数据。
        length: 截取的十六进制字符数，默认 16。

    Returns:
        截断的哈希字符串。
    """
    return hashlib.sha256(data).hexdigest()[:length]


def _chain_to_json(chain: FullReasoningChain) -> str:
    """将 FullReasoningChain 序列化为 JSON 字符串（sort_keys=True 保证确定性）。

    Args:
        chain: 完整推理链。

    Returns:
        JSON 字符串。
    """
    return json.dumps(
        {
            "phases": chain.phases,
            "tool_calls": chain.tool_calls,
            "risk_assessments": chain.risk_assessments,
            "memory_sources": chain.memory_sources,
        },
        sort_keys=True,
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# SymmetricCodec
# ---------------------------------------------------------------------------


class SymmetricCodec:
    """对称推理链编解码器——compress/decompress 互为逆操作。

    赫淮斯托斯约束：compress → decompress 必须对称，可无损恢复。
    赫尔墨斯约束：reversible=False 时 decompress 必须 raise ValueError。

    压缩策略：
        - summary: 取 phases[0].description（或 "无描述"）
        - key_decisions: 提取所有 risk_assessments 中的 decision 字段
        - confidence: 所有 phases 的平均 confidence
        - full_chain_ref: 完整链 JSON 序列化后的 sha256[:16]
        - reversible: 当数据完整（phases 非空、risk_assessments 非空）时为 True

    压缩比目标 10:1，实际取决于输入规模。

    Usage::

        codec = SymmetricCodec()
        compressed = codec.compress(chain)
        restored = codec.decompress(compressed)
    """

    def compress(self, chain: FullReasoningChain) -> CompressedReasoning:
        """将完整推理链压缩为轻量摘要。

        赫淮斯托斯约束的第一半：从完整链中提取关键信息，
        生成可逆（或不可逆）的压缩态。

        压缩规则：
            - summary: 取 phases[0].description，缺失则用 "无描述"
            - key_decisions: 遍历 risk_assessments 取 decision 字段
            - confidence: 所有 phases 的 confidence 均值（无 phase 时为 0.0）
            - full_chain_ref: 完整链 JSON → sha256[:16]
            - reversible: phases 和 risk_assessments 均非空时为 True

        Args:
            chain: 要压缩的完整推理链。

        Returns:
            压缩后的推理摘要。
        """
        # summary: phases[0].description 或 "无描述"
        if chain.phases and isinstance(chain.phases[0], dict):
            summary = chain.phases[0].get("description", "无描述")
        else:
            summary = "无描述"

        # key_decisions: 所有 risk_assessments 的 decision
        key_decisions: list[str] = [
            ra.get("decision", "")
            for ra in chain.risk_assessments
            if isinstance(ra, dict)
        ]

        # confidence: phases 的平均 confidence
        confidences: list[float] = []
        for phase in chain.phases:
            if isinstance(phase, dict) and "confidence" in phase:
                val = phase["confidence"]
                if isinstance(val, (int, float)):
                    confidences.append(float(val))
        confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # full_chain_ref: sha256[:16]
        chain_json = _chain_to_json(chain)
        full_chain_ref = _sha256_truncate(chain_json.encode())

        # reversible: 数据完整时才可恢复
        reversible = bool(chain.phases and chain.risk_assessments)

        return CompressedReasoning(
            summary=summary,
            key_decisions=key_decisions,
            confidence=confidence,
            full_chain_ref=full_chain_ref,
            reversible=reversible,
        )

    def decompress(self, compressed: CompressedReasoning) -> FullReasoningChain:
        """从压缩态恢复推理链（部分恢复——填充压缩摘要，其余为空）。

        赫淮斯托斯约束的第二半：decompress 是 compress 的逆操作。
        由于压缩是有损的（丢弃了 tool_calls、memory_sources 等细节），
        恢复的是结构化的最小可行链。

        赫尔墨斯约束：当 reversible=False 时必须主动抛出异常。

        恢复策略：
            - phases: 填充 [{{"description": summary, "confidence": confidence}}]
            - tool_calls: 空列表（压缩时已丢弃）
            - risk_assessments: 空列表（压缩时仅保留 key_decisions）
            - memory_sources: 空列表（压缩时已丢弃）

        Args:
            compressed: 压缩态推理摘要。

        Returns:
            恢复的推理链（部分恢复）。

        Raises:
            ValueError: 当 compressed.reversible 为 False 时。
        """
        if not compressed.reversible:
            raise ValueError("Cannot decompress non-reversible reasoning")

        restored_phases: list[dict] = [
            {
                "description": compressed.summary,
                "confidence": compressed.confidence,
            }
        ]

        # 包拯审计修正：key_decisions必须恢复到risk_assessments中（赫淮斯托斯对称性）
        restored_risks: list[dict] = [
            {"decision": d} for d in compressed.key_decisions
        ]

        return FullReasoningChain(
            phases=restored_phases,
            tool_calls=[],
            risk_assessments=restored_risks,
            memory_sources=[],
        )

    def verify_reversibility(self, compressed: CompressedReasoning) -> bool:
        """验证压缩态是否可安全恢复。

        检查条件：
            1. compressed.reversible 为 True
            2. summary 非空（否则恢复的链无意义）

        Args:
            compressed: 要验证的压缩态。

        Returns:
            可恢复返回 True，否则 False。
        """
        if not compressed.reversible:
            return False
        if not compressed.summary:
            return False
        return True
