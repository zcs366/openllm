"""
Tests for symmetric_codec.py — 对称编解码器单元测试
"""

import hashlib
import json
import sys
import os

import pytest

# 确保 src 在路径中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from openllm.iko.symmetric_codec import (
    SymmetricCodec,
    CompressedReasoning,
    FullReasoningChain,
    _sha256_truncate,
    _chain_to_json,
)


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def _make_chain(
    phases: list[dict] | None = None,
    risk_assessments: list[dict] | None = None,
    tool_calls: list[dict] | None = None,
    memory_sources: list[str] | None = None,
) -> FullReasoningChain:
    """构造测试用的 FullReasoningChain。"""
    return FullReasoningChain(
        phases=phases if phases is not None else [
            {"description": "分析用户意图", "confidence": 0.9},
            {"description": "生成回答", "confidence": 0.8},
        ],
        tool_calls=tool_calls or [],
        risk_assessments=risk_assessments or [
            {"decision": "采用方案A", "risk": 0.2},
            {"decision": "拒绝敏感操作", "risk": 0.8},
        ],
        memory_sources=memory_sources or ["memory-001", "memory-002"],
    )


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------


def test_compress_basic():
    """测试 compress 正常工作——字段提取正确。"""
    codec = SymmetricCodec()
    chain = _make_chain()
    compressed = codec.compress(chain)

    assert isinstance(compressed, CompressedReasoning)
    assert compressed.summary == "分析用户意图"
    assert compressed.key_decisions == ["采用方案A", "拒绝敏感操作"]
    assert compressed.confidence == pytest.approx(0.85)
    assert len(compressed.full_chain_ref) == 16
    assert compressed.reversible is True


def test_compress_no_phases():
    """测试 compress 处理空 phases——summary 回退到 '无描述'。"""
    codec = SymmetricCodec()
    chain = FullReasoningChain(
        phases=[],
        tool_calls=[],
        risk_assessments=[{"decision": "保留"}],
        memory_sources=[],
    )
    compressed = codec.compress(chain)

    assert compressed.summary == "无描述"
    assert compressed.confidence == 0.0
    # phases 为空 → reversible=False
    assert compressed.reversible is False


def test_decompress_basic():
    """测试 decompress 正常工作——从压缩态恢复链。"""
    codec = SymmetricCodec()
    chain = _make_chain()
    compressed = codec.compress(chain)
    restored = codec.decompress(compressed)

    assert isinstance(restored, FullReasoningChain)
    assert len(restored.phases) == 1
    assert restored.phases[0]["description"] == "分析用户意图"
    assert restored.phases[0]["confidence"] == pytest.approx(0.85)
    assert restored.tool_calls == []
    # 包拯审计修正：key_decisions正确恢复到risk_assessments
    assert len(restored.risk_assessments) == 2
    assert restored.risk_assessments[0] == {"decision": "采用方案A"}
    assert restored.risk_assessments[1] == {"decision": "拒绝敏感操作"}
    assert restored.memory_sources == []


def test_roundtrip_key_fields_consistent():
    """往返测试：compress → decompress → 关键字段一致。"""
    codec = SymmetricCodec()
    chain = _make_chain()
    compressed = codec.compress(chain)
    restored = codec.decompress(compressed)

    # summary 一致
    assert restored.phases[0]["description"] == compressed.summary
    # confidence 一致
    assert restored.phases[0]["confidence"] == compressed.confidence
    # full_chain_ref 可以重新计算并匹配
    recomputed_ref = _sha256_truncate(_chain_to_json(chain).encode())
    assert compressed.full_chain_ref == recomputed_ref


def test_decompress_non_reversible_raises():
    """测试 reversible=False 时 decompress 抛出 ValueError（赫尔墨斯约束）。"""
    codec = SymmetricCodec()
    compressed = CompressedReasoning(
        summary="残缺摘要",
        key_decisions=[],
        confidence=0.5,
        full_chain_ref="abc123def456",
        reversible=False,
    )

    try:
        codec.decompress(compressed)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Cannot decompress non-reversible reasoning" in str(e)


def test_verify_reversibility():
    """测试 verify_reversibility 正确判断可恢复性。"""
    codec = SymmetricCodec()

    # 可恢复
    ok = CompressedReasoning(
        summary="有效摘要",
        key_decisions=["d1"],
        confidence=0.8,
        full_chain_ref="a" * 16,
        reversible=True,
    )
    assert codec.verify_reversibility(ok) is True

    # 不可恢复：reversible=False
    bad1 = CompressedReasoning(
        summary="摘要",
        key_decisions=[],
        confidence=0.5,
        full_chain_ref="b" * 16,
        reversible=False,
    )
    assert codec.verify_reversibility(bad1) is False

    # 不可恢复：summary 为空
    bad2 = CompressedReasoning(
        summary="",
        key_decisions=[],
        confidence=0.5,
        full_chain_ref="c" * 16,
        reversible=True,
    )
    assert codec.verify_reversibility(bad2) is False


def test_empty_chain():
    """测试空 chain 处理——所有字段为默认值。"""
    codec = SymmetricCodec()
    chain = FullReasoningChain()
    compressed = codec.compress(chain)

    assert compressed.summary == "无描述"
    assert compressed.key_decisions == []
    assert compressed.confidence == 0.0
    assert compressed.reversible is False  # phases 和 risk_assessments 均空
    assert len(compressed.full_chain_ref) == 16


def test_sha256_truncate_deterministic():
    """测试 _sha256_truncate 是确定性的——相同输入产生相同输出。"""
    data = b"test data for hashing"
    h1 = _sha256_truncate(data)
    h2 = _sha256_truncate(data)
    assert h1 == h2
    assert len(h1) == 16
    # 验证确实是 sha256 的前 16 位
    full = hashlib.sha256(data).hexdigest()
    assert h1 == full[:16]


def test_compress_custom_phases():
    """测试 compress 处理自定义 phases——description 缺失时回退。"""
    codec = SymmetricCodec()
    chain = FullReasoningChain(
        phases=[
            {"confidence": 0.7},  # 无 description
            {"description": "第二阶段", "confidence": 0.6},
        ],
        risk_assessments=[{"decision": "行动X"}],
    )
    compressed = codec.compress(chain)

    assert compressed.summary == "无描述"
    assert compressed.confidence == pytest.approx(0.65)
    assert compressed.key_decisions == ["行动X"]


def test_compression_ratio():
    """测试压缩比——压缩后数据量显著小于原始链。"""
    codec = SymmetricCodec()
    # 构造一个较大的推理链
    chain = FullReasoningChain(
        phases=[
            {"description": f"阶段{i}", "confidence": 0.5 + i * 0.05, "detail": "x" * 100}
            for i in range(10)
        ],
        tool_calls=[{"tool": f"tool_{i}", "args": {"k": "v" * 50}} for i in range(20)],
        risk_assessments=[{"decision": f"决策{i}", "risk": 0.1 * i} for i in range(10)],
        memory_sources=[f"mem-{i}" for i in range(15)],
    )
    compressed = codec.compress(chain)
    original_size = len(json.dumps(_chain_to_json(chain)))
    compressed_size = len(json.dumps({
        "summary": compressed.summary,
        "key_decisions": compressed.key_decisions,
        "confidence": compressed.confidence,
        "full_chain_ref": compressed.full_chain_ref,
    }))
    ratio = original_size / max(compressed_size, 1)
    # 压缩比应至少 3:1（目标 10:1，但测试数据可能不够大）
    assert ratio >= 3.0, f"Compression ratio too low: {ratio:.1f}x"
