"""
Tests for output_audit.py — 输出审计链单元测试
"""

import hashlib
import sys
import os

# 确保 src 在路径中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from openllm.iko.output_audit import (
    OutputAuditEntry,
    OutputAuditChain,
    _sha256_truncate,
    compute_entry_hash,
)


def test_entry_is_frozen():
    """测试 OutputAuditEntry 是不可变的（frozen=True）。"""
    entry = OutputAuditEntry(
        output_id="out-1",
        intent="test",
        content_hash="abcdef1234567890",
        decision_source="IKO",
        risk_level=0.1,
        confidence=0.9,
        prev_hash="genesis",
        timestamp=1.0,
        reasoning_chain_hash="abc123",
    )
    try:
        entry.output_id = "out-2"  # type: ignore[misc]
        assert False, "Should be immutable"
    except AttributeError:
        pass


def test_append_first_entry():
    """测试追加第一条条目，prev_hash 应为 genesis。"""
    chain = OutputAuditChain()
    entry = chain.append(
        output_id="out-1",
        intent="回答问题",
        content=b"hello world",
        decision_source="IKO",
        risk_level=0.0,
        confidence=0.95,
        reasoning_chain_hash="r1",
    )
    assert entry.prev_hash == "genesis"
    assert entry.content_hash == _sha256_truncate(b"hello world")
    assert len(chain) == 1


def test_append_chain_linking():
    """测试多条条目的 prev_hash 链式链接。"""
    chain = OutputAuditChain()
    e1 = chain.append(
        output_id="out-1", intent="a", content=b"alpha",
        decision_source="IKO", risk_level=0.1, confidence=0.9,
        reasoning_chain_hash="r1",
    )
    e2 = chain.append(
        output_id="out-2", intent="b", content=b"beta",
        decision_source="IKO", risk_level=0.2, confidence=0.8,
        reasoning_chain_hash="r2",
    )
    e3 = chain.append(
        output_id="out-3", intent="c", content=b"gamma",
        decision_source="IKO", risk_level=0.3, confidence=0.7,
        reasoning_chain_hash="r3",
    )

    # 第二条引用第一条的哈希
    assert e2.prev_hash == compute_entry_hash(e1)
    # 第三条引用第二条的哈希
    assert e3.prev_hash == compute_entry_hash(e2)
    assert len(chain) == 3


def test_verify_valid_chain():
    """测试验证有效链返回 True。"""
    chain = OutputAuditChain()
    for i in range(5):
        chain.append(
            output_id=f"out-{i}",
            intent=f"intent-{i}",
            content=f"content-{i}".encode(),
            decision_source="IKO",
            risk_level=0.1 * i,
            confidence=1.0 - 0.1 * i,
            reasoning_chain_hash=f"r{i}",
        )
    assert chain.verify() is True


def test_verify_empty_chain():
    """测试空链验证返回 True。"""
    chain = OutputAuditChain()
    assert chain.verify() is True


def test_get_provenance():
    """测试溯源查询返回正确的正序链。"""
    chain = OutputAuditChain()
    for i in range(3):
        chain.append(
            output_id=f"out-{i}",
            intent=f"intent-{i}",
            content=f"content-{i}".encode(),
            decision_source="IKO",
            risk_level=0.1,
            confidence=0.9,
            reasoning_chain_hash=f"r{i}",
        )
    provenance = chain.get_provenance("out-2")
    assert len(provenance) == 3
    assert provenance[0].output_id == "out-0"
    assert provenance[-1].output_id == "out-2"


def test_get_provenance_first_entry():
    """测试首条条目的溯源只有自身。"""
    chain = OutputAuditChain()
    chain.append(
        output_id="out-0", intent="first", content=b"first",
        decision_source="IKO", risk_level=0.0, confidence=1.0,
        reasoning_chain_hash="r0",
    )
    provenance = chain.get_provenance("out-0")
    assert len(provenance) == 1
    assert provenance[0].output_id == "out-0"


def test_duplicate_output_id_raises():
    """测试重复 output_id 抛出 ValueError。"""
    chain = OutputAuditChain()
    chain.append(
        output_id="out-1", intent="a", content=b"x",
        decision_source="IKO", risk_level=0.0, confidence=1.0,
        reasoning_chain_hash="r",
    )
    try:
        chain.append(
            output_id="out-1", intent="b", content=b"y",
            decision_source="IKO", risk_level=0.0, confidence=1.0,
            reasoning_chain_hash="r2",
        )
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "out-1" in str(e)


def test_sha256_truncate():
    """测试 sha256 截断函数。"""
    h = _sha256_truncate(b"test data")
    assert len(h) == 16
    full = hashlib.sha256(b"test data").hexdigest()
    assert h == full[:16]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
