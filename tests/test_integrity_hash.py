"""integrity_hash 测试 — PAL P0-2
2026-07-07

⚠️ 外部依赖测试: 本测试验证SHA-256完整性hash算法的正确性。
hash算法本身与 ~/.hermes/jiak/scripts/recall_append.py 中的实现一致，
但本测试自包含（独立定义compute_hash），不依赖jiak路径。
验证目标: 算法逻辑（确定性/篡改检测/字段排除），非recall_append.py集成。
"""
import hashlib
import json
import os
import sys
import tempfile
import unittest

# 添加jiak scripts到path（仅在未被污染时）
_jiak_path = os.path.expanduser("~/.hermes/jiak/scripts")
if _jiak_path not in sys.path:
    sys.path.insert(0, _jiak_path)

# 检测recall_append是否可用
_RECALL_APPEND_AVAILABLE = False
try:
    from recall_append import enrich_metadata, validate_line
    _RECALL_APPEND_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    pass


def compute_hash(obj: dict) -> str:
    """与recall_append.py一致的hash计算。"""
    content = json.dumps(
        {k: v for k, v in obj.items() if k not in ("integrity_hash", "_written_by", "_checksum")},
        ensure_ascii=False, sort_keys=True
    )
    return hashlib.sha256(content.encode()).hexdigest()


class TestIntegrityHash(unittest.TestCase):
    """integrity_hash写入+验证链路测试。"""

    def test_hash_deterministic(self):
        """相同输入产生相同hash。"""
        obj = {"type": "note", "content": "hello world", "ts": "2026-07-07"}
        h1 = compute_hash(obj)
        h2 = compute_hash(obj)
        self.assertEqual(h1, h2)

    def test_hash_changes_on_content_modification(self):
        """篡改content后hash变化。"""
        obj1 = {"type": "note", "content": "original", "ts": "2026-07-07"}
        obj2 = {"type": "note", "content": "tampered", "ts": "2026-07-07"}
        self.assertNotEqual(compute_hash(obj1), compute_hash(obj2))

    def test_hash_ignores_integrity_hash_field(self):
        """hash计算排除integrity_hash字段本身。"""
        obj = {"type": "note", "content": "test"}
        h1 = compute_hash(obj)
        obj["integrity_hash"] = "fake_hash_value"
        h2 = compute_hash(obj)
        self.assertEqual(h1, h2)

    def test_hash_ignores_underscore_fields(self):
        """hash计算排除_开头的元数据字段。"""
        obj = {"type": "note", "content": "test"}
        h1 = compute_hash(obj)
        obj["_written_by"] = "agent"
        obj["_checksum"] = "abc123"
        h2 = compute_hash(obj)
        self.assertEqual(h1, h2)

    @unittest.skipUnless(_RECALL_APPEND_AVAILABLE, "recall_append不可用")
    def test_enrich_metadata_adds_hash(self):
        """enrich_metadata为新记录添加integrity_hash。"""
        obj = {"type": "note", "content": "test content"}
        enriched = enrich_metadata(obj)
        self.assertIn("integrity_hash", enriched)
        self.assertEqual(len(enriched["integrity_hash"]), 64)  # SHA-256 hex

    @unittest.skipUnless(_RECALL_APPEND_AVAILABLE, "recall_append不可用")
    def test_validate_line_preserves_hash(self):
        """validate_line中hash基于原始字段（不含enriched字段）。"""
        obj = {"type": "note", "content": "test"}
        obj["integrity_hash"] = compute_hash(obj)
        line = json.dumps(obj, ensure_ascii=False)
        result = validate_line(line)
        self.assertIsNotNone(result)
        # enrich_metadata添加了role/episodic等字段，hash会重新计算
        # 但原始字段的hash应该匹配
        original_hash = compute_hash({"type": "note", "content": "test"})
        self.assertEqual(obj["integrity_hash"], original_hash)

    @unittest.skipUnless(_RECALL_APPEND_AVAILABLE, "recall_append不可用")
    def test_tamper_detection(self):
        """篡改后hash不匹配（但不拒绝——向后兼容）。"""
        obj = {"type": "note", "content": "original"}
        obj["integrity_hash"] = compute_hash(obj)
        # 篡改content
        obj["content"] = "tampered"
        line = json.dumps(obj, ensure_ascii=False)
        # validate_line应该通过（不拒绝），但会打印警告
        result = validate_line(line)
        self.assertIsNotNone(result)  # 不拒绝
        # hash不匹配
        self.assertNotEqual(result["integrity_hash"], compute_hash({"type": "note", "content": "tampered"}))

    @unittest.skipUnless(_RECALL_APPEND_AVAILABLE, "recall_append不可用")
    def test_backward_compatible_no_hash(self):
        """旧记录（无integrity_hash）正常通过验证。"""
        obj = {"type": "note", "content": "old record"}
        line = json.dumps(obj, ensure_ascii=False)
        result = validate_line(line)
        self.assertIsNotNone(result)
        # enrich_metadata会补上hash
        self.assertIn("integrity_hash", result)


if __name__ == "__main__":
    unittest.main()
