"""
IKO ↔ Hermes Bridge 测试
==========================

测试桥接层三个核心接口：
1. status()  — 返回 JSON 字典，包含 version、modules、lambda_value、timestamp
2. classify() — 返回正确 intent 和 reason
3. audit()    — 创建审计链文件（chain.jsonl + <output_id>.hash.json）
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from openllm.iko.iko_hermes_bridge import status, classify, audit


# ═══════════════════════════════════════════════════════════════════════
# 1. status() — 返回 JSON 字典
# ═══════════════════════════════════════════════════════════════════════

class TestStatus:
    """status() 返回合法 JSON 状态摘要。"""

    def test_returns_dict(self):
        """返回值是 dict。"""
        result = status()
        assert isinstance(result, dict)

    def test_has_version(self):
        """包含 version 字段，非空字符串。"""
        result = status()
        assert "version" in result
        assert isinstance(result["version"], str)
        assert len(result["version"]) > 0

    def test_has_modules_list(self):
        """包含 modules 字段，是 8 元素列表。"""
        result = status()
        assert "modules" in result
        assert isinstance(result["modules"], list)
        assert len(result["modules"]) == 8

    def test_has_lambda_value(self):
        """包含 lambda_value 字段，是 0-1 的浮点数。"""
        result = status()
        assert "lambda_value" in result
        lv = result["lambda_value"]
        assert isinstance(lv, float)
        assert 0.0 <= lv <= 1.0

    def test_has_timestamp(self):
        """包含 timestamp 字段，是正浮点数。"""
        result = status()
        assert "timestamp" in result
        assert isinstance(result["timestamp"], float)
        assert result["timestamp"] > 0

    def test_json_serializable(self):
        """返回值可序列化为 JSON。"""
        result = status()
        json_str = json.dumps(result)
        parsed = json.loads(json_str)
        assert parsed["version"] == result["version"]


# ═══════════════════════════════════════════════════════════════════════
# 2. classify() — 返回正确 intent
# ═══════════════════════════════════════════════════════════════════════

class TestClassify:
    """classify() 根据 context/decision 返回正确 intent。"""

    @staticmethod
    def _ctx(
        risk: str = "LOW",
        tool_calls: bool = False,
        side_effects: bool = False,
        options: int = 0,
    ) -> dict:
        return {
            "risk_level": risk,
            "has_tool_calls": tool_calls,
            "has_side_effects": side_effects,
            "option_count": options,
        }

    @staticmethod
    def _dec(content: str = "", dtype: str = "answer") -> dict:
        return {"type": dtype, "content": content}

    def test_inform_intent(self):
        """LOW 风险 + 非空内容 → INFORM。"""
        result = classify(self._ctx(), self._dec("Hello"))
        assert result["intent"] == "inform"
        assert "reason" in result

    def test_error_intent(self):
        """HIGH 风险 → ERROR。"""
        result = classify(self._ctx(risk="HIGH"), self._dec("fail"))
        assert result["intent"] == "error"

    def test_confirm_intent(self):
        """MEDIUM 风险 → CONFIRM。"""
        result = classify(self._ctx(risk="MEDIUM"), self._dec("delete"))
        assert result["intent"] == "confirm"

    def test_act_intent_tool_calls(self):
        """有工具调用 → ACT。"""
        result = classify(self._ctx(tool_calls=True), self._dec("run"))
        assert result["intent"] == "act"

    def test_act_intent_side_effects(self):
        """有副作用 → ACT。"""
        result = classify(self._ctx(side_effects=True), self._dec("deploy"))
        assert result["intent"] == "act"

    def test_decide_intent(self):
        """2+ 选项 → DECIDE。"""
        result = classify(self._ctx(options=3), self._dec("choose"))
        assert result["intent"] == "decide"

    def test_silent_intent(self):
        """空内容 + 无特殊条件 → SILENT。"""
        result = classify(self._ctx(), self._dec(""))
        assert result["intent"] == "silent"

    def test_returns_dict_with_expected_keys(self):
        """返回值包含 intent 和 reason 两个键。"""
        result = classify(self._ctx(), self._dec("test"))
        assert set(result.keys()) == {"intent", "reason"}

    def test_context_missing_field_raises(self):
        """缺少 context 必需字段 → ValueError。"""
        with pytest.raises(ValueError):
            classify({"risk_level": "LOW"}, self._dec("x"))

    def test_decision_missing_content_raises(self):
        """decision 缺少 content 字段 → ValueError。"""
        with pytest.raises(ValueError):
            classify(self._ctx(), {})


# ═══════════════════════════════════════════════════════════════════════
# 3. audit() — 创建审计链文件
# ═══════════════════════════════════════════════════════════════════════

class TestAudit:
    """audit() 创建审计链文件（chain.jsonl + hash 文件）。"""

    def test_creates_chain_file(self):
        """调用后 chain_dir/chain.jsonl 存在且非空。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = audit(tmpdir, "o1", "INFORM", b"hello")
            chain_file = Path(tmpdir) / "chain.jsonl"
            assert chain_file.exists()
            assert chain_file.stat().st_size > 0
            assert result["output_id"] == "o1"

    def test_creates_hash_file(self):
        """调用后 <output_id>.hash.json 存在。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            audit(tmpdir, "o2", "ACT", b"do something")
            hash_file = Path(tmpdir) / "o2.hash.json"
            assert hash_file.exists()
            data = json.loads(hash_file.read_text())
            assert data["output_id"] == "o2"
            assert "entry_hash" in data

    def test_chain_file_is_valid_jsonl(self):
        """chain.jsonl 每行是合法 JSON。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            audit(tmpdir, "o3", "ERROR", b"err")
            chain_file = Path(tmpdir) / "chain.jsonl"
            lines = chain_file.read_text().strip().split("\n")
            for line in lines:
                rec = json.loads(line)
                assert "output_id" in rec
                assert "_entry_hash" in rec

    def test_chain_length_increments(self):
        """多次 audit 后 chain_length 递增。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            r1 = audit(tmpdir, "a1", "INFORM", b"one")
            r2 = audit(tmpdir, "a2", "INFORM", b"two")
            r3 = audit(tmpdir, "a3", "INFORM", b"three")
            assert r1["chain_length"] == 1
            assert r2["chain_length"] == 2
            assert r3["chain_length"] == 3

    def test_duplicate_output_id_raises(self):
        """重复 output_id → ValueError。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            audit(tmpdir, "dup", "INFORM", b"x")
            with pytest.raises(ValueError, match="Duplicate"):
                audit(tmpdir, "dup", "INFORM", b"x")

    def test_returns_expected_keys(self):
        """返回值包含所有预期字段。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = audit(tmpdir, "o4", "DECIDE", b"choose")
            expected = {"output_id", "content_hash", "entry_hash", "chain_length", "path"}
            assert set(result.keys()) == expected

    def test_chain_file_contains_hex_content(self):
        """JSONL 记录包含 content_hex 字段（用于重建）。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            audit(tmpdir, "o5", "INFORM", b"test payload")
            chain_file = Path(tmpdir) / "chain.jsonl"
            rec = json.loads(chain_file.read_text().strip())
            assert "content_hex" in rec
            assert rec["content_hex"] == b"test payload".hex()

    def test_chain_verify_after_multiple_appends(self):
        """多次追加后链完整性验证通过。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            from openllm.iko.output_audit import OutputAuditChain
            for i in range(5):
                audit(tmpdir, f"v{i}", "INFORM", f"item-{i}".encode())
            # 重建链并验证
            chain = OutputAuditChain()
            chain_file = Path(tmpdir) / "chain.jsonl"
            for line in chain_file.read_text().strip().split("\n"):
                rec = json.loads(line)
                chain.append(
                    output_id=rec["output_id"],
                    intent=rec["intent"],
                    content=bytes.fromhex(rec["content_hex"]),
                    decision_source=rec["decision_source"],
                    risk_level=rec["risk_level"],
                    confidence=rec["confidence"],
                    reasoning_chain_hash=rec.get("reasoning_chain_hash", ""),
                )
            assert chain.verify()
            assert len(chain) == 5
