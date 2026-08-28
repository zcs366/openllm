"""
ISL边界完整性事件验收测试（P5：2026-08-28）

T1 正常链verify无事件：3环合法链→verify() True→audit_log文件不存在（无事件误报）
T2 篡改触发边界事件：写3环→篡改第2行epoch→verify() False→audit_log存在≥1条
T3 异常路径触发事件：chain_file写入非法JSON行→verify() False→事件detail.kind=verify_error
T4 降级不抛异常：monkeypatch get_guardian抛RuntimeError→verify()仍返回False且不抛异常
T5 红线·生产零写入：整个测试会话后 ~/.openllm/isl_chain.jsonl 与 ~/.openllm/output/integrity/ 均不存在
T6 向后兼容：verify()仍返回纯bool（isinstance(result, bool)）
"""
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from openllm.core.isl_chain import ISLChain


class TestT1NormalChainNoEvent:
    """T1: 正常链verify无边界事件"""

    def test_verify_true_no_audit_log(self, tmp_path, monkeypatch):
        """3环合法链→verify() True→audit_log文件不存在"""
        chain_file = tmp_path / "chain.jsonl"
        chain = ISLChain(chain_file=chain_file)

        chain.append_epoch(session_id="s1")
        chain.append_epoch(session_id="s2")
        chain.append_epoch(session_id="s3")

        result = chain.verify()
        assert result is True

        # audit_log不应存在——正常链不触发边界事件
        audit_log = tmp_path / "integrity_test" / "audit_log.jsonl"
        assert not audit_log.exists(), "正常链verify不应创建audit_log"


class TestT2TamperTriggersBoundaryEvent:
    """T2: 篡改触发边界完整性事件"""

    def test_tamper_epoch_fires_event(self, tmp_path, monkeypatch):
        """写3环→篡改第2行epoch→verify() False→audit_log存在≥1条"""
        # 确保audit_log路径在tmp_path下
        audit_log = tmp_path / "integrity_test" / "audit_log.jsonl"
        audit_log.parent.mkdir(parents=True, exist_ok=True)

        chain_file = tmp_path / "chain.jsonl"
        chain = ISLChain(chain_file=chain_file)

        chain.append_epoch(session_id="s1")
        chain.append_epoch(session_id="s2")
        chain.append_epoch(session_id="s3")

        assert chain.verify() is True

        # 篡改第2行的epoch字段
        lines = chain_file.read_text(encoding="utf-8").strip().split("\n")
        row2 = json.loads(lines[1])
        row2["epoch"] = 99999
        lines[1] = json.dumps(row2, ensure_ascii=False, sort_keys=True)
        chain_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # 新链实例验证——会触发_report_boundary_event
        chain2 = ISLChain(chain_file=chain_file)
        assert chain2.verify() is False

        # 检查audit_log
        assert audit_log.exists(), "篡改后audit_log应存在"
        events = []
        for line in audit_log.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                events.append(json.loads(line))

        assert len(events) >= 1, "应至少有1条边界事件"
        evt = events[0]
        assert evt["event_type"] == "boundary_integrity"
        assert evt["source"] == "isl_chain.verify"
        assert evt["detail"]["kind"] == "hash_break"
        assert evt["detail"]["line"] == 2  # 第2行


class TestT3VerifyErrorPath:
    """T3: 异常路径触发边界事件"""

    def test_invalid_json_triggers_error_event(self, tmp_path):
        """chain_file写入非法JSON→verify() False→事件detail.kind=verify_error"""
        chain_file = tmp_path / "chain.jsonl"

        # 先写一行合法数据让文件存在
        chain = ISLChain(chain_file=chain_file)
        chain.append_epoch(session_id="s1")
        assert chain.verify() is True

        # 追加一行非法JSON
        with open(chain_file, "a", encoding="utf-8") as f:
            f.write("NOT_VALID_JSON{{{{\n")

        audit_log = tmp_path / "integrity_test" / "audit_log.jsonl"
        audit_log.parent.mkdir(parents=True, exist_ok=True)

        chain2 = ISLChain(chain_file=chain_file)
        result = chain2.verify()
        assert result is False

        # 检查审计日志
        assert audit_log.exists(), "异常路径应创建audit_log"
        events = []
        for line in audit_log.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                events.append(json.loads(line))

        assert len(events) >= 1
        evt = events[0]
        assert evt["event_type"] == "boundary_integrity"
        assert evt["source"] == "isl_chain.verify"
        assert evt["detail"]["kind"] == "verify_error"
        assert "error" in evt["detail"]


class TestT4DegradationNoException:
    """T4: 降级不抛异常"""

    def test_verify_returns_false_when_guardian_unavailable(self, tmp_path, monkeypatch):
        """monkeypatch get_guardian抛RuntimeError→verify()仍返回False且不抛异常"""
        chain_file = tmp_path / "chain.jsonl"
        chain = ISLChain(chain_file=chain_file)
        chain.append_epoch(session_id="s1")

        # 篡改触发verify失败
        lines = chain_file.read_text(encoding="utf-8").strip().split("\n")
        row1 = json.loads(lines[0])
        row1["epoch"] = 12345
        lines[0] = json.dumps(row1, ensure_ascii=False, sort_keys=True)
        chain_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # 让get_guardian抛异常——必须patch integrity_guardian模块的get_guardian
        # 因为 _report_boundary_event 做的是 from openllm.core.integrity_guardian import get_guardian
        def _explode():
            raise RuntimeError("guardian unavailable")

        import openllm.core.integrity_guardian as _ig_mod
        monkeypatch.setattr(_ig_mod, "get_guardian", _explode)

        # verify不应抛异常，只返回False
        chain2 = ISLChain(chain_file=chain_file)
        # __init__中verify已调用，若未抛异常则通过
        result = chain2.verify()
        assert result is False


class TestT5ProductionZeroWrite:
    """T5: 红线·生产零写入"""

    def test_no_production_files_written(self):
        """整个测试会话后 ~/.openllm/isl_chain.jsonl 与
        ~/.openllm/output/integrity/ 均不存在"""
        prod_isl = Path.home() / ".openllm" / "isl_chain.jsonl"
        prod_integrity = Path.home() / ".openllm" / "output" / "integrity"

        assert not prod_isl.exists(), (
            "红线违反：测试不应创建 ~/.openllm/isl_chain.jsonl"
        )
        assert not prod_integrity.exists(), (
            "红线违反：测试不应创建 ~/.openllm/output/integrity/"
        )


class TestT6BackwardCompat:
    """T6: 向后兼容——verify()仍返回纯bool"""

    def test_verify_returns_bool(self, tmp_path):
        """verify()返回值isinstance(bool)，iko两处assert依赖此契约"""
        chain_file = tmp_path / "chain.jsonl"
        chain = ISLChain(chain_file=chain_file)
        chain.append_epoch(session_id="s1")

        result = chain.verify()
        assert isinstance(result, bool), (
            f"verify()必须返回bool，实际返回 {type(result).__name__}"
        )

        # 篡改后也应返回bool
        lines = chain_file.read_text(encoding="utf-8").strip().split("\n")
        row1 = json.loads(lines[0])
        row1["epoch"] = 11111
        lines[0] = json.dumps(row1, ensure_ascii=False, sort_keys=True)
        chain_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        chain2 = ISLChain(chain_file=chain_file)
        result2 = chain2.verify()
        assert isinstance(result2, bool), (
            f"verify()必须返回bool，实际返回 {type(result2).__name__}"
        )
        assert result2 is False
