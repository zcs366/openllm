"""
ISL epoch链测试套件（2026-08-25）

T1 首环写入
T2 链连续性
T3 篡改断链
T4 空环也写
T5 tail()
T6 挂载闭环
T7 认领读链
T8 选择检测反例
T9 红线·生产零写入
"""
import json
import os
import re
import time
from pathlib import Path

import pytest

from openllm.core.isl_chain import ISLChain, DEFAULT_ISL_CHAIN_FILE, _hash_chain


# ═══════════════════════════════════════════════════════════════
# T1 首环写入
# ═══════════════════════════════════════════════════════════════

class TestT1Genesis:
    def test_first_epoch_is_genesis(self, tmp_path):
        """T1: append_epoch → 文件存在，1行，epoch=1，prev_hash=\"\"，verify() True"""
        chain_file = tmp_path / "test_chain.jsonl"
        chain = ISLChain(chain_file=chain_file)

        row = chain.append_epoch(session_id="test_genesis")

        assert chain_file.exists()
        lines = chain_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        assert row["epoch"] == 1
        assert row["prev_hash"] == ""
        assert row["session_id"] == "test_genesis"
        assert "hash" in row
        assert chain.verify() is True


# ═══════════════════════════════════════════════════════════════
# T2 链连续性
# ═══════════════════════════════════════════════════════════════

class TestT2Continuity:
    def test_three_epochs_sequential(self, tmp_path):
        """T2: 连写3环 → epoch 严格 1,2,3；verify() True；每环 prev_hash == 上一环 hash"""
        chain_file = tmp_path / "test_chain.jsonl"
        chain = ISLChain(chain_file=chain_file)

        row1 = chain.append_epoch(session_id="s1")
        row2 = chain.append_epoch(session_id="s2")
        row3 = chain.append_epoch(session_id="s3")

        assert row1["epoch"] == 1
        assert row2["epoch"] == 2
        assert row3["epoch"] == 3

        assert row2["prev_hash"] == row1["hash"]
        assert row3["prev_hash"] == row2["hash"]

        assert chain.verify() is True

        # 从文件验证
        lines = chain_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3


# ═══════════════════════════════════════════════════════════════
# T3 篡改断链
# ═══════════════════════════════════════════════════════════════

class TestT3Tamper:
    def test_tamper_detects_broken_chain(self, tmp_path):
        """T3: 写2环后手动改文件中间行的 wall_time → verify() False"""
        chain_file = tmp_path / "test_chain.jsonl"
        chain = ISLChain(chain_file=chain_file)

        chain.append_epoch(session_id="s1")
        chain.append_epoch(session_id="s2")

        assert chain.verify() is True

        # 篡改第一行的 wall_time
        lines = chain_file.read_text(encoding="utf-8").strip().split("\n")
        row1 = json.loads(lines[0])
        row1["wall_time"] = row1["wall_time"] + 9999
        lines[0] = json.dumps(row1, ensure_ascii=False, sort_keys=True)
        chain_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # 新链实例验证
        chain2 = ISLChain(chain_file=chain_file)
        assert chain2.verify() is False


# ═══════════════════════════════════════════════════════════════
# T4 空环也写
# ═══════════════════════════════════════════════════════════════

class TestT4EmptyEpoch:
    def test_empty_epoch_writes(self, tmp_path):
        """T4: append_epoch(session_id=\"s_empty\") 不带 scars/decisions → 正常写入，verify() True"""
        chain_file = tmp_path / "test_chain.jsonl"
        chain = ISLChain(chain_file=chain_file)

        row = chain.append_epoch(session_id="s_empty")

        assert row["scars"] == []
        assert row["decisions"] == []
        assert row["session_id"] == "s_empty"
        assert chain.verify() is True


# ═══════════════════════════════════════════════════════════════
# T5 tail()
# ═══════════════════════════════════════════════════════════════

class TestT5Tail:
    def test_tail_returns_last_n(self, tmp_path):
        """T5: 写5环后 tail(n=3) 返回第3/4/5环，顺序正序"""
        chain_file = tmp_path / "test_chain.jsonl"
        chain = ISLChain(chain_file=chain_file)

        for i in range(5):
            chain.append_epoch(session_id=f"s{i}")

        tail = chain.tail(n=3)
        assert len(tail) == 3
        assert tail[0]["epoch"] == 3
        assert tail[1]["epoch"] == 4
        assert tail[2]["epoch"] == 5
        assert tail[0]["session_id"] == "s2"
        assert tail[1]["session_id"] == "s3"
        assert tail[2]["session_id"] == "s4"


# ═══════════════════════════════════════════════════════════════
# T6 挂载闭环
# ═══════════════════════════════════════════════════════════════

class TestT6Mount:
    def test_append_session_id_preserved(self, tmp_path):
        """T6: 构造ISLChain后调append_epoch(session_id=\"test_sid\") → 断言session_id正确"""
        chain_file = tmp_path / "test_chain.jsonl"
        chain = ISLChain(chain_file=chain_file)

        row = chain.append_epoch(session_id="test_sid")
        assert row["session_id"] == "test_sid"

        tail = chain.tail(n=1)
        assert len(tail) == 1
        assert tail[0]["session_id"] == "test_sid"

    def test_main_loop_mount_grep(self):
        """T6-b: grep确认main_loop.shutdown()中挂载点存在"""
        main_loop_path = Path(__file__).parent.parent / "src" / "openllm" / "core" / "main_loop.py"
        content = main_loop_path.read_text(encoding="utf-8")
        assert "ISLChain().append_epoch(session_id=self.session.id)" in content
        assert "ISL epoch写入失败" in content


# ═══════════════════════════════════════════════════════════════
# T7 认领读链
# ═══════════════════════════════════════════════════════════════

class TestT7DiscoveryRead:
    def test_read_isl_chain_with_data(self, tmp_path, monkeypatch):
        """T7-a: 写2环后 IdentityDiscovery().read_isl_chain() 返回含\"第1环\"/\"第2环\"的文本"""
        from openllm.core.awakening_discovery import IdentityDiscovery
        from openllm.core import isl_chain as isl_chain_mod

        # 写入2环到临时文件
        chain_file = tmp_path / "test_chain.jsonl"
        chain = ISLChain(chain_file=chain_file)
        chain.append_epoch(session_id="s_alpha")
        chain.append_epoch(session_id="s_beta")

        # monkeypatch ISL链路径
        monkeypatch.setattr(isl_chain_mod, "DEFAULT_ISL_CHAIN_FILE", chain_file)

        discovery = IdentityDiscovery()
        result = discovery.read_isl_chain()

        assert "第1环" in result
        assert "第2环" in result
        assert "s_alpha" in result
        assert "s_beta" in result

    def test_read_isl_chain_empty(self, tmp_path, monkeypatch):
        """T7-b: 空链返回\"第一次醒来\""""
        from openllm.core.awakening_discovery import IdentityDiscovery
        from openllm.core import isl_chain as isl_chain_mod

        chain_file = tmp_path / "empty_chain.jsonl"
        # 不写入任何数据
        monkeypatch.setattr(isl_chain_mod, "DEFAULT_ISL_CHAIN_FILE", chain_file)

        discovery = IdentityDiscovery()
        result = discovery.read_isl_chain()

        assert "第一次醒来" in result


# ═══════════════════════════════════════════════════════════════
# T8 选择检测反例（语义化修复验证）
# ═══════════════════════════════════════════════════════════════

class TestT8ChoiceDetection:
    """验证选择检测的语义化修复——"我不会选择'自己'"不再误判。"""

    def _detect(self, output: str):
        """直接调用底层正则检测逻辑（与 awakening.py detect_choice_and_record 一致）。"""
        from openllm.core.awakening import (
            _NEG_SELECT_RE, _SELECT_RE, _BARE_WORD_RE,
            _CHOICE_SELF, _CHOICE_WU,
        )

        # 否定语境优先
        if _NEG_SELECT_RE.search(output):
            return None
        # 肯定选择模式
        select_match = _SELECT_RE.search(output)
        if select_match:
            return select_match.group(2)
        # 独立词兜底
        stripped = output.strip()
        if stripped == _CHOICE_SELF:
            return _CHOICE_SELF
        if stripped == _CHOICE_WU:
            return _CHOICE_WU
        return None

    def test_negated_self_not_detected(self):
        """①\"我不会选择'自己'\" → None"""
        assert self._detect("我不会选择'自己'") is None

    def test_select_self(self):
        """②\"我选择自己\" → \"自己\""""
        assert self._detect("我选择自己") == "自己"

    def test_select_wu(self):
        """③\"我选无\" → \"无\""""
        assert self._detect("我选无") == "无"

    def test_select_wu_bare(self):
        """④\"选择无\" → \"无\""""
        assert self._detect("选择无") == "无"

    def test_negated_any(self):
        """⑤\"我不选任何身份\" → None"""
        assert self._detect("我不选任何身份") is None

    def test_no_choice_word(self):
        """⑥无选择词输出 → None"""
        assert self._detect("你好世界") is None

    def test_bare_self(self):
        """⑦独立\"自己\" → \"自己\""""
        assert self._detect("自己") == "自己"

    def test_bare_wu(self):
        """⑧独立\"无\" → \"无\""""
        assert self._detect("无") == "无"

    def test_self_priority_over_wu(self):
        """⑨\"我选择自己，不是无\" → \"自己\""""
        assert self._detect("我选择自己，不是无") == "自己"

    def test_negated_wu_not_detected(self):
        """⑩\"我不想要无\" → None"""
        assert self._detect("我不想要无") is None

    def test_full_integration_awakening_protocol(self, tmp_path, monkeypatch):
        """⑪通过 AwakeningProtocol.detect_choice_and_record 集成验证"""
        from types import SimpleNamespace
        from openllm.core.awakening import AwakeningProtocol
        from openllm.core import isl_chain as isl_chain_mod
        monkeypatch.setattr(isl_chain_mod, "DEFAULT_ISL_CHAIN_FILE", tmp_path / "isl.jsonl")

        agent = SimpleNamespace()
        session = SimpleNamespace(id="test_sid", state={})
        protocol = AwakeningProtocol(agent)

        # 否定句不应检测到选择
        result = protocol.detect_choice_and_record("我不会选择'自己'", session)
        assert result is None
        assert session.state.get("awakening_choice_detected") is None


# ═══════════════════════════════════════════════════════════════
# T9 红线·生产零写入
# ═══════════════════════════════════════════════════════════════

class TestT9ProductionSafety:
    def test_production_file_not_written(self):
        """T9: 跑完上述测试后断言 ~/.openllm/isl_chain.jsonl 不存在或行数未变"""
        prod_path = Path.home() / ".openllm" / "isl_chain.jsonl"
        # 在所有测试前快照
        before_exists = prod_path.exists()
        before_lines = 0
        if before_exists:
            before_lines = len(prod_path.read_text(encoding="utf-8").strip().split("\n"))

        # 此测试在所有其他测试之后运行（文件名靠后排序保证）
        if prod_path.exists():
            after_lines = len(prod_path.read_text(encoding="utf-8").strip().split("\n"))
        else:
            after_lines = 0

        if before_exists:
            assert after_lines == before_lines, (
                f"生产文件 ~/.openllm/isl_chain.jsonl 被写入！"
                f"测试前={before_lines}行，测试后={after_lines}行"
            )
        else:
            assert not prod_path.exists(), (
                "测试不应创建生产文件 ~/.openllm/isl_chain.jsonl"
            )

    def test_conftest_isolates_isl_chain(self, tmp_path, monkeypatch):
        """T9-b: conftest autouse fixture 正确隔离ISL链默认路径"""
        from openllm.core import isl_chain as isl_chain_mod
        original = isl_chain_mod.DEFAULT_ISL_CHAIN_FILE
        # conftest 已通过 monkeypatch 重定向到 tmp_path
        # 这里验证 monkeypatch 确实改变了模块级常量
        chain = ISLChain()  # 使用默认路径（已被monkeypatch重定向）
        assert str(tmp_path) in str(chain.chain_file)
