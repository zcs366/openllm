"""厌倦律 IoR 过滤器测试。

覆盖：
  - 正常路径：mark/suppress/get_recent
  - 边界条件：空列表、去重、limit截断
  - 异常路径：损坏JSONL、不存在路径
  - 集成：build_context注入ior_hints
  - append-only 纪律验证
"""
import json
import time
from pathlib import Path

import pytest

from openllm.iai.ior_filter import IoRFilter, _topic_hash


# ── 基础功能 ──────────────────────────────────────


class TestIoRBasics:
    def test_mark_and_get_recent(self, tmp_path: Path):
        f = tmp_path / "ior.jsonl"
        ior = IoRFilter(path=f)
        ior.mark_handled("安全例会")
        ior.mark_handled("三超一疲劳")
        recent = ior.get_recent(limit=10)
        assert len(recent) == 2
        assert "三超一疲劳" in recent
        assert "安全例会" in recent

    def test_mark_dedup(self, tmp_path: Path):
        """同一主题标记两次，只保留一条记录。"""
        f = tmp_path / "ior.jsonl"
        ior = IoRFilter(path=f)
        ior.mark_handled("安全例会")
        ior.mark_handled("安全例会")
        recent = ior.get_recent(limit=10)
        assert len(recent) == 1

    def test_suppress_filters_handled(self, tmp_path: Path):
        f = tmp_path / "ior.jsonl"
        ior = IoRFilter(path=f)
        ior.mark_handled("安全例会")
        ior.mark_handled("三超一疲劳")
        result = ior.suppress(["安全例会", "三超一疲劳", "新主题"])
        assert result == ["新主题"]

    def test_suppress_empty_when_nothing_handled(self, tmp_path: Path):
        f = tmp_path / "ior.jsonl"
        ior = IoRFilter(path=f)
        result = ior.suppress(["安全例会", "新主题"])
        assert result == ["安全例会", "新主题"]

    def test_suppress_empty_input(self, tmp_path: Path):
        f = tmp_path / "ior.jsonl"
        ior = IoRFilter(path=f)
        ior.mark_handled("安全例会")
        result = ior.suppress([])
        assert result == []

    def test_get_recent_limit(self, tmp_path: Path):
        f = tmp_path / "ior.jsonl"
        ior = IoRFilter(path=f)
        for i in range(15):
            ior.mark_handled(f"topic_{i}")
        recent = ior.get_recent(limit=5)
        assert len(recent) == 5
        # 最新的在前
        assert recent[0] == "topic_14"

    def test_suppress_partial_overlap(self, tmp_path: Path):
        """部分匹配——只有完全相同hash的被过滤。"""
        f = tmp_path / "ior.jsonl"
        ior = IoRFilter(path=f)
        ior.mark_handled("安全例会")
        result = ior.suppress(["安全例会", "安全生产", "安全检查"])
        assert result == ["安全生产", "安全检查"]


# ── topic_hash 一致性 ──────────────────────────────


class TestTopicHash:
    def test_same_text_same_hash(self):
        assert _topic_hash("安全例会") == _topic_hash("安全例会")

    def test_case_insensitive(self):
        assert _topic_hash("Hello") == _topic_hash("hello")

    def test_whitespace_insensitive(self):
        assert _topic_hash("  安全  例会  ") == _topic_hash("安全 例会")

    def test_different_text_different_hash(self):
        assert _topic_hash("安全例会") != _topic_hash("安全生产")


# ── append-only 纪律 ───────────────────────────────


class TestIoRAppendOnly:
    def test_file_is_append_only(self, tmp_path: Path):
        """新记录不删除旧记录。"""
        f = tmp_path / "ior.jsonl"
        ior = IoRFilter(path=f)
        ior.mark_handled("topic_a")
        ior.mark_handled("topic_b")
        # 重新读文件验证全部存在
        lines = f.read_text().strip().split("\n")
        assert len(lines) == 2
        topics = [json.loads(l)["topic"] for l in lines]
        assert "topic_a" in topics
        assert "topic_b" in topics

    def test_file_format_jsonl(self, tmp_path: Path):
        """每行一个JSON记录。"""
        f = tmp_path / "ior.jsonl"
        ior = IoRFilter(path=f)
        ior.mark_handled("安全例会")
        ior.mark_handled("三超一疲劳")
        for line in f.read_text().strip().split("\n"):
            record = json.loads(line)
            assert "topic" in record
            assert "hash" in record
            assert "timestamp" in record

    def test_reload_picks_up_existing(self, tmp_path: Path):
        """新实例加载已有记录。"""
        f = tmp_path / "ior.jsonl"
        ior1 = IoRFilter(path=f)
        ior1.mark_handled("topic_x")
        # 新实例
        ior2 = IoRFilter(path=f)
        assert ior2.suppress(["topic_x", "topic_y"]) == ["topic_y"]
        assert ior2.get_recent() == ["topic_x"]


# ── 异常路径 ──────────────────────────────────────


class TestIoREdgeCases:
    def test_corrupted_jsonl_skipped(self, tmp_path: Path):
        """损坏行被跳过，不崩溃。"""
        f = tmp_path / "ior.jsonl"
        f.write_text("not json\n{\"topic\": \"ok\", \"hash\": \"abc\", \"timestamp\": 1}\n\n")
        ior = IoRFilter(path=f)
        recent = ior.get_recent()
        assert recent == ["ok"]

    def test_nonexistent_path_fresh_start(self, tmp_path: Path):
        """不存在的路径 = 全新起点。"""
        f = tmp_path / "nonexistent" / "ior.jsonl"
        ior = IoRFilter(path=f)
        assert ior.get_recent() == []
        assert ior.suppress(["topic_a"]) == ["topic_a"]
        ior.mark_handled("topic_a")
        assert f.exists()


# ── build_context 集成 ─────────────────────────────


class TestIoRBuildContextIntegration:
    def test_build_context_injects_ior_hints(self, tmp_path: Path):
        """有已处理主题时，context.ior_hints 非空。"""
        from openllm.core.isa_impl import ISA
        from openllm.core.models import Message

        isa = ISA(mode="silent")
        # 手动注入IoR滤镜（避免干扰生产路径）
        from openllm.iai.ior_filter import IoRFilter
        isa._ior_filter = IoRFilter(path=tmp_path / "ior.jsonl")
        isa._ior_filter.mark_handled("安全例会")
        isa._ior_filter.mark_handled("三超一疲劳")

        ctx = isa.build_context(Message(text="测试消息"))
        assert len(ctx.ior_hints) == 1
        assert "安全例会" in ctx.ior_hints[0]
        assert "三超一疲劳" in ctx.ior_hints[0]
        assert "不重复" in ctx.ior_hints[0]

    def test_build_context_no_ior_when_empty(self, tmp_path: Path):
        """无已处理主题时，context.ior_hints 为空列表。"""
        from openllm.core.isa_impl import ISA
        from openllm.core.models import Message
        from openllm.iai.ior_filter import IoRFilter

        isa = ISA(mode="silent")
        isa._ior_filter = IoRFilter(path=tmp_path / "ior_empty.jsonl")
        ctx = isa.build_context(Message(text="测试消息"))
        assert ctx.ior_hints == []

    def test_build_context_no_ior_when_filter_unavailable(self):
        """IoR滤镜不可用时，不影响build_context。"""
        from openllm.core.isa_impl import ISA
        from openllm.core.models import Message

        isa = ISA(mode="silent")
        isa._ior_filter = None  # 模拟不可用
        ctx = isa.build_context(Message(text="测试消息"))
        assert ctx.ior_hints == []
        # 其他字段仍正常
        assert ctx.user_message == "测试消息"
        assert isinstance(ctx.identity, dict)


# ── 边界：空文件 ──────────────────────────────────


class TestIoREmptyFile:
    def test_empty_file_returns_empty(self, tmp_path: Path):
        f = tmp_path / "ior.jsonl"
        f.write_text("")
        ior = IoRFilter(path=f)
        assert ior.get_recent() == []
        assert ior.suppress(["a"]) == ["a"]
