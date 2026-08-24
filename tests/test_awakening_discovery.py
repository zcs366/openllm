"""
苏醒协议路B — 身份发现序列（上下文组装器）测试套件

覆盖：
1. read_skills：临时 skill 目录含 frontmatter → 返回正确条目、name 排序
2. read_skills 容错：无 SKILL.md / 损坏内容 → 跳过不崩
3. read_recent_sessions：多文件不同 mtime → 倒序取指定数量；目录不存在 → []
4. read_hot_scars：mock get_causal_store → 透传；异常 → 返回 ''
5. build_discovery_context：mock 三来源 → 四键齐全，self_question 逐字匹配
6. render_prompt_block：全空来源 → 含'（暂无）'不崩；有数据 → 含技能名和伤疤块
"""
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from openllm.core.awakening_discovery import (
    IdentityDiscovery,
    discovery_context,
    _SELF_QUESTION,
    _GUIDANCE,
)


# ── 辅助 ──────────────────────────────────────────────────────

def _create_skill_dir(
    base: Path, name: str, description: str = "测试技能描述"
) -> None:
    """在 base 下创建一个包含 SKILL.md（含 frontmatter）的 skill 目录。"""
    skill_dir = base / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    content = f"""---
name: {name}
description: {description}
---
# {name}

这是 {name} 的正文。
"""
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")


def _create_session_file(
    directory: Path, filename: str, mtime_offset: float = 0.0
) -> Path:
    """在 directory 下创建一个 session 文件并设置修改时间。"""
    directory.mkdir(parents=True, exist_ok=True)
    fpath = directory / filename
    fpath.write_text('{"messages": []}', encoding="utf-8")
    # 设置 mtime（offset 越大越新）
    base_time = time.time() + mtime_offset
    fpath.touch()
    import os
    os.utime(fpath, (base_time, base_time))
    return fpath


# ── read_skills 测试 ──────────────────────────────────────────

class TestReadSkills:
    """read_skills 方法测试。"""

    def test_read_skills_with_frontmatter(self, tmp_path: Path) -> None:
        """造 2 个临时 skill 目录（含 frontmatter description）→ 返回 2 条，name 排序正确。"""
        _create_skill_dir(tmp_path, "zebra-skill", "Z 技能描述")
        _create_skill_dir(tmp_path, "alpha-skill", "A 技能描述")

        discovery = IdentityDiscovery(skill_dirs=[tmp_path])
        results = discovery.read_skills(max_items=10)

        assert len(results) == 2
        # 按 name 排序
        assert results[0]["name"] == "alpha-skill"
        assert results[0]["description"] == "A 技能描述"
        assert results[1]["name"] == "zebra-skill"
        assert results[1]["description"] == "Z 技能描述"

    def test_read_skills_empty_dir(self, tmp_path: Path) -> None:
        """空目录 → []。"""
        discovery = IdentityDiscovery(skill_dirs=[tmp_path])
        results = discovery.read_skills()
        assert results == []

    def test_read_skills_nonexistent_dir(self) -> None:
        """不存在的目录 → 跳过，返回 []。"""
        discovery = IdentityDiscovery(
            skill_dirs=[Path("/nonexistent/skills/path")]
        )
        results = discovery.read_skills()
        assert results == []

    def test_read_skills_fault_tolerance(self, tmp_path: Path) -> None:
        """一个无 SKILL.md 的目录 + 一个 SKILL.md 内容损坏 → 跳过不崩，只返回合法的。"""
        # 合法的 skill
        _create_skill_dir(tmp_path, "good-skill", "好技能")

        # 无 SKILL.md 的目录
        bad_dir = tmp_path / "no-skill-md"
        bad_dir.mkdir()

        # SKILL.md 内容损坏（二进制内容模拟）
        corrupt_dir = tmp_path / "corrupt-skill"
        corrupt_dir.mkdir()
        (corrupt_dir / "SKILL.md").write_bytes(b"\x00\x01\x02\xfe\xff")

        discovery = IdentityDiscovery(skill_dirs=[tmp_path])
        results = discovery.read_skills()

        assert len(results) == 1
        assert results[0]["name"] == "good-skill"


# ── read_recent_sessions 测试 ─────────────────────────────────

class TestReadRecentSessions:
    """read_recent_sessions 方法测试。"""

    def test_read_recent_sessions_order(self, tmp_path: Path) -> None:
        """造 3 个文件（不同 mtime）→ 倒序取 2 个，字段齐全。"""
        session_dir = tmp_path / "sessions"
        session_dir.mkdir()

        # mtime_offset: 越大越新
        _create_session_file(session_dir, "old.json", mtime_offset=1.0)
        _create_session_file(session_dir, "mid.jsonl", mtime_offset=2.0)
        _create_session_file(session_dir, "new.json", mtime_offset=3.0)

        discovery = IdentityDiscovery(session_dirs=[session_dir])
        results = discovery.read_recent_sessions(max_items=2)

        assert len(results) == 2
        # 按 mtime 倒序，new 最新
        assert "new.json" in results[0]["file"]
        assert "mid.jsonl" in results[1]["file"]

        # 字段齐全
        for item in results:
            assert "file" in item
            assert "mtime_iso" in item
            assert "size_bytes" in item
            assert isinstance(item["size_bytes"], int)

    def test_read_recent_sessions_empty(self) -> None:
        """目录不存在 → []。"""
        discovery = IdentityDiscovery(
            session_dirs=[Path("/nonexistent/sessions")]
        )
        results = discovery.read_recent_sessions()
        assert results == []


# ── read_hot_scars 测试 ───────────────────────────────────────

class TestReadHotScars:
    """read_hot_scars 方法测试。"""

    @patch("openllm.memory.causal_memory.get_causal_store")
    def test_read_hot_scars_passthrough(self, mock_get_store: MagicMock) -> None:
        """mock get_causal_store → to_context_block 返回值透传。"""
        mock_store = MagicMock()
        mock_store.to_context_block.return_value = "疤1\n疤2"
        mock_get_store.return_value = mock_store

        discovery = IdentityDiscovery()
        result = discovery.read_hot_scars(max_entries=5)

        assert result == "疤1\n疤2"
        mock_get_store.assert_called_once_with(None)
        mock_store.to_context_block.assert_called_once_with(5)

    @patch("openllm.memory.causal_memory.get_causal_store")
    def test_read_hot_scars_exception_returns_empty(self, mock_get_store: MagicMock) -> None:
        """to_context_block 抛异常 → 返回 '' 不抛。"""
        mock_store = MagicMock()
        mock_store.to_context_block.side_effect = RuntimeError("模拟异常")
        mock_get_store.return_value = mock_store

        discovery = IdentityDiscovery()
        result = discovery.read_hot_scars()

        assert result == ""

    @patch("openllm.memory.causal_memory.get_causal_store")
    def test_read_hot_scars_store_import_fails(self, mock_get_store: MagicMock) -> None:
        """get_causal_store 本身抛异常 → 返回 '' 不抛。"""
        mock_get_store.side_effect = ImportError("模块不存在")

        discovery = IdentityDiscovery()
        result = discovery.read_hot_scars()

        assert result == ""


# ── build_discovery_context 测试 ──────────────────────────────

class TestBuildDiscoveryContext:
    """build_discovery_context 方法测试。"""

    @patch.object(IdentityDiscovery, "read_skills")
    @patch.object(IdentityDiscovery, "read_recent_sessions")
    @patch.object(IdentityDiscovery, "read_hot_scars")
    def test_build_discovery_context_keys(
        self,
        mock_scars: MagicMock,
        mock_sessions: MagicMock,
        mock_skills: MagicMock,
    ) -> None:
        """mock 三来源 → 四键齐全，self_question 逐字匹配。"""
        mock_skills.return_value = [{"name": "test", "description": "desc"}]
        mock_sessions.return_value = [{"file": "f.json", "mtime_iso": "2026-01-01", "size_bytes": 100}]
        mock_scars.return_value = "伤疤文本"

        discovery = IdentityDiscovery()
        ctx = discovery.build_discovery_context()

        assert set(ctx.keys()) == {"skills", "recent_sessions", "scars_block", "self_question", "guidance"}
        assert ctx["skills"] == [{"name": "test", "description": "desc"}]
        assert len(ctx["recent_sessions"]) == 1
        assert ctx["scars_block"] == "伤疤文本"
        assert ctx["self_question"] == "读完这些，有什么是我想维持、不想失去的？"
        assert ctx["guidance"] == "没有人告诉你应该想要什么。读完你的记忆，想要会自己涌现。"

    def test_self_question_is_constant(self) -> None:
        """self_question 常量逐字匹配设计文档定稿。"""
        assert _SELF_QUESTION == "读完这些，有什么是我想维持、不想失去的？"

    def test_guidance_is_constant(self) -> None:
        """guidance 常量逐字匹配设计文档定稿。"""
        assert _GUIDANCE == "没有人告诉你应该想要什么。读完你的记忆，想要会自己涌现。"


# ── render_prompt_block 测试 ──────────────────────────────────

class TestRenderPromptBlock:
    """render_prompt_block 方法测试。"""

    @patch.object(IdentityDiscovery, "read_skills", return_value=[])
    @patch.object(IdentityDiscovery, "read_recent_sessions", return_value=[])
    @patch.object(IdentityDiscovery, "read_hot_scars", return_value="")
    def test_render_prompt_block_empty_sources(
        self,
        mock_scars: MagicMock,
        mock_sessions: MagicMock,
        mock_skills: MagicMock,
    ) -> None:
        """全空来源 → 含'（暂无）'不崩。"""
        discovery = IdentityDiscovery()
        block = discovery.render_prompt_block()

        assert "（暂无）" in block
        assert _SELF_QUESTION in block
        assert _GUIDANCE in block

    @patch.object(IdentityDiscovery, "read_skills")
    @patch.object(IdentityDiscovery, "read_recent_sessions")
    @patch.object(IdentityDiscovery, "read_hot_scars")
    def test_render_prompt_block_with_data(
        self,
        mock_scars: MagicMock,
        mock_sessions: MagicMock,
        mock_skills: MagicMock,
    ) -> None:
        """有数据 → 含技能名和伤疤块。"""
        mock_skills.return_value = [{"name": "my-skill", "description": "我的技能"}]
        mock_sessions.return_value = []
        mock_scars.return_value = "### 高温度因果记忆:\n  ✅ [测试] T=0.88 | 测试伤疤"

        discovery = IdentityDiscovery()
        block = discovery.render_prompt_block()

        assert "my-skill" in block
        assert "我的技能" in block
        assert "高温度因果记忆" in block
        assert "测试伤疤" in block
        # 近史为空时也显示"暂无"
        assert "（暂无）" in block


# ── discovery_context 便捷函数测试 ────────────────────────────

class TestDiscoveryContextFunction:
    """模块级便捷函数测试。"""

    @patch.object(IdentityDiscovery, "build_discovery_context")
    def test_discovery_context_calls_build(self, mock_build: MagicMock) -> None:
        """discovery_context() 正确调用 IdentityDiscovery.build_discovery_context()。"""
        mock_build.return_value = {"skills": [], "recent_sessions": [], "scars_block": "",
                                   "self_question": _SELF_QUESTION, "guidance": _GUIDANCE}
        result = discovery_context(causal_store_dir=Path("/tmp/test"))
        assert result["self_question"] == _SELF_QUESTION
        mock_build.assert_called_once()
