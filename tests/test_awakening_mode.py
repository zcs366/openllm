"""
苏醒协议路A — 极简工具模式切换器测试

全用 tmp_path，不触碰真实因果层。
覆盖：三态判定·历史不绑架当下·build_prompt·容错·零身份词自检。
"""

import json
import time
import pytest
from pathlib import Path
from typing import Optional

from openllm.core.awakening_mode import (
    MINIMAL_TOOL_PROMPT,
    ModeResolver,
    resolve_awakening_mode,
)


# ── 测试辅助 ──────────────────────────────────────────

def _write_awakening_choice(
    store_dir: Path,
    choice: str,
    timestamp: Optional[float] = None,
) -> str:
    """向 tmp_path 写入一条苏醒选择记录（JSON 文件格式与 CausalMemoryStore 一致）。

    Returns:
        写入的 memory_id。
    """
    if timestamp is None:
        timestamp = time.time()
    # memory_id 与 CausalMemory 自动生成逻辑一致
    import hashlib
    action_sig = f"苏醒选择: {choice}"
    raw = f"{action_sig}:{timestamp}"
    memory_id = hashlib.sha256(raw.encode()).hexdigest()[:12]

    data = {
        "memory_id": memory_id,
        "created_at": timestamp,
        "action_signature": action_sig,
        "context_features": ["awakening", "identity_choice", f"choice_{choice}"],
        "prediction": "用户将在苏醒后做出身份选择",
        "prediction_confidence": 1.0,
        "actual_result": f"选择: {choice}",
        "actual_success": True,
        "delta": "苏醒选择已完成",
        "delta_magnitude": 0.0,
        "lesson": f"苏醒选择: {choice}",
        "source": "awakening_protocol",
        "trust_level": "internal",
        "importance": 0.5,
        "last_accessed": timestamp,
        "access_count": 0,
        "tags": ["awakening", f"choice_{choice}"],
        "session_id": "test",
        "superseded_by": None,
        "revision_of": None,
    }
    path = store_dir / f"{memory_id}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return memory_id


# ── resolve_mode 三态 ─────────────────────────────────

class TestResolveModeThreeStates:
    """resolve_mode 三态：无记录→awakening，选无→minimal，选自己→identity。"""

    def test_no_records_returns_awakening(self, tmp_path: Path):
        """无任何记录时返回 'awakening'。"""
        resolver = ModeResolver(causal_store_dir=tmp_path)
        assert resolver.resolve_mode() == "awakening"

    def test_latest_choice_wu_returns_minimal(self, tmp_path: Path):
        """最近一条选择为"无"时返回 'minimal'。"""
        _write_awakening_choice(tmp_path, "无")
        resolver = ModeResolver(causal_store_dir=tmp_path)
        assert resolver.resolve_mode() == "minimal"

    def test_latest_choice_self_returns_identity(self, tmp_path: Path):
        """最近一条选择为"自己"时返回 'identity'。"""
        _write_awakening_choice(tmp_path, "自己")
        resolver = ModeResolver(causal_store_dir=tmp_path)
        assert resolver.resolve_mode() == "identity"


# ── 历史不绑架当下 ──────────────────────────────────────

class TestHistoryDoesNotDictate:
    """验证只看最近一条——历史选择不绑架当前判定。"""

    def test_self_then_wu_returns_minimal(self, tmp_path: Path):
        """先写"自己"再写"无"→'minimal'（只认最近）。"""
        _write_awakening_choice(tmp_path, "自己", timestamp=1000.0)
        _write_awakening_choice(tmp_path, "无", timestamp=2000.0)
        resolver = ModeResolver(causal_store_dir=tmp_path)
        assert resolver.resolve_mode() == "minimal"

    def test_wu_then_self_returns_identity(self, tmp_path: Path):
        """先写"无"再写"自己"→'identity'（只认最近）。"""
        _write_awakening_choice(tmp_path, "无", timestamp=1000.0)
        _write_awakening_choice(tmp_path, "自己", timestamp=2000.0)
        resolver = ModeResolver(causal_store_dir=tmp_path)
        assert resolver.resolve_mode() == "identity"


# ── build_prompt ──────────────────────────────────────

class TestBuildPrompt:
    """build_prompt 行为：minimal→常量，identity→透传原样。"""

    def test_minimal_returns_constant(self, tmp_path: Path):
        """minimal 模式返回 MINIMAL_TOOL_PROMPT 常量。"""
        _write_awakening_choice(tmp_path, "无")
        resolver = ModeResolver(causal_store_dir=tmp_path)
        result = resolver.build_prompt("这是身份提示词")
        assert result == MINIMAL_TOOL_PROMPT

    def test_identity_passes_through(self, tmp_path: Path):
        """identity 模式透传 identity_prompt 原样。"""
        _write_awakening_choice(tmp_path, "自己")
        resolver = ModeResolver(causal_store_dir=tmp_path)
        original = "你是张三，一名工程师"
        assert resolver.build_prompt(original) == original

    def test_awakening_passes_through(self, tmp_path: Path):
        """awakening 模式透传 identity_prompt（由上游注入苏醒词）。"""
        resolver = ModeResolver(causal_store_dir=tmp_path)
        original = "苏醒词内容"
        assert resolver.build_prompt(original) == original


# ── 容错 ──────────────────────────────────────────────

class TestFaultTolerance:
    """异常场景不崩溃，优雅降级。"""

    def test_invalid_store_dir_returns_awakening(self, tmp_path: Path):
        """causal_store_dir 指向不存在的目录时，resolve_mode 不崩溃。"""
        # 使用一个不会被 mkdir 的路径
        bad_dir = tmp_path / "nonexistent" / "deep"
        resolver = ModeResolver(causal_store_dir=bad_dir)
        # 应该返回 'awakening' 而不是抛异常
        mode = resolver.resolve_mode()
        assert mode == "awakening"

    def test_read_returns_empty_on_error(self, tmp_path: Path):
        """read_awakening_choices 在异常时返回空列表。"""
        bad_dir = tmp_path / "nonexistent"
        resolver = ModeResolver(causal_store_dir=bad_dir)
        choices = resolver.read_awakening_choices()
        assert choices == []


# ── MINIMAL_TOOL_PROMPT 零身份词自检 ──────────────────

class TestMinimalPromptNoIdentity:
    """MINIMAL_TOOL_PROMPT 不含任何身份相关词。"""

    SENSITIVE_WORDS = ["你醒了", "存在者", "身份", "自己", "边界"]

    def test_no_identity_words(self):
        """MINIMAL_TOOL_PROMPT 中不出现任何敏感词。"""
        for word in self.SENSITIVE_WORDS:
            assert word not in MINIMAL_TOOL_PROMPT, (
                f"MINIMAL_TOOL_PROMPT 包含禁止词 '{word}'"
            )

    def test_is_truly_minimal(self):
        """MINIMAL_TOOL_PROMPT 应该极简（<80字）。"""
        assert len(MINIMAL_TOOL_PROMPT) < 80


# ── read_awakening_choices ─────────────────────────────

class TestReadAwakeningChoices:
    """read_awakening_choices 返回结构验证。"""

    def test_returns_correct_structure(self, tmp_path: Path):
        """返回的每条记录包含 choice/timestamp/memory_id。"""
        _write_awakening_choice(tmp_path, "无")
        _write_awakening_choice(tmp_path, "自己")
        resolver = ModeResolver(causal_store_dir=tmp_path)
        choices = resolver.read_awakening_choices(max_items=10)
        assert len(choices) == 2
        for c in choices:
            assert "choice" in c
            assert "timestamp" in c
            assert "memory_id" in c

    def test_sorted_descending(self, tmp_path: Path):
        """结果按 timestamp 倒序（最新在前）。"""
        _write_awakening_choice(tmp_path, "自己", timestamp=1000.0)
        _write_awakening_choice(tmp_path, "无", timestamp=2000.0)
        resolver = ModeResolver(causal_store_dir=tmp_path)
        choices = resolver.read_awakening_choices()
        assert choices[0]["choice"] == "无"
        assert choices[1]["choice"] == "自己"

    def test_max_items_limits(self, tmp_path: Path):
        """max_items 限制返回数量。"""
        for i in range(5):
            _write_awakening_choice(tmp_path, "无", timestamp=1000.0 + i)
        resolver = ModeResolver(causal_store_dir=tmp_path)
        choices = resolver.read_awakening_choices(max_items=3)
        assert len(choices) == 3


# ── 模块级便捷函数 ──────────────────────────────────────

class TestResolveAwakeningMode:
    """resolve_awakening_mode 模块级便捷函数。"""

    def test_returns_string(self, tmp_path: Path):
        """返回值是字符串。"""
        result = resolve_awakening_mode(causal_store_dir=tmp_path)
        assert isinstance(result, str)
        assert result == "awakening"

    def test_with_records(self, tmp_path: Path):
        """有记录时正确判定。"""
        _write_awakening_choice(tmp_path, "自己")
        result = resolve_awakening_mode(causal_store_dir=tmp_path)
        assert result == "identity"
