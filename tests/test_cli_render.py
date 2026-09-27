"""P1-1 CLI Markdown 渲染测试。

覆盖：rich 渲染 / 原样回退 / 开关 / monkeypatch / 异常回退 / 端到端 AgentShell。
"""
import pytest

# ── 快速冒烟：import 能通过 ──
from openllm.cli import render
from openllm.cli.render import render_answer


# ══════════════════════════════════════════════════════════════════
# T1 — MD 文本：渲染后无裸星号、无原始表格行
# ══════════════════════════════════════════════════════════════════

class TestRenderRichMd:

    MD_TEXT = (
        "## 概述\n\n"
        "**IAI** 是感知层，负责眼睛耳朵。\n\n"
        "| 模块 | 功能 |\n"
        "|------|------|\n"
        "| IAI  | 感知 |\n"
        "| ISA  | 记忆 |\n"
    )

    def test_rich_renders_no_raw_stars(self, capsys):
        """T1a: 渲染后输出中不含 '**' 裸星号（rich 解析为 Style 对象）。"""
        render_answer(self.MD_TEXT)
        out = capsys.readouterr().out
        assert "**" not in out, f"输出仍有裸星号:\n{out}"

    def test_rich_renders_no_raw_table_pipe(self, capsys):
        """T1b: 渲染后不含原始 '| module |' 类型表格源码行。"""
        render_answer(self.MD_TEXT)
        out = capsys.readouterr().out
        # 检查：去除 ANSI escape 后不应出现 "| IAI" 这种表格源码
        import re
        clean = re.sub(r"\x1b\[[0-9;]*m", "", out)
        assert "| IAI" not in clean, f"输出仍有原始表格源码:\n{clean}"


# ══════════════════════════════════════════════════════════════════
# T2 — 纯文本：输出与输入逐字符一致
# ══════════════════════════════════════════════════════════════════

class TestRenderPlain:

    PLAIN = "你好，这是一段普通回答。\n"

    def test_plain_text_passthrough(self, capsys):
        """T2: 无 md 特征 → 原样 print，逐字符一致。"""
        render_answer(self.PLAIN)
        out = capsys.readouterr().out
        assert out == self.PLAIN, f"输出不一致:\n{out!r}"


# ══════════════════════════════════════════════════════════════════
# T3 — OPENLLM_RENDER=0 → 即使 md 也原样 print
# ══════════════════════════════════════════════════════════════════

class TestRenderSwitch:

    MD_TEXT = "## 标题\n\n**加粗** 内容\n"

    def test_render_off_passthrough(self, monkeypatch, capsys):
        """T3: OPENLLM_RENDER=0 → md 原样 print（星号在）。"""
        monkeypatch.setenv("OPENLLM_RENDER", "0")
        render_answer(self.MD_TEXT)
        out = capsys.readouterr().out
        assert "**加粗**" in out, f"开关关闭后仍有渲染:\n{out}"


# ══════════════════════════════════════════════════════════════════
# T4 — HAS_RICH=False → 原样 print
# ══════════════════════════════════════════════════════════════════

class TestNoRich:

    MD_TEXT = "## 标题\n\n**加粗** 内容\n"

    def test_no_rich_passthrough(self, monkeypatch, capsys):
        """T4: HAS_RICH=False → md 原样 print。"""
        monkeypatch.setattr(render, "HAS_RICH", False, raising=False)
        render_answer(self.MD_TEXT)
        out = capsys.readouterr().out
        assert "**加粗**" in out, f"无 rich 时仍有渲染:\n{out}"
        # 恢复
        monkeypatch.setattr(render, "HAS_RICH", True, raising=False)


# ══════════════════════════════════════════════════════════════════
# T5 — 渲染异常 → 回退原样 print
# ══════════════════════════════════════════════════════════════════

class TestRenderFallback:

    MD_TEXT = "## 标题\n\n**加粗** 内容\n"

    def test_exception_fallback_print(self, monkeypatch, capsys):
        """T5: Markdown 构造抛异常 → 完整 print 原文不崩。"""
        orig_md_cls = render.Markdown
        monkeypatch.setattr(render, "Markdown", type(
            "BoomMarkdown",
            (),
            {"__init__": lambda self, *a, **kw: (_ for _ in ()).throw(RuntimeError("boom"))},
        ))
        render_answer(self.MD_TEXT)
        out = capsys.readouterr().out
        assert "**加粗**" in out, f"异常回退后原文丢失:\n{out}"
        # 恢复
        monkeypatch.setattr(render, "Markdown", orig_md_cls, raising=False)


# ══════════════════════════════════════════════════════════════════
# T6 — 端到端 AgentShell.default() → MD 通过 render_answer 出口
# ══════════════════════════════════════════════════════════════════

class TestAgentShellRender:

    MD_REPLY = (
        "## 你好\n\n"
        "**欢迎**使用 openLLM。\n\n"
        "| 模块 | 功能 |\n"
        "|------|------|\n"
        "| IAI  | 感知 |\n"
    )

    def test_shell_default_renders_md(self, monkeypatch, capsys):
        """T6: stub AgentShell → provider 返回 md → shell.default() →
        stdout 包含各段内容且无裸 '**'。"""
        from types import SimpleNamespace
        import re

        # stub provider 直接返回 md
        class _StubProvider:
            def chat(self, messages):
                return TestAgentShellRender.MD_REPLY

        # 构造 shell（跳过 __init__）
        from openllm.cli.main import AgentShell
        shell = AgentShell.__new__(AgentShell)
        side = SimpleNamespace(provider=_StubProvider())
        setattr(shell, "agent", SimpleNamespace(
            octopus=SimpleNamespace(left=side, right=side),
        ))
        shell._history = []

        shell.default("你好")
        out = capsys.readouterr().out

        # 去除 ANSI escape 后检查
        clean = re.sub(r"\x1b\[[0-9;]*m", "", out)

        # 断言 1：各段文字内容都在（去空格包含式匹配防 wrap）
        for seg in ("欢迎", "openLLM", "IAI", "感知"):
            # 去空格匹配，防 rich 换行拆断
            compact = clean.replace(" ", "").replace("\n", "")
            assert seg in compact, f"端到端输出缺 {seg!r}:\n{clean}"

        # 断言 2：无裸 "**"
        assert "**" not in clean, f"端到端输出仍有裸星号:\n{clean}"
