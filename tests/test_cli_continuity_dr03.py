"""DR-20260927-03 对话连贯性——快路径"无声无息"三断点修复测试。

老搭档实测：openLLM 说"去翻文件/查资料"，然后无声无息。
断点A: provider.chat 抛异常穿透 → 炸出 REPL（界面冻结/退出）
断点B: 3轮工具上限打满仍在要工具 → 剥掉TOOL_CALLS后正文空 → "(无输出)"
断点A2: 回路中途 chat 抛异常 → 同A

修复语义：
A/A2 → result="[LLM错误]..."，走既有错误显示+墓碑入账，shell 不崩
B    → "[回合上限]..."一句话交代，正文保留时附尾部
"""
import re
import sys
import importlib
from types import SimpleNamespace

import pytest

TOOLCALL_RE = re.compile(
    r"^\s*\*{0,2}\s*TOOL_CALLS:\s*(\{.*\})\s*\*{0,2}\s*$", re.MULTILINE)


def _stub_fast(monkeypatch):
    import types as _t
    fake_isa = _t.ModuleType("openllm.core.isa_impl")
    fake_isa.FULL_TOOLS = []
    fake_isa.BASE_TOOLS = []
    fake_isa.TOOL_DESCRIPTIONS = {}
    monkeypatch.setitem(sys.modules, "openllm.core.isa_impl", fake_isa)
    fake_models = _t.ModuleType("openllm.core.models")
    fake_models.Decision = lambda **kw: SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, "openllm.core.models", fake_models)
    fake_oct = _t.ModuleType("openllm.iai.octopus")
    fake_oct._LeftBrain = type("_LB", (), {"_TOOLCALL_RE": TOOLCALL_RE})()
    monkeypatch.setitem(sys.modules, "openllm.iai.octopus", fake_oct)
    fake_ml = _t.ModuleType("openllm.core.main_loop")
    fake_ml.Agent = type("Agent", (), {})
    monkeypatch.setitem(sys.modules, "openllm.core.main_loop", fake_ml)


def _build_shell(monkeypatch, provider):
    _stub_fast(monkeypatch)
    from openllm.cli.main import AgentShell
    shell = AgentShell.__new__(AgentShell)
    left = SimpleNamespace(provider=provider,
                           _extract_tool_calls=_extract)
    shell.agent = SimpleNamespace(
        octopus=SimpleNamespace(left=left, right=left),
        isn=SimpleNamespace(execute=lambda d: "工具结果ok"),
        session=SimpleNamespace(active_turn=None, turns=[]))
    shell._history = []
    shell._session_file = None
    return shell


def _extract(result):
    """真实提取语义：有 TOOL_CALLS 行就回一个读文件调用。"""
    if "TOOL_CALLS" in (result or ""):
        return [{"name": "read_file", "args": {"path": "x"}}]
    return []


class _BoomFirst:
    """第一次 chat 就炸。"""
    model = "stub"
    def chat(self, messages):
        raise RuntimeError("连接中断")


class _BoomOnSecond:
    """第一次给 TOOL_CALLS，回路中途炸。"""
    model = "stub"
    def __init__(self):
        self.calls = 0
    def chat(self, messages):
        self.calls += 1
        if self.calls == 1:
            return '我去读文件。\nTOOL_CALLS: {"tool_calls": [{"name": "read_file", "args": {"path": "x"}}]}'
        raise RuntimeError("回路中断")


class _AlwaysTools:
    """永远要工具——3轮上限打满场景。"""
    model = "stub"
    def __init__(self):
        self.calls = 0
    def chat(self, messages):
        self.calls += 1
        return ('继续\nTOOL_CALLS: {"tool_calls": '
                '[{"name": "read_file", "args": {"path": "x"}}]}')


class TestBreakpointA:
    """断点A：chat 异常不炸场，转为错误正文+墓碑入账。"""

    def test_first_chat_boom_no_crash(self, monkeypatch, capsys):
        shell = _build_shell(monkeypatch, _BoomFirst())
        shell.default("帮我看下那份文件")  # 不应抛异常（无关键词走快路径）
        out = capsys.readouterr().out
        assert "[LLM错误]" in out, f"错误应显示给用户: {out}"
        assert "连接中断" in out
        # 墓碑入账（问题不从历史消失）
        assert len(shell._hist()) == 2
        assert shell._hist()[1]["role"] == "assistant"

    def test_loop_chat_boom_no_crash(self, monkeypatch, capsys):
        p = _BoomOnSecond()
        shell = _build_shell(monkeypatch, p)
        shell.default("看下那份文件然后总结")
        out = capsys.readouterr().out
        assert "[LLM错误]" in out, f"回路中断应显示: {out}"
        assert p.calls == 2  # 回路中途断，不再续


class TestBreakpointB:
    """断点B：3轮上限打满 → [回合上限] 交代，不再(无输出)。"""

    def test_round_limit_told(self, monkeypatch, capsys):
        p = _AlwaysTools()
        shell = _build_shell(monkeypatch, p)
        shell.default("做个大任务")
        out = capsys.readouterr().out
        assert "[回合上限]" in out, f"应有上限交代: {out}"
        assert p.calls == 4  # 初始1 + 回路3
        # 交代入账（不是空轮）
        assert "[回合上限]" in shell._hist()[1]["content"]

    def test_body_kept_when_present(self, monkeypatch, capsys):
        """上限触发但模型正文非空 → 正文保留在交代之后。"""
        class _BodyThenTools(_AlwaysTools):
            def chat(self, messages):
                self.calls += 1
                if self.calls == 4:
                    return ("已完成第一部分。\n"
                            'TOOL_CALLS: {"tool_calls": '
                            '[{"name": "read_file", "args": {"path": "x"}}]}')
                return ('继续\nTOOL_CALLS: {"tool_calls": '
                        '[{"name": "read_file", "args": {"path": "x"}}]}')
        shell = _build_shell(monkeypatch, _BodyThenTools())
        shell.default("做个大任务")
        out = capsys.readouterr().out
        assert "[回合上限]" in out
        assert "已完成第一部分" in out, f"正文应保留: {out}"
