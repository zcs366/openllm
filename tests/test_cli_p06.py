"""
P0.6 散文承诺→真工具行（快路径协议刀）+ 错误串历史卫生。

T1: 散文承诺追问——模型用散文表达继续意图时触发追问
T2: 纯闲聊无意图词→不追问
T3: OPENLLM_NUDGE=0 → 同T1输入不追问
T4: 追问抛异常→原result保住
T5: 错误卫生——[LLM错误]串不入账
T6: 正常对话照常成对入账（回归DR-20260917-04语义）
"""
import re
import sys
import importlib
from types import SimpleNamespace

import pytest


# -- 公共工具：stub 快路径所需 import --

def _stub_fast_path_imports(monkeypatch):
    """Mock 快路径所需 import，防止真实模块加载。"""
    import types as _t

    fake_isa = _t.ModuleType("openllm.core.isa_impl")
    fake_isa.FULL_TOOLS = []
    fake_isa.BASE_TOOLS = []
    fake_isa.TOOL_DESCRIPTIONS = {}
    monkeypatch.setitem(sys.modules, "openllm.core.isa_impl", fake_isa)

    fake_models = _t.ModuleType("openllm.core.models")
    fake_models.Decision = lambda **kw: SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, "openllm.core.models", fake_models)

    fake_octopus = _t.ModuleType("openllm.iai.octopus")
    fake_octopus._LeftBrain = type("_LB", (), {
        "_TOOLCALL_RE": re.compile(
            r"^\s*\*{0,2}\s*TOOL_CALLS:\s*(\{.*\})\s*\*{0,2}\s*$", re.MULTILINE)
    })()
    monkeypatch.setitem(sys.modules, "openllm.iai.octopus", fake_octopus)

    fake_main_loop = _t.ModuleType("openllm.core.main_loop")
    fake_main_loop.Agent = type("Agent", (), {})
    monkeypatch.setitem(sys.modules, "openllm.core.main_loop", fake_main_loop)


def _stub_main(monkeypatch):
    """确保 main 模块加载到最新代码。"""
    _stub_fast_path_imports(monkeypatch)
    mod = sys.modules.get("openllm.cli.main")
    if mod is not None:
        importlib.reload(mod)


def _build_shell(provider, monkeypatch, isn_exec=None, extract_fn=None):
    """构造带 stub agent 的 AgentShell。

    快路径条件: len(line) < 50 AND 无触发词 (搜/搜索/写/读/执行/分析)。
    因此测试输入必须 < 50 字且不含这些触发词。
    """
    _stub_main(monkeypatch)
    from openllm.cli.main import AgentShell

    shell = AgentShell.__new__(AgentShell)

    _left = SimpleNamespace(
        provider=provider,
        _extract_tool_calls=extract_fn,
    )
    _right = SimpleNamespace(provider=provider)
    _octopus = SimpleNamespace(left=_left, right=_right)

    if isn_exec is not None:
        _isn = SimpleNamespace(execute=isn_exec)
    else:
        _isn = SimpleNamespace()

    _session = SimpleNamespace(active_turn=None, turns=[])
    _agent = SimpleNamespace(octopus=_octopus, isn=_isn, session=_session)
    shell.agent = _agent
    shell._history = []
    return shell


# ============================================================
# T1-T4: Continue-intent nudge tests
# ============================================================

class TestContinueIntentNudge:
    def test_prose_intent_triggers_nudge(self, capsys, monkeypatch):
        """第一轮有 TOOL_CALLS 被执行，末轮正文含'我再' -> 追问被触发。"""
        monkeypatch.delenv("OPENLLM_NUDGE", raising=False)

        chat_returns = [
            "searching...\nTOOL_CALLS: {\"tool_calls\": [{\"name\": \"read_file\", \"args\": {\"path\": \"/tmp/x\"}}]}",
            "我再补读几份关键的总件和收官件，才能把全貌说准。",
            "最终分析结果如上。",
        ]
        call_log = []

        def fake_chat(messages):
            idx = len(call_log)
            call_log.append(messages)
            return chat_returns[idx]

        def fake_extract(text):
            if "TOOL_CALLS:" in text:
                return [{"name": "read_file", "args": {"path": "/tmp/x"}}]
            return []

        shell = _build_shell(
            SimpleNamespace(chat=fake_chat),
            monkeypatch,
            isn_exec=lambda d: "tool result",
            extract_fn=fake_extract,
        )

        shell.default("hello")

        assert len(call_log) >= 3, f"expected >= 3 chat calls, got {len(call_log)}"
        nudge_msg = call_log[2][-1]
        assert nudge_msg["role"] == "user"
        assert "TOOL_CALLS" in nudge_msg["content"]

        out = capsys.readouterr().out
        assert "最终分析结果如上" in out

    def test_no_intent_no_nudge(self, capsys, monkeypatch):
        """纯闲聊不含意图词 -> 不追问。"""
        monkeypatch.delenv("OPENLLM_NUDGE", raising=False)

        call_log = []

        def fake_chat(messages):
            call_log.append(messages)
            if len(call_log) == 1:
                return "searching...\nTOOL_CALLS: {\"tool_calls\": [{\"name\": \"read_file\", \"args\": {\"path\": \"/tmp/x\"}}]}"
            return "task done"

        def fake_extract(text):
            if "TOOL_CALLS:" in text:
                return [{"name": "read_file", "args": {"path": "/tmp/x"}}]
            return []

        shell = _build_shell(
            SimpleNamespace(chat=fake_chat),
            monkeypatch,
            isn_exec=lambda d: "ok",
            extract_fn=fake_extract,
        )

        shell.default("hello")

        assert len(call_log) == 2, f"expected 2 chat calls, got {len(call_log)}"
        out = capsys.readouterr().out
        assert "task done" in out

    def test_nudge_switch_off(self, capsys, monkeypatch):
        """OPENLLM_NUDGE=0 -> 不追问。"""
        monkeypatch.setenv("OPENLLM_NUDGE", "0")

        call_log = []

        def fake_chat(messages):
            call_log.append(messages)
            if len(call_log) == 1:
                return "searching...\nTOOL_CALLS: {\"tool_calls\": [{\"name\": \"read_file\", \"args\": {\"path\": \"/tmp/x\"}}]}"
            return "我再读一份文件看看。"

        shell = _build_shell(
            SimpleNamespace(chat=fake_chat),
            monkeypatch,
            isn_exec=lambda d: "ok",
            extract_fn=lambda t: [{"name": "read_file", "args": {}}] if "TOOL_CALLS:" in t else [],
        )

        shell.default("hello")

        assert len(call_log) == 2, f"OPENLLM_NUDGE=0: expected 2, got {len(call_log)}"

    def test_nudge_exception_preserves_original(self, capsys, monkeypatch):
        """追问抛异常 -> 原result保住、不崩、正文照出。"""
        monkeypatch.delenv("OPENLLM_NUDGE", raising=False)

        call_log = []

        def fake_chat(messages):
            call_log.append(messages)
            if len(call_log) == 1:
                return "searching...\nTOOL_CALLS: {\"tool_calls\": [{\"name\": \"read_file\", \"args\": {\"path\": \"/tmp/x\"}}]}"
            elif len(call_log) == 2:
                return "我再读一份文件。"
            else:
                raise RuntimeError("nudge network error")

        shell = _build_shell(
            SimpleNamespace(chat=fake_chat),
            monkeypatch,
            isn_exec=lambda d: "ok",
            extract_fn=lambda t: [{"name": "read_file", "args": {}}] if "TOOL_CALLS:" in t else [],
        )

        shell.default("hello")

        out = capsys.readouterr().out
        assert "我再读一份文件" in out
        assert len(call_log) == 3


# ============================================================
# T5-T6: Error string history hygiene
# ============================================================

class TestErrorStringHygiene:
    def test_error_tombstone_in_history(self, capsys, monkeypatch):
        """[LLM错误] → 刀⑥-2 墓碑对入账（旧语义"整对不入账"已废）。

        2026-09-26 会话连续性三修·刀⑥-2：报错轮的 user 提问不再从历史
        消失——assistant 记 TOMBSTONE_ERROR 真状态陈述；错误原文照常显示。
        """
        from openllm.cli.main import TOMBSTONE_ERROR
        error_text = "[LLM错误] 网络没连通（SSLError，已重试3次）"

        shell = _build_shell(
            SimpleNamespace(chat=lambda m: error_text),
            monkeypatch,
        )

        assert len(shell._hist()) == 0
        shell.default("hello")
        assert len(shell._hist()) == 2, \
            f"error round must book tombstone pair, got {shell._hist()}"
        assert shell._hist()[0] == {"role": "user", "content": "hello"}
        assert shell._hist()[1] == {"role": "assistant", "content": TOMBSTONE_ERROR}
        assert error_text not in shell._hist()[1]["content"], \
            "raw error text must not enter history (screen only)"

        out = capsys.readouterr().out
        assert "SSLError" in out, f"screen should show error: {out}"

    def test_normal_dialogue_pairs_in_history(self, capsys, monkeypatch):
        """正常对话照常成对入账（回归DR-20260917-04语义）。"""
        shell = _build_shell(
            SimpleNamespace(chat=lambda m: "normal reply"),
            monkeypatch,
        )

        assert len(shell._hist()) == 0
        shell.default("hello")

        assert len(shell._hist()) == 2, f"normal dialogue should pair, got {len(shell._hist())}"
        assert shell._hist()[0] == {"role": "user", "content": "hello"}
        assert shell._hist()[1] == {"role": "assistant", "content": "normal reply"}
