"""DR-20260927-04 spinner + 视觉现代化测试。

S1 非 tty 哑火：不炸、不写流
S2 tty 形态：start/stop 写转轮帧与清行码
S3 pause/resume 共屏
S4 快路径接线：default() 期间 spinner 起停（tty 模拟下）
S5 环境开关 OPENLLM_SPINNER=0
"""
import io
import os
import re

import pytest

from openllm.cli import spinner as sp_mod
from openllm.cli.spinner import Spinner


class _FakeTTY(io.StringIO):
    """模拟 tty 流（isatty→True），capture write 内容。"""
    def __init__(self):
        super().__init__()
        self.frames = []

    def isatty(self):
        return True

    def write(self, s):
        if "\r" in s:
            self.frames.append(s)
        return super().write(s)


class TestSpinnerBasics:

    def test_non_tty_silent(self):
        """非 tty：哑火——start 不起线程，不写任何字节。"""
        buf = io.StringIO()
        sp = Spinner("思考中", stream=buf).start()
        sp.stop()
        assert buf.getvalue() == ""

    def test_tty_spins_and_clears(self):
        buf = _FakeTTY()
        sp = Spinner("思考中", stream=buf, interval=0.01).start()
        import time
        time.sleep(0.08)
        sp.stop()
        assert len(buf.frames) >= 2, "应有多次转轮帧"
        assert "思考中" in buf.frames[-2] or "思考中" in buf.getvalue()
        assert "\x1b[K" in buf.getvalue(), "stop 应清行"

    def test_pause_resume(self):
        buf = _FakeTTY()
        sp = Spinner("思考中", stream=buf, interval=0.01).start()
        import time
        time.sleep(0.05)
        sp.pause()
        mark = len(buf.getvalue())
        time.sleep(0.05)
        assert len(buf.getvalue()) == mark, "paused 期间不应写"
        sp.resume()
        time.sleep(0.05)
        sp.stop()
        assert len(buf.getvalue()) > mark

    def test_update_text(self):
        buf = _FakeTTY()
        sp = Spinner("思考中", stream=buf, interval=0.01).start()
        sp.update("工具中")
        import time
        time.sleep(0.05)
        sp.stop()
        assert "工具中" in buf.getvalue()


class TestAnsiOk:

    def test_ansi_ok_fake_tty(self):
        assert sp_mod.ansi_ok(_FakeTTY()) is True

    def test_ansi_ok_pipe(self):
        assert sp_mod.ansi_ok(io.StringIO()) is False


# ── 快路径接线（stub 套路同 dr03）──

TOOLCALL_RE = re.compile(
    r"^\s*\*{0,2}\s*TOOL_CALLS:\s*(\{.*\})\s*\*{0,2}\s*$", re.MULTILINE)


def _stub_fast(monkeypatch):
    import sys as _s
    import types as _t
    fake_isa = _t.ModuleType("openllm.core.isa_impl")
    fake_isa.FULL_TOOLS = []
    fake_isa.BASE_TOOLS = []
    fake_isa.TOOL_DESCRIPTIONS = {}
    monkeypatch.setitem(_s.modules, "openllm.core.isa_impl", fake_isa)
    fake_models = _t.ModuleType("openllm.core.models")
    fake_models.Decision = lambda **kw: None
    monkeypatch.setitem(_s.modules, "openllm.core.models", fake_models)
    fake_oct = _t.ModuleType("openllm.iai.octopus")
    fake_oct._LeftBrain = type("_LB", (), {"_TOOLCALL_RE": TOOLCALL_RE})()
    monkeypatch.setitem(_s.modules, "openllm.iai.octopus", fake_oct)
    fake_ml = _t.ModuleType("openllm.core.main_loop")
    fake_ml.Agent = type("Agent", (), {})
    monkeypatch.setitem(_s.modules, "openllm.core.main_loop", fake_ml)


class TestWiring:

    def _shell(self, monkeypatch, reply="普通回答"):
        _stub_fast(monkeypatch)
        from types import SimpleNamespace
        from openllm.cli.main import AgentShell

        class _P:
            model = "stub"
            _available = False
            def chat(self, messages):
                return reply

        shell = AgentShell.__new__(AgentShell)
        left = SimpleNamespace(provider=_P(), _extract_tool_calls=lambda r: [])
        shell.agent = SimpleNamespace(
            octopus=SimpleNamespace(left=left, right=left),
            isn=SimpleNamespace(),
            session=SimpleNamespace(active_turn=None, turns=[]))
        shell._history = []
        shell._session_file = None
        return shell

    def test_fast_path_with_spinner_default_on(self, monkeypatch, capsys):
        """默认开：capsys 的流非 tty → spinner 哑火但接线不炸。"""
        shell = self._shell(monkeypatch)
        shell.default("hi")
        out = capsys.readouterr().out
        assert "普通回答" in out

    def test_spinner_off_env(self, monkeypatch, capsys):
        monkeypatch.setenv("OPENLLM_SPINNER", "0")
        shell = self._shell(monkeypatch)
        shell.default("hi")
        out = capsys.readouterr().out
        assert "普通回答" in out
