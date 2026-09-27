"""DR-20260927-05 语气与分寸——问候输入不再客服腔。

钉子：问候类输入走快路径时，system prompt 必须携带语气段
（老搭档口吻/禁催任务清单/禁服务口号/措辞求变化）。
"""
import re
import sys
from types import SimpleNamespace

TOOLCALL_RE = re.compile(
    r"^\s*\*{0,2}\s*TOOL_CALLS:\s*(\{.*\})\s*\*{0,2}\s*$", re.MULTILINE)


def test_greeting_fast_path_carries_tone(monkeypatch):
    import types as _t
    fake_isa = _t.ModuleType("openllm.core.isa_impl")
    fake_isa.FULL_TOOLS = []
    fake_isa.BASE_TOOLS = ["read_file"]
    fake_isa.TOOL_DESCRIPTIONS = {"read_file": "读文件"}
    monkeypatch.setitem(sys.modules, "openllm.core.isa_impl", fake_isa)
    fake_models = _t.ModuleType("openllm.core.models")
    fake_models.Decision = lambda **kw: None
    monkeypatch.setitem(sys.modules, "openllm.core.models", fake_models)
    fake_oct = _t.ModuleType("openllm.iai.octopus")
    fake_oct._LeftBrain = type("_LB", (), {"_TOOLCALL_RE": TOOLCALL_RE})()
    monkeypatch.setitem(sys.modules, "openllm.iai.octopus", fake_oct)
    fake_ml = _t.ModuleType("openllm.core.main_loop")
    fake_ml.Agent = type("Agent", (), {})
    monkeypatch.setitem(sys.modules, "openllm.core.main_loop", fake_ml)

    from openllm.cli.main import AgentShell

    captured = {}

    class _P:
        model = "stub"
        _available = False
        def chat(self, messages):
            captured["system"] = messages[0]["content"]
            return "下午好，老搭档。刚醒，眯着一会儿。"

    shell = AgentShell.__new__(AgentShell)
    left = SimpleNamespace(provider=_P(), _extract_tool_calls=lambda r: [])
    shell.agent = SimpleNamespace(
        octopus=SimpleNamespace(left=left, right=left),
        isn=SimpleNamespace(),
        session=SimpleNamespace(active_turn=None, turns=[]))
    shell._history = []
    shell._session_file = None

    shell.default("老搭档，醒来啦，下午好")  # 无关键词 → 快路径

    sysmsg = captured.get("system", "")
    assert "语气与分寸" in sysmsg, "system 缺语气段"
    assert "老搭档" in sysmsg
    assert "不要催任务" in sysmsg
    assert "客服腔" in sysmsg
