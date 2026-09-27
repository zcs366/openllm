"""DR-20260927-06 记忆写通道 + 快路径读侧召回。

T1 打分启发式：问候低分/任务高分/上限封顶
T2 写通道接线：任务轮后 bus.write 被调用且 content 双方在场
T3 问候轮不写（低于 0.35 阈值）
T4 读侧：召回命中 → messages 注入记忆 system；未命中 → 不注入
"""
import re
import sys
from types import SimpleNamespace

import pytest

TOOLCALL_RE = re.compile(
    r"^\s*\*{0,2}\s*TOOL_CALLS:\s*(\{.*\})\s*\*{0,2}\s*$", re.MULTILINE)

from openllm.cli.main import _score_importance, _bus_of


class TestScoring:

    def test_greeting_low(self):
        assert _score_importance("你好", "你好呀") == 0.2
        assert _score_importance("下午好", "下午好，在呢") == 0.2

    def test_task_higher_than_chat(self):
        chat = _score_importance("随便聊聊天气", "好的呀")
        task = _score_importance("帮我分析这个架构方案", "好的，分析如下" + "x" * 400)
        assert task > chat
        assert task <= 0.8  # 封顶

    def test_bounds(self):
        assert _score_importance("为什么", "短") >= 0.45


class FakeBus:
    def __init__(self):
        self.writes = []
        self.recall = []

    def write(self, req):
        self.writes.append(req)
        return SimpleNamespace(success=True, record_id="r1")

    def query(self, q):
        return self.recall


class _P:
    model = "stub"
    _available = False
    def __init__(self):
        self.seen = []
    def chat(self, messages):
        self.seen.append(list(messages))
        return "任务完成，结论如下。"


def _shell(monkeypatch, bus=None):
    import types as _t
    fake_isa_mod = _t.ModuleType("openllm.core.isa_impl")
    fake_isa_mod.FULL_TOOLS = []
    fake_isa_mod.BASE_TOOLS = []
    fake_isa_mod.TOOL_DESCRIPTIONS = {}
    monkeypatch.setitem(sys.modules, "openllm.core.isa_impl", fake_isa_mod)
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
    p = _P()
    # _bus_of 走 agent.isa._get_memory_bus()
    isa = SimpleNamespace(_get_memory_bus=lambda: bus) if bus else None
    left = SimpleNamespace(provider=p, _extract_tool_calls=lambda r: [])
    agent_ns = SimpleNamespace(
        octopus=SimpleNamespace(left=left, right=left),
        isn=SimpleNamespace(),
        session=SimpleNamespace(active_turn=None, turns=[]))
    if isa is not None:
        agent_ns.isa = isa
    shell = AgentShell.__new__(AgentShell)
    shell.agent = agent_ns
    shell._history = []
    shell._session_file = None
    return shell, p


class TestWritePath:

    def test_task_turn_writes_bus(self, monkeypatch):
        bus = FakeBus()
        shell, _ = _shell(monkeypatch, bus)
        shell.default("看看这个方案有什么问题")  # 无关键词 → 快路径
        assert len(bus.writes) == 1
        req = bus.writes[0]
        assert "方案" in req.content and "用户" in req.content
        assert req.source == "cli_dialogue"
        assert req.importance >= 0.5

    def test_greeting_turn_skips_write(self, monkeypatch):
        bus = FakeBus()
        shell, _ = _shell(monkeypatch, bus)
        shell.default("下午好")
        assert bus.writes == [], "问候轮不应写记忆"

    def test_write_failure_silent(self, monkeypatch):
        class _BoomBus(FakeBus):
            def write(self, req):
                raise RuntimeError("bus down")
        shell, _ = _shell(monkeypatch, _BoomBus())
        shell.default("看看这个方案有什么问题")  # 不应抛
        assert shell._hist()[0]["content"].startswith("看看这个方案")


class TestReadPath:

    def test_recall_injected(self, monkeypatch):
        bus = FakeBus()
        bus.recall = [SimpleNamespace(
            content="中国政治通史：两条轴——正当性与央地", score=0.9)]
        shell, p = _shell(monkeypatch, bus)
        shell.default("接着聊通史，秦那一章讲什么")
        sysmsgs = [m for m in p.seen[0] if m["role"] == "system"]
        assert any("相关历史记忆" in m["content"] for m in sysmsgs), \
            "召回应注入 system"
        assert any("两条轴" in m["content"] for m in sysmsgs)

    def test_no_recall_no_injection(self, monkeypatch):
        bus = FakeBus()
        shell, p = _shell(monkeypatch, bus)
        shell.default("看看这个方案有什么问题")
        sysmsgs = [m for m in p.seen[0]
                   if m["role"] == "system" and "相关历史记忆" in m.get("content", "")]
        assert sysmsgs == []


class TestBusOf:

    def test_stub_agent_none(self):
        assert _bus_of(SimpleNamespace()) is None

    def test_isa_without_getter_none(self):
        assert _bus_of(SimpleNamespace(isa=SimpleNamespace())) is None
