"""六体入口契约钉子（加固一批 ①②③）。

依据：承重账反事实实验——把 IAI 的 predict_consequences 换成返回 list、把 IOS 的
arbitrate 换成返回 Proposal，系统**当场崩**，崩点都是"下游直接取属性"。
= 六体边界此前不是被强制的，是靠"大家都记得该返回什么"维系的（「边界假」的机器证据）。

三批钉子验的是同一件事：**约定必须变成可机判的东西**——
  ① 命名协议（成员存在）② 书面返回契约（注解非空）③ 形状（返回类型含下游要读的成员）。
"""
from types import SimpleNamespace

import pytest

from openllm.core import body_protocol as bp
from openllm.core.iko_impl import IKO
from openllm.core.ios_impl import IOS
from openllm.core.isa_impl import ISA
from openllm.core.isn_impl import ISN
from openllm.iai.octopus import 章鱼I


# ── 真实系统：入口必须写着返回契约（类级内省，不实例化，快且无副作用）──

REAL_CLASSES = {"isa": ISA, "octopus": 章鱼I, "ios": IOS, "isn": ISN, "iko": IKO}


@pytest.mark.parametrize("attr,method", sorted(bp.ENTRY_CONTRACTS))
def test_real_entry_declares_return_contract(attr, method):
    import inspect
    fn = getattr(REAL_CLASSES[attr], method, None)
    assert fn is not None, f"{attr}.{method} 不存在"
    ann = inspect.signature(fn).return_annotation
    assert ann is not inspect.Signature.empty, f"{attr}.{method} 没写返回注解＝口头约定"
    assert bp._resolve_return_name(ann) == bp.ENTRY_CONTRACTS[(attr, method)]["returns"]


@pytest.mark.parametrize("attr,method", sorted(bp.ENTRY_CONTRACTS))
def test_declared_shape_carries_attrs_downstream_reads(attr, method):
    """声明的返回类型必须真的含下游会读的成员（否则就是把崩溃留到运行时）。"""
    import dataclasses
    from openllm.core import models as m
    spec = bp.ENTRY_CONTRACTS[(attr, method)]
    cls = getattr(m, spec["returns"], None)
    if cls is None or not dataclasses.is_dataclass(cls):
        pytest.skip("非仓内 dataclass（None/bool/str/dict/tuple）")
    names = {f.name for f in dataclasses.fields(cls)}
    for a in spec["attrs"]:
        assert a in names or hasattr(cls, a), f"{spec['returns']} 缺 {a}"


# ── 验证器本身：缺注解、错形状、坏字段都必须拦下 ──

def _agent(isa=None):
    return SimpleNamespace(
        isa=isa or SimpleNamespace(),
        octopus=SimpleNamespace(),
        ios=SimpleNamespace(),
        isn=SimpleNamespace(),
        iko=SimpleNamespace(),
    )


def test_validator_refuses_entry_without_return_annotation():
    def build_context(msg, session=None, octopus=None, ios=None) -> object: ...
    def respond(text, phase_times=None): ...          # ★ 故意不写返回注解
    isa = SimpleNamespace(build_context=build_context, respond=respond)
    rep = bp.validate_entry_contracts(_agent(isa), strict=False)
    assert rep["agent.isa.respond()"]["ok"] is False
    with pytest.raises(RuntimeError) as ei:
        bp.validate_entry_contracts(_agent(isa))
    assert "agent.isa.respond()" in str(ei.value)


def test_validator_refuses_wrong_declared_contract():
    """声明成别的类型 → 契约不符（这正是"把 Proposal 当 Decision 用"的事故形态）。"""
    def arbitrate(proposal, critique, risk) -> str:   # 期望 Decision
        return "随便"
    ios = SimpleNamespace(arbitrate=arbitrate)
    rep = bp.validate_entry_contracts(
        SimpleNamespace(ios=ios, isa=SimpleNamespace(), octopus=SimpleNamespace(),
                        isn=SimpleNamespace(), iko=SimpleNamespace()), strict=False)
    assert rep["agent.ios.arbitrate()"]["ok"] is False
    assert "契约不符" in rep["agent.ios.arbitrate()"]["why"]


def test_shape_check_bites_when_attr_list_is_wrong(monkeypatch):
    """把"下游要读的成员"故意写错 → 形状检查必须报出来（证明③真在跑）。"""
    from openllm.core.models import Decision
    monkeypatch.setitem(bp.ENTRY_CONTRACTS, ("ios", "arbitrate"),
                        {"returns": "Decision", "attrs": ("不存在的字段",)})
    def arbitrate(proposal, critique, risk) -> Decision: ...
    rep = bp.validate_entry_contracts(
        SimpleNamespace(ios=SimpleNamespace(arbitrate=arbitrate), isa=SimpleNamespace(),
                        octopus=SimpleNamespace(), isn=SimpleNamespace(), iko=SimpleNamespace()),
        strict=False)
    assert rep["agent.ios.arbitrate()"]["ok"] is False
    assert "Decision.不存在的字段" in rep["agent.ios.arbitrate()"]["why"]


# ── ① 命名协议 ──

def test_protocols_recognise_real_shaped_bodies():
    isa = SimpleNamespace(build_context=lambda *a, **k: None, respond=lambda *a, **k: None)
    iai = SimpleNamespace(gate=lambda *a, **k: None, emit=lambda *a, **k: None)
    ios = SimpleNamespace(risk_check=lambda *a, **k: None, arbitrate=lambda *a, **k: None)
    isn = SimpleNamespace(execute=lambda *a, **k: None)
    iko = SimpleNamespace(trace=lambda *a, **k: None, process_output=lambda *a, **k: None)
    rep = bp.validate_protocols(SimpleNamespace(isa=isa, iai=iai, ios=ios, isn=isn, iko=iko))
    assert all(rep.values()), rep


def test_protocols_reject_incomplete_body():
    isn = SimpleNamespace()          # 没有 execute
    rep = bp.validate_protocols(SimpleNamespace(isa=SimpleNamespace(), iai=SimpleNamespace(),
                                                ios=SimpleNamespace(), isn=isn,
                                                iko=SimpleNamespace()))
    assert rep["ISN"] is False
