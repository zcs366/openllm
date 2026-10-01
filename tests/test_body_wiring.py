"""六体接线钉子（配平动作4 的施工件）——治「隐式接线、断了不响」。

病（实例）：ISN 靠 `getattr(getattr(agent,"isn",None),"execute",None)` 取，
装不上就是 None，然后工具循环**静默跳过**——「工具异动，下面没了」的机制。
钉子验的是：**缺件必须炸（启动）/ 必须吼（运行时）**，不许静默。
"""
import logging
from types import SimpleNamespace

import pytest

from openllm.core.body_protocol import (SIX_BODIES, resolve_body_method,
                                        validate_bodies)


def _full_agent():
    """六体齐全的替身：只需被验的入口方法存在。"""
    return SimpleNamespace(
        iai=SimpleNamespace(gate=lambda *a, **k: None, emit=lambda *a, **k: None),
        isa=SimpleNamespace(build_context=lambda *a, **k: None, respond=lambda *a, **k: None),
        ios=SimpleNamespace(risk_check=lambda *a, **k: None, arbitrate=lambda *a, **k: None),
        isn=SimpleNamespace(execute=lambda *a, **k: None),
        iko=SimpleNamespace(trace=lambda *a, **k: None),
    )


def test_six_bodies_named():
    assert set(SIX_BODIES) == {"IAI", "ISA", "IOS", "ISN", "IKO", "IAX"}


def test_validate_passes_when_all_wired():
    rep = validate_bodies(_full_agent())
    assert all(v["ok"] for v in rep.values())
    assert rep["IAX"]["kind"] == "机制体"   # IAX 无实例，本身就是心跳
    assert rep["ISN"]["kind"] == "实例体"


def test_validate_raises_naming_the_missing_body():
    a = _full_agent()
    del a.isn
    with pytest.raises(RuntimeError) as ei:
        validate_bodies(a)
    assert "ISN" in str(ei.value)
    assert "agent.isn" in str(ei.value)


def test_validate_raises_when_method_missing():
    a = _full_agent()
    a.isn = SimpleNamespace()          # 实例在、入口不在
    with pytest.raises(RuntimeError) as ei:
        validate_bodies(a)
    assert "agent.isn.execute()" in str(ei.value)


def test_validate_non_strict_reports_instead_of_raising():
    a = _full_agent()
    del a.isn
    rep = validate_bodies(a, strict=False)
    assert rep["ISN"]["ok"] is False
    assert rep["ISA"]["ok"] is True


def test_resolve_returns_callable_when_wired():
    fn = resolve_body_method(_full_agent(), "ISN", "execute")
    assert callable(fn)


def test_resolve_shouts_instead_of_silently_returning_none(caplog):
    a = _full_agent()
    del a.isn
    with caplog.at_level(logging.ERROR):
        fn = resolve_body_method(a, "ISN", "execute", logger=logging.getLogger("t"))
    assert fn is None
    assert any("六体接线缺口" in r.message for r in caplog.records), "缺件必须吼一声，不许静默"
