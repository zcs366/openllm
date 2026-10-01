"""六体统一接口协议。"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
import time as _time


class BodyInterface(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    def status(self) -> dict: ...
    def execute(self, hc, **kwargs) -> Optional[Any]:
        return None


class BodyRegistry:
    def __init__(self):
        self._bodies: dict[str, BodyInterface] = {}
    def register(self, body):
        self._bodies[body.name] = body
    def get(self, name):
        return self._bodies.get(name)
    def status(self):
        return {n: b.status() for n, b in self._bodies.items()}
    def names(self):
        return list(self._bodies.keys())


@dataclass
class UnifiedHeartbeat:
    user_message: str = ""
    tick_id: str = ""
    registry: Optional[BodyRegistry] = None
    identity: dict = field(default_factory=dict)
    memory: dict = field(default_factory=dict)
    search_results: list = field(default_factory=list)
    prediction: Any = None
    left_proposal: Any = None
    right_critique: Any = None
    risk: Any = None
    decision: Any = None
    result: Any = None
    output: str = ""
    research: Optional[dict] = None
    phase_log: list = field(default_factory=list)
    def log_phase(self, phase, status, detail=""):
        self.phase_log.append({"phase": phase, "status": status, "detail": detail, "time": _time.time()})


class IAIBody(BodyInterface):
    name = "IAI"
    def __init__(self, brain=None):
        self._brain = brain
    def status(self):
        return {"health": self._brain.health_check() if self._brain else "offline"}
    def execute(self, hc, **kwargs):
        if not self._brain:
            return
        ctx = kwargs.get("ctx")
        if ctx and hc.prediction is None:
            hc.prediction = self._brain.left.predict(ctx)
        if ctx and hc.left_proposal is None:
            hc.left_proposal, hc.right_critique = self._brain.reason(ctx, hc.prediction, kwargs.get("risk") or hc.risk)


class IOSBody(BodyInterface):
    name = "IOS"
    def __init__(self, ios=None):
        self._ios = ios
    def status(self):
        return {"health": "online" if self._ios else "offline"}
    def execute(self, hc, **kwargs):
        if not self._ios:
            return
        ctx = kwargs.get("ctx")
        if ctx and hc.prediction and hc.risk is None:
            hc.risk = self._ios.risk_check(ctx, hc.prediction)
        if hc.left_proposal and hc.right_critique and hc.decision is None:
            hc.decision = self._ios.arbitrate(hc.left_proposal, hc.right_critique, hc.risk)


class ISNBody(BodyInterface):
    name = "ISN"
    def __init__(self, isn=None):
        self._isn = isn
    def status(self):
        return {"health": "online" if self._isn else "offline", "tools": len(getattr(self._isn, 'tools', {})) if self._isn else 0}
    def execute(self, hc, **kwargs):
        if self._isn and hc.decision and hc.result is None:
            hc.result = self._isn.execute(hc.decision)


class IKOBody(BodyInterface):
    name = "IKO"
    def __init__(self, iko=None):
        self._iko = iko
    def status(self):
        return {"health": "online" if self._iko else "offline"}
    def execute(self, hc, **kwargs):
        if self._iko and hc.result:
            hc.output = hc.result.output if hasattr(hc.result, 'output') else str(hc.result)


# ═══════════════════════════════════════════════
# 六体显式接线（2026-10-01 配平 · 治「隐式接线、断了不响」）
# ═══════════════════════════════════════════════
# 病（实测）：ISN 以前是 `getattr(getattr(agent,"isn",None),"execute",None)` 取的——
#   改名不报错、装不上就返回 None、然后静默跳过整个工具循环。
#   这就是「工具异动，下面没了」能在系统里存活的机制。
# 药（两条，缺一不可）：
#   ① 启动即验 validate_bodies()：缺件**当场抛**，不许带病上工；
#   ② 运行时取件 resolve_body_method()：缺件**必吼一声**（ERROR），不许静默。
# 诚实标注：IAX 是**机制体**（本身就是心跳，无实例），故其「接线」= 本模块被加载。
SIX_BODIES = {
    "IAI": ("iai", ("gate", "emit")),
    "ISA": ("isa", ("build_context", "respond")),
    "IOS": ("ios", ("risk_check", "arbitrate")),
    "ISN": ("isn", ("execute",)),
    "IKO": ("iko", ("trace",)),
    "IAX": (None, ()),   # 机制体：无实例
}


def validate_bodies(agent, strict: bool = True) -> dict:
    """启动即验六体是否真的接上了。返回体检表；strict 时缺件抛 RuntimeError。"""
    report: dict = {}
    missing: list = []
    for body, (attr, methods) in SIX_BODIES.items():
        if attr is None:
            report[body] = {"kind": "机制体", "ok": True, "missing": []}
            continue
        obj = getattr(agent, attr, None)
        lack = []
        if obj is None:
            lack.append(f"agent.{attr}（实例缺失）")
        else:
            for m in methods:
                if getattr(obj, m, None) is None:
                    lack.append(f"agent.{attr}.{m}()")
        report[body] = {"kind": "实例体", "ok": not lack, "missing": lack}
        if lack:
            missing.append(body)
    if missing and strict:
        raise RuntimeError(
            "六体接线不全，拒绝带病启动：" + "；".join(f"{b} → {report[b]['missing']}" for b in missing))
    return report


def resolve_body_method(agent, body: str, method: str, *, logger=None):
    """运行时显式取入口方法。缺件**不静默**：打 ERROR 后返回 None。"""
    attr = SIX_BODIES.get(body.upper(), (body.lower(), ()))[0] or body.lower()
    fn = getattr(getattr(agent, attr, None), method, None)
    if fn is None and logger is not None:
        logger.error("六体接线缺口：agent.%s.%s 不存在——本阶段该项能力缺失（非静默告警）",
                     attr, method)
    return fn
