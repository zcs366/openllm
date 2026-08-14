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
