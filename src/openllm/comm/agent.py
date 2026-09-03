"""comm/agent.py — 通信体核心：AgentComm 栈（多agent可靠通信落地）

把四条腿组装成一个 agent 可直接使用的通信栈：
  - P0-2 event_bus 语义路由（传得准）——broadcast 按兴趣语义找人
  - P0-1 裁决模块（该不该传）——send/deliver 双闸门
  - P1-3 SessionManager（传得稳）——可靠会话 ACK/超时/重试
  - P1-4 温度脉冲（温度）——主动振动（comm 外，agent 可持有 pulse 引擎）

设计：
  - Registry：agent 寻址（id → CommAgent），进程内共享
  - CommAgent：身份 + send/reply/broadcast/on_message + mailbox
  - 消息 = CommMessage 轻量信封（区别于 protocol.MessageEnvelope——
    那是六体协议层 HMAC/hash 链；CommMessage 是通信体层的语义消息）
  - 全程 comm.* 事件（bus 可选）+ 裁决审计 + 会话快照

关系说明：本栈 = 通信基础设施（治理协调层 IO-S 的盟友），裁决器是
governance.adjudication（IO-S 治理层），会话是 iai.session（工程宝石）。
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("openllm.comm")
DEFAULT_COMM_DIR = Path.home() / ".openllm" / "comm"


# ═══════════════════════════════════════════════════════════════
# 消息信封
# ═══════════════════════════════════════════════════════════════

@dataclass
class CommMessage:
    """通信体消息：语义信封（非 protocol 层 HMAC 信封）。"""
    msg_id: str = field(default_factory=lambda: f"msg-{uuid.uuid4().hex[:10]}")
    from_id: str = ""
    to_id: str = ""                  # "" = 广播（broadcast 专用）
    kind: str = "tell"               # tell / ask / reply / notify / broadcast
    body: str = ""
    importance: float = 0.5
    urgency: float = 0.0
    session_id: str = ""             # 关联可靠会话（reply 必带）
    reply_to: str = ""               # 回复的 msg_id
    ts: float = 0.0

    def __post_init__(self):
        if not self.ts:
            self.ts = time.time()


@dataclass
class SendResult:
    """send 结果：裁决/投递状态。"""
    ok: bool
    message: Optional[CommMessage] = None
    verdict_action: str = ""         # PASS / BLOCK
    reason: str = ""
    session_id: str = ""


# ═══════════════════════════════════════════════════════════════
# 注册表（寻址）
# ═══════════════════════════════════════════════════════════════

class Registry:
    """agent 寻址表：agent_id → CommAgent。进程内共享。

    可选持有共享基础设施（sessions/bus/adjudicator/log_dir）——注册的
    agent 未自带时自动继承。会话管理器必须共享：A 建的 session B 要能
    ack/respond（可靠会话跨 agent 同一状态机）。
    """

    def __init__(self, *, sessions: Any = None, bus: Any = None,
                 adjudicator: Any = None, log_dir: Optional[Path] = None) -> None:
        self._agents: dict[str, "CommAgent"] = {}
        self._sessions = sessions
        self._bus = bus
        self._adjudicator = adjudicator
        self._log_dir = log_dir

    def register(self, agent: "CommAgent") -> None:
        if agent.agent_id in self._agents:
            raise ValueError(f"agent 已注册: {agent.agent_id}")
        # 共享基础设施注入（agent 未自带时）
        if agent._sessions is None and self._sessions is not None:
            agent._sessions = self._sessions
        if agent._bus is None and self._bus is not None:
            agent._bus = self._bus
        if agent._adjudicator is None and self._adjudicator is not None:
            agent._adjudicator = self._adjudicator
        if agent._log_dir is None and self._log_dir is not None:
            agent._log_dir = self._log_dir
        self._agents[agent.agent_id] = agent

    def unregister(self, agent_id: str) -> None:
        self._agents.pop(agent_id, None)

    def get(self, agent_id: str) -> Optional["CommAgent"]:
        return self._agents.get(agent_id)

    def agents(self) -> list[str]:
        return list(self._agents)

    def deliver(self, msg: CommMessage) -> bool:
        """按 to_id 投递到 mailbox。找不到目标返回 False。"""
        target = self._agents.get(msg.to_id)
        if target is None:
            return False
        target._inbox.append(msg)
        return True

    def __contains__(self, agent_id: str) -> bool:
        return agent_id in self._agents


# ═══════════════════════════════════════════════════════════════
# 通信体 Agent
# ═══════════════════════════════════════════════════════════════

class CommAgent:
    """接入通信体的 agent 身份 + 收发能力。

    用法：
        reg = Registry()
        a = CommAgent("IAI", registry=reg, interests=["模型训练"])
        b = CommAgent("ISN", registry=reg)
        a.send("ISN", "技能更新了", kind="tell")
        b.run_once()          # 收 mailbox：auto_ack → handler

    消息流（可靠对话）：
        send → 裁决(send) → registry.deliver → mailbox
        run_once → 裁决(deliver) → session.ack → handler → (reply → session.respond)
    """

    def __init__(
        self,
        agent_id: str,
        *,
        registry: Registry,
        adjudicator: Any = None,        # TransmissionAdjudicator（P0-1）
        sessions: Any = None,           # SessionManager（P1-3）
        bus: Any = None,                # EventBus（P0-2，事件可见性）
        interests: Optional[list[str]] = None,  # 语义兴趣（broadcast 路由用）
        auto_ack: bool = True,
        log_dir: Optional[Path] = None,
    ) -> None:
        self.agent_id = agent_id
        self._registry = registry
        self._log_dir = log_dir or (DEFAULT_COMM_DIR / agent_id)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._adjudicator = adjudicator
        self._sessions = sessions
        self._bus = bus
        self._interests: list[str] = list(interests or [])
        self._auto_ack = auto_ack
        self._handlers: list[Callable[[CommAgent, CommMessage], Optional[str]]] = []
        self._inbox: list[CommMessage] = []
        self._sent: list[CommMessage] = []
        self._stats = {"sent": 0, "received": 0, "replied": 0, "blocked": 0}
        self._semantic_engine: Any = None  # 惰性缓存（模型加载贵，勿每次 new）
        registry.register(self)

    # -- 兴趣/处理注册 ------------------------------------------------

    def register_interest(self, text: str) -> None:
        """声明语义兴趣（broadcast 路由 + P0-2 语义匹配用）。"""
        if text not in self._interests:
            self._interests.append(text)

    def on_message(self, handler: Callable[["CommAgent", CommMessage], Optional[str]]) -> None:
        """注册消息处理器。handler 返回 str = 自动 reply 内容；None = 不自动回。"""
        self._handlers.append(handler)

    # -- 发送 ---------------------------------------------------------

    def send(self, to_id: str, body: str, kind: str = "tell",
             importance: float = 0.5, urgency: float = 0.0,
             reply_to: str = "", session_id: str = "") -> SendResult:
        """点对点发送：裁决 → 投递。ask 语义 = kind="ask"（期待 reply）。"""
        msg = CommMessage(from_id=self.agent_id, to_id=to_id, kind=kind,
                          body=body, importance=importance, urgency=urgency,
                          reply_to=reply_to, session_id=session_id)
        # 裁决：该不该传（P0-1）
        verdict = self._adjudicate_send(msg)
        if verdict is None or verdict.action != "PASS":
            self._stats["blocked"] += 1
            self._emit("comm.blocked", msg, verdict)
            return SendResult(ok=False, message=msg,
                              verdict_action=getattr(verdict, "action", ""),
                              reason=getattr(verdict, "reason", "裁决拦截"))
        # 可靠会话：仅新消息建会话（reply 沿用原 session，勿以自己为
        # initiator 重复创建——那是发方 A 的会话，B 只 respond）
        if not msg.session_id:
            session = self._ensure_session(msg)
            if session:
                msg.session_id = session.session_id
        # 投递
        delivered = self._registry.deliver(msg)
        if not delivered:
            return SendResult(ok=False, message=msg, reason=f"目标不存在: {to_id}")
        self._sent.append(msg)
        self._stats["sent"] += 1
        self._emit("comm.sent", msg, verdict)
        return SendResult(ok=True, message=msg, verdict_action="PASS",
                          session_id=msg.session_id)

    def reply(self, to_msg: CommMessage, body: str,
              importance: float = 0.5) -> SendResult:
        """回复消息：自动带 session respond 关联。"""
        return self.send(to_id=to_msg.from_id, body=body, kind="reply",
                         importance=importance, reply_to=to_msg.msg_id,
                         session_id=to_msg.session_id)

    def broadcast(self, body: str, kind: str = "broadcast",
                  importance: float = 0.3, semantic: bool = True) -> list[SendResult]:
        """广播：semantic=True 时只发给兴趣语义匹配的 agent（P0-2 路由）。"""
        results: list[SendResult] = []
        for agent_id in self._registry.agents():
            if agent_id == self.agent_id:
                continue
            if semantic and not self._interest_match(agent_id, body):
                continue
            results.append(self.send(agent_id, body, kind=kind, importance=importance))
        return results

    def _interest_match(self, agent_id: str, body: str) -> bool:
        """语义匹配：目标 agent 的兴趣 × 广播内容（无 embedding 时退化为关键词）。"""
        target = self._registry.get(agent_id)
        if target is None or not target._interests:
            return False
        try:
            if self._semantic is not None:
                for interest in target._interests:
                    sim = self._semantic.similarity(interest, body)
                    if sim is not None and sim >= 0.45:
                        return True
                return False
        except Exception:
            pass
        # 降级：关键词包含
        return any(kw.lower() in body.lower() for kw in target._interests)

    # -- 接收 ---------------------------------------------------------

    def run_once(self) -> int:
        """处理 mailbox：每条消息 裁决(deliver) → auto_ack → handler → reply。返回处理条数。"""
        processed = 0
        while self._inbox:
            msg = self._inbox.pop(0)
            # 裁决：收方打扰（P0-1；deliver 侧闸门）
            verdict = self._adjudicate_deliver(msg)
            if verdict is not None and verdict.action != "PASS":
                self._stats["blocked"] += 1
                self._emit("comm.blocked", msg, verdict)
                continue
            # 可靠会话 ACK
            if self._auto_ack and msg.session_id:
                self._ack_session(msg.session_id)
            self._stats["received"] += 1
            self._emit("comm.delivered", msg, verdict)
            # handler 处理（返回 str = 自动 reply）
            for handler in self._handlers:
                try:
                    auto_body = handler(self, msg)
                    if auto_body and msg.kind in ("ask", "tell", "notify"):
                        self.reply(msg, auto_body)
                        self._stats["replied"] += 1
                        if msg.session_id:
                            self._respond_session(msg.session_id)
                except Exception as exc:
                    logger.warning("[comm:%s] handler 异常: %s", self.agent_id, exc)
            # 收到 reply 且本 agent 是会话发起方 → 闭环 close
            if msg.kind == "reply" and msg.session_id and self._is_initiator(msg.session_id):
                self.close_session(msg.session_id)
            processed += 1
        return processed

    def _is_initiator(self, session_id: str) -> bool:
        if self._sessions is None:
            return False
        try:
            s = self._sessions.get(session_id)
            return s is not None and s.initiator == self.agent_id
        except Exception:
            return False

    # -- 会话 ---------------------------------------------------------

    def _ensure_session(self, msg: CommMessage) -> Any:
        mgr = self._sessions
        if mgr is None:
            try:
                from openllm.iai.session import SessionManager
                mgr = SessionManager(sessions_dir=self._log_dir / "sessions")
                self._sessions = mgr
            except ImportError:
                return None
        try:
            return mgr.create(initiator=self.agent_id, target=msg.to_id,
                              init_signal_id=msg.msg_id, meta={"kind": msg.kind})
        except Exception as exc:
            logger.warning("[comm:%s] 会话创建失败: %s", self.agent_id, exc)
            return None

    def _ack_session(self, session_id: str) -> None:
        if self._sessions is None:
            return
        try:
            self._sessions.set_ack(session_id, f"ack-{uuid.uuid4().hex[:8]}")
        except Exception:
            pass  # 已 ack/终态等——忽略

    def _respond_session(self, session_id: str) -> None:
        if self._sessions is None:
            return
        try:
            self._sessions.set_response(session_id, f"resp-{uuid.uuid4().hex[:8]}")
        except Exception:
            pass

    def close_session(self, session_id: str) -> None:
        if self._sessions is None:
            return
        try:
            self._sessions.set_closed(session_id, f"close-{uuid.uuid4().hex[:8]}")
        except Exception:
            pass

    def check_timeouts(self) -> int:
        """超时检查：返回新超时数（session watchdog）。"""
        if self._sessions is None:
            return 0
        try:
            return len(self._sessions.timeout_check())
        except Exception:
            return 0

    # -- 裁决 ---------------------------------------------------------

    def _ensure_adjudicator(self):
        if self._adjudicator is None:
            try:
                from openllm.governance.adjudication import TransmissionAdjudicator
                self._adjudicator = TransmissionAdjudicator(
                    log_dir=self._log_dir / "adjudication")
            except ImportError:
                return None
        return self._adjudicator

    def _adjudicate_send(self, msg: CommMessage) -> Any:
        adj = self._ensure_adjudicator()
        if adj is None:
            return None
        from openllm.governance.adjudication import TransmissionRequest
        return adj.decide(TransmissionRequest(
            from_agent=self.agent_id, to_agent=msg.to_id, msg_type="comm." + msg.kind,
            body=msg.body, importance=msg.importance, urgency=msg.urgency,
            channel="internal"))

    def _adjudicate_deliver(self, msg: CommMessage) -> Any:
        # v0：deliver 侧闸门复用同一裁决器（频控对收发双侧生效）
        adj = self._ensure_adjudicator()
        if adj is None:
            return None
        from openllm.governance.adjudication import TransmissionRequest
        return adj.decide(TransmissionRequest(
            from_agent=msg.from_id, to_agent=self.agent_id, msg_type="comm.deliver",
            body=msg.body, importance=msg.importance, channel="internal"))

    # -- 事件/审计 ----------------------------------------------------

    def _emit(self, event_type: str, msg: CommMessage, verdict: Any) -> None:
        if self._bus is None:
            return
        try:
            from openllm.iai.event_bus import Event
            self._bus.publish(Event(
                source=self.agent_id, type=event_type, timestamp=time.time(),
                entropy_score=0.0,
                payload={"msg_id": msg.msg_id, "from": msg.from_id, "to": msg.to_id,
                         "kind": msg.kind, "body": msg.body[:200],
                         "session_id": msg.session_id,
                         "verdict": getattr(verdict, "action", "")}))
        except Exception as exc:
            logger.warning("[comm:%s] 事件发布失败: %s", self.agent_id, exc)

    def stats(self) -> dict:
        return {**self._stats, "inbox": len(self._inbox),
                "interests": list(self._interests)}

    # -- 语义引擎（P0-2 可选接入）--------------------------------------

    @property
    def _semantic(self):
        """惰性取全局语义匹配器（broadcast 语义路由用，无则关键词降级）。
        实例缓存——SemanticMatcher 模型加载 ~7s，每次 new 是灾难。"""
        if self._semantic_engine is None:
            try:
                from openllm.iai.event_semantic import SemanticMatcher
                self._semantic_engine = SemanticMatcher()
            except Exception:
                return None
        return self._semantic_engine
