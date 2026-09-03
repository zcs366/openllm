"""session.py — 可靠会话状态机（P1-3 · 工程宝石平移）

出处：ISA v0.6.0 isa_session.py（赫淮斯托斯 2026-09-04 钉的工程宝石——
"ISA 真正可移植的价值 = SessionManager 可靠会话状态机，正好补 Hermes
message_agent fire-and-forget 的缺失，双回复即缺会话绑定的病征"）。

平移原则（七神终裁 行动3）：
  - 状态机核心原样移植（SessionStatus/AgentSession/合法转移表/超时重试）
  - 去 ISA 依赖：默认快照路径 ISA_HOME → ~/.openllm/sessions
  - 适配 event_bus：状态变更 publish 会话事件（source=SESSION），让五体
    连续感知会话生命周期；不传 bus 则纯本地（行为与 ISA 原版一致）

状态机：INITIATED → ACKED → RESPONDED → CLOSED；超时→重试(≤3)→ESCALATED。
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("openllm.iai.session")
DEFAULT_SESSIONS_DIR = Path.home() / ".openllm" / "sessions"


# ═══════════════════════════════════════════════════════════════
# 会话状态枚举
# ═══════════════════════════════════════════════════════════════

class SessionStatus(Enum):
    """会话状态——赫耳墨斯信使的生命周期。"""
    INITIATED  = auto()   # 刚刚创建，等待ACK
    ACKED      = auto()   # 目标已确认收到
    RESPONDED  = auto()   # 目标已回复
    CLOSED     = auto()   # 发起方确认关闭
    TIMEOUT    = auto()   # 超时未收到预期响应
    ESCALATED  = auto()   # 3次重试后宣告失败

    def __str__(self) -> str:
        return self.name.lower()


# 合法状态转移表（原样保留——宝石的契约）
_VALID_TRANSITIONS: dict[SessionStatus, set[SessionStatus]] = {
    SessionStatus.INITIATED:  {SessionStatus.ACKED, SessionStatus.TIMEOUT},
    SessionStatus.ACKED:      {SessionStatus.RESPONDED, SessionStatus.TIMEOUT},
    SessionStatus.RESPONDED:  {SessionStatus.CLOSED, SessionStatus.TIMEOUT},
    SessionStatus.TIMEOUT:    {SessionStatus.INITIATED, SessionStatus.ESCALATED},
    SessionStatus.ESCALATED:  set(),   # 终态，不可转移
    SessionStatus.CLOSED:     set(),   # 终态，不可转移
}


@dataclass
class AgentSession:
    """Agent可靠通信会话：一次可靠消息交换的全生命周期追踪。"""
    session_id: str                     # 唯一会话ID: ses-xxxxxxxx
    initiator: str                      # 发起方Agent ID
    target: str                         # 目标Agent ID
    init_signal_id: str                 # 初始信号ID（s-xxxxxxxx）
    sla_seconds: int = 30               # 超时阈值（秒）
    status: SessionStatus = SessionStatus.INITIATED
    created_at: float = 0.0             # 创建时间戳（epoch秒）
    last_update: float = 0.0            # 最后更新时间
    retry_count: int = 0                # 当前重试次数
    max_retries: int = 3                # 最大重试次数
    ack_signal_id: str = ""             # ACK信号ID
    response_signal_id: str = ""        # 回复信号ID
    close_signal_id: str = ""           # 关闭信号ID
    meta: dict = field(default_factory=dict)  # 扩展元数据

    def __post_init__(self):
        if not self.session_id:
            self.session_id = f"ses-{uuid.uuid4().hex[:12]}"
        if not self.created_at:
            now = time.time()
            self.created_at = now
            self.last_update = now

    def __str__(self) -> str:
        return (f"AgentSession(session_id={self.session_id}, "
                f"status={self.status}, initiator={self.initiator}, "
                f"target={self.target}, sla={self.sla_seconds}s, "
                f"retry={self.retry_count}/{self.max_retries})")

    def __repr__(self) -> str:
        return self.__str__()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = str(self.status)
        return d


# ═══════════════════════════════════════════════════════════════
# 会话管理器
# ═══════════════════════════════════════════════════════════════

class SessionManager:
    """会话管理器——可靠通信的核心调度器。

    职责：创建/更新/查询 · 合法状态转移强制 · 超时检测自动重试 ·
    快照持久化（JSONL）· 统计。可选接 event_bus 发布会话生命周期事件。

    bus=None 时行为与 ISA 原版完全一致（纯本地 + JSONL）。
    """

    def __init__(self, sessions_dir: Optional[Path] = None,
                 bus: Any = None):
        self._sessions_dir = Path(sessions_dir) if sessions_dir else DEFAULT_SESSIONS_DIR
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        self._snapshot_path = self._sessions_dir / "sessions.jsonl"
        self._sessions: dict[str, AgentSession] = {}
        self._lock = threading.Lock()
        self._sla_configs: list[dict] = []
        self._bus = bus  # 可选 EventBus：状态变更发布会话事件
        self._load_snapshot()

    # ── 创建 ──

    def create(self, initiator: str, target: str,
               init_signal_id: str, sla_seconds: int = 30,
               meta: Optional[dict] = None) -> AgentSession:
        """创建新会话（INITIATED状态）。"""
        session = AgentSession(
            session_id=f"ses-{uuid.uuid4().hex[:12]}",
            initiator=initiator,
            target=target,
            init_signal_id=init_signal_id,
            sla_seconds=sla_seconds,
            status=SessionStatus.INITIATED,
            created_at=time.time(),
            last_update=time.time(),
            meta=meta or {},
        )
        with self._lock:
            self._sessions[session.session_id] = session
            self._append_snapshot(session)
        self._emit(session, "session.created")
        return session

    # ── 状态更新 ──

    def update_status(self, session_id: str,
                      new_status: SessionStatus) -> AgentSession:
        """更新会话状态，强制执行合法转移。"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"会话不存在: {session_id}")
            old_status = session.status
            allowed = _VALID_TRANSITIONS.get(old_status, set())
            if new_status not in allowed:
                raise ValueError(
                    f"非法状态转移: {old_status} → {new_status} "
                    f"(允许: {[s.name for s in allowed]})"
                )
            session.status = new_status
            session.last_update = time.time()
            self._append_snapshot(session)
        self._emit(session, f"session.{new_status}")
        return session

    def set_ack(self, session_id: str, ack_signal_id: str) -> AgentSession:
        """标记ACKED并记录ACK信号ID。"""
        session = self.update_status(session_id, SessionStatus.ACKED)
        with self._lock:
            session.ack_signal_id = ack_signal_id
            self._append_snapshot(session)
        return session

    def set_response(self, session_id: str, response_signal_id: str) -> AgentSession:
        """标记RESPONDED并记录回复信号ID。"""
        session = self.update_status(session_id, SessionStatus.RESPONDED)
        with self._lock:
            session.response_signal_id = response_signal_id
            self._append_snapshot(session)
        return session

    def set_closed(self, session_id: str, close_signal_id: str) -> AgentSession:
        """标记CLOSED并记录关闭信号ID。"""
        session = self.update_status(session_id, SessionStatus.CLOSED)
        with self._lock:
            session.close_signal_id = close_signal_id
            self._append_snapshot(session)
        return session

    def set_escalated(self, session_id: str) -> AgentSession:
        """标记ESCALATED（交付失败）。"""
        session = self.update_status(session_id, SessionStatus.ESCALATED)
        self._append_snapshot(session)
        return session

    def increment_retry(self, session_id: str) -> Optional[AgentSession]:
        """递增重试计数。返回更新后的session，若达到上限返回None。"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            session.retry_count += 1
            if session.retry_count >= session.max_retries:
                session.status = SessionStatus.ESCALATED  # 已达上限——升级
            else:
                session.status = SessionStatus.INITIATED  # 重置重发
            session.last_update = time.time()
            self._append_snapshot(session)
            new_status = session.status
        self._emit(session, f"session.{new_status}")
        return session

    # ── 查询 ──

    def get(self, session_id: str) -> Optional[AgentSession]:
        with self._lock:
            return self._sessions.get(session_id)

    def get_active(self) -> list[AgentSession]:
        """获取所有活跃（未终结）会话。"""
        terminal = {SessionStatus.CLOSED, SessionStatus.ESCALATED}
        with self._lock:
            return [s for s in self._sessions.values()
                    if s.status not in terminal]

    def get_by_initiator(self, initiator: str) -> list[AgentSession]:
        with self._lock:
            return [s for s in self._sessions.values()
                    if s.initiator == initiator]

    def get_by_target(self, target: str) -> list[AgentSession]:
        with self._lock:
            return [s for s in self._sessions.values()
                    if s.target == target]

    # ── 超时检测 ──

    def timeout_check(self) -> list[AgentSession]:
        """检查所有活跃会话是否超时，返回本次新超时的会话。"""
        now = time.time()
        timed_out: list[AgentSession] = []
        active = self.get_active()
        for session in active:
            if session.status in (SessionStatus.INITIATED,
                                  SessionStatus.ACKED,
                                  SessionStatus.RESPONDED):
                elapsed = now - session.last_update
                if elapsed > session.sla_seconds:
                    try:
                        if session.retry_count < session.max_retries:
                            self.increment_retry(session.session_id)
                        else:
                            self.update_status(session.session_id,
                                               SessionStatus.TIMEOUT)
                            self.set_escalated(session.session_id)
                        timed_out.append(session)
                    except (KeyError, ValueError):
                        pass
        return timed_out

    # ── 统计 ──

    def get_stats(self) -> dict:
        with self._lock:
            total = len(self._sessions)
            counts = {s: 0 for s in SessionStatus}
            for sess in self._sessions.values():
                counts[sess.status] = counts.get(sess.status, 0) + 1
            return {
                "total": total,
                "active": total - counts[SessionStatus.CLOSED]
                           - counts[SessionStatus.ESCALATED],
                "by_status": {str(k): v for k, v in counts.items()},
                "snapshot_path": str(self._snapshot_path),
            }

    # ── SLA注册 ──

    def register_sla(self, sla_seconds: int,
                     max_concurrent: int = 50) -> dict:
        config = {
            "sla_seconds": sla_seconds,
            "max_concurrent": max_concurrent,
            "registered_at": time.time(),
        }
        self._sla_configs.append(config)
        return config

    # ── event_bus 适配（平移新增）──

    def _emit(self, session: AgentSession, event_type: str) -> None:
        """状态变更发布会话事件（source=SESSION）。bus=None 时静默。"""
        if self._bus is None:
            return
        try:
            from openllm.iai.event_bus import Event
            self._bus.publish(Event(
                source="SESSION", type=event_type,
                timestamp=time.time(), entropy_score=0.0,
                payload={
                    "session_id": session.session_id,
                    "initiator": session.initiator,
                    "target": session.target,
                    "status": str(session.status),
                    "retry_count": session.retry_count,
                    "sla_seconds": session.sla_seconds,
                }))
        except Exception as exc:
            logger.warning("[session] 会话事件发布失败: %s", exc)

    # ── 持久化 ──

    def _append_snapshot(self, session: AgentSession):
        """追加快照到JSONL。"""
        try:
            d = session.to_dict()
            line = json.dumps(d, ensure_ascii=False) + "\n"
            with open(self._snapshot_path, "a") as f:
                f.write(line)
                f.flush()
        except Exception:
            pass  # 快照写入失败不影响内存操作

    def _load_snapshot(self):
        """从JSONL快照恢复会话。"""
        if not self._snapshot_path.exists():
            return
        try:
            with open(self._snapshot_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        d["status"] = SessionStatus[d["status"].upper()]
                        session = AgentSession(**d)
                        # 保留最新状态（JSONL按时间顺序）
                        self._sessions[session.session_id] = session
                    except Exception:
                        continue
        except Exception:
            pass
