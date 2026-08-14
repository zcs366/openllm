"""
action_trace.py — ETAS风格的Action Trace系统
=============================================

灵感来源：ETAS (arXiv 2607.17780) 的核心insight：
  "Handling ≠ Hiding — 一个handler拦截了action，但trace仍然记录了
   程序请求了那个权限。"

四种事件类型（对齐ETAS dynamic semantics）：
  request  — 程序请求了一个action（无论是否被handler拦截）
  handled  — handler拦截了这个action，返回了mock/dry-run结果
  commit   — action通过了所有检查，实际执行
  denied   — action被策略/handler拒绝

设计约束（对齐ETAS设计限制）：
  1. 每个perform action必须产生request事件
  2. handler不能grant authority（只interpret不authorize）
  3. trace是append-only的，不可修改已有事件
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class EventPhase(str, Enum):
    """trace事件阶段——对齐ETAS event phases."""
    REQUEST = "request"
    HANDLED = "handled"
    COMMIT = "commit"
    DENIED = "denied"


class DenialCause(str, Enum):
    """拒绝原因——对齐ETAS Denial causes."""
    POLICY = "policy"
    BOUNDARY = "boundary"
    SANDBOX = "sandbox"
    SCHEMA = "schema"
    TOOL_EXPOSURE = "tool_exposure"
    DEPLOYMENT = "deployment"
    HANDLER = "handler"


@dataclass(frozen=True)
class TraceEvent:
    """
    一个不可变的trace事件。
    
    对齐ETAS：events distinguish request from handled outcome, 
    external commit, or denial.
    """
    event_id: str
    phase: EventPhase
    action_name: str
    timestamp: float
    # request/commit的参数
    params: dict = field(default_factory=dict)
    # handled事件：哪个handler处理了
    handler_name: Optional[str] = None
    # commit事件：结果摘要
    result_summary: Optional[str] = None
    # denied事件：拒绝原因
    denial_cause: Optional[DenialCause] = None
    denial_reason: Optional[str] = None
    # 元数据（effects annotation等）
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "event_id": self.event_id,
            "phase": self.phase.value,
            "action": self.action_name,
            "ts": self.timestamp,
        }
        if self.params:
            d["params"] = self.params
        if self.handler_name:
            d["handler"] = self.handler_name
        if self.result_summary:
            d["result"] = self.result_summary
        if self.denial_cause:
            d["denial_cause"] = self.denial_cause.value
        if self.denial_reason:
            d["denial_reason"] = self.denial_reason
        if self.metadata:
            d["meta"] = self.metadata
        return d


def make_trace_event(
    phase: EventPhase,
    action_name: str,
    *,
    params: Optional[dict] = None,
    handler_name: Optional[str] = None,
    result_summary: Optional[str] = None,
    denial_cause: Optional[DenialCause] = None,
    denial_reason: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> TraceEvent:
    """创建trace事件的工厂函数。"""
    return TraceEvent(
        event_id=uuid.uuid4().hex[:12],
        phase=phase,
        action_name=action_name,
        timestamp=time.time(),
        params=params or {},
        handler_name=handler_name,
        result_summary=result_summary,
        denial_cause=denial_cause,
        denial_reason=denial_reason,
        metadata=metadata or {},
    )


class ActionTraceStore:
    """
    append-only的trace存储。
    
    对齐ETAS：trace τ ∈ Ev* 是一个事件序列；
    τ·η 是追加事件。
    """
    
    def __init__(self, max_events: int = 10000, session_id: str = "", writer=None):
        self._events: list[TraceEvent] = []
        self._max_events = max_events
        self._session_id = session_id
        self._writer = writer  # ActionTraceWriter or None
    
    @property
    def events(self) -> list[TraceEvent]:
        """返回当前trace（只读视图）。"""
        return list(self._events)
    
    @property
    def size(self) -> int:
        return len(self._events)
    
    def append(self, event: TraceEvent) -> None:
        """追加事件（内存+持久化）。"""
        if len(self._events) >= self._max_events:
            # 老化：丢弃最老的1/4
            self._events = self._events[self._max_events // 4:]
        self._events.append(event)
        # 持久化（如果有writer）
        if self._writer:
            self._writer.append(event)
    
    def last_event(self) -> Optional[TraceEvent]:
        """返回最后一个事件。"""
        return self._events[-1] if self._events else None
    
    def filter_by_action(self, action_name: str) -> list[TraceEvent]:
        """按action名过滤事件。"""
        return [e for e in self._events if e.action_name == action_name]
    
    def filter_by_phase(self, phase: EventPhase) -> list[TraceEvent]:
        """按phase过滤事件。"""
        return [e for e in self._events if e.phase == phase]
    
    def has_request(self, action_name: str) -> bool:
        """检查某个action是否被请求过。"""
        return any(
            e.phase == EventPhase.REQUEST and e.action_name == action_name
            for e in self._events
        )
    
    def has_commit(self, action_name: str) -> bool:
        """检查某个action是否被提交执行。"""
        return any(
            e.phase == EventPhase.COMMIT and e.action_name == action_name
            for e in self._events
        )
    
    def request_commit_pairs(self) -> list[tuple[TraceEvent, Optional[TraceEvent]]]:
        """
        返回(request, commit)对——对审计最有价值的视图。
        
        对齐ETAS：每个commit应该有对应的request。
        """
        pairs = []
        for e in self._events:
            if e.phase == EventPhase.REQUEST:
                # 找同一action的最近commit
                commit = None
                for later in self._events:
                    if (later.phase == EventPhase.COMMIT 
                        and later.action_name == e.action_name
                        and later.timestamp >= e.timestamp):
                        commit = later
                        break
                pairs.append((e, commit))
        return pairs
    
    def to_summary(self) -> dict:
        """生成trace摘要——用于审计和IOS检查。"""
        requests = self.filter_by_phase(EventPhase.REQUEST)
        commits = self.filter_by_phase(EventPhase.COMMIT)
        denials = self.filter_by_phase(EventPhase.DENIED)
        handled = self.filter_by_phase(EventPhase.HANDLED)
        
        # 统计各action的request/commit/denied
        action_stats = {}
        for e in self._events:
            a = e.action_name
            if a not in action_stats:
                action_stats[a] = {"request": 0, "commit": 0, "handled": 0, "denied": 0}
            action_stats[a][e.phase.value] += 1
        
        return {
            "total_events": self.size,
            "requests": len(requests),
            "commits": len(commits),
            "handled": len(handled),
            "denials": len(denials),
            "action_stats": action_stats,
        }
    
    def clear(self) -> None:
        """清空trace（仅用于测试/重置）。"""
        self._events.clear()


# ── T1: SecurityFilter（敏感参数过滤）────────────────────────

class SecurityFilter:
    """
    敏感参数过滤器——无状态classmethod。
    
    过滤tool调用参数中的密码/token/密钥等敏感值。
    复用ios_privacy_guard的redact_pii模式。
    """
    
    # 需要过滤的参数名模式（小写匹配）
    _SENSITIVE_KEYS = frozenset({
        "password", "passwd", "token", "secret", "key", "api_key",
        "apikey", "authorization", "auth", "cookie", "credential",
        "private_key", "access_token", "refresh_token",
    })
    
    # 需要过滤的值模式（正则）
    _SENSITIVE_PATTERNS = [
        r"^(sk-|pk-|ak-|rk-)[a-zA-Z0-9]{20,}",   # API keys
        r"^(Bearer|Basic)\s+",                        # Auth headers
        r"-----BEGIN.*PRIVATE KEY-----",              # PEM keys
    ]
    
    @classmethod
    def filter_params(cls, params: dict) -> dict:
        """过滤敏感参数。返回脱敏后的副本。"""
        if not params:
            return {}
        
        filtered = {}
        for k, v in params.items():
            if cls._is_sensitive_key(k):
                filtered[k] = "[FILTERED]"
            elif isinstance(v, str) and cls._matches_sensitive_pattern(v):
                filtered[k] = "[FILTERED]"
            elif isinstance(v, dict):
                filtered[k] = cls.filter_params(v)  # 递归过滤
            else:
                filtered[k] = v
        return filtered
    
    @classmethod
    def _is_sensitive_key(cls, key: str) -> bool:
        """检查参数名是否是敏感字段。"""
        lower = key.lower().replace("-", "_")
        return any(s in lower for s in cls._SENSITIVE_KEYS)
    
    @classmethod
    def _matches_sensitive_pattern(cls, value: str) -> bool:
        """检查值是否匹配敏感模式。"""
        import re
        for pattern in cls._SENSITIVE_PATTERNS:
            if re.search(pattern, value):
                return True
        return False


# ── T2: ActionTraceWriter（JSONL持久化）────────────────────────

_ARCHIVE_SUFFIX = ".arc"
_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB归档阈值
_VERSION = 1


class ActionTraceWriter:
    """
    ActionTrace的JSONL持久化写入器。
    
    设计约束：
      1. append-only，永不修改已有行
      2. 每次append立即flush（防进程崩溃丢数据）
      3. 文件超10MB自动归档（rename为.arc）
      4. session_id嵌入每行，支持跨session查询
      5. security filter在写入前脱敏参数
    """
    
    def __init__(
        self,
        session_id: str,
        storage_dir: Optional[Path] = None,
        max_file_size: int = _MAX_FILE_SIZE,
    ):
        self._session_id = session_id
        self._dir = storage_dir or Path.home() / ".hermes" / "state"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max_file_size = max_file_size
        self._file_path = self._dir / "action_trace.jsonl"
        self._writer = None
    
    def _ensure_writer(self):
        """延迟打开文件句柄。"""
        if self._writer is None:
            self._writer = open(self._file_path, "a", encoding="utf-8")
    
    def append(self, event: TraceEvent) -> None:
        """追加事件到JSONL文件。"""
        try:
            self._ensure_writer()
            
            # 归档检查
            if self._file_path.exists() and self._file_path.stat().st_size > self._max_file_size:
                self._rotate()
            
            # 构建JSONL行
            entry = {
                "v": _VERSION,
                "sid": self._session_id,
                "ts": event.timestamp,
                "eid": event.event_id,
                "phase": event.phase.value,
                "action": event.action_name,
            }
            
            # 安全过滤后写入params
            if event.params:
                entry["params"] = SecurityFilter.filter_params(event.params)
            if event.handler_name:
                entry["handler"] = event.handler_name
            if event.result_summary:
                entry["result"] = event.result_summary[:200]
            if event.denial_cause:
                entry["denial_cause"] = event.denial_cause.value
            if event.denial_reason:
                entry["denial_reason"] = event.denial_reason[:200]
            
            self._writer.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._writer.flush()
            
        except Exception:
            pass  # 静默降级——持久化失败不阻塞主流程
    
    def flush(self) -> None:
        """显式flush。"""
        if self._writer:
            self._writer.flush()
    
    def _rotate(self) -> None:
        """归档当前文件。rename原子性。"""
        if self._writer:
            self._writer.close()
            self._writer = None
        
        import time as _time
        ts = _time.strftime("%Y%m%d-%H%M%S")
        archive_path = self._file_path.with_suffix(f".{ts}{_ARCHIVE_SUFFIX}")
        
        try:
            self._file_path.rename(archive_path)
        except Exception:
            pass  # rename失败不阻塞
    
    def close(self) -> None:
        """关闭文件句柄。"""
        if self._writer:
            self._writer.close()
            self._writer = None
    
    @classmethod
    def load_recent(
        cls,
        n: int = 100,
        storage_dir: Optional[Path] = None,
    ) -> list[dict]:
        """加载最近N条trace事件（从JSONL读取）。"""
        d = storage_dir or Path.home() / ".hermes" / "state"
        path = d / "action_trace.jsonl"
        
        if not path.exists():
            return []
        
        entries = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if "v" in entry and "sid" in entry:
                            entries.append(entry)
                    except json.JSONDecodeError:
                        continue  # 损坏行跳过
        except Exception:
            return []
        
        return entries[-n:]  # 返回最近N条
