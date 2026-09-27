"""openLLM DSH — Session 事件溯源日志（append-only）— DR-20260927-07。

对标 pi 侧 dsh-session-log（JS）的核心设计，Python 版给 openLLM：

不变式（照搬 DSH 哲学）：
1. append-only：事件写入后不可变，永不改写历史行
2. seq 单调递增，从 0 起，由 append 时分配（写盘时落定）
3. 模型可见即已记录（Model-visible means logged）
4. 坏行容错：断电半行只丢半行，重放跳过并记录

事件类型（openLLM 语境裁剪）：
  session/start   会话启动（model, pid, cwd）
  session/end     会话结束（reason: eof|exit|crash-resume）
  turn/start      一轮开始（input 原文）
  turn/end        一轮结束（duration_ms, status, tool_rounds）
  user/message    用户消息原文
  assistant/message  助手最终正文（含墓碑——它们是真实历史）
  tool/call       本轮工具调用清单（快路径 trace / 心跳 turn.actions）
  memory/write    MemoryBus 写入结果（record_id, importance）
  recall/inject   本轮 ISA 召回注入条数
  command         /命令 执行（web/clear/model...，args 原文）
  note            系统注记（异常、降级，人可读）

文件布局：
  ~/.openllm/dsh/<YYYYMMDD>/<HHMMSS>-<rand4>.jsonl
  一天一目录，文件名即 session id（可读可排序）。

与 sessions/（DR-02，Web UI 数据源）的关系：
  sessions/ 是给浏览器看的派生视图（简单、两行一轮）；
  dsh/ 是完整事件真相（含工具调用、召回、命令、墓碑、打分）。
  重建 session 用 dsh/，浏览用 sessions/。日志器失败静默——
  日志是义务不是关卡，绝不阻塞对话主循环。
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterator, Optional

__all__ = [
    "DshLogger", "dsh_root", "list_dsh_sessions", "load_events",
    "replay_session", "rebuild_history",
]

_SID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}$")
_lock = threading.Lock()


def dsh_root() -> Path:
    """事件日志根目录（测试可用 OPENLLM_DSH_DIR 覆写）。"""
    d = Path(os.environ.get(
        "OPENLLM_DSH_DIR", str(Path.home() / ".openllm" / "dsh")))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _new_rel_path() -> str:
    now = time.localtime()
    sid = (time.strftime("%Y%m%d-%H%M%S", now)
           + "-" + uuid.uuid4().hex[:4])
    return f"{sid[:8]}/{sid[9:]}.jsonl"  # YYYYMMDD/HHMMSS-rand4.jsonl


def _sid_from_rel(rel: str) -> str:
    # YYYYMMDD/HHMMSS-rand4.jsonl → YYYYMMDD-HHMMSS-rand4
    return rel.replace("/", "-").removesuffix(".jsonl")


class DshLogger:
    """每会话一个实例：append-only 事件写器。

    线程安全（CLI 单线程写，锁是防御性的）；每事件立即 flush——
    崩溃最多丢当前半行，不丢已 flush 的事件。
    """

    def __init__(self, model: str = "", rel_path: Optional[str] = None):
        self._failed = False
        try:
            rel = rel_path or _new_rel_path()
            self.rel_path = rel
            self.session_id = _sid_from_rel(rel)
            self.path = dsh_root() / rel
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._seq = -1
            self._fh = open(self.path, "a", encoding="utf-8")
            self.start(model)
        except Exception:
            self._failed = True
            self._fh = None
            self.session_id = "dsh-disabled"

    # ── 核心：append ──

    def append(self, etype: str, **data: Any) -> int:
        """追加事件，返回 seq。失败静默返回 -1（日志非关卡）。"""
        if self._failed or self._fh is None:
            return -1
        with _lock:
            try:
                self._seq += 1
                rec = {"seq": self._seq, "ts": round(time.time(), 3),
                       "type": etype}
                rec.update(data)
                self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                self._fh.flush()
                return self._seq
            except Exception:
                self._failed = True
                return -1

    # ── 语义事件 ──

    def start(self, model: str = "") -> int:
        return self.append("session/start", model=model,
                           pid=os.getpid())

    def end(self, reason: str = "exit") -> int:
        return self.append("session/end", reason=reason)

    def turn_start(self, user_input: str) -> int:
        return self.append("turn/start", input=user_input)

    def turn_end(self, duration_ms: float = 0.0, status: str = "ok",
                 tool_rounds: int = 0) -> int:
        return self.append("turn/end", duration_ms=round(duration_ms, 1),
                           status=status, tool_rounds=tool_rounds)

    def user_message(self, content: str) -> int:
        return self.append("user/message", content=content)

    def assistant_message(self, content: str) -> int:
        return self.append("assistant/message", content=content)

    def tool_calls(self, calls: list) -> int:
        return self.append("tool/call", calls=calls)

    def memory_write(self, record_id: str, importance: float) -> int:
        return self.append("memory/write", record_id=record_id,
                           importance=importance)

    def recall_inject(self, n: int, query: str = "") -> int:
        return self.append("recall/inject", n=n, query=query[:80])

    def command(self, cmd: str, args: str = "") -> int:
        return self.append("command", cmd=cmd, args=args[:120])

    def note(self, text: str) -> int:
        return self.append("note", text=text[:200])

    def close(self) -> None:
        try:
            if self._fh is not None:
                self._fh.close()
        except Exception:
            pass
        self._fh = None


# ═══════════════════════════════════════════════
# 读取与重建
# ═══════════════════════════════════════════════

def list_dsh_sessions(limit: int = 50) -> list:
    """事件日志会话列表（新→旧）：id/事件数/首条用户输入/起止时间。"""
    out = []
    root = dsh_root()
    files = sorted(root.rglob("*.jsonl"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for p in files[:limit]:
        sid = _sid_from_rel(str(p.relative_to(root)))
        events, first_user = [], ""
        started = ended = None
        for ev in _iter_events(p):
            events.append(ev)
            if ev["type"] == "user/message" and not first_user:
                first_user = str(ev.get("content", ""))[:40]
            if ev["type"] == "session/start":
                started = ev.get("ts")
            elif ev["type"] == "session/end":
                ended = ev.get("ts")
        out.append({"id": sid, "events": len(events),
                    "title": first_user or "（无用户输入）",
                    "started": started, "ended": ended})
    return out


def _iter_events(path: Path) -> Iterator[dict]:
    """逐行读事件；坏行跳过（断电半行容错）。"""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict) and "type" in rec:
                    yield rec
    except OSError:
        return


def load_events(sid: str) -> Optional[list]:
    """按 session id 读全部事件（id 白名单校验防路径穿越）。"""
    if not _SID_RE.match(sid or ""):
        return None
    p = dsh_root() / sid[:8] / f"{sid[9:]}.jsonl"
    if not p.exists():
        return None
    return list(_iter_events(p))


def replay_session(sid: str) -> Optional[list]:
    """重放事件流 → 对话历史（user/assistant 对），可直接喂 provider。

    墓碑（[LLM错误]/[回合上限]/TOMBSTONE_*）保留为 assistant 消息——
    它们是真实历史，重建时如实还原。tool/call 召回等事件不进对话流，
    但在 rebuild_history 的元数据里可见。
    """
    events = load_events(sid)
    if events is None:
        return None
    msgs = []
    for ev in events:
        if ev["type"] == "user/message":
            msgs.append({"role": "user", "content": ev.get("content", "")})
        elif ev["type"] == "assistant/message":
            msgs.append({"role": "assistant",
                         "content": ev.get("content", "")})
    return msgs


def rebuild_history(sid: str) -> Optional[dict]:
    """完整重建包：对话历史 + 会话元数据 + 逐轮统计。

    这是"重建任何 session"的入口：历史可直接喂 provider.chat 续聊。
    """
    events = load_events(sid)
    if events is None:
        return None
    model = turns = None
    started = ended = None
    turns_out, cur = [], None
    tools_in_round = []
    for ev in events:
        t = ev["type"]
        if t == "session/start":
            model = ev.get("model", "")
            started = ev.get("ts")
        elif t == "session/end":
            ended = ev.get("ts")
        elif t == "user/message":
            cur = {"user": ev.get("content", ""), "ts": ev.get("ts"),
                   "tools": [], "assistant": None}
            turns_out.append(cur)
        elif t == "tool/call" and cur is not None:
            cur["tools"].extend(ev.get("calls", []))
        elif t == "assistant/message" and cur is not None:
            cur["assistant"] = ev.get("content", "")
    turns = [{"user": t_["user"], "assistant": t_["assistant"],
              "tools": [c if isinstance(c, dict) else {"raw": str(c)}
                        for c in t_["tools"]]}
             for t_ in turns_out]
    history = []
    for t_ in turns_out:
        history.append({"role": "user", "content": t_["user"]})
        if t_["assistant"] is not None:
            history.append({"role": "assistant", "content": t_["assistant"]})
    return {"session_id": sid, "model": model, "started": started,
            "ended": ended, "turns": turns, "history": history}
