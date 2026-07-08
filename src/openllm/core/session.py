"""
Session/Turn 模型 — 阿波罗神启：存在与行动分离

Session = 存在（持久状态·配置·连接）
Turn = 行动（独立token预算·独立生命周期·独立安全上下文）

生命周期：
  Session.start() → Session.new_turn() → Turn.complete() → Session.end()

设计原则：
  - Session跨Turn持久化状态
  - Turn独立token预算，互不干扰
  - 压缩入口由Session统一管理
  - checkpoint定时保存，支持断点恢复
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class TurnStatus(Enum):
    """Turn状态机。"""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class Turn:
    """
    Turn = 行动（独立token预算·独立生命周期·独立安全上下文）
    
    每个Turn有自己的：
    - token_budget: 本轮可用token上限
    - actions: 本轮内的操作记录
    - status: 状态机 (pending → active → complete/failed)
    - start_time/end_time: 生命周期时间戳
    """
    id: str
    session: 'Session'
    token_budget: int
    actions: list = field(default_factory=list)
    status: str = TurnStatus.PENDING.value
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    # 本轮元数据
    phase_metrics: list = field(default_factory=list)
    prediction: Optional[str] = None
    risk_level: str = "low"
    
    def complete(self):
        """完成Turn。"""
        if self.status != TurnStatus.ACTIVE.value:
            raise ValueError(f"Turn {self.id} cannot complete from status {self.status}")
        self.status = TurnStatus.COMPLETE.value
        self.end_time = time.time()
        # 记录到Session
        self.session.turns.append(self)
        self.session.active_turn = None
    
    def fail(self, reason: str = ""):
        """失败Turn。"""
        if self.status != TurnStatus.ACTIVE.value:
            raise ValueError(f"Turn {self.id} cannot fail from status {self.status}")
        self.status = TurnStatus.FAILED.value
        self.end_time = time.time()
        self.actions.append({
            "type": "error",
            "reason": reason,
            "timestamp": time.time(),
        })
        # 记录到Session
        self.session.turns.append(self)
        self.session.active_turn = None
    
    def add_action(self, action_type: str, detail: str = "", **kwargs):
        """添加操作记录。"""
        self.actions.append({
            "type": action_type,
            "detail": detail[:200],  # 截断防溢
            "timestamp": time.time(),
            **kwargs,
        })
    
    def trace_phase(self, phase: str, status: str, duration_ms: float = 0.0, detail: str = ""):
        """记录阶段观测。"""
        self.phase_metrics.append({
            "phase": phase,
            "status": status,
            "duration_ms": duration_ms,
            "detail": detail[:100],
            "timestamp": time.time(),
        })
    
    @property
    def duration_ms(self) -> float:
        """Turn总耗时。"""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return (time.time() - self.start_time) * 1000
    
    def summary(self) -> dict:
        """Turn摘要。"""
        return {
            "id": self.id,
            "status": self.status,
            "actions_count": len(self.actions),
            "phases": len(self.phase_metrics),
            "duration_ms": round(self.duration_ms, 1),
            "token_budget": self.token_budget,
        }


@dataclass
class Session:
    """
    Session = 存在（持久状态·配置·连接）
    
    跨Turn持久化：
    - state: 跨Turn持久状态（记忆、偏好等）
    - config: 全局配置
    - turns: 已完成的所有Turn
    - active_turn: 当前活跃Turn
    - max_context_tokens: token上限
    
    压缩入口：
    - compact_if_needed(): 当context接近上限时触发压缩
    """
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    state: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    turns: list = field(default_factory=list)
    active_turn: Optional[Turn] = None
    max_context_tokens: int = 100000
    
    # Session级元数据
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    _turn_count: int = 0
    
    def start(self):
        """启动Session。"""
        if self.start_time is not None:
            raise ValueError(f"Session {self.id} already started")
        self.start_time = time.time()
        self.state["status"] = "running"
        self.state["created_at"] = self.start_time
        return self
    
    def new_turn(self, token_budget: Optional[int] = None) -> Turn:
        """创建新Turn。"""
        if self.active_turn is not None:
            raise ValueError(f"Session {self.id} has active turn {self.active_turn.id}")
        if self.state.get("status") != "running":
            raise ValueError(f"Session {self.id} not running (status={self.state.get('status')})")
        
        # token预算：默认为max_context_tokens的10%
        if token_budget is None:
            token_budget = self.max_context_tokens // 10
        
        turn = Turn(
            id=f"t{self._turn_count}",
            session=self,
            token_budget=token_budget,
        )
        turn.status = TurnStatus.ACTIVE.value
        self.active_turn = turn
        self._turn_count += 1
        return turn
    
    def compact_if_needed(self) -> bool:
        """
        三层压缩 Layer 1: Tool Result Trimming
        参考Claude Code的QueryEngine压缩策略。

        触发条件：已完成Turn数 > 5 且 最近Turn的actions总字符数 > max_context的50%
        压缩操作：将旧Turn中的工具结果替换为 "[Old tool result content cleared]"
        保留：Turn的元数据（id/status/duration）、决策信息、因果链
        """
        if len(self.turns) < 6:
            return False

        # 估算：最近10个Turn的actions体积
        recent_turns = self.turns[-10:]
        total_chars = sum(len(str(t.actions)) for t in recent_turns)

        # 触发阈值：50%
        threshold = self.max_context_tokens * 0.5  # 粗略：1 token ≈ 4 chars
        if total_chars < threshold:
            return False

        # 压缩：最旧的5个Turn中，标记工具结果为已清除
        # 保留：Turn元数据、决策信息、最近5个Turn的完整内容
        compact_count = 0
        for t in self.turns[:-5]:  # 保留最近5轮
            new_actions = []
            for action in t.actions:
                if action.get("type") in ("execute", "tool_result", "search_result"):
                    # 工具结果→占位符（保留"知道调用过"）
                    new_actions.append({
                        "type": "trimmed",
                        "original_type": action.get("type"),
                        "detail": "[Old tool result content cleared]",
                        "timestamp": action.get("timestamp", time.time()),
                    })
                    compact_count += 1
                else:
                    # 决策/错误/预测→保留
                    new_actions.append(action)
            t.actions = new_actions

        self.state["last_compact"] = time.time()
        self.state["trimmed_items"] = compact_count
        return True
    
    def checkpoint(self) -> dict:
        """
        定时保存Session状态。
        
        返回快照字典，可用于断点恢复。
        """
        snapshot = {
            "session_id": self.id,
            "state": self.state.copy(),
            "turn_count": self._turn_count,
            "completed_turns": len(self.turns),
            "timestamp": time.time(),
        }
        # 写入磁盘
        checkpoint_dir = Path.home() / ".openllm" / "output" / "iai" / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"session_{self.id}.json"
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)
        return snapshot
    
    def end(self):
        """结束Session。"""
        if self.active_turn is not None:
            self.active_turn.fail("Session ended")
        self.end_time = time.time()
        self.state["status"] = "ended"
        self.state["ended_at"] = self.end_time
    
    @property
    def duration_ms(self) -> float:
        """Session总耗时。"""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        if self.start_time:
            return (time.time() - self.start_time) * 1000
        return 0.0
    
    def summary(self) -> dict:
        """Session摘要。"""
        return {
            "id": self.id,
            "status": self.state.get("status", "unknown"),
            "total_turns": self._turn_count,
            "completed_turns": len(self.turns),
            "duration_ms": round(self.duration_ms, 1),
            "max_context_tokens": self.max_context_tokens,
        }


# ═══════════════════════════════════════════════════════
# 工厂函数
# ═══════════════════════════════════════════════════════

def create_session(max_context_tokens: int = 100000, **kwargs) -> Session:
    """创建并启动Session。"""
    session = Session(max_context_tokens=max_context_tokens, **kwargs)
    session.start()
    return session


# ═══════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    # 快速测试
    print("=== Session/Turn 测试 ===\n")
    
    # 测试1: 创建并启动
    session = create_session(max_context_tokens=10000)
    assert session.start_time is not None
    print(f"✅ 测试1: Session启动 {session.id}")
    
    # 测试2: 创建Turn
    turn = session.new_turn(token_budget=5000)
    assert turn.status == "active"
    assert turn.token_budget == 5000
    print(f"✅ 测试2: Turn创建 {turn.id} budget={turn.token_budget}")
    
    # 测试3: 添加操作
    turn.add_action("reason", "生成提案")
    turn.trace_phase("reason", "ok", 100.0)
    assert len(turn.actions) == 1
    print(f"✅ 测试3: 操作记录 actions={len(turn.actions)}")
    
    # 测试4: 完成Turn
    turn.complete()
    assert turn.status == "complete"
    assert len(session.turns) == 1
    assert session.active_turn is None
    print(f"✅ 测试4: Turn完成 {turn.summary()}")
    
    # 测试5: 新Turn
    turn2 = session.new_turn()
    turn2.add_action("test", "测试操作")
    turn2.complete()
    assert len(session.turns) == 2
    print(f"✅ 测试5: 多Turn total={len(session.turns)}")
    
    # 测试6: 压缩
    compacted = session.compact_if_needed()
    print(f"✅ 测试6: 压缩检查 compacted={compacted}")
    
    # 测试7: Checkpoint
    snapshot = session.checkpoint()
    assert "session_id" in snapshot
    print(f"✅ 测试7: Checkpoint {snapshot['session_id']}")
    
    # 测试8: 结束
    session.end()
    assert session.state["status"] == "ended"
    print(f"✅ 测试8: Session结束 {session.summary()}")
    
    print(f"\n全部 8/8 测试通过 ✅")
