#!/usr/bin/env python3
"""
左右脑（Hemispheres）— openLLM 自我对弈架构

AlphaGo 第37手的工程化：
不是"一个Agent分两个角色扮演"，是"两个Agent真正地对弈，
像AlphaGo自我对弈一样进化"。

左脑（正手）：主干活——写代码、做研究、执行任务
右脑（反手）：主监控——挑刺、找漏洞、左脑崩溃时接管

五体 × 左右脑 = 十个视角，对弈中涌现。

依赖：
- ISA OfflineManager（心跳检测）
- Δ胶囊 MemoryOS（checkpoint）
- ISN 技能系统（工具执行）
- IKO 输出系统（可观测）
"""

import json
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional


# ─── 常量 ──────────────────────────────────────────────
HEMISPHERE_DIR = Path.home() / ".hermes" / "hemispheres"
HEARTBEAT_INTERVAL = 5       # 秒，心跳频率
HEARTBEAT_TIMEOUT = 20       # 秒，超过此时间无心跳 → 判死
DEFAULT_CHECKPOINT_DIR = HEMISPHERE_DIR / "checkpoints"


# ─── 枚举 ──────────────────────────────────────────────

class HemisphereSide(Enum):
    LEFT = "left"    # 正手：主干活
    RIGHT = "right"  # 反手：主监控

class HemisphereState(Enum):
    ACTIVE = "active"     # 正常运行
    DEGRADED = "degraded" # 性能下降但仍在运行
    FAILOVER = "failover" # 正在接管对方
    OFFLINE = "offline"   # 已离线

class ArbiterVerdict(Enum):
    LEFT_WINS = "left_wins"
    RIGHT_WINS = "right_wins"
    COMPROMISE = "compromise"     # 折中方案
    INCONCLUSIVE = "inconclusive" # 无法裁决，走保守路由


# ─── 数据结构 ──────────────────────────────────────────

@dataclass
class Heartbeat:
    """心跳信号"""
    side: str                 # "left" | "right"
    timestamp: float          # 发送时间
    seq: int                  # 序列号（用于检测漏跳）
    state: str                # 当前状态
    context_pct: float = 0.0  # 上下文占用百分比
    last_action: str = ""     # 最近执行的动作描述

@dataclass
class Checkpoint:
    """左右脑检查点（用于崩溃恢复）"""
    side: str
    seq: int
    timestamp: float
    action: str               # 正在执行的动作
    state_snapshot: dict      # 状态快照
    context_snapshot: str     # 上下文摘要
    tool_call_id: str = ""    # 正在执行的工具调用ID

@dataclass
class ArbiterRecord:
    """对弈仲裁记录"""
    id: str
    timestamp: float
    left_proposal: str
    right_critique: str
    verdict: str
    evidence_left: list[str]
    evidence_right: list[str]
    resolution: str


# ─── 左右脑核心 ────────────────────────────────────────

class Hemisphere:
    """
    单侧大脑（左脑或右脑）。
    
    每侧大脑是一个独立的Agent会话，有：
    - 自己的心跳信号
    - 自己的checkpoint序列
    - 自己对侧脑的监控
    """
    
    def __init__(self, side: HemisphereSide,
                 heartbeat_callback: Optional[Callable] = None,
                 checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR):
        self.side = side
        self.side_str = side.value
        self.checkpoint_dir = checkpoint_dir / self.side_str
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.state = HemisphereState.ACTIVE
        self.seq = 0
        self.last_heartbeat: float = time.time()
        self._opponent_last_heartbeat: float = time.time()
        self._heartbeat_callback = heartbeat_callback
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # 对弈记录
        self.arbiter_records: list[ArbiterRecord] = []
        
    # ── 心跳 ──
    
    def start(self):
        """启动心跳线程"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()
        
    def stop(self):
        """停止心跳线程"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
            
    def _heartbeat_loop(self):
        """心跳发送循环"""
        while self._running:
            self._send_heartbeat()
            time.sleep(HEARTBEAT_INTERVAL)
            
    def _send_heartbeat(self):
        """发送一次心跳"""
        hb = Heartbeat(
            side=self.side_str,
            timestamp=time.time(),
            seq=self.seq,
            state=self.state.value,
        )
        self.seq += 1
        # 写入共享心跳文件
        hb_path = HEMISPHERE_DIR / f"heartbeat_{self.side_str}.json"
        hb_path.parent.mkdir(parents=True, exist_ok=True)
        hb_path.write_text(json.dumps(asdict(hb)))
        
        if self._heartbeat_callback:
            self._heartbeat_callback(hb)
            
    def check_opponent_alive(self) -> bool:
        """检查对侧大脑是否存活"""
        opp_path = HEMISPHERE_DIR / f"heartbeat_{'right' if self.side_str == 'left' else 'left'}.json"
        if not opp_path.exists():
            return False
        try:
            hb = json.loads(opp_path.read_text())
            age = time.time() - hb["timestamp"]
            return age < HEARTBEAT_TIMEOUT
        except (json.JSONDecodeError, OSError):
            return False
            
    def check_opponent_healthy(self) -> tuple[bool, str]:
        """
        检查对侧大脑健康状态。
        返回：(是否健康, 状态描述)
        """
        opp_side = 'right' if self.side_str == 'left' else 'left'
        opp_path = HEMISPHERE_DIR / f"heartbeat_{opp_side}.json"
        if not opp_path.exists():
            return False, f"对侧({opp_side})无心跳文件"
        try:
            hb = json.loads(opp_path.read_text())
            age = time.time() - hb["timestamp"]
            if age < HEARTBEAT_TIMEOUT:
                return True, f"健康(距今{age:.0f}秒, 状态={hb['state']})"
            else:
                return False, f"心跳超时(距今{age:.0f}秒, 阈值={HEARTBEAT_TIMEOUT}秒)"
        except (json.JSONDecodeError, OSError) as e:
            return False, f"心跳读取失败: {e}"
            
    # ── Checkpoint（崩溃恢复用） ──
    
    def save_checkpoint(self, action: str, state_snapshot: dict,
                        context_snapshot: str = "",
                        tool_call_id: str = "") -> Checkpoint:
        """保存检查点"""
        cp = Checkpoint(
            side=self.side_str,
            seq=len(list(self.checkpoint_dir.glob("*.json"))),
            timestamp=time.time(),
            action=action,
            state_snapshot=state_snapshot or {"note": "no state captured"},
            context_snapshot=context_snapshot[:500],
            tool_call_id=tool_call_id,
        )
        cp_path = self.checkpoint_dir / f"cp_{cp.seq:04d}.json"
        cp_path.write_text(json.dumps(asdict(cp), ensure_ascii=False, indent=2))
        return cp
        
    def load_latest_checkpoint(self) -> Optional[Checkpoint]:
        """加载最近的checkpoint"""
        files = sorted(self.checkpoint_dir.glob("cp_*.json"), reverse=True)
        if not files:
            return None
        try:
            return Checkpoint(**json.loads(files[0].read_text()))
        except (json.JSONDecodeError, OSError, TypeError):
            return None
            
    # ── 对弈 ──
    
    def record_arbitration(self, left: str, right: str,
                           verdict: ArbiterVerdict, resolution: str,
                           evidence_left: list[str] = None,
                           evidence_right: list[str] = None) -> ArbiterRecord:
        """记录一次对弈仲裁"""
        record = ArbiterRecord(
            id=uuid.uuid4().hex[:8],
            timestamp=time.time(),
            left_proposal=left[:200],
            right_critique=right[:200],
            verdict=verdict.value,
            evidence_left=evidence_left or [],
            evidence_right=evidence_right or [],
            resolution=resolution[:500],
        )
        self.arbiter_records.append(record)
        return record


class HemispherePair:
    """
    左右脑对 — 完整的自我对弈单元。
    
    职责：
    - 启动左右脑
    - 交换心跳
    - 检测崩溃 → 触发failover
    - 提供对弈仲裁接口
    """
    
    def __init__(self, left_name: str = "左脑",
                 right_name: str = "右脑",
                 arbiter: Optional[Callable] = None):
        self.left = Hemisphere(HemisphereSide.LEFT)
        self.right = Hemisphere(HemisphereSide.RIGHT)
        self.left_name = left_name
        self.right_name = right_name
        self._arbiter = arbiter or self._default_arbiter
        self._running = False
        
    def start(self):
        """启动左右脑（启动心跳）"""
        self.left.start()
        self.right.start()
        self._running = True
        print(f"🧠 左右脑已启动: {self.left_name}(左) ↔ {self.right_name}(右)")
        
    def stop(self):
        """停止左右脑"""
        self._running = False
        self.left.stop()
        self.right.stop()
        print(f"🧠 左右脑已停止")
        
    def health_check(self) -> dict:
        """完整健康检查"""
        left_ok, left_status = self.left.check_opponent_healthy()
        right_ok, right_status = self.right.check_opponent_healthy()
        
        # 注意：left.check_opponent 检查的是 right，反之亦然
        return {
            "left": {"alive": right_ok, "status": right_status},
            "right": {"alive": left_ok, "status": left_status},
            "timestamp": time.time(),
        }
        
    def failover_if_needed(self) -> bool:
        """
        检测崩溃并触发failover。
        返回：True=发生了failover, False=一切正常
        """
        check = self.health_check()
        
        # 左脑崩溃
        if not check["left"]["alive"] and check["right"]["alive"]:
            print(f"⚠️ 左脑崩溃！右脑接管中...")
            self.right.state = HemisphereState.FAILOVER
            # 读左脑checkpoint
            cp = self.left.load_latest_checkpoint()
            if cp:
                print(f"  恢复点: {cp.action} (seq={cp.seq})")
            return True
            
        # 右脑崩溃
        if not check["right"]["alive"] and check["left"]["alive"]:
            print(f"⚠️ 右脑崩溃！左脑继续运行(失去监控)")
            self.left.state = HemisphereState.DEGRADED
            return True
            
        # 双脑崩溃
        if not check["left"]["alive"] and not check["right"]["alive"]:
            print(f"🔴 双脑均无心跳！紧急模式")
            return True
            
        return False
        
    # ── 对弈 ──
    
    def arbitrate(self, left_proposal: str, right_critique: str,
                  evidence_left: list[str] = None,
                  evidence_right: list[str] = None) -> ArbiterRecord:
        """
        左右脑对弈仲裁。
        
        left_proposal: 左脑提出的方案
        right_critique: 右脑对方案的批评
        evidence: 各自证据
        
        返回：仲裁记录（含裁决结果）
        """
        verdict, resolution = self._arbiter(left_proposal, right_critique,
                                            evidence_left, evidence_right)
        
        # 记录到左右脑
        record = self.left.record_arbitration(
            left_proposal, right_critique, verdict, resolution,
            evidence_left, evidence_right,
        )
        self.right.arbiter_records.append(record)
        return record
        
    @staticmethod
    def _default_arbiter(left: str, right: str,
                          ev_left: list[str], ev_right: list[str]) -> tuple:
        """
        默认仲裁策略：谁有证据跟谁，平局走保守路线。
        
        这是最简单的仲裁器——后续可以替换为IO-S策略引擎。
        """
        if not right.strip() or right == left:
            # 右脑没意见 → 左脑赢
            return ArbiterVerdict.LEFT_WINS, f"采纳左脑方案: {left[:100]}"
        
        if not ev_right and ev_left:
            # 左脑有证据，右脑没有 → 左脑赢
            return ArbiterVerdict.LEFT_WINS, f"左脑有证据支持: {left[:100]}"
        
        if ev_right and not ev_left:
            # 右脑有证据，左脑没有 → 右脑赢
            return ArbiterVerdict.RIGHT_WINS, f"右脑有证据反驳: {right[:100]}"
        
        if ev_right and ev_left:
            # 双方都有证据 → 折中
            return ArbiterVerdict.COMPROMISE, (
                f"双方各有证据。折中方案:\n"
                f"  左: {left[:100]}\n"
                f"  右: {right[:100]}"
            )
        
        # 都没证据 → 走保守路线
        return ArbiterVerdict.INCONCLUSIVE, (
            f"双方均无充分证据。走保守路线:\n"
            f"  暂不执行，收集更多数据后再议"
        )


# ── 简易测试 ──────────────────────────────────────────

def demo():
    """演示左右脑对弈"""
    import time
    
    pair = HemispherePair()
    pair.start()
    
    # 等几次心跳
    time.sleep(2)
    
    # 健康检查
    health = pair.health_check()
    print(f"\n健康检查: {json.dumps(health, indent=2)}")
    
    # 模拟对弈
    record = pair.arbitrate(
        left_proposal="重构ISA五层架构，用bolt替代fLock",
        right_critique="fLock已经跑了1000+次测试全过，bolt未经验证。重构风险>收益。",
        evidence_left=["fLock在并发写入时性能下降70%", "fLock不支持跨进程"],
        evidence_right=["bolt在ragflow中发生过数据损坏", "当前fLock测试覆盖率100%"],
    )
    print(f"\n对弈仲裁: {record.verdict}")
    print(f"  左脑: {record.left_proposal}")
    print(f"  右脑: {record.right_critique}")
    print(f"  裁决: {record.resolution[:100]}...")
    
    # 检查对侧存活
    for side_name, h in [("左脑", pair.left), ("右脑", pair.right)]:
        alive, status = h.check_opponent_healthy()
        print(f"  {side_name} 看对侧: {'🟢' if alive else '🔴'} {status}")
    
    # 模拟failover
    print(f"\n模拟failover...")
    # 手动删除左脑心跳文件
    hb_path = Path.home() / ".hermes" / "hemispheres" / "heartbeat_left.json"
    if hb_path.exists():
        hb_path.unlink()
    time.sleep(1)
    happened = pair.failover_if_needed()
    print(f"  Failover触发: {'✅' if happened else '❌'}")
    
    pair.stop()
    print(f"\n对弈记录: {len(pair.left.arbiter_records)} 条")
    print(f"✅ 左右脑演示完成")


if __name__ == "__main__":
    demo()
