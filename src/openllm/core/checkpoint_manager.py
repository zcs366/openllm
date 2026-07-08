"""checkpoint_manager.py — Region注册+增量Checkpoint+AOF

来自老IO-S syscall/checkpoint.py：
  借鉴Concordia (arXiv 2606.23521)的GPU-resident checkpoint模式：
  - Region注册：每个Process注册自己的快照/恢复函数
  - 增量checkpoint：只快照变化的Region
  - AOF日志：append-only log，只追加不修改
  - 自动checkpoint：定时触发

七神划界:
  - 赫淮斯托斯: 只管快照不缓存执行结果
  - 克洛诺斯: checkpoint周期可配置
  - 阿波罗: AOF日志只追加不修改
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("openllm.checkpoint")

OPENLLM_HOME = Path.home() / ".openllm"
CHECKPOINT_DIR = OPENLLM_HOME / "checkpoints"
REGISTRY_PATH = CHECKPOINT_DIR / "registry.json"
AOF_DIR = CHECKPOINT_DIR / "aof"
SNAPSHOT_DIR = CHECKPOINT_DIR / "snapshots"


@dataclass
class Region:
    """Checkpoint Region — 每个Process的快照单元。"""
    region_id: str
    snapshot_fn: Callable
    restore_fn: Callable
    metadata: dict = field(default_factory=dict)
    last_snapshot_at: float = 0.0
    last_snapshot_size: int = 0


class CheckpointManager:
    """Checkpoint管理器 — 统一管理checkpoint生命周期。

    Concordia的软件映射:
      - Region注册 ← Concordia的GPU state region
      - 增量checkpoint ← Concordia的JIT delta handler
      - AOF日志 ← Concordia的append-only log
      - 恢复 ← Concordia的recovery applier
    """

    def __init__(self, interval: float = 300.0,
                 auto_start: bool = True):
        self.interval = interval
        self._regions: dict[str, Region] = {}
        self._lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self._running = False
        self._aof_seq: int = 0

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        AOF_DIR.mkdir(parents=True, exist_ok=True)
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

        if auto_start:
            self.start()

    # ── Region注册 ──

    def register(self, region_id: str,
                 snapshot_fn: Callable,
                 restore_fn: Callable,
                 metadata: Optional[dict] = None) -> bool:
        with self._lock:
            if region_id in self._regions:
                return False
            self._regions[region_id] = Region(
                region_id=region_id,
                snapshot_fn=snapshot_fn,
                restore_fn=restore_fn,
                metadata=metadata or {},
            )
            logger.info(f"✅ region注册: {region_id}")
            return True

    def unregister(self, region_id: str) -> bool:
        with self._lock:
            if region_id not in self._regions:
                return False
            del self._regions[region_id]
            logger.info(f"region注销: {region_id}")
            return True

    def list_regions(self) -> list[dict]:
        with self._lock:
            return [{
                "region_id": r.region_id,
                "last_snapshot_at": r.last_snapshot_at,
                "last_snapshot_size": r.last_snapshot_size,
                "metadata": r.metadata,
            } for r in self._regions.values()]

    # ── 快照 ──

    def snapshot(self, region_id: str) -> Optional[dict]:
        with self._lock:
            region = self._regions.get(region_id)
            if region is None:
                logger.warning(f"未知region: {region_id}")
                return None

        try:
            data = region.snapshot_fn()
            ts = time.time()

            # 写入快照文件
            snap_file = (
                SNAPSHOT_DIR / f"{region_id}_{int(ts)}.json")
            snap_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2))

            # 写AOF
            self._aof_append("snapshot", region_id, {
                "file": str(snap_file),
                "size": snap_file.stat().st_size,
            })

            region.last_snapshot_at = ts
            region.last_snapshot_size = snap_file.stat().st_size

            return data
        except Exception as e:
            logger.error(f"snapshot失败 {region_id}: {e}")
            return None

    def snapshot_all(self) -> dict:
        results = {}
        with self._lock:
            region_ids = list(self._regions.keys())
        for rid in region_ids:
            results[rid] = self.snapshot(rid)
        return results

    # ── 恢复 ──

    def restore(self, region_id: str,
                snapshot_data: Optional[dict] = None) -> bool:
        with self._lock:
            region = self._regions.get(region_id)
            if region is None:
                return False

        try:
            if snapshot_data is None:
                # 从最新快照文件恢复
                snaps = sorted(
                    SNAPSHOT_DIR.glob(f"{region_id}_*.json"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True)
                if not snaps:
                    return False
                snapshot_data = json.loads(snaps[0].read_text())

            region.restore_fn(snapshot_data)
            self._aof_append("restore", region_id, {})
            return True
        except Exception as e:
            logger.error(f"restore失败 {region_id}: {e}")
            return False

    # ── AOF ──

    def _aof_append(self, operation: str, region_id: str,
                    data: dict):
        self._aof_seq += 1
        entry = {
            "seq": self._aof_seq,
            "operation": operation,
            "region_id": region_id,
            "data": data,
            "timestamp": datetime.now(
                timezone.utc).isoformat(),
        }
        aof_file = AOF_DIR / "checkpoint.aof"
        with open(aof_file, "a") as f:
            f.write(
                json.dumps(entry, ensure_ascii=False) + "\n")

    # ── 自动checkpoint线程 ──

    def start(self):
        if self._running:
            return
        self._running = True
        self._worker = threading.Thread(
            target=self._loop, daemon=True)
        self._worker.start()
        logger.info(
            f"✅ CheckpointManager启动 "
            f"(间隔{self.interval}s)")

    def stop(self):
        self._running = False
        if self._worker:
            self._worker.join(timeout=5)

    def _loop(self):
        while self._running:
            time.sleep(self.interval)
            if self._running:
                self.snapshot_all()


# ══════════════════════════════════════════════════════════════
# Layer 14 — DeltaCheckpointManager (Concordia JIT delta)
# ══════════════════════════════════════════════════════════════

class DeltaCheckpointManager:
    """JIT增量Checkpoint管理器 — 只快照脏Region，跳过未变更部分。

    Concordia JIT Delta 映射：
      - dirty flag  ← Concordia的change-tracking bitmap
      - JIT快照     ← 仅序列化dirty region，跳过clean region
      - 恢复时间估算 ← 基于累计快照大小 / 累计耗时的滑动均值
      - SLA检查     ← 恢复时间 ≤ SLA阈值

    设计约束：
      - 不修改CheckpointManager现有代码（组合，非继承）
      - dirty flag纯内存存储，重启后自动清零（重新标记）
      - 恢复时间估算基于历史数据，冷启动时使用保守默认值
    """

    # 冷启动保守默认值：假设1MB/s恢复速度
    _DEFAULT_RECOVERY_SPEED: float = 1024 * 1024  # bytes/s
    # 恢复时间估算的滑动窗口大小
    _HISTORY_WINDOW: int = 20

    def __init__(self, manager: CheckpointManager):
        """
        Args:
            manager: 底层CheckpointManager实例，提供Region注册和快照能力。
        """
        self._mgr = manager
        # region_id → True 表示该region已脏，纯内存不持久化
        self._dirty: dict[str, bool] = {}
        self._lock = threading.Lock()
        # 恢复时间估算历史：[(快照字节数, 耗时秒数), ...]
        self._recovery_history: list[tuple[int, float]] = []
        # SLA阈值（秒），默认无限大（不检查）
        self._recovery_sla: float = float("inf")

    # ── 脏标记管理 ──

    def mark_dirty(self, region_name: str) -> None:
        """标记指定Region为dirty，下次snapshot_dirty时会被快照。

        Args:
            region_name: Region ID（须已在CheckpointManager中注册）。
        """
        with self._lock:
            self._dirty[region_name] = True
            logger.debug(f"region标记为dirty: {region_name}")

    def clear_dirty(self, region_name: str) -> None:
        """手动清除Region的dirty标记。快照完成后自动调用。

        Args:
            region_name: Region ID。
        """
        with self._lock:
            self._dirty.pop(region_name, None)

    def is_dirty(self, region_name: str) -> bool:
        """查询Region是否为dirty。"""
        with self._lock:
            return self._dirty.get(region_name, False)

    def dirty_regions(self) -> list[str]:
        """返回所有dirty region的ID列表。"""
        with self._lock:
            return [rid for rid, d in self._dirty.items() if d]

    # ── JIT Delta快照 ──

    def snapshot_dirty_regions(self) -> dict[str, Optional[dict]]:
        """仅快照dirty regions（JIT delta模式）。

        遍历所有dirty region，调用底层CheckpointManager.snapshot()，
        成功后自动清除dirty flag。

        Returns:
            {region_id: snapshot_data | None} — 只包含本次实际快照的region。
        """
        dirty_list = self.dirty_regions()
        if not dirty_list:
            logger.debug("无dirty region，跳过JIT delta快照")
            return {}

        results = {}
        for rid in dirty_list:
            t0 = time.time()
            data = self._mgr.snapshot(rid)
            elapsed = time.time() - t0

            if data is not None:
                # 从Region获取快照大小，用于恢复时间估算
                with self._mgr._lock:
                    region = self._mgr._regions.get(rid)
                    snap_size = (region.last_snapshot_size
                                 if region else 0)
                self._record_snapshot_cost(snap_size, elapsed)
                self.clear_dirty(rid)

            results[rid] = data

        logger.info(
            f"JIT delta快照完成: "
            f"{len(results)}/{len(dirty_list)} regions"
        )
        return results

    # ── 恢复时间估算 ──

    def _record_snapshot_cost(self, size_bytes: int,
                              elapsed: float) -> None:
        """记录一次快照的成本数据，用于恢复时间估算。"""
        if elapsed <= 0 or size_bytes <= 0:
            return
        with self._lock:
            self._recovery_history.append((size_bytes, elapsed))
            # 保持滑动窗口
            if len(self._recovery_history) > self._HISTORY_WINDOW:
                self._recovery_history = (
                    self._recovery_history[-self._HISTORY_WINDOW:]
                )

    def get_recovery_time_estimate(self) -> float:
        """估算当前脏数据的恢复时间（秒）。

        算法：
          1. 统计所有dirty region的累计快照大小
          2. 用历史快照速度（字节/秒）外推恢复时间
          3. 无历史数据时使用保守默认值（1MB/s）

        Returns:
            预估恢复时间（秒）。
        """
        # 累计dirty region的快照大小
        total_dirty_bytes = 0
        with self._lock:
            for rid in list(self._dirty.keys()):
                region = self._mgr._regions.get(rid)
                if region:
                    total_dirty_bytes += region.last_snapshot_size

        if total_dirty_bytes <= 0:
            return 0.0

        # 计算历史恢复速度
        speed = self._compute_speed()
        return total_dirty_bytes / speed

    def _compute_speed(self) -> float:
        """根据历史数据计算恢复速度（bytes/s）。"""
        with self._lock:
            if not self._recovery_history:
                return self._DEFAULT_RECOVERY_SPEED
            total_bytes = sum(s for s, _ in self._recovery_history)
            total_time = sum(t for _, t in self._recovery_history)
            if total_time <= 0:
                return self._DEFAULT_RECOVERY_SPEED
            return total_bytes / total_time

    # ── SLA管理 ──

    def set_recovery_sla(self, seconds: float) -> None:
        """设置恢复时间SLA阈值。

        Args:
            seconds: 最大允许恢复时间（秒），0或负数表示不检查。
        """
        self._recovery_sla = max(seconds, 0.0)
        logger.info(f"恢复SLA设置: {self._recovery_sla}s")

    def check_sla(self) -> tuple[bool, float]:
        """检查当前恢复时间估算是否满足SLA。

        Returns:
            (satisfied, estimated_seconds)
            - satisfied: True表示预估恢复时间 ≤ SLA阈值
            - estimated_seconds: 当前预估恢复时间
        """
        estimate = self.get_recovery_time_estimate()
        satisfied = estimate <= self._recovery_sla
        return satisfied, estimate

    # ── 状态报告 ──

    def status(self) -> dict:
        """返回DeltaCheckpointManager的当前状态摘要。"""
        dirty_list = self.dirty_regions()
        speed = self._compute_speed()
        estimate = self.get_recovery_time_estimate()
        satisfied, _ = self.check_sla()
        return {
            "dirty_count": len(dirty_list),
            "dirty_regions": dirty_list,
            "recovery_speed_bps": round(speed),
            "recovery_estimate_s": round(estimate, 3),
            "sla_threshold_s": (self._recovery_sla
                                if self._recovery_sla != float("inf")
                                else None),
            "sla_satisfied": satisfied,
            "history_samples": len(self._recovery_history),
        }
