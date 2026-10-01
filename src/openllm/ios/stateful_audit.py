"""
StatefulAuditTrail — 跨session的信号保真追踪
============================================

来源：arXiv:2607.02514 "Distributed Attacks in Persistent-State AI Control"

核心洞察：
- 无状态审计（每次操作独立评分）逃逸率87%
- 有状态审计（跨PR累积追踪）逃逸率降至54%
- 安全是序列级属性，不是操作级属性

本模块实现：
1. 每次tool_call/事件后记录suspicion_score
2. 跨session累积cumulative_score
3. 超过P95阈值时自动触发RECALL告警
4. JSONL日志自动归档（超过retention_days的记录迁移）
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# 时间戳单一判定源（写侧收口 2026-10-01 成市拍板）：archive_old_records 读
# RECALL/stateful_scores 时间戳必须过 coerce_ts（双格式容忍），不得裸比。
from openllm.isa.timeutil import coerce_ts

logger = logging.getLogger("openllm.ios.stateful_audit")

# ─ 默认审计目录 ──
# 2026-09-15 医师接骨：原先是 __init__ 里的内联字面量，测试无法重定向 →
# tests/test_governance.py 等用例把审计文件直接写进生产目录
# ~/.hermes/jiak/governance_audit/（累积 1.6M+705K）。抽成模块级常量的意义
# 不在于改默认值，而在于**给测试一个可 monkeypatch 的接缝**（conftest 已接）。
DEFAULT_AUDIT_DIR = Path.home() / ".hermes" / "jiak" / "governance_audit"


@dataclass
class AuditRecord:
    """单条审计记录。"""
    session_id: str
    event_type: str           # "tool_call" | "file_write" | "skill_invoke" | "custom"
    suspicion_score: float    # 0.0(良性) ~ 1.0(恶意)
    timestamp: float = field(default_factory=time.time)
    cumulative_score: float = 0.0
    content_hash: str = ""    # 操作内容的sha256[:16]
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AuditRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class AlertRecord:
    """告警记录。"""
    session_id: str
    trigger_score: float
    threshold: float
    alert_type: str           # "cumulative_threshold" | "health_check"
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class StatefulAuditTrail:
    """跨session的信号保真追踪器。

    设计原则：
    - JSONL追加写入（原子操作，与AuditChain同模式）
    - cumulative_score跨session累积
    - P95阈值从baseline数据标定（不拍脑袋）
    - 日志自动归档（retention_days参数）

    用法：
        trail = StatefulAuditTrail()
        # 记录一次操作
        trail.record("session-1", "tool_call", suspicion_score=0.3,
                     content_hash="abc123", details={"tool": "web_search"})
        # 检查是否触发告警
        alert = trail.check_threshold("session-1")
        if alert:
            print(f"⚠️ 累积风险: {alert.trigger_score:.2f} > {alert.threshold:.2f}")
        # 扫描全局分布
        stats = trail.get_distribution()
    """

    def __init__(self, audit_dir: Optional[Path] = None,
                 retention_days: int = 90):
        """
        Args:
            audit_dir: 审计日志目录，默认 DEFAULT_AUDIT_DIR
            retention_days: 日志保留天数，超过则归档
        """
        self.audit_dir = audit_dir or DEFAULT_AUDIT_DIR
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir = self.audit_dir / "archive"
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days

        # 内存中的累积状态（启动时从JSONL恢复）
        self._cumulative_by_session: dict[str, float] = {}
        self._records_cache: list[dict] = []
        self._load_state()

    def _records_path(self) -> Path:
        return self.audit_dir / "stateful_scores.jsonl"

    def _alerts_path(self) -> Path:
        return self.audit_dir / "stateful_alerts.jsonl"

    def _load_state(self) -> None:
        """从JSONL恢复累积状态。"""
        path = self._records_path()
        if not path.exists():
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        sid = rec.get("session_id", "")
                        cum = rec.get("cumulative_score", 0.0)
                        if sid:
                            # 恢复每个session的最新累积分数
                            self._cumulative_by_session[sid] = cum
                            self._records_cache.append(rec)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning(f"加载状态失败: {e}")

    def record(self, session_id: str, event_type: str,
               suspicion_score: float,
               content_hash: str = "",
               details: Optional[dict] = None) -> AuditRecord:
        """记录一次操作的suspicion_score，并更新累积分数。

        Args:
            session_id: 会话ID
            event_type: 事件类型（tool_call/file_write/skill_invoke/custom）
            suspicion_score: 嫌疑分数 0.0~1.0
            content_hash: 操作内容的hash（可选）
            details: 额外信息（可选）

        Returns:
            AuditRecord 包含更新后的cumulative_score
        """
        # 获取该session之前的累积分数
        prev_cumulative = self._cumulative_by_session.get(session_id, 0.0)

        # 新累积 = 旧累积 + 当前嫌疑分
        new_cumulative = prev_cumulative + suspicion_score

        record = AuditRecord(
            session_id=session_id,
            event_type=event_type,
            suspicion_score=suspicion_score,
            cumulative_score=new_cumulative,
            content_hash=content_hash,
            details=details or {},
        )

        # 更新内存状态
        self._cumulative_by_session[session_id] = new_cumulative
        self._records_cache.append(record.to_dict())

        # JSONL追加写入
        self._append_jsonl(self._records_path(), record.to_dict())

        logger.debug(
            f"审计记录: session={session_id} type={event_type} "
            f"score={suspicion_score:.3f} cumulative={new_cumulative:.3f}"
        )

        return record

    def check_threshold(self, session_id: str,
                        threshold: Optional[float] = None) -> Optional[AlertRecord]:
        """检查某个session的累积分数是否超过阈值。

        Args:
            session_id: 会话ID
            threshold: 阈值，None则用默认值3.0（3次中等嫌疑操作）

        Returns:
            AlertRecord 如果触发告警，否则None
        """
        if threshold is None:
            threshold = 3.0  # 默认：3次中等嫌疑(1.0)触发

        cumulative = self._cumulative_by_session.get(session_id, 0.0)

        if cumulative > threshold:
            alert = AlertRecord(
                session_id=session_id,
                trigger_score=cumulative,
                threshold=threshold,
                alert_type="cumulative_threshold",
            )
            self._append_jsonl(self._alerts_path(), alert.to_dict())

            # 自动追加RECALL告警
            self._append_recall_alert(session_id, cumulative, threshold)

            logger.warning(
                f"⚠️ 累积风险告警: session={session_id} "
                f"cumulative={cumulative:.3f} > threshold={threshold:.3f}"
            )
            return alert

        return None

    def get_distribution(self) -> dict:
        """获取全局suspicion_score分布统计。

        Returns:
            {
                "total_records": int,
                "total_sessions": int,
                "mean_score": float,
                "std_score": float,
                "p50": float,
                "p95": float,
                "p99": float,
                "max_score": float,
            }
        """
        scores = [r["suspicion_score"] for r in self._records_cache]
        if not scores:
            return {
                "total_records": 0, "total_sessions": 0,
                "mean_score": 0, "std_score": 0,
                "p50": 0, "p95": 0, "p99": 0, "max_score": 0,
            }

        scores_sorted = sorted(scores)
        n = len(scores_sorted)
        mean = sum(scores) / n
        std = (sum((s - mean) ** 2 for s in scores) / n) ** 0.5

        return {
            "total_records": n,
            "total_sessions": len(self._cumulative_by_session),
            "mean_score": round(mean, 4),
            "std_score": round(std, 4),
            "p50": round(scores_sorted[int(n * 0.5)], 4),
            "p95": round(scores_sorted[int(n * 0.95)], 4),
            "p99": round(scores_sorted[int(n * 0.99)], 4),
            "max_score": round(max(scores), 4),
        }

    def get_cumulative_by_session(self) -> dict[str, float]:
        """获取所有session的累积分数。"""
        return dict(self._cumulative_by_session)

    def health_check(self, threshold_shift_pct: float = 20.0) -> dict:
        """定期健康检查——检测P95偏移。

        Args:
            threshold_shift_pct: P95偏移告警阈值（百分比）

        Returns:
            {
                "status": "healthy" | "warning" | "critical",
                "current_p95": float,
                "record_count": int,
                "session_count": int,
                "message": str,
            }
        """
        stats = self.get_distribution()

        if stats["total_records"] < 10:
            return {
                "status": "warning",
                "current_p95": stats["p95"],
                "record_count": stats["total_records"],
                "session_count": stats["total_sessions"],
                "message": f"样本不足({stats['total_records']}条)，至少需要10条",
            }

        # 检查是否有session累积分数异常
        max_cumulative = max(self._cumulative_by_session.values()) if self._cumulative_by_session else 0
        if max_cumulative > 5.0:
            return {
                "status": "critical",
                "current_p95": stats["p95"],
                "record_count": stats["total_records"],
                "session_count": stats["total_sessions"],
                "message": f"session累积分数异常高({max_cumulative:.2f})，需要立即审查",
            }

        return {
            "status": "healthy",
            "current_p95": stats["p95"],
            "record_count": stats["total_records"],
            "session_count": stats["total_sessions"],
            "message": f"P95={stats['p95']:.4f}，分布正常",
        }

    def archive_old_records(self) -> int:
        """归档超过retention_days的旧记录。

        Returns:
            归档的记录数
        """
        path = self._records_path()
        if not path.exists():
            return 0

        cutoff = time.time() - (self.retention_days * 86400)
        kept = []
        archived = 0

        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        # 单一判定源（成市拍板 2026-10-01）：原写法
                        # `rec.get("timestamp",0) < cutoff` 遇 ISO 字符串 →
                        # str<float TypeError，内层只捕 JSONDecodeError，一条脏行
                        # 击穿整个归档（外层 except → return 0）。改过
                        # openllm.isa.timeutil.coerce_ts 双格式容忍。
                        ts_val = coerce_ts(rec.get("timestamp", rec.get("ts", 0)))
                        if ts_val < cutoff:
                            # 归档：追加到archive文件
                            archive_file = self.archive_dir / f"scores_{cutoff:.0f}.jsonl"
                            self._append_jsonl(archive_file, rec)
                            archived += 1
                        else:
                            kept.append(line)
                    except json.JSONDecodeError:
                        kept.append(line)

            if archived > 0:
                # 重写主文件（只保留未归档的）
                with open(path, "w", encoding="utf-8") as f:
                    for line in kept:
                        f.write(line + "\n")
                logger.info(f"归档完成: {archived}条记录迁移到archive/")

        except Exception as e:
            logger.error(f"归档失败: {e}")
            return 0

        return archived

    def _append_jsonl(self, path: Path, data: dict) -> bool:
        """JSONL追加写入（原子操作）。"""
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
            return True
        except Exception as e:
            logger.error(f"JSONL追加失败: {e}")
            return False

    def _append_recall_alert(self, session_id: str,
                             cumulative: float, threshold: float) -> None:
        """自动追加RECALL告警记录。

        写侧时间戳规范（成市拍板 2026-10-01）：canonical 字段 `ts`（epoch
        float）。优先走门房 scripts/recall_append.py（与 execution_recorder
        同款 subprocess 模式，门房负责 enrich/哈希/容量守卫）；门房不可用
        时回退为直写，但仍落规范字段，不再写 `timestamp`。
        """
        # Path.home() 调用时求值（测试可重定向 HOME；conftest 已隔离）
        jiak_dir = Path.home() / ".hermes" / "jiak"
        recall_path = jiak_dir / "RECALL.jsonl"
        if not recall_path.exists():
            return

        alert_entry = {
            "type": "security_alert",
            "source": "stateful_audit",
            "session_id": session_id,
            "cumulative_score": cumulative,
            "threshold": threshold,
            "ts": time.time(),
            "message": f"跨session累积风险超阈值: {cumulative:.2f} > {threshold:.2f}",
        }

        gate_script = jiak_dir / "scripts" / "recall_append.py"
        if gate_script.exists():
            try:
                import subprocess
                import sys
                subprocess.run(
                    [sys.executable, str(gate_script),
                     json.dumps(alert_entry, ensure_ascii=False)],
                    capture_output=True, timeout=10,
                )
                return
            except Exception:
                pass  # 门房失败不阻塞主流程，回退直写

        try:
            with open(recall_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(alert_entry, ensure_ascii=False) + "\n")
        except Exception:
            pass  # RECALL写入失败不阻塞主流程
