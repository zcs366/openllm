"""
路由可观测化 — JSONL日志记录器

每次路由决策写入一条JSONL记录，包含：
- timestamp: 时间戳
- intent: 用户输入
- turn_count: 第几轮
- decisions: 各阶段决策列表
- stats: 本轮路由统计
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger("openllm.router_logger")


class RouterLogger:
    """路由决策日志记录器。"""
    
    def __init__(self, log_path: str = "~/.openllm/router_log.jsonl", 
                 max_bytes: int = 1_000_000, enabled: bool = True):
        self.log_path = Path(log_path).expanduser()
        self.max_bytes = max_bytes
        self.enabled = enabled
        if enabled:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def log_decision(self, intent: str, turn_count: int, decisions: list, 
                     context_pct: float = 0.0, stats: Optional[dict] = None):
        """记录一次路由决策。"""
        if not self.enabled:
            return
        record = {
            "timestamp": time.time(),
            "datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "intent": intent[:200],  # 截断防止超长输入
            "turn_count": turn_count,
            "context_pct": round(context_pct * 100, 1),
            "decisions": [
                {
                    "phase": d.phase,
                    "action": d.action.name,
                    "reason": d.reason,
                }
                for d in decisions
            ],
            "stats": stats or {},
        }
        
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        # 轮转检查
        self._rotate_if_needed()
    
    def _rotate_if_needed(self):
        """日志超过max_bytes时轮转。"""
        try:
            if self.log_path.exists() and self.log_path.stat().st_size > self.max_bytes:
                rotated = self.log_path.with_suffix(f".{int(time.time())}.jsonl")
                self.log_path.rename(rotated)
                # 压缩旧日志
                try:
                    import gzip
                    with open(rotated, "rb") as f_in:
                        with gzip.open(str(rotated) + ".gz", "wb") as f_out:
                            f_out.write(f_in.read())
                    rotated.unlink()
                except Exception as e:
                    logger.warning("Router日志压缩失败: %s", e)
        except Exception as e:
            logger.warning("Router日志轮转失败: %s", e)
    
    def get_recent(self, n: int = 10) -> list:
        """读取最近n条路由决策。"""
        if not self.log_path.exists():
            return []
        lines = self.log_path.read_text(encoding="utf-8").strip().split("\n")
        recent = []
        for line in lines[-n:]:
            if line.strip():
                try:
                    recent.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return recent
    
    def get_summary(self) -> dict:
        """获取路由决策摘要。"""
        if not self.log_path.exists():
            return {"total_decisions": 0}
        
        lines = self.log_path.read_text(encoding="utf-8").strip().split("\n")
        total = len([l for l in lines if l.strip()])
        
        if total == 0:
            return {"total_decisions": 0}
        
        # 统计各阶段跳过/降级频率
        phase_stats = {}
        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                for d in record.get("decisions", []):
                    phase = d["phase"]
                    action = d["action"]
                    if phase not in phase_stats:
                        phase_stats[phase] = {"RUN": 0, "SKIP": 0, "DEGRADED": 0}
                    phase_stats[phase][action] = phase_stats[phase].get(action, 0) + 1
            except json.JSONDecodeError:
                continue
        
        return {
            "total_decisions": total,
            "phase_stats": phase_stats,
            "log_size_bytes": self.log_path.stat().st_size,
        }
