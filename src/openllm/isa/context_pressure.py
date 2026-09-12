"""
ISA Layer 8 — 上下文压力监控（SelfCompact）

纯规则引擎，零LLM调用。监控会话上下文窗口使用率，
当压力超过阈值时触发自主压缩策略。

压力分级：
  normal   < 70%  → 无操作
  caution  70-85% → 轻量压缩（丢弃旧消息）
  critical > 85%  → 标准/深度压缩（摘要+保留关键信息）

核心格言：
  上下文窗口是Agent的短期记忆，压力是它的体温。
  体温过高必须降温，否则认知系统崩溃。
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("openllm.context_pressure")

# ── 压力阈值 ──────────────────────────────────────────
THRESHOLD_CAUTION = 0.70   # 70% → 轻量压缩
THRESHOLD_CRITICAL = 0.85  # 85% → 标准/深度压缩

# ── 压缩策略定义 ──────────────────────────────────────
STRATEGIES: Dict[str, Dict[str, Any]] = {
    "light": {
        "name": "light",
        "description": "轻量压缩——仅丢弃最旧的消息，保留最近对话",
        "action": "truncate_oldest",
        "max_drop_ratio": 0.3,        # 最多丢弃30%的消息
        "keep_recent": 5,              # 至少保留最近5条
        "summary": False,
    },
    "standard": {
        "name": "standard",
        "description": "标准压缩——旧消息摘要化+保留关键信息",
        "action": "summarize_old",
        "summarize_from_percentile": 0.5,  # 从中间位置开始摘要
        "summary_max_tokens": 500,
        "keep_recent": 8,
        "summary": True,
    },
    "deep": {
        "name": "deep",
        "description": "深度压缩——全量重压缩，仅保留关键决策和最新上下文",
        "action": "full_recompress",
        "keep_recent": 10,
        "keep_decisions": True,        # 保留所有决策点
        "keep_tool_results": False,    # 丢弃工具调用结果
        "summary_max_tokens": 800,
        "summary": True,
    },
}


class ContextPressureMonitor:
    """
    上下文压力监控器——Layer 8 SelfCompact 核心组件。

    跟踪每个会话的上下文使用率，根据压力级别决定压缩策略。
    所有判断基于纯规则，无LLM调用。
    """

    def __init__(self, state_path: Optional[Path] = None):
        """
        初始化压力监控器。

        Args:
            state_path: 持久化文件路径，默认 ~/.hermes/memory/context_pressure_state.json
        """
        if state_path is None:
            state_path = Path.home() / ".hermes" / "memory" / "context_pressure_state.json"
        self._state_path = state_path
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._load_state()

    # ── 核心API ───────────────────────────────────────

    def record_usage(self, session_id: str, tokens_used: int, tokens_budget: int) -> None:
        """
        记录会话的上下文使用情况。

        Args:
            session_id: 会话标识符
            tokens_used: 已使用的token数
            tokens_budget: 上下文窗口总预算（token上限）
        """
        if tokens_budget <= 0:
            logger.warning(f"[SelfCompact] 非法预算值: {tokens_budget}, session={session_id}")
            return

        ratio = tokens_used / tokens_budget
        self._sessions[session_id] = {
            "tokens_used": tokens_used,
            "tokens_budget": tokens_budget,
            "ratio": round(ratio, 4),
            "level": self._classify(ratio),
            "updated_at": time.time(),
            "compressions": self._sessions.get(session_id, {}).get("compressions", 0),
        }
        self._save_state()

        level = self._sessions[session_id]["level"]
        if level != "normal":
            logger.info(f"[SelfCompact] 压力检测 session={session_id} level={level} ratio={ratio:.1%}")

    def get_pressure_level(self, session_id: str) -> str:
        """
        获取会话当前压力级别。

        Returns:
            'normal' | 'caution' | 'critical' | 'unknown'
        """
        session = self._sessions.get(session_id)
        if not session:
            return "unknown"
        return session["level"]

    def should_compress(self, session_id: str) -> Tuple[bool, str]:
        """
        判断会话是否应触发压缩。

        Returns:
            (是否应压缩, 推荐策略级别) — 如 (False, '') 或 (True, 'light')
        """
        level = self.get_pressure_level(session_id)
        if level in ("caution", "critical"):
            strategy = self._strategy_for_level(level)
            return True, strategy
        return False, ""

    def get_compression_strategy(self, level: str) -> Dict[str, Any]:
        """
        获取指定压力级别的压缩策略。

        Args:
            level: 'light' | 'standard' | 'deep'

        Returns:
            压缩策略字典
        """
        return STRATEGIES.get(level, STRATEGIES["light"])

    def clear_session(self, session_id: str) -> None:
        """清除指定会话的压力监控数据。"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            self._save_state()
            logger.info(f"[SelfCompact] 已清除 session={session_id}")

    def get_stats(self) -> Dict[str, Any]:
        """获取所有会话的压力概览（调试用）。"""
        return {
            "active_sessions": len(self._sessions),
            "sessions": {
                sid: {"level": s["level"], "ratio": s["ratio"], "compressions": s["compressions"]}
                for sid, s in self._sessions.items()
            },
        }

    # ── 内部方法 ──────────────────────────────────────

    @staticmethod
    def _classify(ratio: float) -> str:
        """根据使用率分类压力级别。"""
        if ratio >= THRESHOLD_CRITICAL:
            return "critical"
        elif ratio >= THRESHOLD_CAUTION:
            return "caution"
        return "normal"

    @staticmethod
    def _strategy_for_level(level: str) -> str:
        """压力级别→压缩策略映射。"""
        return {
            "caution": "light",
            "critical": "standard",
        }.get(level, "light")

    def _load_state(self) -> None:
        """从JSON文件恢复状态。"""
        if not self._state_path.exists():
            return
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            self._sessions = data.get("sessions", {})
            logger.debug(f"[SelfCompact] 加载 {len(self._sessions)} 个会话状态")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"[SelfCompact] 状态加载失败: {e}")
            self._sessions = {}

    def _save_state(self) -> None:
        """将状态持久化到JSON文件。"""
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "updated_at": time.time(),
                "sessions": self._sessions,
            }
            self._state_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning(f"[SelfCompact] 状态保存失败: {e}")
