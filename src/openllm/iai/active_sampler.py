"""ActiveSampler — 主动采样律⑧

超级Agent十律⑧：感知是主动采样非被动接收。
蜻蜓微扫视：agent主动搜索/主动提问（非预测确认）。

核心机制：
  - 赫淮斯托斯硬上限：每轮预算3次，防API成本膨胀
  - 克洛诺斯反预测扰动：explore模式随机偏离焦点采样，破自证预言闭环
  - 赫拉克勒斯日志：append-only ~/.openllm/iai/sample_log.jsonl

集成点：章鱼I感知阶段可选接入（sampler is None时零开销）。
"""
import json
import random
import time
import uuid
from pathlib import Path
from typing import Optional


class ActiveSampler:
    """主动采样器 — 每轮预算硬上限 + 反预测扰动。

    Attributes:
        budget: 每轮最大采样次数（硬上限）
        _used: 本轮已用次数
        _round_id: 本轮唯一标识
    """

    DEFAULT_BUDGET = 3  # 赫淮斯托斯硬上限

    def __init__(self, budget: int = DEFAULT_BUDGET,
                 log_dir: Optional[Path] = None):
        """
        Args:
            budget: 每轮采样预算（硬上限），默认3次
            log_dir: 日志目录，默认 ~/.openllm/iai/
        """
        self.budget = budget
        self._used = 0
        self._round_id = f"round_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        self._log_dir = log_dir or (Path.home() / ".openllm" / "iai")
        self._log_file = self._log_dir / "sample_log.jsonl"

    def sample(self, mode: str = "verify") -> bool:
        """检查是否可以发起一次主动采样。

        Args:
            mode: 'explore'=反预测扰动（随机偏离焦点）,
                  'verify'=验证预测（常规确认）

        Returns:
            True=可以采样（调用方应执行采样并调用mark_used）
            False=预算已用尽，本轮不可再采样
        """
        if self._used >= self.budget:
            return False
        if mode not in ("explore", "verify"):
            raise ValueError(f"mode必须是'explore'或'verify'，收到'{mode}'")
        return True

    def mark_used(self, mode: str, focus_topic: str = "",
                  deviation: float = 0.0):
        """消耗一次预算并记录日志。

        Args:
            mode: 采样模式（explore/verify）
            focus_topic: 当前焦点主题（explore模式偏离此主题）
            deviation: 偏离度 0.0~1.0（explore时>0表示偏离焦点的距离）

        Raises:
            RuntimeError: 预算已用尽时抛出
        """
        if self._used >= self.budget:
            raise RuntimeError(
                f"采样预算已用尽（{self._used}/{self.budget}）"
            )
        self._used += 1
        self._append_log(mode, focus_topic, deviation)

    def _append_log(self, mode: str, focus_topic: str,
                    deviation: float):
        """append-only日志写入。写失败静默不抛异常。"""
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            entry = {
                "timestamp": time.time(),
                "round_id": self._round_id,
                "mode": mode,
                "focus_topic": focus_topic,
                "deviation": deviation,
                "budget_used": self._used,
                "budget_total": self.budget,
                "anti_prediction": mode == "explore",
            }
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass  # 写失败不阻塞主循环

    def reset_budget(self):
        """重置预算（新轮次开始时调用）。"""
        self._used = 0
        self._round_id = f"round_{int(time.time())}_{uuid.uuid4().hex[:6]}"

    def generate_deviation(self, focus_topic: str = "") -> float:
        """克洛诺斯反预测扰动：explore模式生成随机偏离。

        不在预测焦点位置采样——随机选一个偏离方向。
        返回偏离度（0.0~1.0），调用方可据此选择采样方向。

        Returns:
            随机偏离度，>0.3为显著偏离
        """
        return round(random.random(), 4)

    def get_status(self) -> dict:
        """当前状态摘要。"""
        return {
            "budget": self.budget,
            "used": self._used,
            "remaining": max(0, self.budget - self._used),
            "round_id": self._round_id,
            "exhausted": self._used >= self.budget,
        }

    @property
    def remaining(self) -> int:
        """剩余预算。"""
        return max(0, self.budget - self._used)

    def __repr__(self) -> str:
        return (f"ActiveSampler(budget={self.budget}, "
                f"used={self._used}, round={self._round_id})")
