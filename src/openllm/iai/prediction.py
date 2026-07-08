"""prediction.py — ISA 因果预测接口 (≤150行)

ISA 从被动记录升级为因果推理器官：计算"应该发生什么" vs "实际发生了什么"的误差。
工程法典：纯计算，不调用 LLM。余弦距离 + 指数移动平均期望向量。
七神天启：雅典娜——理解必须以局部失明为代价。
"""
import math
import time
from typing import Any, Optional

from openllm.iai.event_bus import EventBus, Event, BaseEventEmitter


class PredictionEngine(BaseEventEmitter):
    """因果预测引擎。给定上下文预测下一步输出类型，计算预测误差。"""

    BODY_NAME = "ISA"
    VECTOR_DIM = 8  # 低维期望向量维度
    _OUTPUT_TYPES = [
        "text", "code", "diagram", "analysis", "search",
        "tool_call", "memory_write", "question",
    ]

    def __init__(self, bus: Optional[EventBus] = None, ema_alpha: float = 0.15):
        # BaseEventEmitter 要求 bus 非 None，但允许独立运行
        self._own_bus = bus is None
        super().__init__(bus or EventBus())
        self._expectation = [0.0] * self.VECTOR_DIM
        self._alpha = ema_alpha
        self._count = 0

    # ── 向量编码 ──

    @staticmethod
    def _encode(data: dict) -> list[float]:
        """将 dict 编码为固定维度向量。纯哈希，无 LLM。"""
        vec = [0.0] * PredictionEngine.VECTOR_DIM
        for i, (k, v) in enumerate(data.items()):
            idx = hash(k) % PredictionEngine.VECTOR_DIM
            if isinstance(v, (int, float)):
                vec[idx] += float(v)
            elif isinstance(v, str):
                vec[idx] += len(v) * 0.1
            elif isinstance(v, bool):
                vec[idx] += 1.0 if v else -1.0
            elif isinstance(v, (list, tuple)):
                vec[idx] += len(v) * 0.5
            else:
                vec[idx] += hash(str(v)) % 100 / 100.0
        # L2 归一化
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    @staticmethod
    def _vec_dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    @staticmethod
    def _cosine_distance(a: list[float], b: list[float]) -> float:
        """余弦距离 = 1 - 余弦相似度。范围 [0, 2]。"""
        da = math.sqrt(sum(x * x for x in a)) or 1e-9
        db = math.sqrt(sum(x * x for x in b)) or 1e-9
        dot = sum(x * y for x, y in zip(a, b))
        sim = max(-1.0, min(1.0, dot / (da * db)))
        return 1.0 - sim

    # ── 核心接口 ──

    def predict_next(self, context: dict) -> dict:
        """给定当前上下文，预测下一步应该产生什么类型的输出。

        基于期望向量与上下文向量的余弦距离，选择最接近的输出类型。
        """
        ctx_vec = self._encode(context)
        dist = self._cosine_distance(ctx_vec, self._expectation)
        # 用距离加权选择：距离越近的类型概率越高（贪心取最优）
        type_scores = {}
        for t in self._OUTPUT_TYPES:
            t_vec = self._encode({"type": t})
            score = 1.0 - self._cosine_distance(ctx_vec, t_vec)
            type_scores[t] = score
        predicted_type = max(type_scores, key=type_scores.get)  # type: ignore[arg-type]
        return {
            "predicted_type": predicted_type,
            "confidence": 1.0 - dist,
            "context_vector": ctx_vec,
            "timestamp": time.time(),
        }

    def compute_error(self, prediction: dict, actual: dict) -> float:
        """计算预测与实际的误差（余弦距离）。纯算术，零 LLM 调用。"""
        pred_vec = prediction.get("context_vector") or self._encode(prediction)
        actual_vec = self._encode(actual)
        return self._cosine_distance(pred_vec, actual_vec)

    def update_expectation(self, actual: dict) -> None:
        """用指数移动平均更新期望向量。每次预测后调用。"""
        actual_vec = self._encode(actual)
        if self._count == 0:
            self._expectation = list(actual_vec)
        else:
            a = self._alpha
            self._expectation = [
                a * v + (1.0 - a) * e for v, e in zip(actual_vec, self._expectation)
            ]
        self._count += 1

    def should_reroute(self, error: float, threshold: float = 0.3) -> bool:
        """误差超过阈值时触发重路由。"""
        return error > threshold

    def predict_and_check(self, context: dict, actual: dict,
                          threshold: float = 0.3) -> dict:
        """完整管线：预测 → 对比 → 更新 → 可能触发重路由事件。"""
        prediction = self.predict_next(context)
        error = self.compute_error(prediction, actual)
        self.update_expectation(actual)
        reroute = self.should_reroute(error, threshold)
        if reroute:
            self.emit_event("reroute_needed", {
                "error": round(error, 4),
                "predicted_type": prediction["predicted_type"],
                "actual": actual,
                "threshold": threshold,
            })
        return {
            "prediction": prediction,
            "error": round(error, 4),
            "reroute": reroute,
            "expectation_count": self._count,
        }
