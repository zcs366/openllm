"""prediction.py — ISA 因果预测接口 (≤150行)

ISA 从被动记录升级为因果推理器官：计算"应该发生什么" vs "实际发生了什么"的误差。
工程法典：纯计算，不调用 LLM。余弦距离 + 指数移动平均期望向量。
七神天启：雅典娜——理解必须以局部失明为代价。
"""
import math
import time
import json
from pathlib import Path
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
        """将 dict 编码为固定维度向量。纯哈希，无 LLM。

        2026-08-14修复：原用内置hash()（PYTHONHASHSEED随机化，字符串hash
        跨进程不稳定）→ test_different_contexts_positive_error偶发失败
        （两段不同文本碰巧hash到同一idx）。改用zlib.crc32确定性哈希。
        """
        def _stable_hash(s: str) -> int:
            import zlib
            return zlib.crc32(s.encode("utf-8"))

        vec = [0.0] * PredictionEngine.VECTOR_DIM
        for i, (k, v) in enumerate(data.items()):
            idx = _stable_hash(k) % PredictionEngine.VECTOR_DIM
            if isinstance(v, (int, float)):
                vec[idx] += float(v)
            elif isinstance(v, str):
                vec[idx] += len(v) * 0.1
            elif isinstance(v, bool):
                vec[idx] += 1.0 if v else -1.0
            elif isinstance(v, (list, tuple)):
                vec[idx] += len(v) * 0.5
            else:
                vec[idx] += _stable_hash(str(v)) % 100 / 100.0
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

    # ── 预测偏差记录（PAL T-F-6：预测律闭环·2026-09-02） ──

    @staticmethod
    def _errors_file(path: Optional[Path] = None) -> Path:
        """返回预测偏差JSONL文件路径。"""
        return path or (Path.home() / ".openllm" / "iai" / "prediction_errors.jsonl")

    def record_error(self, prediction: dict, actual: dict,
                     source: str = "compare",
                     errors_file: Optional[Path] = None) -> dict:
        """记录预测偏差。独立验证：user_feedback直录，compare需match=False。

        预测律闭环（赫尔墨斯铁律）：
        - source='user_feedback' → 已独立验证，直接记录+emit
        - source='compare' → 依赖 delta.prediction_match：True=预测正确不记录
          （match 字段从 prediction dict 取，不存在则计算）

        返回: {recorded: bool, match: bool, error: float, event_id: str}
        """
        # 独立验证：user_feedback 视为外部已确认
        if source == "user_feedback":
            match = False  # 外部确认有偏差
        else:
            # compare 路径：检查 prediction dict 中的 match 字段
            # （由 octopus.compare 通过 heartbeat_context 传入）
            match = prediction.get("match", prediction.get("prediction_match", None))
            if match is None:
                # 无显式标记：自行计算偏差
                error = self.compute_error(prediction, actual)
                match = error < 0.3  # 低于阈值=预测正确

        if match:
            return {"recorded": False, "match": True, "error": 0.0, "event_id": ""}

        # 计算偏差值
        error = self.compute_error(prediction, actual)
        confidence = prediction.get("confidence", 0.0)

        # 构造偏差记录
        record = {
            "predicted": prediction.get("predicted_type", prediction),
            "actual": actual,
            "match": False,
            "source": source,
            "timestamp": time.time(),
            "confidence": round(confidence, 4),
            "error": round(error, 4),
        }

        # emit prediction.error 事件
        evt = self.emit_event("prediction.error", record, entropy_score=error)

        # append-only 写 JSONL
        self._append_error_record(record, errors_file)

        return {
            "recorded": True,
            "match": False,
            "error": round(error, 4),
            "event_id": evt.event_id,
        }

    def _append_error_record(self, record: dict,
                             errors_file: Optional[Path] = None) -> None:
        """append-only 写预测偏差到 JSONL 文件。失败不阻塞主流程。"""
        fpath = self._errors_file(errors_file)
        try:
            fpath.parent.mkdir(parents=True, exist_ok=True)
            with open(fpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except IOError as e:
            pass  # 不阻塞主循环

    # ── 状态持久化（IAI融合·2026-09-02：EMA跨调用/跨重启累积） ──

    def to_state(self) -> dict:
        """导出预测器状态（期望向量+计数+alpha）。"""
        return {
            "expectation": self._expectation,
            "count": self._count,
            "alpha": self._alpha,
        }

    def from_state(self, state: dict) -> None:
        """恢复预测器状态。"""
        if not state:
            return
        self._expectation = state.get(
            "expectation", [0.0] * self.VECTOR_DIM)
        self._count = state.get("count", 0)
        self._alpha = state.get("alpha", self._alpha)

    def save(self, path=None) -> bool:
        """持久化到JSON文件。"""
        import json
        path = Path(path) if path else (
            Path.home() / ".openllm" / "iai" / "predictor_state.json")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_state(), f)
            return True
        except Exception:
            return False

    def load(self, path=None) -> bool:
        """从JSON文件恢复状态。返回是否成功。"""
        import json
        path = Path(path) if path else (
            Path.home() / ".openllm" / "iai" / "predictor_state.json")
        try:
            with open(path, encoding="utf-8") as f:
                self.from_state(json.load(f))
            return True
        except Exception:
            return False
