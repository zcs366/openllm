"""
TriggerBenchmark — IAI Layer 9 前瞻触发评测
============================================
前瞻触发：在用户意图未明确时，根据上下文模式预测下一步需求并提前触发。
predict_fn由外部注入，本模块不直接调用LLM。JSON持久化。
"""
import json, time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

@dataclass
class TestCase:
    """单条触发评测用例。"""
    pattern: str
    expected_prediction: bool
    context: Optional[str] = None
    id: str = ""
    def __post_init__(self):
        if not self.id:
            self.id = f"tc_{hash(self.pattern) & 0xFFFFFF:06x}"

@dataclass
class EvalResult:
    """单次评测运行结果。"""
    run_id: str
    timestamp: float
    total: int
    correct: int
    accuracy: float
    details: list[dict] = field(default_factory=list)

class TriggerBenchmark:
    """IAI Layer 9 前瞻触发评测器。

    用法:
        bench = TriggerBenchmark()
        bench.add_test_case("用户提到deadline临近", True, context="明天交报告")
        bench.add_test_case("用户说今天天气不错", False)
        result = bench.evaluate(my_predict_fn)
        print(bench.get_accuracy())
    """
    def __init__(self, persist_path: Optional[str] = None):
        self._cases: list[TestCase] = []
        self._results: list[EvalResult] = []
        self._persist_path = Path(persist_path) if persist_path else None
        if self._persist_path:
            self._load()

    def add_test_case(self, pattern: str, expected_prediction: bool,
                      context: Optional[str] = None) -> str:
        """添加评测用例。返回用例ID。"""
        tc = TestCase(pattern=pattern, expected_prediction=expected_prediction,
                      context=context)
        self._cases.append(tc)
        return tc.id

    @property
    def case_count(self) -> int:
        return len(self._cases)

    def evaluate(self, predict_fn: Callable[[str, Optional[str]], bool]) -> dict:
        """运行一轮评测。predict_fn签名: (pattern, context) -> bool。"""
        if not self._cases:
            return {"error": "无评测用例", "accuracy": 0.0}
        details, correct = [], 0
        for tc in self._cases:
            predicted = predict_fn(tc.pattern, tc.context)
            ok = predicted == tc.expected_prediction
            correct += int(ok)
            details.append({"id": tc.id, "pattern": tc.pattern,
                            "expected": tc.expected_prediction, "predicted": predicted,
                            "correct": ok})
        accuracy = correct / len(self._cases)
        result = EvalResult(run_id=f"run_{int(time.time() * 1000)}",
                            timestamp=time.time(), total=len(self._cases),
                            correct=correct, accuracy=accuracy, details=details)
        self._results.append(result)
        self._save()
        return {"run_id": result.run_id, "accuracy": accuracy,
                "correct": correct, "total": len(self._cases),
                "failed": len(self._cases) - correct}

    def get_accuracy(self) -> float:
        """最近一次评测准确率，无结果返回0.0。"""
        return self._results[-1].accuracy if self._results else 0.0

    def get_trend(self, n_runs: int = 10) -> list[float]:
        """最近N次评测的准确率趋势。"""
        return [r.accuracy for r in self._results[-n_runs:]]

    def export_results(self, path: str) -> None:
        """导出评测结果为JSON文件。"""
        data = {
            "cases": [asdict(tc) for tc in self._cases],
            "results": [{"run_id": r.run_id, "timestamp": r.timestamp,
                         "total": r.total, "correct": r.correct,
                         "accuracy": r.accuracy, "details": r.details}
                        for r in self._results]}
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _save(self) -> None:
        if self._persist_path:
            self.export_results(str(self._persist_path))

    def _load(self) -> None:
        """从持久化文件恢复。"""
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            data = json.loads(self._persist_path.read_text(encoding="utf-8"))
            self._cases = [TestCase(**c) for c in data.get("cases", [])]
            self._results = [EvalResult(**r) for r in data.get("results", [])]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
