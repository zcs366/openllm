"""
实验引擎 — 研究agent的核心工具。

实验生命周期: design → run → collect → analyze → conclude
区别于编程agent的"写代码→测试"：实验引擎处理的是假说验证，不是代码正确性。

数据模型:
  Experiment: 一次实验的完整定义
  ExperimentResult: 实验结果+统计分析
  Hypothesis: 假说及其验证状态
"""

import json
import time
import statistics
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional


# ── 假说状态 ──

class HypothesisStatus(Enum):
    UNVERIFIED = "unverified"
    TESTING = "testing"
    SUPPORTED_1 = "supported(1)"
    SUPPORTED_2 = "supported(2)"
    SUPPORTED_3 = "supported(3)"
    PROVEN = "proven"
    CHALLENGED = "challenged"
    DISPROVEN = "disproven"


@dataclass
class Hypothesis:
    """一条假说。"""
    id: str
    claim: str  # 一句话描述
    prediction: str  # 可证伪的预测
    status: HypothesisStatus = HypothesisStatus.UNVERIFIED
    evidence_count: int = 0
    refutations: int = 0
    created_at: float = field(default_factory=time.time)
    last_tested: Optional[float] = None
    notes: list[str] = field(default_factory=list)

    def record_test(self, supported: bool, note: str = ""):
        """记录一次实验结果。"""
        self.last_tested = time.time()
        self.notes.append(f"[{'✅' if supported else '❌'}] {note}")
        if supported:
            self.evidence_count += 1
            if self.evidence_count >= 3 and self.refutations == 0:
                self.status = HypothesisStatus.PROVEN
            elif self.evidence_count >= 2:
                self.status = HypothesisStatus.SUPPORTED_2
            elif self.evidence_count >= 1:
                self.status = HypothesisStatus.SUPPORTED_1
        else:
            self.refutations += 1
            if self.refutations >= 2:
                self.status = HypothesisStatus.DISPROVEN
            else:
                self.status = HypothesisStatus.CHALLENGED

    def dict(self) -> dict:
        return {
            "id": self.id,
            "claim": self.claim,
            "prediction": self.prediction,
            "status": self.status.value,
            "evidence_count": self.evidence_count,
            "refutations": self.refutations,
            "notes": self.notes[-5:],  # 最近5条
        }


# ── 实验定义 ──

@dataclass
class Experiment:
    """一次实验。"""
    id: str
    name: str
    hypothesis_id: str  # 关联的假说
    description: str = ""
    method: str = ""  # 实验方法
    inputs: dict = field(default_factory=dict)  # 输入参数
    run_fn: Optional[str] = None  # 可序列化的运行函数名
    created_at: float = field(default_factory=time.time)
    status: str = "designed"  # designed/running/completed/failed

    def dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "hypothesis_id": self.hypothesis_id,
            "status": self.status,
            "method": self.method,
        }


@dataclass
class ExperimentResult:
    """实验结果。"""
    experiment_id: str
    success: bool
    metrics: dict = field(default_factory=dict)  # 量化指标
    raw_data: Any = None  # 原始数据
    conclusion: str = ""
    duration_s: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "success": self.success,
            "metrics": self.metrics,
            "conclusion": self.conclusion,
            "duration_s": round(self.duration_s, 2),
        }


# ── 实验引擎 ──

class ExperimentEngine:
    """
    研究agent的实验管理核心。

    职责:
    1. 管理假说生命周期（UNVERIFIED→PROVEN/DISPROVEN）
    2. 设计和记录实验
    3. 收集和分析实验结果
    4. 自动更新假说状态
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        self.hypotheses: dict[str, Hypothesis] = {}
        self.experiments: dict[str, Experiment] = {}
        self.results: list[ExperimentResult] = []
        self._storage_dir = storage_dir or Path.home() / ".openllm" / "research"
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self):
        """从磁盘加载状态。"""
        h_path = self._storage_dir / "hypotheses.json"
        if h_path.exists():
            try:
                data = json.loads(h_path.read_text())
                for h in data:
                    hyp = Hypothesis(
                        id=h["id"], claim=h["claim"], prediction=h["prediction"],
                        status=HypothesisStatus(h["status"]),
                        evidence_count=h.get("evidence_count", 0),
                        refutations=h.get("refutations", 0),
                        notes=h.get("notes", []),
                    )
                    self.hypotheses[hyp.id] = hyp
            except Exception:
                pass

    def _save(self):
        """保存状态到磁盘。"""
        h_path = self._storage_dir / "hypotheses.json"
        h_path.write_text(json.dumps(
            [h.dict() for h in self.hypotheses.values()],
            indent=2, ensure_ascii=False
        ))

    # ── 假说管理 ──

    def add_hypothesis(self, id: str, claim: str, prediction: str) -> Hypothesis:
        """注册一条新假说。"""
        hyp = Hypothesis(id=id, claim=claim, prediction=prediction)
        self.hypotheses[id] = hyp
        self._save()
        return hyp

    def get_hypothesis(self, id: str) -> Optional[Hypothesis]:
        return self.hypotheses.get(id)

    def list_hypotheses(self, status_filter: Optional[str] = None) -> list[dict]:
        """列出所有假说。"""
        hyps = list(self.hypotheses.values())
        if status_filter:
            hyps = [h for h in hyps if h.status.value.startswith(status_filter)]
        return [h.dict() for h in hyps]

    # ── 实验管理 ──

    def design_experiment(
        self, name: str, hypothesis_id: str, method: str,
        inputs: Optional[dict] = None, description: str = ""
    ) -> Experiment:
        """设计一个实验。"""
        if hypothesis_id not in self.hypotheses:
            raise ValueError(f"Unknown hypothesis: {hypothesis_id}")

        exp_id = f"exp_{len(self.experiments)+1:03d}"
        exp = Experiment(
            id=exp_id, name=name, hypothesis_id=hypothesis_id,
            method=method, inputs=inputs or {}, description=description,
        )
        self.experiments[exp_id] = exp
        self.hypotheses[hypothesis_id].status = HypothesisStatus.TESTING
        self._save()
        return exp

    def record_result(
        self, experiment_id: str, success: bool,
        metrics: Optional[dict] = None, conclusion: str = "",
        raw_data: Any = None, duration_s: float = 0.0
    ) -> ExperimentResult:
        """记录实验结果，自动更新关联假说状态。"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Unknown experiment: {experiment_id}")

        exp = self.experiments[experiment_id]
        exp.status = "completed" if success else "failed"

        result = ExperimentResult(
            experiment_id=experiment_id, success=success,
            metrics=metrics or {}, raw_data=raw_data,
            conclusion=conclusion, duration_s=duration_s,
        )
        self.results.append(result)

        # 自动更新假说
        hyp = self.hypotheses[exp.hypothesis_id]
        hyp.record_test(supported=success, note=f"{exp.name}: {conclusion}")
        self._save()
        return result

    # ── 分析 ──

    def summary(self) -> dict:
        """研究进展摘要。"""
        status_counts = {}
        for h in self.hypotheses.values():
            s = h.status.value
            status_counts[s] = status_counts.get(s, 0) + 1

        return {
            "total_hypotheses": len(self.hypotheses),
            "total_experiments": len(self.experiments),
            "total_results": len(self.results),
            "status_distribution": status_counts,
            "proven": [h.dict() for h in self.hypotheses.values()
                       if h.status == HypothesisStatus.PROVEN],
            "disproven": [h.dict() for h in self.hypotheses.values()
                         if h.status == HypothesisStatus.DISPROVEN],
        }


# ── CLI ──

def main():
    import sys
    engine = ExperimentEngine()

    if len(sys.argv) < 2:
        print(json.dumps(engine.summary(), indent=2, ensure_ascii=False))
        return

    cmd = sys.argv[1]
    if cmd == "add":
        _, hid, claim, pred = sys.argv[:4]
        h = engine.add_hypothesis(hid, claim, pred)
        print(f"✅ Added hypothesis: {h.id} — {h.claim}")
    elif cmd == "list":
        for h in engine.list_hypotheses():
            print(f"  [{h['status']}] {h['id']}: {h['claim']}")
    elif cmd == "summary":
        print(json.dumps(engine.summary(), indent=2, ensure_ascii=False))
    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
