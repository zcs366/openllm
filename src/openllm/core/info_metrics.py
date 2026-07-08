"""IKO Layer 12 信息论度量 — Agentic as Compressor

纯数学计算，零LLM调用。度量Agent推理的信息压缩质量：
  - Shannon熵 H = -Σ p(x)*log₂p(x)
  - 压缩比 = compressed_size / original_size
  - 信息增益 = H(new) - [H(context+new) - H(context)]
JSON持久化到 ~/.hermes/core/info_metrics_state.json
架构归属：IKO (输出层) → Layer 12 Agentic as Compressor
"""
import json, math, time
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

_STATE_DIR = Path.home() / ".hermes" / "core"
_STATE_PATH = _STATE_DIR / "info_metrics_state.json"

@dataclass
class InferenceRecord:
    """单次推理的信息度量记录"""
    session_id: str
    input_len: int
    output_len: int
    input_entropy: float
    output_entropy: float
    compression_ratio: float
    info_gain: float
    timestamp: float = field(default_factory=time.time)
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def _char_freq(text: str) -> dict[str, float]:
    """字符级频率分布→概率字典。"""
    if not text: return {}
    counter = Counter(text)
    total = len(text)
    return {ch: count / total for ch, count in counter.items()}

def shannon_entropy(text: str) -> float:
    """Shannon熵 H = -Σ p(x)*log₂p(x)，单位：比特。"""
    freq = _char_freq(text)
    if not freq: return 0.0
    return -sum(p * math.log2(p) for p in freq.values() if p > 0)

def combined_entropy(a: str, b: str) -> float:
    """联合熵 H(A,B)。"""
    return shannon_entropy(a + b)

class InfoMetrics:
    """信息论度量器 — Layer 12 Agentic as Compressor。

    Agent本质是信息压缩器。度量熵/压缩比/信息增益，
    量化Agent将高熵输入压缩为低熵输出的能力。
    纯数学计算，无LLM调用。JSON持久化。
    """
    def __init__(self, max_records: int = 5000):
        self._records: list[InferenceRecord] = []
        self._max_records = max_records
        self._load()

    def _load(self) -> None:
        try:
            if _STATE_PATH.exists():
                data = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
                self._records = [InferenceRecord(**r) for r in data.get("records", [])]
        except (json.JSONDecodeError, TypeError, OSError):
            self._records = []

    def _save(self) -> None:
        try:
            _STATE_DIR.mkdir(parents=True, exist_ok=True)
            payload = {"records": [r.to_dict() for r in self._records[-self._max_records:]]}
            _STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass

    def compute_codelength(self, text: str) -> float:
        """文本Shannon码长（熵），单位比特。"""
        return round(shannon_entropy(text), 4)

    def compute_compression_ratio(self, original: str, compressed: str) -> float:
        """压缩比 = len(compressed)/len(original)，越小越好。"""
        if not original: return 0.0
        return round(len(compressed) / len(original), 4)

    def compute_information_gain(self, context: str, new_info: str) -> float:
        """信息增益：IG = H(new) - [H(context+new) - H(context)]，单位比特。"""
        if not new_info: return 0.0
        h_new = shannon_entropy(new_info)
        h_ctx = shannon_entropy(context) if context else 0.0
        return round(max(h_new - (combined_entropy(context, new_info) - h_ctx), 0.0), 4)

    def record_inference(self, session_id: str, input_text: str, output_text: str) -> InferenceRecord:
        """记录一次推理，自动计算全部信息度量并持久化。"""
        rec = InferenceRecord(
            session_id=session_id, input_len=len(input_text), output_len=len(output_text),
            input_entropy=self.compute_codelength(input_text),
            output_entropy=self.compute_codelength(output_text),
            compression_ratio=self.compute_compression_ratio(input_text, output_text),
            info_gain=self.compute_information_gain(input_text, output_text),
        )
        self._records.append(rec)
        if len(self._records) > self._max_records:
            self._records = self._records[-self._max_records:]
        self._save()
        return rec

    def get_session_metrics(self, session_id: str) -> dict[str, Any]:
        """指定会话的聚合度量：推理次数/平均熵/压缩比/信息增益。"""
        recs = [r for r in self._records if r.session_id == session_id]
        if not recs:
            return {"session_id": session_id, "total_inferences": 0}
        avg_in = sum(r.input_entropy for r in recs) / len(recs)
        avg_out = sum(r.output_entropy for r in recs) / len(recs)
        return {
            "session_id": session_id,
            "total_inferences": len(recs),
            "avg_input_entropy": round(avg_in, 4),
            "avg_output_entropy": round(avg_out, 4),
            "avg_compression_ratio": round(sum(r.compression_ratio for r in recs) / len(recs), 4),
            "total_info_gain": round(sum(r.info_gain for r in recs), 4),
            "entropy_reduction": round(avg_in - avg_out, 4),
        }

    def get_global_stats(self) -> dict[str, Any]:
        """全局统计：总推理数/平均压缩比/最佳最差/总信息增益。"""
        if not self._records:
            return {"total_inferences": 0}
        ratios = [r.compression_ratio for r in self._records]
        return {
            "total_inferences": len(self._records),
            "avg_compression_ratio": round(sum(ratios) / len(ratios), 4),
            "best_compression": round(min(ratios), 4),
            "worst_compression": round(max(ratios), 4),
            "total_info_gain": round(sum(r.info_gain for r in self._records), 4),
        }
