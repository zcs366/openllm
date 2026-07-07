"""FeedbackLoop — 六体自监督的统一反馈接口。

核心设计：
- FeedbackRecord: 不可变反馈记录（source→target·score·issues·suggestion）
- FeedbackStore: 跨轮反馈持久化（JSONL·链式hash审计）
- FeedbackLoop: 收集+评分+应用三步闭环

每个体既能评估其他体的输出质量，又能接受反馈并调整行为。
监督者矩阵：IAX→ISN, IAI→IKO, ISA→IOS, IOS→IAI, ISN→IAX, IKO→ISN
"""
import json
import time
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class FeedbackRecord:
    """一条反馈记录——不可变"""
    source_body: str       # 谁发出的反馈 (IAI/ISA/IOS/ISN/IKO)
    target_body: str       # 谁被评估
    round_number: int      # 心跳轮次
    score: float           # 0.0-1.0 质量评分
    issues: list[str]      # 发现的问题
    suggestion: str        # 改进建议
    timestamp: float = field(default_factory=time.time)
    prev_hash: str = ""    # 链式审计
    record_hash: str = ""  # 自身hash

    def compute_hash(self) -> str:
        """计算本条记录的hash"""
        data = f"{self.source_body}:{self.target_body}:{self.round_number}:{self.score}:{self.prev_hash}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


class FeedbackStore:
    """反馈持久化存储（JSONL格式，链式hash审计）"""

    def __init__(self, store_path: str = None):
        self.path = Path(store_path or Path.home() / ".openllm" / "feedback_store.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._last_hash = ""

    def append(self, record: FeedbackRecord) -> None:
        """追加一条反馈记录，维护hash链"""
        r = FeedbackRecord(
            source_body=record.source_body,
            target_body=record.target_body,
            round_number=record.round_number,
            score=record.score,
            issues=record.issues,
            suggestion=record.suggestion,
            timestamp=record.timestamp,
            prev_hash=self._last_hash,
        )
        h = r.compute_hash()
        r = FeedbackRecord(
            source_body=r.source_body, target_body=r.target_body,
            round_number=r.round_number, score=r.score,
            issues=r.issues, suggestion=r.suggestion,
            timestamp=r.timestamp, prev_hash=r.prev_hash, record_hash=h,
        )
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
        self._last_hash = h

    def get_recent(self, n: int = 10) -> list[dict]:
        """获取最近n条反馈"""
        if not self.path.exists():
            return []
        lines = self.path.read_text().strip().split("\n")
        return [json.loads(l) for l in lines[-n:] if l.strip()]

    def get_for_body(self, body_name: str, n: int = 20) -> list[dict]:
        """获取关于某个体的最近反馈"""
        all_records = self.get_recent(100)
        return [r for r in all_records if r["target_body"] == body_name][-n:]

    def verify_chain(self) -> bool:
        """验证hash链完整性"""
        if not self.path.exists():
            return True
        lines = self.path.read_text().strip().split("\n")
        prev = ""
        for line in lines:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("prev_hash", "") != prev:
                return False
            prev = rec.get("record_hash", "")
        return True

    def clear(self) -> None:
        """清空存储（测试用）"""
        if self.path.exists():
            self.path.unlink()
        self._last_hash = ""


class FeedbackLoop:
    """六体自监督的统一反馈闭环"""

    # 核心监督者映射（子产监督矩阵）
    CORE_SUPERVISORS = {
        "IAX": "ISN",   # 韩信监督心跳
        "IAI": "IKO",   # 子贡监督感知
        "ISA": "IOS",   # 子产监督记忆
        "IOS": "IAI",   # 萧何监督决策
        "ISN": "IAX",   # 军师监督执行
        "IKO": "ISN",   # 鲁班监督输出
    }

    def __init__(self, store_path: str = None):
        self.store = FeedbackStore(store_path)
        self._round = 0

    def collect_feedback(self, body_outputs: dict[str, dict]) -> list[FeedbackRecord]:
        """收集本轮各体的相互评估。

        body_outputs: {body_name: last_output_dict}
        """
        self._round += 1
        records = []

        for target, output in body_outputs.items():
            # 核心监督者评估
            supervisor = self.CORE_SUPERVISORS.get(target)
            if supervisor and supervisor in body_outputs:
                score, issues, suggestion = self._evaluate(
                    supervisor, target, output
                )
                rec = FeedbackRecord(
                    source_body=supervisor,
                    target_body=target,
                    round_number=self._round,
                    score=score,
                    issues=issues,
                    suggestion=suggestion,
                )
                self.store.append(rec)
                records.append(rec)

        return records

    def _evaluate(self, evaluator: str, target: str, output: dict) -> tuple[float, list[str], str]:
        """评估一个体的输出质量。

        初版用规则评估，未来可升级为LLM评估。
        """
        score = 0.8  # 默认分数
        issues: list[str] = []
        suggestion = ""

        # 基本健康检查
        if not output:
            score = 0.3
            issues.append("输出为空")
            suggestion = f"{target}产生了空输出，检查内部状态"
        elif len(str(output)) < 10:
            score = 0.5
            issues.append("输出过短")
            suggestion = f"{target}输出信息量不足"

        # 特定体的检查逻辑
        if target == "IAX" and output.get("heartbeat_ok") is False:
            score = 0.2
            issues.append("心跳异常")
            suggestion = "IAX心跳失败，检查调度器"

        if target == "ISA" and output.get("memory_count", 0) == 0:
            score = 0.4
            issues.append("记忆为空")
            suggestion = "ISA无记忆数据，检查写入管道"

        if target == "ISN" and output.get("error_rate", 0) > 0.3:
            score = 0.4
            issues.append(f"工具错误率过高: {output.get('error_rate', 0):.0%}")
            suggestion = "ISN工具调用错误率超标，检查工具注册"

        return score, issues, suggestion

    def apply_feedback(self) -> dict[str, float]:
        """将积累的反馈应用到下一轮行为调整。

        返回每个体的调整因子（score < 0.7的体需要更多自审）。
        """
        adjustments = {}
        for body in ["IAX", "IAI", "ISA", "IOS", "ISN", "IKO"]:
            recent = self.store.get_for_body(body, n=5)
            if recent:
                avg_score = sum(r["score"] for r in recent) / len(recent)
                adjustments[body] = avg_score
            else:
                adjustments[body] = 1.0  # 无反馈默认正常
        return adjustments

    def get_health_report(self) -> dict:
        """生成六体健康报告"""
        report = {"round": self._round, "bodies": {}, "chain_valid": self.store.verify_chain()}
        for body in ["IAX", "IAI", "ISA", "IOS", "ISN", "IKO"]:
            recent = self.store.get_for_body(body, n=5)
            if recent:
                avg = sum(r["score"] for r in recent) / len(recent)
                issues_count = sum(len(r["issues"]) for r in recent)
                report["bodies"][body] = {
                    "avg_score": round(avg, 2),
                    "issues_total": issues_count,
                    "status": "healthy" if avg >= 0.7 else "degraded" if avg >= 0.5 else "critical",
                }
            else:
                report["bodies"][body] = {"avg_score": 1.0, "issues_total": 0, "status": "no_data"}
        return report
