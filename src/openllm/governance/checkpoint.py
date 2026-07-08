"""
Constitutional Checkpoint — P3宪法检查点监控
============================================

12个月后评估：宪法是否已被任何外部项目引用？
本模块提供检查点机制——定期扫描外部引用。
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class CheckpointResult:
    """检查点结果。"""
    checkpoint_id: str
    timestamp: float
    month: int                          # 第几个月
    external_references: int            # 外部引用数
    precedent_count: int                # 积累判例数
    test_pass_rate: float               # 测试通过率
    recommendation: str                 # 建议
    signature: str = ""

    def compute_hash(self) -> str:
        content = json.dumps({
            "checkpoint_id": self.checkpoint_id,
            "month": self.month,
            "external_references": self.external_references,
            "precedent_count": self.precedent_count,
            "test_pass_rate": self.test_pass_rate,
            "recommendation": self.recommendation,
        }, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content.encode()).hexdigest()


class ConstitutionalCheckpoint:
    """宪法检查点引擎。

    用法：
        cp = ConstitutionalCheckpoint()
        result = cp.run_checkpoint(
            month=1,
            external_references=0,
            precedent_count=30,
            test_pass_rate=0.95,
        )
        if result.recommendation == "continue":
            # 继续积累判例
        elif result.recommendation == "pivot":
            # 需要重新评估方向
    """

    def __init__(self):
        self._checkpoints: list[CheckpointResult] = []

    def run_checkpoint(
        self,
        month: int,
        external_references: int,
        precedent_count: int,
        test_pass_rate: float,
    ) -> CheckpointResult:
        """运行检查点评估。"""
        # 根据指标决定建议
        if external_references >= 5 and test_pass_rate >= 0.9:
            recommendation = "continue"  # 继续积累判例
        elif external_references == 0 and month >= 6:
            recommendation = "pivot"     # 6个月零引用→需要重新评估
        elif precedent_count >= 100 and test_pass_rate >= 0.85:
            recommendation = "expand"    # 判例充足+测试通过→扩展到更多场景
        else:
            recommendation = "monitor"   # 继续监控

        result = CheckpointResult(
            checkpoint_id=f"cp-month{month}-{int(time.time()*1000)}",
            timestamp=time.time(),
            month=month,
            external_references=external_references,
            precedent_count=precedent_count,
            test_pass_rate=test_pass_rate,
            recommendation=recommendation,
        )

        signature = result.compute_hash()
        result = CheckpointResult(
            checkpoint_id=result.checkpoint_id,
            timestamp=result.timestamp,
            month=result.month,
            external_references=result.external_references,
            precedent_count=result.precedent_count,
            test_pass_rate=result.test_pass_rate,
            recommendation=result.recommendation,
            signature=signature,
        )

        self._checkpoints.append(result)
        return result

    def get_checkpoints(self) -> list[CheckpointResult]:
        return list(self._checkpoints)

    def get_trajectory(self) -> dict:
        """获取检查点轨迹——判断方向是否正确。"""
        if not self._checkpoints:
            return {"trend": "no_data"}

        refs = [c.external_references for c in self._checkpoints]
        precedents = [c.precedent_count for c in self._checkpoints]

        # 简单趋势判断
        if len(refs) >= 2:
            ref_trend = "growing" if refs[-1] > refs[0] else "flat"
        else:
            ref_trend = "insufficient_data"

        return {
            "checkpoints": len(self._checkpoints),
            "ref_trend": ref_trend,
            "latest_recommendation": self._checkpoints[-1].recommendation,
            "total_precedents": precedents[-1] if precedents else 0,
        }
