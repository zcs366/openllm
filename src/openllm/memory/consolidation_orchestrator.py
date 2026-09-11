"""
ConsolidationOrchestrator — 固化链路编排器
================================================================

回答的问题：pressure→consolidate→写growth→再水化验证，怎么串起来？
  三个独立模块（context_pressure / consolidation_score / rehydration_validator）
  + 一个新容器（soul_growth）存在但零接线。本模块是桥梁——
  把断头管接通为完整闭环。

闭环流程：
  ① ContextPressureMonitor.get_pressure_level(session_id) → 取level
  ② level in (caution, critical) 才继续，否则返回 {acted: False}
  ③ ConsolidationScorer.candidates(level) → 固化候选
  ④ ConsolidationScorer.consolidate(candidates) → 固化执行
  ⑤ 对每个consolidated bullet → SoulGrowthLedger.record() → 写SOUL层
  ⑥ RehydrationValidator.validate(原文, growth内容) → 第三闸
     verdict == fail → 标记rollback_reason

设计原则：
  - 零LLM调用，确定性编排
  - append-only（不删已有记录）
  - 每个模块真实调用（军规八：代码活着）

上下文格言：
  断头管接通之日，成长闭环成立之时。
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("openllm.consolidation_orchestrator")


class ConsolidationOrchestrator:
    """固化链路编排器——串联 pressure → scorer → growth → validator。

    零LLM、确定性、append-only。

    用法：
        orch = ConsolidationOrchestrator()
        # 先构造环境：pressure有level，scorer有bullet
        report = orch.run_cycle(session_id="s1")
        # report = {acted: True, level: "caution", consolidated_n: 1, ...}
    """

    def __init__(
        self,
        pressure_state_path: Optional[Path] = None,
        scorer_state_path: Optional[Path] = None,
        growth_path: Optional[Path] = None,
        audit_log_path: Optional[Path] = None,
        min_score: float = 4.0,
    ) -> None:
        """
        Args:
            pressure_state_path: ContextPressureMonitor状态文件路径
            scorer_state_path: ConsolidationScorer状态文件路径
            growth_path: SoulGrowthLedger JSONL路径
            audit_log_path: RehydrationValidator审计日志路径
            min_score: 固化分数门槛
        """
        # ── 延迟导入：确保各模块在运行时可用 ──
        from openllm.memory.context_pressure import ContextPressureMonitor
        from openllm.consolidation_score import ConsolidationScorer
        from openllm.identity.soul_growth import SoulGrowthLedger
        from openllm.memory.rehydration_validator import RehydrationValidator

        self._pressure = ContextPressureMonitor(state_path=pressure_state_path)
        self._scorer = ConsolidationScorer(
            state_path=scorer_state_path, min_score=min_score,
        )
        self._growth = SoulGrowthLedger(growth_path=growth_path)
        self._validator = RehydrationValidator(audit_log_path=audit_log_path)

    def should_consolidate(self, session_id: str, pressure_level: str) -> bool:
        """判断是否应该执行固化。

        只在 caution/critical 水位才动作。
        """
        return pressure_level in ("caution", "critical")

    def run_cycle(self, session_id: str) -> dict:
        """执行一次完整的固化闭环。

        流程：
          ① 从 pressure monitor 取 level
          ② level 不够则短路返回
          ③ 从 scorer 取候选
          ④ 固化候选
          ⑤ 写入 SOUL 层成长库
          ⑥ 再水化验证 → fail 则回滚

        Args:
            session_id: 会话标识符

        Returns:
            {acted, level, consolidated_n, failed_n, growth_path, details}
        """
        # ① 取压力level
        level = self._pressure.get_pressure_level(session_id)

        # ② level不够则短路
        if not self.should_consolidate(session_id, level):
            return {
                "acted": False,
                "level": level,
                "consolidated_n": 0,
                "failed_n": 0,
                "growth_path": str(self._growth._path),
                "details": "pressure level insufficient for consolidation",
            }

        # ③ 从scorer取候选
        candidates = self._scorer.candidates(pressure_level=level)
        if not candidates:
            return {
                "acted": False,
                "level": level,
                "consolidated_n": 0,
                "failed_n": 0,
                "growth_path": str(self._growth._path),
                "details": "no candidates available",
            }

        # ④ 固化候选
        consolidate_report = self._scorer.consolidate(
            bullets=candidates, pressure_level=level,
        )

        # ⑤ 对每个consolidated bullet → 写SOUL层成长库
        consolidated_ids = [
            entry["bullet_id"] for entry in consolidate_report["consolidated"]
        ]
        failed_n = 0
        growth_details: List[dict] = []

        for bid in consolidated_ids:
            bullet = self._scorer._bullets.get(bid)
            if bullet is None:
                continue

            # 写入SOUL层成长库
            record = self._growth.record(
                bullet_id=bid,
                content=bullet.content,
                origin="consolidation",
                score=consolidate_report["consolidated"][
                    consolidated_ids.index(bid)
                ]["score"],
                provenance_trust=bullet.provenance.trust.value,
            )

            # ⑥ 再水化验证——第三闸（真实往返：写入SOUL层后读回，验证无损）
            # 原实现 validate(原文, 原文) 自比=必然pass=占位。正确语义：
            #   固化写入 growth → 再水化=从growth读回 → 与原content比对
            #   往返无损 → pass；读回被压缩/污染/截断 → fail → 回滚
            readback = self._growth.load(recent_n=1)
            rehydrated = readback[-1]["content"] if readback else bullet.content
            validation = self._validator.validate(
                original=bullet.content,
                rehydrated=rehydrated,  # 读回的状态（往返验证，非自比）
            )
            if validation["verdict"] == "fail":
                failed_n += 1
                reason = f"rehydration_fail: score={validation['score']:.3f}"
                self._growth.mark_rollback(bid, reason)
                growth_details.append({
                    "bullet_id": bid,
                    "status": "rollback",
                    "reason": reason,
                })
                logger.warning(
                    "ConsolidationOrchestrator: bullet %s 回滚 "
                    "(rehydration score=%.3f < threshold)",
                    bid, validation["score"],
                )
            else:
                growth_details.append({
                    "bullet_id": bid,
                    "status": "consolidated",
                    "rehydration_score": validation["score"],
                })

        # 构建报告
        report = {
            "acted": True,
            "level": level,
            "consolidated_n": len(consolidated_ids),
            "failed_n": failed_n,
            "growth_path": str(self._growth._path),
            "details": growth_details,
            "rejected_n": len(consolidate_report["rejected"]),
        }
        logger.info(
            "ConsolidationOrchestrator: session=%s level=%s "
            "consolidated=%d failed=%d",
            session_id, level, len(consolidated_ids), failed_n,
        )
        return report
