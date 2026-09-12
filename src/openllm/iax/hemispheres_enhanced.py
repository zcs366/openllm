"""
增强版左右脑（Hemispheres）— openLLM 自我对弈架构 v2.0

在原有hemispheres.py基础上增强：
1. 多种对弈策略（保守/激进/平衡/IO-S驱动）
2. IO-S集成：策略引擎驱动仲裁
3. 更智能的证据评估
4. 对弈历史分析和学习

依赖：
- hemispheres.py（基础架构）
- IO-S（可选，策略引擎）
- ISA（可选，记忆系统）
"""

import json
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Dict, List
import logging

logger = logging.getLogger("openllm.hemispheres_enhanced")

# ── 从原版导入 ──
from .hemispheres import (
    HemisphereSide, HemisphereState, ArbiterVerdict,
    Heartbeat, Checkpoint, ArbiterRecord,
    Hemisphere, HemispherePair,
    HEMISPHERE_DIR, HEARTBEAT_INTERVAL, HEARTBEAT_TIMEOUT,
)


# ── 新增枚举 ──

class ArbitrationStrategy(Enum):
    """仲裁策略枚举"""
    CONSERVATIVE = "conservative"    # 保守：无证据不行动
    BALANCED = "balanced"            # 平衡：权衡利弊
    AGGRESSIVE = "aggressive"        # 激进：有想法就干
    IOS_DRIVEN = "ios_driven"        # IO-S驱动：策略引擎决策
    EVIDENCE_BASED = "evidence_based" # 证据优先：谁证据多听谁


# ── 新增数据结构 ──

@dataclass
class ArbitrationContext:
    """仲裁上下文"""
    task_id: str                     # 任务ID
    task_description: str            # 任务描述
    risk_level: str = "medium"       # 风险等级
    time_pressure: str = "normal"    # 时间压力
    user_preference: str = ""        # 用户偏好
    historical_data: List[Dict] = field(default_factory=list)  # 历史数据
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_description": self.task_description,
            "risk_level": self.risk_level,
            "time_pressure": self.time_pressure,
            "user_preference": self.user_preference,
            "historical_data_count": len(self.historical_data),
        }


@dataclass
class ArbitrationResult:
    """仲裁结果"""
    verdict: str                     # 裁决结果
    resolution: str                  # 决议内容
    confidence: float                # 置信度 (0.0-1.0)
    reasoning: str                   # 推理过程
    evidence_summary: Dict[str, Any] = field(default_factory=dict)
    strategy_used: str = ""          # 使用的策略
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "resolution": self.resolution,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "evidence_summary": self.evidence_summary,
            "strategy_used": self.strategy_used,
        }


# ── 增强仲裁器 ──

class EnhancedArbiter:
    """
    增强版仲裁器。
    
    支持多种仲裁策略，可与IO-S集成。
    """
    
    def __init__(self, default_strategy: ArbitrationStrategy = ArbitrationStrategy.BALANCED):
        """
        初始化增强仲裁器。
        
        Args:
            default_strategy: 默认仲裁策略
        """
        self.default_strategy = default_strategy
        self._ios_callback: Optional[Callable] = None
        self._history: List[ArbitrationResult] = []
        self._strategy_performance: Dict[str, Dict[str, int]] = {}
        
    def set_ios_callback(self, callback: Callable):
        """设置IO-S回调函数"""
        self._ios_callback = callback
        
    def arbitrate(
        self,
        left_proposal: str,
        right_critique: str,
        evidence_left: List[str] = None,
        evidence_right: List[str] = None,
        context: ArbitrationContext = None,
        strategy: ArbitrationStrategy = None,
    ) -> ArbitrationResult:
        """
        执行仲裁。
        
        Args:
            left_proposal: 左脑提案
            right_critique: 右脑批评
            evidence_left: 左脑证据
            evidence_right: 右脑证据
            context: 仲裁上下文
            strategy: 指定策略（覆盖默认）
            
        Returns:
            ArbitrationResult: 仲裁结果
        """
        strategy = strategy or self.default_strategy
        evidence_left = evidence_left or []
        evidence_right = evidence_right or []
        
        # 根据策略选择仲裁方法
        if strategy == ArbitrationStrategy.CONSERVATIVE:
            result = self._arbitrate_conservative(
                left_proposal, right_critique, evidence_left, evidence_right, context
            )
        elif strategy == ArbitrationStrategy.AGGRESSIVE:
            result = self._arbitrate_aggressive(
                left_proposal, right_critique, evidence_left, evidence_right, context
            )
        elif strategy == ArbitrationStrategy.EVIDENCE_BASED:
            result = self._arbitrate_evidence_based(
                left_proposal, right_critique, evidence_left, evidence_right, context
            )
        elif strategy == ArbitrationStrategy.IOS_DRIVEN:
            result = self._arbitrate_ios_driven(
                left_proposal, right_critique, evidence_left, evidence_right, context
            )
        else:  # BALANCED
            result = self._arbitrate_balanced(
                left_proposal, right_critique, evidence_left, evidence_right, context
            )
        
        # 记录历史
        self._history.append(result)
        self._update_strategy_performance(strategy.value, result.verdict)
        
        return result
    
    def _arbitrate_conservative(
        self,
        left: str,
        right: str,
        ev_left: List[str],
        ev_right: List[str],
        context: ArbitrationContext = None,
    ) -> ArbitrationResult:
        """保守策略：无证据不行动"""
        if not right.strip() or right == left:
            return ArbitrationResult(
                verdict=ArbiterVerdict.LEFT_WINS.value,
                resolution=f"采纳左脑方案: {left[:100]}",
                confidence=0.8,
                reasoning="右脑无反对意见",
                evidence_summary={"left": len(ev_left), "right": len(ev_right)},
                strategy_used="conservative",
            )
        
        if not ev_right and ev_left:
            return ArbitrationResult(
                verdict=ArbiterVerdict.LEFT_WINS.value,
                resolution=f"左脑有证据支持: {left[:100]}",
                confidence=0.9,
                reasoning="左脑提供了证据，右脑没有",
                evidence_summary={"left": len(ev_left), "right": len(ev_right)},
                strategy_used="conservative",
            )
        
        if ev_right and not ev_left:
            return ArbitrationResult(
                verdict=ArbiterVerdict.RIGHT_WINS.value,
                resolution=f"右脑有证据反驳: {right[:100]}",
                confidence=0.9,
                reasoning="右脑提供了证据，左脑没有",
                evidence_summary={"left": len(ev_left), "right": len(ev_right)},
                strategy_used="conservative",
            )
        
        # 双方都有证据或都没证据 → 走保守路线
        return ArbitrationResult(
            verdict=ArbiterVerdict.INCONCLUSIVE.value,
            resolution="双方均无充分证据。走保守路线：暂不执行，收集更多数据后再议",
            confidence=0.5,
            reasoning="保守策略：证据不足时选择等待",
            evidence_summary={"left": len(ev_left), "right": len(ev_right)},
            strategy_used="conservative",
        )
    
    def _arbitrate_aggressive(
        self,
        left: str,
        right: str,
        ev_left: List[str],
        ev_right: List[str],
        context: ArbitrationContext = None,
    ) -> ArbitrationResult:
        """激进策略：有想法就干"""
        if not right.strip() or right == left:
            return ArbitrationResult(
                verdict=ArbiterVerdict.LEFT_WINS.value,
                resolution=f"采纳左脑方案: {left[:100]}",
                confidence=0.7,
                reasoning="右脑无反对意见，激进执行",
                evidence_summary={"left": len(ev_left), "right": len(ev_right)},
                strategy_used="aggressive",
            )
        
        # 即使有反对，也倾向于执行
        if ev_left:
            return ArbitrationResult(
                verdict=ArbiterVerdict.LEFT_WINS.value,
                resolution=f"左脑有方案且有证据，执行: {left[:100]}",
                confidence=0.6,
                reasoning="激进策略：有方案就执行，风险后置",
                evidence_summary={"left": len(ev_left), "right": len(ev_right)},
                strategy_used="aggressive",
            )
        
        # 都没证据 → 也执行左脑方案
        return ArbitrationResult(
            verdict=ArbiterVerdict.LEFT_WINS.value,
            resolution=f"激进执行左脑方案: {left[:100]}",
            confidence=0.5,
            reasoning="激进策略：无证据也执行，快速迭代",
            evidence_summary={"left": len(ev_left), "right": len(ev_right)},
            strategy_used="aggressive",
        )
    
    def _arbitrate_balanced(
        self,
        left: str,
        right: str,
        ev_left: List[str],
        ev_right: List[str],
        context: ArbitrationContext = None,
    ) -> ArbitrationResult:
        """平衡策略：权衡利弊"""
        if not right.strip() or right == left:
            return ArbitrationResult(
                verdict=ArbiterVerdict.LEFT_WINS.value,
                resolution=f"采纳左脑方案: {left[:100]}",
                confidence=0.8,
                reasoning="右脑无反对意见",
                evidence_summary={"left": len(ev_left), "right": len(ev_right)},
                strategy_used="balanced",
            )
        
        # 计算证据权重
        left_weight = len(ev_left) * 1.0
        right_weight = len(ev_right) * 1.2  # 右脑批评权重略高
        
        if left_weight > right_weight:
            return ArbitrationResult(
                verdict=ArbiterVerdict.LEFT_WINS.value,
                resolution=f"左脑方案证据更充分: {left[:100]}",
                confidence=0.7,
                reasoning=f"平衡策略：左脑权重{left_weight:.1f} vs 右脑权重{right_weight:.1f}",
                evidence_summary={"left": len(ev_left), "right": len(ev_right)},
                strategy_used="balanced",
            )
        
        if right_weight > left_weight:
            return ArbitrationResult(
                verdict=ArbiterVerdict.RIGHT_WINS.value,
                resolution=f"右脑批评更有说服力: {right[:100]}",
                confidence=0.7,
                reasoning=f"平衡策略：右脑权重{right_weight:.1f} vs 左脑权重{left_weight:.1f}",
                evidence_summary={"left": len(ev_left), "right": len(ev_right)},
                strategy_used="balanced",
            )
        
        # 权重相等 → 折中
        return ArbitrationResult(
            verdict=ArbiterVerdict.COMPROMISE.value,
            resolution=f"双方势均力敌，折中方案：\n  左: {left[:100]}\n  右: {right[:100]}",
            confidence=0.6,
            reasoning="平衡策略：权重相等，折中处理",
            evidence_summary={"left": len(ev_left), "right": len(ev_right)},
            strategy_used="balanced",
        )
    
    def _arbitrate_evidence_based(
        self,
        left: str,
        right: str,
        ev_left: List[str],
        ev_right: List[str],
        context: ArbitrationContext = None,
    ) -> ArbitrationResult:
        """证据优先策略：谁证据多听谁"""
        if not right.strip() or right == left:
            return ArbitrationResult(
                verdict=ArbiterVerdict.LEFT_WINS.value,
                resolution=f"采纳左脑方案: {left[:100]}",
                confidence=0.8,
                reasoning="右脑无反对意见",
                evidence_summary={"left": len(ev_left), "right": len(ev_right)},
                strategy_used="evidence_based",
            )
        
        # 纯粹基于证据数量
        if len(ev_left) > len(ev_right):
            return ArbitrationResult(
                verdict=ArbiterVerdict.LEFT_WINS.value,
                resolution=f"左脑证据更多({len(ev_left)} vs {len(ev_right)}): {left[:100]}",
                confidence=0.8,
                reasoning="证据优先策略：证据数量决定胜负",
                evidence_summary={"left": len(ev_left), "right": len(ev_right)},
                strategy_used="evidence_based",
            )
        
        if len(ev_right) > len(ev_left):
            return ArbitrationResult(
                verdict=ArbiterVerdict.RIGHT_WINS.value,
                resolution=f"右脑证据更多({len(ev_right)} vs {len(ev_left)}): {right[:100]}",
                confidence=0.8,
                reasoning="证据优先策略：证据数量决定胜负",
                evidence_summary={"left": len(ev_left), "right": len(ev_right)},
                strategy_used="evidence_based",
            )
        
        # 证据数量相等
        return ArbitrationResult(
            verdict=ArbiterVerdict.COMPROMISE.value,
            resolution=f"双方证据数量相等，折中处理",
            confidence=0.5,
            reasoning="证据优先策略：证据数量相等",
            evidence_summary={"left": len(ev_left), "right": len(ev_right)},
            strategy_used="evidence_based",
        )
    
    def _arbitrate_ios_driven(
        self,
        left: str,
        right: str,
        ev_left: List[str],
        ev_right: List[str],
        context: ArbitrationContext = None,
    ) -> ArbitrationResult:
        """IO-S驱动策略：策略引擎决策"""
        # 如果有IO-S回调，使用IO-S决策
        if self._ios_callback and context:
            try:
                ios_result = self._ios_callback(
                    left_proposal=left,
                    right_critique=right,
                    evidence_left=ev_left,
                    evidence_right=ev_right,
                    context=context.to_dict(),
                )
                
                if ios_result and isinstance(ios_result, dict):
                    return ArbitrationResult(
                        verdict=ios_result.get("verdict", ArbiterVerdict.INCONCLUSIVE.value),
                        resolution=ios_result.get("resolution", "IO-S决策"),
                        confidence=ios_result.get("confidence", 0.7),
                        reasoning=ios_result.get("reasoning", "IO-S策略引擎驱动"),
                        evidence_summary={"left": len(ev_left), "right": len(ev_right)},
                        strategy_used="ios_driven",
                    )
            except Exception as e:
                logger.warning(f"IO-S回调失败，回退到平衡策略: {e}")
        
        # 回退到平衡策略
        return self._arbitrate_balanced(left, right, ev_left, ev_right, context)
    
    def _update_strategy_performance(self, strategy: str, verdict: str):
        """更新策略性能统计"""
        if strategy not in self._strategy_performance:
            self._strategy_performance[strategy] = {}
        
        self._strategy_performance[strategy][verdict] = (
            self._strategy_performance[strategy].get(verdict, 0) + 1
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取策略性能统计"""
        return {
            "total_arbitrations": len(self._history),
            "strategy_performance": self._strategy_performance,
            "average_confidence": (
                sum(r.confidence for r in self._history) / len(self._history)
                if self._history else 0.0
            ),
        }


# ── 增强左右脑对 ──

class EnhancedHemispherePair(HemispherePair):
    """
    增强版左右脑对。
    
    继承原版HemispherePair，增加：
    1. 多策略仲裁
    2. IO-S集成
    3. 对弈历史分析
    """
    
    def __init__(
        self,
        left_name: str = "左脑",
        right_name: str = "右脑",
        default_strategy: ArbitrationStrategy = ArbitrationStrategy.BALANCED,
    ):
        """
        初始化增强版左右脑对。
        
        Args:
            left_name: 左脑名称
            right_name: 右脑名称
            default_strategy: 默认仲裁策略
        """
        # 创建增强仲裁器
        self.enhanced_arbiter = EnhancedArbiter(default_strategy)
        
        # 调用父类初始化，传入自定义仲裁器
        super().__init__(
            left_name=left_name,
            right_name=right_name,
            arbiter=self._enhanced_arbitrate,
        )
        
        self._ios_callback: Optional[Callable] = None
        
    def set_ios_callback(self, callback: Callable):
        """设置IO-S回调函数"""
        self._ios_callback = callback
        self.enhanced_arbiter.set_ios_callback(callback)
        
    def _enhanced_arbitrate(
        self,
        left: str,
        right: str,
        ev_left: List[str],
        ev_right: List[str],
    ) -> tuple:
        """
        增强版仲裁方法。
        
        返回：(verdict, resolution) 格式，兼容原版接口
        """
        # 创建仲裁上下文
        context = ArbitrationContext(
            task_id=uuid.uuid4().hex[:8],
            task_description=left[:200],
        )
        
        # 执行增强仲裁
        result = self.enhanced_arbiter.arbitrate(
            left_proposal=left,
            right_critique=right,
            evidence_left=ev_left,
            evidence_right=ev_right,
            context=context,
        )
        
        # 转换为原版格式
        verdict = ArbiterVerdict(result.verdict)
        resolution = result.resolution
        
        return verdict, resolution
    
    def arbitrate_with_strategy(
        self,
        left_proposal: str,
        right_critique: str,
        evidence_left: List[str] = None,
        evidence_right: List[str] = None,
        context: ArbitrationContext = None,
        strategy: ArbitrationStrategy = None,
    ) -> ArbitrationResult:
        """
        使用指定策略执行仲裁。
        
        Args:
            left_proposal: 左脑提案
            right_critique: 右脑批评
            evidence_left: 左脑证据
            evidence_right: 右脑证据
            context: 仲裁上下文
            strategy: 指定策略
            
        Returns:
            ArbitrationResult: 仲裁结果
        """
        return self.enhanced_arbiter.arbitrate(
            left_proposal=left_proposal,
            right_critique=right_critique,
            evidence_left=evidence_left,
            evidence_right=evidence_right,
            context=context,
            strategy=strategy,
        )
    
    def get_arbitration_stats(self) -> Dict[str, Any]:
        """获取仲裁统计信息"""
        return self.enhanced_arbiter.get_performance_stats()


# ── 便捷函数 ──

def create_enhanced_hemisphere_pair(
    left_name: str = "左脑",
    right_name: str = "右脑",
    default_strategy: ArbitrationStrategy = ArbitrationStrategy.BALANCED,
) -> EnhancedHemispherePair:
    """
    创建增强版左右脑对。
    
    Args:
        left_name: 左脑名称
        right_name: 右脑名称
        default_strategy: 默认仲裁策略
        
    Returns:
        EnhancedHemispherePair: 增强版左右脑对实例
    """
    return EnhancedHemispherePair(
        left_name=left_name,
        right_name=right_name,
        default_strategy=default_strategy,
    )


def quick_arbitrate(
    left_proposal: str,
    right_critique: str,
    evidence_left: List[str] = None,
    evidence_right: List[str] = None,
    strategy: ArbitrationStrategy = ArbitrationStrategy.BALANCED,
) -> ArbitrationResult:
    """
    快速仲裁函数。
    
    Args:
        left_proposal: 左脑提案
        right_critique: 右脑批评
        evidence_left: 左脑证据
        evidence_right: 右脑证据
        strategy: 仲裁策略
        
    Returns:
        ArbitrationResult: 仲裁结果
    """
    arbiter = EnhancedArbiter(strategy)
    return arbiter.arbitrate(
        left_proposal=left_proposal,
        right_critique=right_critique,
        evidence_left=evidence_left,
        evidence_right=evidence_right,
    )