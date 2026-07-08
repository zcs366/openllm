"""
技能自进化模块 — 将WorldEvolver集成到openLLM的ISN系统

功能：
1. 技能预测与观察记录
2. 失败模式识别与规则提炼
3. 技能置信度管理
4. 自进化世界模型

依赖：
- ~/.hermes/jiak/world_model_evolver.py
- jiak卡片系统

用法：
    from openllm.tools.skill_evolution import SkillEvolution
    
    evolution = SkillEvolution()
    
    # 记录预测
    prediction = evolution.record_prediction(
        card_id="skill-xxx",
        prediction="这个技能会成功",
        confidence=0.8,
    )
    
    # 记录观察
    evolution.record_observation(
        card_id="skill-xxx",
        actual="技能执行成功",
        prediction_ref=prediction["prediction_id"],
    )
    
    # 分析失败模式
    patterns = evolution.analyze_failure_patterns("skill-xxx")
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import json

logger = logging.getLogger("openllm.skill_evolution")

# ── 路径配置 ──
JIAK_HOME = Path.home() / ".hermes" / "jiak"
WORLD_MODEL_EVOLVER_PATH = JIAK_HOME / "world_model_evolver.py"


class SkillEvolution:
    """
    技能自进化类。
    
    将WorldEvolver集成到openLLM的ISN系统中。
    懒加载：只在第一次调用时导入world_model_evolver模块。
    """
    
    def __init__(self, auto_load: bool = True):
        """
        初始化技能自进化模块。
        
        Args:
            auto_load: 是否自动加载world_model_evolver模块（默认True）
        """
        self._loaded = False
        self._record_prediction = None
        self._record_observation = None
        self._cluster_failures = None
        self._update_card_confidence = None
        self._get_card_confidence = None
        self._record_mismatch = None
        self._extract_rules_from_mismatches = None
        
        if auto_load:
            self._load_world_model_evolver()
    
    def _load_world_model_evolver(self) -> bool:
        """
        懒加载world_model_evolver模块。
        
        Returns:
            bool: 是否加载成功
        """
        if self._loaded:
            return True
        
        if not WORLD_MODEL_EVOLVER_PATH.exists():
            logger.warning(f"world_model_evolver.py不存在: {WORLD_MODEL_EVOLVER_PATH}")
            return False
        
        try:
            import importlib.util as _ilu
            _spec = _ilu.spec_from_file_location("world_model_evolver", JIAK_HOME / "world_model_evolver.py")
            if _spec and _spec.loader:
                _mod = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                record_prediction = _mod.record_prediction
                record_observation = _mod.record_observation
                cluster_failures = _mod.cluster_failures
                set_confidence = _mod.set_confidence
                get_card_confidence = _mod.get_card_confidence
                get_mismatches = _mod.get_mismatches
                refine_card = _mod.refine_card
            
            self._record_prediction = record_prediction
            self._record_observation = record_observation
            self._cluster_failures = cluster_failures
            self._update_card_confidence = set_confidence
            self._get_card_confidence = get_card_confidence
            self._record_mismatch = None  # 暂时不支持
            self._extract_rules_from_mismatches = refine_card
            
            self._loaded = True
            logger.info("✅ world_model_evolver 已加载")
            return True
            
        except Exception as e:
            logger.warning(f"world_model_evolver 加载失败: {e}")
            return False
    
    def record_prediction(
        self,
        card_id: str,
        prediction: str,
        confidence: float = 0.7,
        context: str = "",
    ) -> Dict[str, Any]:
        """
        记录一条预测（任务执行前调用）。
        
        Args:
            card_id: 卡片ID
            prediction: 预测内容
            confidence: 置信度 (0.0-1.0)
            context: 上下文信息
            
        Returns:
            Dict: 预测记录
        """
        if not self._loaded:
            logger.warning("world_model_evolver未加载，返回空字典")
            return {}
        
        try:
            return self._record_prediction(card_id, prediction, confidence, context)
        except Exception as e:
            logger.error(f"记录预测失败: {e}")
            return {}
    
    def record_observation(
        self,
        card_id: str,
        actual: str,
        prediction_ref: str = "",
        success: bool = True,
    ) -> Dict[str, Any]:
        """
        记录观察结果（任务执行后调用）。
        
        Args:
            card_id: 卡片ID
            actual: 实际结果
            prediction_ref: 预测记录ID
            success: 是否成功
            
        Returns:
            Dict: 观察记录
        """
        if not self._loaded:
            logger.warning("world_model_evolver未加载，返回空字典")
            return {}
        
        try:
            return self._record_observation(card_id, actual, prediction_ref, success)
        except Exception as e:
            logger.error(f"记录观察失败: {e}")
            return {}
    
    def analyze_failure_patterns(self, card_id: str = None) -> List[Dict[str, Any]]:
        """
        分析失败模式。
        
        Args:
            card_id: 卡片ID（可选，不指定则分析所有）
            
        Returns:
            List[Dict]: 失败模式列表
        """
        if not self._loaded:
            logger.warning("world_model_evolver未加载，返回空列表")
            return []
        
        try:
            return self._cluster_failures(card_id)
        except Exception as e:
            logger.error(f"分析失败模式失败: {e}")
            return []
    
    def update_confidence(
        self,
        card_id: str,
        confidence: float,
        reason: str = "",
    ) -> bool:
        """
        更新卡片置信度。
        
        Args:
            card_id: 卡片ID
            confidence: 新置信度 (0.0-1.0)
            reason: 更新原因
            
        Returns:
            bool: 是否更新成功
        """
        if not self._loaded:
            logger.warning("world_model_evolver未加载，返回False")
            return False
        
        try:
            self._update_card_confidence(card_id, confidence, reason)
            return True
        except Exception as e:
            logger.error(f"更新置信度失败: {e}")
            return False
    
    def get_confidence(self, card_id: str) -> float:
        """
        获取卡片置信度。
        
        Args:
            card_id: 卡片ID
            
        Returns:
            float: 置信度 (0.0-1.0)
        """
        if not self._loaded:
            logger.warning("world_model_evolver未加载，返回默认值")
            return 0.7
        
        try:
            return self._get_card_confidence(card_id)
        except Exception as e:
            logger.error(f"获取置信度失败: {e}")
            return 0.7
    
    def record_mismatch(
        self,
        card_id: str,
        prediction: str,
        actual: str,
        severity: str = "medium",
        context: str = "",
    ) -> Dict[str, Any]:
        """
        记录预测与实际的不匹配。
        
        注意：此功能暂未实现，返回空字典。
        
        Args:
            card_id: 卡片ID
            prediction: 预测内容
            actual: 实际结果
            severity: 严重程度 (low/medium/high)
            context: 上下文信息
            
        Returns:
            Dict: 不匹配记录（暂为空）
        """
        logger.warning("record_mismatch功能暂未实现")
        return {}
    
    def extract_rules(self, card_id: str = None) -> List[Dict[str, Any]]:
        """
        从不匹配记录中提取规则。
        
        Args:
            card_id: 卡片ID（可选，不指定则提取所有）
            
        Returns:
            List[Dict]: 提取的规则列表
        """
        if not self._loaded:
            logger.warning("world_model_evolver未加载，返回空列表")
            return []
        
        try:
            # 使用refine_card函数来提取规则
            if card_id:
                result = self._extract_rules_from_mismatches(card_id)
                return [result] if result else []
            else:
                # 如果没有指定card_id，返回空列表
                return []
        except Exception as e:
            logger.error(f"提取规则失败: {e}")
            return []
    
    def is_loaded(self) -> bool:
        """
        检查world_model_evolver是否已加载。
        
        Returns:
            bool: 是否已加载
        """
        return self._loaded
    
    def get_evolution_status(self, card_id: str = None) -> Dict[str, Any]:
        """
        获取进化状态。
        
        Args:
            card_id: 卡片ID（可选）
            
        Returns:
            Dict: 进化状态信息
        """
        if not self._loaded:
            return {"loaded": False, "error": "world_model_evolver未加载"}
        
        try:
            confidence = self.get_confidence(card_id) if card_id else None
            patterns = self.analyze_failure_patterns(card_id)
            rules = self.extract_rules(card_id)
            
            return {
                "loaded": True,
                "card_id": card_id,
                "confidence": confidence,
                "failure_patterns_count": len(patterns),
                "extracted_rules_count": len(rules),
                "status": "active",
            }
        except Exception as e:
            return {"loaded": True, "error": str(e)}


# ── 便捷函数 ──

def create_skill_evolution(auto_load: bool = True) -> SkillEvolution:
    """
    创建SkillEvolution实例。
    
    Args:
        auto_load: 是否自动加载world_model_evolver模块
        
    Returns:
        SkillEvolution: 实例
    """
    return SkillEvolution(auto_load=auto_load)


def quick_predict(
    card_id: str,
    prediction: str,
    confidence: float = 0.7,
) -> Dict[str, Any]:
    """
    便捷函数：快速记录预测。
    
    Args:
        card_id: 卡片ID
        prediction: 预测内容
        confidence: 置信度
        
    Returns:
        Dict: 预测记录
    """
    evolution = create_skill_evolution()
    return evolution.record_prediction(card_id, prediction, confidence)


def quick_observe(
    card_id: str,
    actual: str,
    prediction_ref: str = "",
    success: bool = True,
) -> Dict[str, Any]:
    """
    便捷函数：快速记录观察。
    
    Args:
        card_id: 卡片ID
        actual: 实际结果
        prediction_ref: 预测记录ID
        success: 是否成功
        
    Returns:
        Dict: 观察记录
    """
    evolution = create_skill_evolution()
    return evolution.record_observation(card_id, actual, prediction_ref, success)