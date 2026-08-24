"""
PredictionLayers — 多层预测框架
=================================

每个预测层是一个predict函数，对记忆按该维度的相关性打分。
组合预测 = 多层加权求和。

设计原则：
  - 每个预测层实现同一个接口
  - 增加新层只需要：写predict函数 + 注册 + 调权重
  - 不改MemoryBus，不改CausalMemoryStore

Usage:
    layers = PredictionLayers()
    ranked = layers.combined_predict("搜索VISTA论文", memories)
    # ranked = [(score, memory), ...] 按综合分数降序
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Protocol
import math
import time
import logging

logger = logging.getLogger("openllm.prediction_layers")


# ── 接口 ──

class PredictionLayer(Protocol):
    """预测层协议——每个预测层实现predict方法"""
    name: str
    weight: float
    
    def predict(self, query: str, memory) -> float:
        """返回0-1的相关性分数"""
        ...


# ── 具体实现 ──

class TokenPredictor:
    """token预测层——文本相似度"""
    name = "token"
    weight = 0.30
    
    def predict(self, query: str, memory) -> float:
        content = getattr(memory, "content", "") or getattr(memory, "lesson", "") or ""
        if not content or not query:
            return 0.0
        # 简单关键词匹配（生产环境用embedding）
        q_words = set(query.lower().split())
        c_words = set(content.lower().split())
        overlap = q_words & c_words
        return min(1.0, len(overlap) / max(len(q_words), 1))


class CausalPredictor:
    """因果预测层——因果记忆的lesson与query的相关性"""
    name = "causal"
    weight = 0.20
    
    def predict(self, query: str, memory) -> float:
        lesson = getattr(memory, "lesson", "") or ""
        if not lesson:
            return 0.0
        q_words = set(query.lower().split())
        l_words = set(lesson.lower().split())
        overlap = q_words & l_words
        return min(1.0, len(overlap) / max(len(q_words), 1))


class EmotionPredictor:
    """情感预测层——温度越高越重要"""
    name = "emotion"
    weight = 0.15
    
    def predict(self, query: str, memory) -> float:
        temp = getattr(memory, "temperature", None)
        if temp is None:
            # 从其他属性推断
            confidence = getattr(memory, "prediction_confidence", 0.5)
            temp = confidence
        return max(0.0, min(1.0, temp))


class TemporalPredictor:
    """时间预测层——越近的记忆越相关"""
    name = "temporal"
    weight = 0.10
    
    def predict(self, query: str, memory) -> float:
        created = getattr(memory, "created_at", 0)
        if not created:
            return 0.5
        age_hours = (time.time() - created) / 3600
        # 指数衰减：24小时内高相关，7天后低相关
        return math.exp(-age_hours / 168)  # 168小时=7天


class PreferencePredictor:
    """偏好预测层——匹配用户偏好"""
    name = "preference"
    weight = 0.15
    
    def __init__(self):
        self.user_keywords = set()  # 从用户画像加载
    
    def predict(self, query: str, memory) -> float:
        if not self.user_keywords:
            return 0.5  # 无偏好数据时给中性分
        content = getattr(memory, "content", "") or getattr(memory, "lesson", "") or ""
        matches = sum(1 for kw in self.user_keywords if kw in content.lower())
        return min(1.0, matches / max(len(self.user_keywords), 1))


class SocialPredictor:
    """社会预测层——jage共享记忆的相关性"""
    name = "social"
    weight = 0.10
    
    def predict(self, query: str, memory) -> float:
        source = getattr(memory, "source", "") or ""
        tags = getattr(memory, "tags", []) or []
        # 社会来源（jage/shared）权重更高
        if "social" in source or "jage" in source or "shared" in str(tags):
            return 0.8
        return 0.3


# ── 组合预测器 ──

@dataclass
class ScoredMemory:
    """带分数的记忆"""
    memory: object
    score: float
    layer_scores: Dict[str, float] = field(default_factory=dict)


class PredictionLayers:
    """
    多层预测框架。
    
    每个预测层是一个predict函数。
    组合预测 = 多层加权求和。
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.layers: List[PredictionLayer] = [
            TokenPredictor(),
            CausalPredictor(),
            EmotionPredictor(),
            TemporalPredictor(),
            PreferencePredictor(),
            SocialPredictor(),
        ]
        
        # 自定义权重（覆盖默认）
        if weights:
            for layer in self.layers:
                if layer.name in weights:
                    layer.weight = weights[layer.name]
        
        # 归一化权重
        total_weight = sum(l.weight for l in self.layers)
        if total_weight > 0:
            for l in self.layers:
                l.weight /= total_weight
    
    def combined_predict(self, query: str, memories: list, top_k: int = 10) -> List[ScoredMemory]:
        """
        多层组合预测。
        
        Args:
            query: 查询文本
            memories: 记忆列表
            top_k: 返回前k个
        
        Returns:
            按综合分数降序排列的ScoredMemory列表
        """
        scored = []
        for mem in memories:
            layer_scores = {}
            total = 0.0
            for layer in self.layers:
                s = layer.predict(query, mem)
                layer_scores[layer.name] = s
                total += layer.weight * s
            
            scored.append(ScoredMemory(
                memory=mem,
                score=total,
                layer_scores=layer_scores,
            ))
        
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]
    
    def explain(self, scored_mem: ScoredMemory) -> str:
        """解释某个记忆的评分来源"""
        parts = []
        for layer in self.layers:
            s = scored_mem.layer_scores.get(layer.name, 0)
            parts.append(f"{layer.name}={s:.2f}×{layer.weight:.2f}={s*layer.weight:.3f}")
        return f"score={scored_mem.score:.3f} [{', '.join(parts)}]"


# ── 集成到context_engine ──

def predict_and_rank(query: str, memories: list, weights: Optional[Dict[str, float]] = None) -> list:
    """
    集成函数——在context_engine中调用。
    
    Args:
        query: 用户查询
        memories: 候选记忆列表
        weights: 自定义权重（可选）
    
    Returns:
        按综合分数降序排列的记忆列表
    """
    predictor = PredictionLayers(weights=weights)
    scored = predictor.combined_predict(query, memories)
    return [sm.memory for sm in scored]


# ── CLI测试 ──

if __name__ == "__main__":
    import time
    
    # 模拟记忆
    class MockMemory:
        def __init__(self, content, lesson="", temperature=0.5, created_at=0, source="", tags=None):
            self.content = content
            self.lesson = lesson
            self.temperature = temperature
            self.created_at = created_at or time.time()
            self.source = source
            self.tags = tags or []
            self.prediction_confidence = temperature
    
    memories = [
        MockMemory("搜索VISTA论文", lesson="搜索策略需要调整", temperature=0.9, source="causal"),
        MockMemory("ISA记忆系统v2.0", lesson="情感温度函数有效", temperature=0.8, source="causal"),
        MockMemory("今天天气不错", temperature=0.3),
        MockMemory("用户偏好：反AI味写作", temperature=0.95, source="preference"),
        MockMemory("jage共享：其他Agent的教训", temperature=0.7, source="jage", tags=["shared"]),
    ]
    
    predictor = PredictionLayers()
    scored = predictor.combined_predict("搜索VISTA", memories, top_k=5)
    
    print("=== 多层预测结果 ===")
    for sm in scored:
        print(f"  score={sm.score:.3f} | {sm.memory.content[:30]}")
        print(f"    {predictor.explain(sm)}")
