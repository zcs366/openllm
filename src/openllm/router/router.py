"""
路由决策模块 - 根据query复杂度和budget选择执行器
"""
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
from .registry import Provider, get_registry


class Budget(Enum):
    """预算档位"""
    FREE = 'free'
    BALANCED = 'balanced'
    QUALITY = 'quality'


@dataclass
class Decision:
    """路由决策结果"""
    provider: Provider
    model: str
    reason: str
    fallback_chain: List[str]  # 降级链：执行器名称列表
    
    def __str__(self) -> str:
        return f"Decision(provider={self.provider.name}, model={self.model}, reason={self.reason})"


# 数学/代码复杂度关键词
COMPLEXITY_MARKERS = [
    # 数学相关
    r'[∑∫∂∇√∞±×÷=<>≤≥≠≈]',
    r'\b(?:求导|积分|微分|极限|矩阵|向量|方程|不等式)\b',
    r'\b(?:derivative|integral|matrix|vector|equation)\b',
    # 代码相关
    r'\b(?:def|class|import|from|return|if|else|for|while|try|except)\b',
    r'\b(?:函数|类|导入|返回|循环|条件)\b',
    r'\b(?:algorithm|recursion|binary|sort|search)\b',
    r'[{}\[\]();]',
]


def analyze_complexity(query: str) -> str:
    """
    分析query复杂度
    返回: 'high' | 'medium' | 'low'
    """
    # 检查是否有复杂度标记
    for pattern in COMPLEXITY_MARKERS:
        if re.search(pattern, query, re.IGNORECASE):
            return 'high'
    
    # 简单长度启发
    query_len = len(query.strip())
    if query_len <= 15:
        return 'low'
    
    return 'medium'


def build_fallback_chain(selected_provider: Provider, budget: Budget) -> List[str]:
    """
    构建降级链：质量降序，最多3跳
    """
    registry = get_registry()
    all_providers = registry.get_all_providers()
    
    # 按质量档降序排序
    sorted_providers = sorted(all_providers, key=lambda p: p.quality_tier, reverse=True)
    
    # 找到当前执行器在排序后列表中的位置
    selected_idx = -1
    for idx, p in enumerate(sorted_providers):
        if p.name == selected_provider.name:
            selected_idx = idx
            break
    
    if selected_idx == -1:
        # 如果找不到，返回当前执行器自身
        return [selected_provider.name]
    
    # 从当前执行器开始，质量降序取最多3个
    fallback_chain = []
    for p in sorted_providers[selected_idx:selected_idx + 3]:
        if p.name != selected_provider.name:
            fallback_chain.append(p.name)
    
    return fallback_chain


def select_by_budget(budget: Budget) -> List[Provider]:
    """
    根据预算选择执行器列表（质量降序）
    """
    registry = get_registry()
    all_providers = registry.get_all_providers()
    
    if budget == Budget.FREE:
        # 免费档：cost_tier == 0 或 cost_tier == 1
        candidates = [p for p in all_providers if p.cost_tier <= 1]
    elif budget == Budget.BALANCED:
        # 均衡档：cost_tier <= 2
        candidates = [p for p in all_providers if p.cost_tier <= 2]
    else:  # QUALITY
        # 质量档：全部
        candidates = all_providers[:]
    
    # 按质量档降序排序
    candidates.sort(key=lambda p: p.quality_tier, reverse=True)
    
    return candidates


def parse_explicit_model(query: str) -> tuple[Optional[str], str]:
    """
    解析显式@模型名前缀
    返回: (模型名, 剩余query)
    """
    match = re.match(r'^@(\S+)\s+(.+)$', query)
    if match:
        return match.group(1), match.group(2)
    return None, query


def route(query: str, budget: Optional[Budget] = None) -> Decision:
    """
    路由决策主函数
    
    Args:
        query: 用户查询
        budget: 预算档位（默认balanced）
    
    Returns:
        Decision: 路由决策结果
    """
    if budget is None:
        budget = Budget.BALANCED
    
    registry = get_registry()
    
    # 检查显式@执行器名或@模型名
    explicit_model, remaining_query = parse_explicit_model(query)
    if explicit_model:
        # 首先尝试作为执行器名
        provider = registry.get_provider(explicit_model)
        if provider:
            fallback_chain = build_fallback_chain(provider, budget)
            # 选择执行器的第一个模型
            model = provider.available_models[0] if provider.available_models else provider.model or 'default'
            return Decision(
                provider=provider,
                model=model,
                reason=f"显式指定执行器 @{explicit_model}",
                fallback_chain=fallback_chain
            )
        # 然后尝试作为模型名
        provider = registry.get_provider_for_model(explicit_model)
        if provider:
            fallback_chain = build_fallback_chain(provider, budget)
            return Decision(
                provider=provider,
                model=explicit_model,
                reason=f"显式指定模型 @{explicit_model}",
                fallback_chain=fallback_chain
            )
    
    # 分析复杂度（仅balanced档跟随复杂度升降；显式free/quality是硬约束）
    complexity = analyze_complexity(remaining_query)
    
    if budget == Budget.BALANCED:
        if complexity == 'high':
            effective_budget = Budget.QUALITY
            reason_suffix = "（复杂任务，升级到质量档）"
        elif complexity == 'low':
            effective_budget = Budget.FREE
            reason_suffix = "（简单任务，使用免费档）"
        else:
            effective_budget = budget
            reason_suffix = ""
    else:
        effective_budget = budget
        reason_suffix = "" if complexity == 'medium' else f"，显式{budget.value}档不因复杂度{complexity}调档"
    
    # 选择执行器
    candidates = select_by_budget(effective_budget)
    
    if not candidates:
        # 降级到免费本地执行器
        ollama = registry.get_provider('ollama')
        if ollama:
            return Decision(
                provider=ollama,
                model='ilm-v6',
                reason="无可用执行器，降级到本地ollama",
                fallback_chain=[]
            )
        raise RuntimeError("无可用执行器")
    
    # 选择质量最高的
    selected = candidates[0]
    fallback_chain = build_fallback_chain(selected, effective_budget)
    
    # 选择模型
    if selected.available_models:
        model = selected.available_models[0]
    else:
        model = selected.model or 'default'
    
    reason = f"budget={budget.value}, complexity={complexity}{reason_suffix}"
    
    return Decision(
        provider=selected,
        model=model,
        reason=reason,
        fallback_chain=fallback_chain
    )