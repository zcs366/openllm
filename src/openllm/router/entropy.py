"""
Shannon熵计算 - 从账本计算路由分布熵H(r)
"""
import math
from collections import Counter
from typing import Dict, List, Optional
from .ledger import Ledger


def shannon_entropy(distribution: Dict[str, int]) -> float:
    """
    计算Shannon熵
    
    Args:
        distribution: {项目: 计数} 字典
    
    Returns:
        熵值（bits）
    """
    total = sum(distribution.values())
    
    if total == 0:
        return 0.0
    
    entropy = 0.0
    for count in distribution.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    
    return entropy


def h_of(n: int = 100, ledger: Optional[Ledger] = None) -> float:
    """
    从账本滑动窗口计算路由分布熵H(r)
    
    Args:
        n: 滑动窗口大小（默认最近100次成功调用）
        ledger: 账本实例（可选，默认使用全局账本）
    
    Returns:
        Shannon熵值（bits），范围[0, log2(k)]，k为执行器数量
    """
    if ledger is None:
        ledger = Ledger()
    
    records = ledger.read_records()
    
    # 只统计成功调用（result == 'ok'）
    successful_records = [r for r in records if r.get('result') == 'ok']
    
    # 取最近n条
    recent_records = successful_records[-n:]
    
    if not recent_records:
        return 0.0
    
    # 统计各执行器调用次数
    provider_counts: Dict[str, int] = Counter()
    for record in recent_records:
        provider = record.get('provider', 'unknown')
        provider_counts[provider] += 1
    
    # 计算Shannon熵
    return shannon_entropy(dict(provider_counts))


def get_distribution(n: int = 100, ledger: Optional[Ledger] = None) -> Dict[str, float]:
    """
    获取路由分布（概率）
    
    Args:
        n: 滑动窗口大小
        ledger: 账本实例
    
    Returns:
        {执行器: 概率} 字典
    """
    if ledger is None:
        ledger = Ledger()
    
    records = ledger.read_records()
    
    # 只统计成功调用
    successful_records = [r for r in records if r.get('result') == 'ok']
    
    # 取最近n条
    recent_records = successful_records[-n:]
    
    if not recent_records:
        return {}
    
    # 统计各执行器调用次数
    provider_counts: Dict[str, int] = Counter()
    for record in recent_records:
        provider = record.get('provider', 'unknown')
        provider_counts[provider] += 1
    
    total = sum(provider_counts.values())
    
    if total == 0:
        return {}
    
    # 转换为概率
    return {provider: count / total for provider, count in provider_counts.items()}