"""
路由决策模块 - 根据query复杂度和budget选择执行器

v0.5 延迟维（2026-09-22 总律，机制映射见工程任务书）：
- 信号分层：route() 第一跳按 scenario 定候选池，第二跳按 budget 筛选
- 延迟形态定池：interactive 永不落 batch-only 执行器（免费档结构性慢是商业模式）
- 延迟容忍池专吃慢：batch 场景按 cost_tier 升序（本地0→免费1→网关2→直连3）
- 本地路由自给自足：local=true 执行器禁止外呼，本地调用
- 注册表是宪法：池归属全在 providers.json，本模块只读
"""
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
from .registry import Provider, get_registry


class Budget(Enum):
    """预算档位"""
    FREE = 'free'
    BALANCED = 'balanced'
    QUALITY = 'quality'


class Scenario(Enum):
    """场景档位（延迟形态，第一路由信号）"""
    INTERACTIVE = 'interactive'  # 交互：人在等，快即体验，付费快池
    BATCH = 'batch'              # 批处理：夜间产线，慢换零边际，免费慢池


@dataclass
class Decision:
    """路由决策结果"""
    provider: Provider
    model: str
    reason: str
    fallback_chain: List[str]  # 降级链：执行器名称列表
    scenario: str = 'interactive'  # v0.5: 路由时的场景档

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


def _provider_serves_scenario(provider: Provider, scenario: Scenario) -> bool:
    """执行器在指定场景下是否有可用模型"""
    if scenario == Scenario.BATCH:
        return len(provider.batch_models()) > 0
    return len(provider.interactive_models()) > 0


def _models_for_scenario(provider: Provider, scenario: Scenario) -> List[str]:
    """执行器在指定场景下的可用模型列表"""
    if scenario == Scenario.BATCH:
        return provider.batch_models()
    return provider.interactive_models()


def build_fallback_chain(selected_provider: Provider, budget: Budget,
                         scenario: Optional[Scenario] = None,
                         selected_model: Optional[str] = None) -> List[str]:
    """
    构建降级链：质量降序，最多3跳
    v0.5：同场景内构建——跨池降级禁止（interactive链不混batch-only，反之亦然）
    """
    if scenario is None:
        scenario = Scenario.INTERACTIVE
    registry = get_registry()
    all_providers = registry.get_all_providers()

    # 场景过滤：只允许能在该场景服务的执行器（跨池禁止）
    in_scenario = [p for p in all_providers if _provider_serves_scenario(p, scenario)]

    # 按质量档降序排序
    sorted_providers = sorted(in_scenario, key=lambda p: p.quality_tier, reverse=True)

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


def select_by_budget(budget: Budget, scenario: Optional[Scenario] = None) -> List[Provider]:
    """
    根据预算选择执行器列表
    v0.5：第一跳场景过滤已内嵌——batch按cost升序（本地0→免费1→网关2），interactive按质量降序
    """
    if scenario is None:
        scenario = Scenario.INTERACTIVE
    registry = get_registry()
    all_providers = registry.get_all_providers()

    # 第一跳：场景池过滤（跨池禁止的总闸）
    candidates = [p for p in all_providers if _provider_serves_scenario(p, scenario)]

    if budget == Budget.FREE:
        # 免费档：cost_tier == 0 或 cost_tier == 1
        candidates = [p for p in candidates if p.cost_tier <= 1]
    elif budget == Budget.BALANCED:
        # 均衡档：cost_tier <= 2
        candidates = [p for p in candidates if p.cost_tier <= 2]
    else:  # QUALITY
        pass  # 全部场景内候选

    # 排序：batch场景成本升序（专吃慢，零边际优先）；interactive质量降序（快即体验）
    if scenario == Scenario.BATCH:
        candidates.sort(key=lambda p: (p.cost_tier, -p.quality_tier))
    else:
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


def route(query: str, budget: Optional[Budget] = None,
          scenario: Optional[Scenario] = None) -> Decision:
    """
    路由决策主函数（v0.5 两跳路由：先场景后预算）

    总律映射：
    - 第一跳 scenario 定候选池（信号分层：先路由任务，再路由模型）
    - 第二跳 budget 在池内筛选（延迟形态定池：interactive永不落batch-only）
    - batch 按 cost 升序选（延迟容忍池专吃慢）；interactive 按 quality 降序选（快即体验）

    Args:
        query: 用户查询
        budget: 预算档位（默认balanced）
        scenario: 场景档位（默认interactive）

    Returns:
        Decision: 路由决策结果
    """
    if budget is None:
        budget = Budget.BALANCED
    if scenario is None:
        scenario = Scenario.INTERACTIVE

    registry = get_registry()

    # 检查显式@执行器名或@模型名（显式指定>一切路由判断）
    explicit_model, remaining_query = parse_explicit_model(query)
    if explicit_model:
        # 首先尝试作为执行器名
        provider = registry.get_provider(explicit_model)
        if provider:
            model = _models_for_scenario(provider, scenario)
            model = model[0] if model else (provider.available_models[0] if provider.available_models else provider.model or 'default')
            fallback_chain = build_fallback_chain(provider, budget, scenario, model)
            return Decision(
                provider=provider,
                model=model,
                reason=f"显式指定执行器 @{explicit_model}",
                fallback_chain=fallback_chain,
                scenario=scenario.value
            )
        # 然后尝试作为模型名
        provider = registry.get_provider_for_model(explicit_model)
        if provider:
            fallback_chain = build_fallback_chain(provider, budget, scenario, explicit_model)
            return Decision(
                provider=provider,
                model=explicit_model,
                reason=f"显式指定模型 @{explicit_model}",
                fallback_chain=fallback_chain,
                scenario=scenario.value
            )

    # 分析复杂度（仅balanced档跟随复杂度升降；显式free/quality是硬约束）
    complexity = analyze_complexity(remaining_query)

    scn_slow_warning = ''
    if budget == Budget.BALANCED:
        if scenario == Scenario.BATCH:
            # 批处理balanced：不做质量升降级，成本优先（夜间产线逻辑）
            effective_budget = Budget.FREE
            reason_suffix = "（batch场景balanced降为free，成本优先）"
        elif complexity == 'high':
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
        # 总律3：interactive+free=免费档结构性慢，警告但放行（local优先已在select中体现）
        if scenario == Scenario.INTERACTIVE and budget == Budget.FREE:
            scn_slow_warning = "（警告：免费档结构性慢，交互场景慎用）"

    # 第二跳：池内选择（第一跳场景过滤在select_by_budget内）
    candidates = select_by_budget(effective_budget, scenario)

    if not candidates:
        # 场景内无候选：interactive降级到本地，batch降级到ollama（两者同体）
        ollama = registry.get_provider('ollama')
        if ollama:
            return Decision(
                provider=ollama,
                model='ilm-v6',
                reason="场景内无可用执行器，降级到本地ollama",
                fallback_chain=[],
                scenario=scenario.value
            )
        raise RuntimeError("无可用执行器")

    # 选择排序首位（interactive=质量最高；batch=成本最低）
    selected = candidates[0]
    fallback_chain = build_fallback_chain(selected, effective_budget, scenario)

    # 选择模型（场景内可用模型首位）
    scn_models = _models_for_scenario(selected, scenario)
    if scn_models:
        model = scn_models[0]
    elif selected.available_models:
        model = selected.available_models[0]
    else:
        model = selected.model or 'default'

    reason = f"scn={scenario.value}, budget={budget.value}, complexity={complexity}{reason_suffix}{scn_slow_warning}"

    return Decision(
        provider=selected,
        model=model,
        reason=reason,
        fallback_chain=fallback_chain,
        scenario=scenario.value
    )