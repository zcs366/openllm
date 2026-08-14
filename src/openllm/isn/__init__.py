"""ISN — 技能系统（Skill System）。

ISN技能系统负责：
1. 技能存储与检索
2. 工具动态注册（ToolRegistryBridge）
3. 技能自进化（SkillEvolution）
4. MAS Prompt评测
5. 统一技能配置（UnifiedSkillConfig）— 五源融合共享数据模型
"""
from .tool_registry_bridge import ToolRegistryBridge, ToolConfig, BridgeResult
from .unified_skill_config import (
    MetaSkillConfig,
    OptimizationHints,
    SkillCompositionContract,
    SkillFrameworkSource,
    SkillLifecycleState,
    SkillUsageMetrics,
    SourceTraceability,
    UnifiedSkillConfig,
)

__all__ = [
    "ToolRegistryBridge",
    "ToolConfig",
    "BridgeResult",
    "UnifiedSkillConfig",
    "SkillLifecycleState",
    "SkillFrameworkSource",
    "OptimizationHints",
    "SourceTraceability",
    "SkillUsageMetrics",
    "SkillCompositionContract",
    "MetaSkillConfig",
]

# ═══════════════════════════════════════════════════════════════════
# 待集成模块（沉默代码·不可删除·等待集成）
# ═══════════════════════════════════════════════════════════════════
# 以下模块包含有价值的代码，但尚未集成到公共API中。
# 铁律：不可删除沉默代码。这些模块等待后续集成激活。
#
# ISN 技能生命周期：
#   - skill_lifecycle.py      — 技能生命周期管理（创建→活跃→退役）
#   - skill_retirement.py     — 技能退役与归档机制
#   - profile_consistency.py  — SA身份一致性校验
#   - mas_prompt_bench.py     — MAS Prompt评测基准
#
# ISN 适配器层（adapters/）：
#   - openskill_adapter.py    — OpenSkill框架适配器
#   - skillos_adapter.py      — SkillOS框架适配器
#   - skcc_adapter.py         — SKCC框架适配器
#   - metaskills_adapter.py   — MetaSkills框架适配器
#   - skillopt_adapter.py     — SkillOpt框架适配器
#
# 集成优先级：见 isan/skill_lifecycle.py 和 isn/skill_retirement.py
# ═══════════════════════════════════════════════════════════════════
