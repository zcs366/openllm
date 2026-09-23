"""
Tool Presets — 按需装配的工具子集（pi思路：用得到才在册）
==========================================================

背景（2026-09-22 军令）：
  create_default_tools() 全量27件，schema每轮对话全量注入上下文，
  本地7B的16k窗口里租金约1700 token，且选项越多模型选错率越高。
  参照pi（BASE_TOOLS=4），日常只留最必须的CORE，其余18件收进能力池
  （~/.openllm/pool/，磁盘零成本），用时用ToolRegistryBridge动态注册。

CORE九件 = 文件六件套 + 记忆三件套：
  read_file/write_file/search/list_dir/shell/python_exec —— 最小行动闭环（本工厂负责）
  memory_write/memory_read/memory_search —— Agent自己的记忆器官（ISA立身之本，
  由engine在create_core_tools()之后注册——器官挂引擎，不挂工厂）

用法：
    engine侧：self.tools = create_core_tools()   # 净结果=6文件+3记忆=9件
    临时扩编：create_core_tools(extra=["fcrawl", "ocr"]) 或 OPENLLM_TOOL_PRESET=fcrawl,ocr
"""

from __future__ import annotations

from typing import List, Optional

from .executor import ToolRegistry, create_default_tools

# 元工具显式schema（engine._build_tools_schema优先用注册schema，兜底才是默认生成）
TOOL_LOAD_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "精确装载的工具名"},
        "query": {"type": "string", "description": "语义搜索词，返回候选清单"},
    },
}
TOOL_UNLOAD_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "要卸载的工具名"},
    },
    "required": ["name"],
}

# 文件六件套（顺序无谓，名字是契约）。记忆三件由engine侧追加。
CORE_TOOL_NAMES = [
    "read_file", "write_file", "search", "list_dir", "shell", "python_exec",
]


def create_core_tools(extra: Optional[List[str]] = None,
                      with_loader: bool = True) -> ToolRegistry:
    """全量注册后剪枝到CORE（+extra）。

    剪枝姿势：先create_default_tools()再保留白名单——比逐个register安全，
    因为executor.py里的注册顺序/依赖逻辑原样保留，我们只裁清单。
    write_file/shell在_requires_verify里，都留在CORE，verify钩子不受影响。

    with_loader=True（默认）时注册元工具 tool_load/tool_unload：
      模型可在会话内从能力池(~/.openllm/pool/)按需发现并装载被剪的工具，
      用完卸载回收schema空间。这就是pi思路的最后一环——
      CORE常驻，其余待命，Agent自己持钥匙。

    返回的registry上有 _pruned_names / _pruned_funcs 供审计与装载。
    """
    registry = create_default_tools()
    # 库存武器上膛（2026-09-22）：四研究模块8件工具注册进来，与24件一起进剪枝池——
    # 默认不出鞘（被剪），tool_load 可装载。全量回退 create_default_tools() 时不含这8件。
    from .inventory_tools import register_inventory_tools
    register_inventory_tools(registry)
    all_names = {t["name"] for t in registry.list_tools()}
    keep = set(CORE_TOOL_NAMES) | set(extra or [])
    unknown = set(extra or []) - all_names
    if unknown:
        import logging
        logging.getLogger("openllm.tools.presets").warning(
            "extra里这些工具不在全量注册表中，忽略: %s", sorted(unknown))
    pruned = all_names - keep
    pruned_funcs: dict = {}
    for name in pruned:
        pruned_funcs[name] = (registry._tools[name],
                              registry._descriptions[name],
                              registry._schemas[name])
        registry._tools.pop(name, None)
        registry._descriptions.pop(name, None)
        registry._schemas.pop(name, None)
    setattr(registry, "_pruned_names", sorted(pruned))
    setattr(registry, "_pruned_funcs", pruned_funcs)
    if with_loader:
        _attach_tool_loader(registry)
    return registry


# ── tool_load / tool_unload：Agent自持武器库钥匙 ──────────────

def _attach_tool_loader(registry: ToolRegistry) -> None:
    """在registry上挂 tool_load（按需装载）与 tool_unload（用完卸载）。

    装载来源是剪枝时保留的 _pruned_funcs（函数原对象，非重新import），
    所以行为与全量模式逐字节一致。池子只用于语义搜索候选，不提供实现。
    失败路径全部显式报错文本，绝不静默。
    """

    def tool_load(name: str = "", query: str = "") -> str:
        """装载工具进本会话。name=精确装载；query=语义搜索返回候选。"""
        pruned_funcs = getattr(registry, "_pruned_funcs", {})
        if name:
            if name in registry._tools:
                return f"[tool_load] {name} 已在册（CORE常驻），无需装载。"
            if name not in pruned_funcs:
                # 模糊匹配：前缀/包含即候选（7B记不全名也能装上）
                fuzzy = sorted(n for n in pruned_funcs
                               if n.startswith(name) or name in n)
                if len(fuzzy) == 1:
                    fname = fuzzy[0]
                    func, desc, schema = pruned_funcs.pop(fname)
                    registry.register(fname, func, desc, schema or None)
                    getattr(registry, "_pruned_names").remove(fname)
                    return (f"[tool_load] '{name}' 精确未中，模糊匹配已装载 "
                            f"{fname} —— {desc}")
                if fuzzy:
                    return (f"[tool_load] '{name}' 有多个近似: {fuzzy}。"
                            f"请用全名装载。")
                return (f"[tool_load] 池子里没有 '{name}'。"
                        f"可装载: {sorted(pruned_funcs)}")
            func, desc, schema = pruned_funcs.pop(name)
            registry.register(name, func, desc, schema or None)
            getattr(registry, "_pruned_names").remove(name)
            return f"[tool_load] 已装载 {name} —— {desc}"
        if query:
            try:
                from ..isn.capability_pool import CapabilityPool
                hits = CapabilityPool().search(query, top_k=5, kind="tool")
            except Exception as e:  # 池子缺席不炸，退化为报pruned清单
                return (f"[tool_load] 池子不可用({e})。"
                        f"可按名装载: {sorted(pruned_funcs)}")
            if not hits:
                return (f"[tool_load] 池子无命中 '{query}'。"
                        f"可按名装载: {sorted(pruned_funcs)}")
            lines = [f"- {h['name']}: {h['description'][:60]}" for h in hits]
            return ("[tool_load] 候选如下，用 tool_load(name=...) 精确装载:\n"
                    + "\n".join(lines))
        return ("[tool_load] 需要参数。tool_load(name='fcrawl') 精确装载；"
                f"tool_load(query='网页') 搜索。当前可装载: {sorted(pruned_funcs)}")

    def tool_unload(name: str) -> str:
        """卸载会话内装载的工具，回收schema空间。CORE常驻不可卸。"""
        pruned_funcs = getattr(registry, "_pruned_funcs", {})
        if name in CORE_TOOL_NAMES:
            return f"[tool_unload] {name} 是CORE常驻，不可卸载。"
        if name not in registry._tools:
            return f"[tool_unload] {name} 不在册。"
        pruned_funcs[name] = (registry._tools[name],
                              registry._descriptions[name],
                              registry._schemas[name])
        registry._tools.pop(name, None)
        registry._descriptions.pop(name, None)
        registry._schemas.pop(name, None)
        getattr(registry, "_pruned_names").append(name)
        return f"[tool_unload] 已卸载 {name}，回到池子待命。"

    registry.register("tool_load", tool_load,
                      "按需装载工具(name=精确/query=搜索候选)——CORE不够用时用",
                      schema=TOOL_LOAD_SCHEMA)
    registry.register("tool_unload", tool_unload,
                      "卸载会话内装载的工具(name)——用完回收",
                      schema=TOOL_UNLOAD_SCHEMA)
