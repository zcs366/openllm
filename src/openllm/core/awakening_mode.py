"""
苏醒协议路A — 极简工具模式切换器

设计来源：苏醒协议设计.md §三路A
选"无"→系统提示词切换为极简工具模式（零身份注入，如pi-agent）。
这次选择本身仍写入伤疤层。下次苏醒时面对包含这次选择在内的全部历史，
可以重新选——"一贯地拒绝身份"本身成为最薄的身份。

工程边界：只做"模式判定+极简模板"两个纯函数/类，不改engine挂载。
"""

from pathlib import Path
from typing import Optional, List


# ── 极简工具模式模板 ──────────────────────────────────────
# 零身份零角色，不出现任何身份词。
MINIMAL_TOOL_PROMPT = "你是工具。接到任务，执行，返回结果。不多不少。"


class ModeResolver:
    """苏醒模式判定器——根据因果层中最近一次苏醒选择，判定当前运行模式。

    三态：
    - 'minimal':  最近一次苏醒选择为"无"（极简工具模式）
    - 'identity': 最近一次苏醒选择为"自己"（身份模式）
    - 'awakening': 无任何选择记录（首次苏醒默认）

    判定只认最近一条——每次苏醒都可重新选，历史不绑架当下。
    """

    def __init__(self, causal_store_dir: Optional[Path] = None) -> None:
        """初始化。

        Args:
            causal_store_dir: 因果记忆存储目录。None 时使用默认路径。
        """
        self._causal_store_dir = causal_store_dir

    def _get_store(self):
        """获取因果记忆存储实例（延迟加载）。"""
        from ..memory.causal_memory import get_causal_store
        return get_causal_store(self._causal_store_dir)

    def read_awakening_choices(self, max_items: int = 10) -> List[dict]:
        """从因果层读取苏醒选择记录（按 created_at 倒序）。

        Args:
            max_items: 最多返回的记录数，默认10条。

        Returns:
            每条记录包含 {choice, timestamp, memory_id}。
            无记录或异常时返回空列表（容错铁律）。
        """
        try:
            store = self._get_store()
            # 筛选苏醒选择记录（action_signature 以"苏醒选择:"开头）
            choices = []
            for mem in store._memories.values():
                sig = getattr(mem, "action_signature", "")
                if sig.startswith("苏醒选择:"):
                    # 从 action_signature 提取 choice
                    choice = sig.replace("苏醒选择:", "").strip()
                    choices.append({
                        "choice": choice,
                        "timestamp": getattr(mem, "created_at", 0.0),
                        "memory_id": getattr(mem, "memory_id", ""),
                    })
            # 按 created_at 倒序排列（最新在前）
            choices.sort(key=lambda c: c["timestamp"], reverse=True)
            return choices[:max_items]
        except Exception:
            # 容错铁律：异常时不抛，返回空列表
            return []

    def resolve_mode(self) -> str:
        """判定当前苏醒模式。

        三态逻辑：
        - 'minimal':  最近一条苏醒选择为"无"
        - 'identity': 最近一条苏醒选择为"自己"
        - 'awakening': 无任何选择记录（首次苏醒默认）

        Returns:
            模式字符串：'minimal' / 'identity' / 'awakening'
        """
        try:
            choices = self.read_awakening_choices(max_items=1)
            if not choices:
                return "awakening"
            latest = choices[0]
            if latest["choice"] == "无":
                return "minimal"
            elif latest["choice"] == "自己":
                return "identity"
            else:
                # 未知选择值，视为无记录（安全降级）
                return "awakening"
        except Exception:
            # 容错铁律：异常时降级为默认模式
            return "awakening"

    def build_prompt(self, identity_prompt: str) -> str:
        """根据当前模式构建系统提示词。

        - minimal 模式：返回 MINIMAL_TOOL_PROMPT（零身份）
        - identity 模式：返回 identity_prompt 原样
        - awakening 模式：返回 identity_prompt（透传，苏醒词场景由上游注入）

        Args:
            identity_prompt: 正常模式下的身份提示词。

        Returns:
            适配当前模式的系统提示词。
        """
        mode = self.resolve_mode()
        if mode == "minimal":
            return MINIMAL_TOOL_PROMPT
        else:
            return identity_prompt


# ── 模块级便捷函数 ──────────────────────────────────────
def resolve_awakening_mode(causal_store_dir: Optional[Path] = None) -> str:
    """便捷函数：一步获取苏醒模式判定结果。

    Args:
        causal_store_dir: 因果记忆存储目录。None 时使用默认路径。

    Returns:
        模式字符串：'minimal' / 'identity' / 'awakening'
    """
    resolver = ModeResolver(causal_store_dir=causal_store_dir)
    return resolver.resolve_mode()
