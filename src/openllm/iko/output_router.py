"""
IKO Output Router — 输出路由引擎
=================================

将 OutputIntent 映射到具体的渲染器，生成 RenderPlan。
RenderPlan 指导下游渲染逻辑选择正确的输出格式和展示策略。

核心设计：
- RenderPlan 不可变数据类，描述渲染策略
- BaseRenderer 抽象基类，定义渲染器接口
- RendererRegistry 插件注册表，支持运行时注册新渲染器
- OutputRouter 路由引擎，根据 OutputIntent + user_prefs 选择渲染器
- ROUTE_TABLE 路由表，6种 OutputIntent → 渲染配置映射

设计约束：
- 路由逻辑零LLM调用，纯确定性规则
- 渲染器通过注册表解耦，支持插件化扩展
- RenderPlan 是路由的最终产出，不可变

用法：
    router = OutputRouter()
    plan = router.route(
        intent=OutputIntent.INFORM,
        content={"text": "Hello, world!"},
        user_prefs={"verbosity": "concise"},
    )
    # plan.renderer_name == "text"
    # plan.format_hint == "plain"
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from openllm.iko.intent_classifier import OutputIntent

logger = logging.getLogger(__name__)


# ── 输出格式枚举 ──

class OutputFormat(Enum):
    """输出格式类型。"""
    PLAIN = "plain"
    """纯文本格式。"""

    STRUCTURED = "structured"
    """结构化数据格式（JSON/YAML/表格）。"""

    DIFF = "diff"
    """差异对比格式（变更预览）。"""

    CARD = "card"
    """卡片格式（确认/错误）。"""

    SILENT = "silent"
    """静默，无可见输出。"""


# ── RenderPlan 数据类 ──

@dataclass(frozen=True)
class RenderPlan:
    """渲染计划——路由引擎的最终产出，不可变。

    描述渲染器选择、格式提示、置信度展示策略和溯源信息揭示策略。

    Attributes:
        renderer_name: 渲染器名称，对应 RendererRegistry 中的注册名。
        format_hint: 输出格式提示，指导渲染器选择具体输出形态。
        confidence_display: 置信度展示策略，如 "always"/"threshold"/"never"。
        trace_reveal: 溯源信息揭示策略，如 "full"/"summary"/"none"。
    """
    renderer_name: str
    format_hint: OutputFormat
    confidence_display: str = "threshold"
    trace_reveal: str = "summary"


# ── 渲染器基类 ──

class BaseRenderer(ABC):
    """渲染器抽象基类——所有渲染器的接口契约。

    子类必须实现：
    - render(): 将内容渲染为字符串
    - get_render_plan(): 返回渲染计划参数
    """

    @abstractmethod
    def render(
        self,
        intent: OutputIntent,
        content: dict[str, Any],
        user_prefs: dict[str, Any],
        confidence: float,
    ) -> str:
        """将内容渲染为最终输出字符串。

        Args:
            intent: 输出意图类型。
            content: 待渲染的内容数据。
            user_prefs: 用户偏好设置。
            confidence: 置信度，0.0 到 1.0。

        Returns:
            渲染后的字符串输出。
        """
        ...

    @abstractmethod
    def get_render_plan(self, intent: OutputIntent, content: dict[str, Any]) -> dict[str, Any]:
        """返回此渲染器的渲染计划参数。

        Args:
            intent: 输出意图类型。
            content: 待渲染的内容数据。

        Returns:
            包含 format_hint, confidence_display, trace_reveal 的字典。
        """
        ...


# ── 具体渲染器 ──

class TextRenderer(BaseRenderer):
    """纯文本渲染器——简洁模式，适合 INFORM 意图。

    将内容直接以纯文本形式输出，无额外格式化。
    """

    def render(
        self,
        intent: OutputIntent,
        content: dict[str, Any],
        user_prefs: dict[str, Any],
        confidence: float,
    ) -> str:
        """渲染为纯文本。

        Args:
            intent: 输出意图类型。
            content: 待渲染内容，期望包含 'text' 字段。
            user_prefs: 用户偏好，可包含 'verbosity' 控制详细度。
            confidence: 置信度。

        Returns:
            纯文本字符串。
        """
        text = content.get("text", "")
        verbosity = user_prefs.get("verbosity", "normal")

        if verbosity == "concise":
            # 简洁模式：截取前200字符
            if len(text) > 200:
                return text[:200] + "..."
            return text
        return text

    def get_render_plan(self, intent: OutputIntent, content: dict[str, Any]) -> dict[str, Any]:
        """返回纯文本渲染计划参数。"""
        return {
            "format_hint": OutputFormat.PLAIN,
            "confidence_display": "threshold",
            "trace_reveal": "none",
        }


class StructuredRenderer(BaseRenderer):
    """结构化数据渲染器——表格+JSON渲染，适合 DECIDE 意图。

    将内容以结构化格式（表格/列表）输出。
    """

    def render(
        self,
        intent: OutputIntent,
        content: dict[str, Any],
        user_prefs: dict[str, Any],
        confidence: float,
    ) -> str:
        """渲染为结构化文本。

        将 content 中的 'items'（列表）渲染为带编号的表格行。
        如果 content 包含 'table'（列表的列表），渲染为 Markdown 表格。

        Args:
            intent: 输出意图类型。
            content: 待渲染内容，包含 'items' 或 'table'。
            user_prefs: 用户偏好。
            confidence: 置信度。

        Returns:
            结构化文本字符串。
        """
        lines: list[str] = []

        # 渲染表格
        if "table" in content:
            table_data = content["table"]
            if isinstance(table_data, list) and table_data:
                # 第一行作为表头
                header = table_data[0]
                lines.append("| " + " | ".join(str(c) for c in header) + " |")
                lines.append("| " + " | ".join("---" for _ in header) + " |")
                for row in table_data[1:]:
                    lines.append("| " + " | ".join(str(c) for c in row) + " |")

        # 渲染选项列表
        if "items" in content:
            items = content["items"]
            for i, item in enumerate(items, 1):
                label = item.get("label", str(item)) if isinstance(item, dict) else str(item)
                lines.append(f"  {i}. {label}")

        return "\n".join(lines)

    def get_render_plan(self, intent: OutputIntent, content: dict[str, Any]) -> dict[str, Any]:
        """返回结构化渲染计划参数。"""
        return {
            "format_hint": OutputFormat.STRUCTURED,
            "confidence_display": "threshold",
            "trace_reveal": "summary",
        }


class DiffRenderer(BaseRenderer):
    """变更预览渲染器——Diff语义+理由链，适合 ACT 意图。

    借鉴 Cursor 的 diff 语义：展示变更前后的对比和变更理由。
    """

    def render(
        self,
        intent: OutputIntent,
        content: dict[str, Any],
        user_prefs: dict[str, Any],
        confidence: float,
    ) -> str:
        """渲染变更预览。

        content 结构：
        - changes: list[dict]，每项包含 'file', 'old', 'new'
        - reason: str，变更理由

        Args:
            intent: 输出意图类型。
            content: 待渲染内容。
            user_prefs: 用户偏好。
            confidence: 置信度。

        Returns:
            diff 预览字符串。
        """
        lines: list[str] = []
        changes = content.get("changes", [])
        reason = content.get("reason", "")

        if reason:
            lines.append(f"# 变更理由: {reason}")
            lines.append("")

        for change in changes:
            file_path = change.get("file", "unknown")
            old_text = change.get("old", "")
            new_text = change.get("new", "")
            lines.append(f"--- {file_path}")
            lines.append(f"+++ {file_path}")
            if old_text:
                for line in old_text.splitlines():
                    lines.append(f"- {line}")
            if new_text:
                for line in new_text.splitlines():
                    lines.append(f"+ {line}")
            lines.append("")

        return "\n".join(lines).rstrip()

    def get_render_plan(self, intent: OutputIntent, content: dict[str, Any]) -> dict[str, Any]:
        """返回 diff 渲染计划参数。"""
        return {
            "format_hint": OutputFormat.DIFF,
            "confidence_display": "always",
            "trace_reveal": "full",
        }


class ConfirmationRenderer(BaseRenderer):
    """确认卡片渲染器——操作确认+风险标签，适合 CONFIRM 意图。

    渲染操作确认卡片，包含操作描述、风险标签和确认/取消选项。
    """

    def render(
        self,
        intent: OutputIntent,
        content: dict[str, Any],
        user_prefs: dict[str, Any],
        confidence: float,
    ) -> str:
        """渲染确认卡片。

        content 结构：
        - action: str，操作描述
        - risk_level: str，风险等级
        - details: str，操作详情
        - options: list[str]，可选操作

        Args:
            intent: 输出意图类型。
            content: 待渲染内容。
            user_prefs: 用户偏好。
            confidence: 置信度。

        Returns:
            确认卡片字符串。
        """
        lines: list[str] = []
        action = content.get("action", "未知操作")
        risk_level = content.get("risk_level", "UNKNOWN")
        details = content.get("details", "")
        options = content.get("options", ["确认", "取消"])

        # 风险标签映射
        risk_labels = {
            "HIGH": "⚠️ 高风险",
            "MEDIUM": "⚡ 中风险",
            "LOW": "✅ 低风险",
        }
        risk_label = risk_labels.get(risk_level, f"? 未知风险({risk_level})")

        lines.append("┌─────────────────────────────────┐")
        lines.append("│        操作确认                   │")
        lines.append("├─────────────────────────────────┤")
        lines.append(f"│ 操作: {action}")
        lines.append(f"│ 风险: {risk_label}")
        if details:
            lines.append(f"│ 详情: {details}")
        lines.append("├─────────────────────────────────┤")
        for i, opt in enumerate(options, 1):
            lines.append(f"│ [{i}] {opt}")
        lines.append("└─────────────────────────────────┘")

        return "\n".join(lines)

    def get_render_plan(self, intent: OutputIntent, content: dict[str, Any]) -> dict[str, Any]:
        """返回确认渲染计划参数。"""
        return {
            "format_hint": OutputFormat.CARD,
            "confidence_display": "always",
            "trace_reveal": "summary",
        }


class ErrorCardRenderer(BaseRenderer):
    """错误卡片渲染器——错误原因+恢复建议，适合 ERROR 意图。

    渲染错误信息卡片，包含错误描述、原因分析和恢复建议。
    """

    def render(
        self,
        intent: OutputIntent,
        content: dict[str, Any],
        user_prefs: dict[str, Any],
        confidence: float,
    ) -> str:
        """渲染错误卡片。

        content 结构：
        - error: str，错误描述
        - cause: str，错误原因
        - recovery: list[str]，恢复建议列表

        Args:
            intent: 输出意图类型。
            content: 待渲染内容。
            user_prefs: 用户偏好。
            confidence: 置信度。

        Returns:
            错误卡片字符串。
        """
        lines: list[str] = []
        error = content.get("error", "未知错误")
        cause = content.get("cause", "")
        recovery = content.get("recovery", [])

        lines.append("╔═══════════════════════════════════╗")
        lines.append("║         ❌ 错误报告               ║")
        lines.append("╠═══════════════════════════════════╣")
        lines.append(f"║ 错误: {error}")
        if cause:
            lines.append(f"║ 原因: {cause}")
        if recovery:
            lines.append("║───────────────────────────────────║")
            lines.append("║ 恢复建议:")
            for i, tip in enumerate(recovery, 1):
                lines.append(f"║   {i}. {tip}")
        lines.append("╚═══════════════════════════════════╝")

        return "\n".join(lines)

    def get_render_plan(self, intent: OutputIntent, content: dict[str, Any]) -> dict[str, Any]:
        """返回错误渲染计划参数。"""
        return {
            "format_hint": OutputFormat.CARD,
            "confidence_display": "always",
            "trace_reveal": "full",
        }


class SilentRenderer(BaseRenderer):
    """静默渲染器——无可见输出，适合 SILENT 意图。

    不产生任何可见输出，但记录渲染事件供审计使用。
    """

    def render(
        self,
        intent: OutputIntent,
        content: dict[str, Any],
        user_prefs: dict[str, Any],
        confidence: float,
    ) -> str:
        """返回空字符串（静默输出）。

        Args:
            intent: 输出意图类型。
            content: 待渲染内容（忽略）。
            user_prefs: 用户偏好（忽略）。
            confidence: 置信度（忽略）。

        Returns:
            空字符串。
        """
        return ""

    def get_render_plan(self, intent: OutputIntent, content: dict[str, Any]) -> dict[str, Any]:
        """返回静默渲染计划参数。"""
        return {
            "format_hint": OutputFormat.SILENT,
            "confidence_display": "never",
            "trace_reveal": "none",
        }


# ── 渲染器注册表 ──

class RendererRegistry:
    """渲染器插件注册表——管理渲染器实例的注册和查找。

    支持运行时动态注册新渲染器，解耦路由逻辑和具体渲染实现。

    Attributes:
        _renderers: 渲染器名称 → 渲染器实例的映射。
    """

    def __init__(self) -> None:
        """初始化空注册表。"""
        self._renderers: dict[str, BaseRenderer] = {}

    def register(self, name: str, renderer: BaseRenderer) -> None:
        """注册一个渲染器。

        Args:
            name: 渲染器唯一名称（小写+下划线）。
            renderer: 渲染器实例。

        Raises:
            TypeError: renderer 不是 BaseRenderer 子类实例。
            ValueError: name 已被注册。
        """
        if not isinstance(renderer, BaseRenderer):
            raise TypeError(
                f"renderer must be a BaseRenderer instance, got {type(renderer).__name__}"
            )
        if name in self._renderers:
            raise ValueError(f"Renderer '{name}' already registered")
        self._renderers[name] = renderer
        logger.debug("Registered renderer: %s (%s)", name, type(renderer).__name__)

    def get(self, name: str) -> Optional[BaseRenderer]:
        """根据名称获取渲染器。

        Args:
            name: 渲染器名称。

        Returns:
            渲染器实例，未找到返回 None。
        """
        return self._renderers.get(name)

    def has(self, name: str) -> bool:
        """检查渲染器是否已注册。

        Args:
            name: 渲染器名称。

        Returns:
            已注册返回 True。
        """
        return name in self._renderers

    def list_names(self) -> list[str]:
        """列出所有已注册渲染器名称。

        Returns:
            渲染器名称列表。
        """
        return sorted(self._renderers.keys())

    def __len__(self) -> int:
        """返回已注册渲染器数量。"""
        return len(self._renderers)

    def __repr__(self) -> str:
        """返回注册表字符串表示。"""
        names = ", ".join(self.list_names())
        return f"RendererRegistry(count={len(self)}, names=[{names}])"


# ── 路由表 ──

ROUTE_TABLE: dict[OutputIntent, dict[str, Any]] = {
    OutputIntent.ERROR: {
        "renderer": "error_card",
        "format": OutputFormat.CARD,
        "confidence_display": "always",
        "trace_reveal": "full",
    },
    OutputIntent.CONFIRM: {
        "renderer": "confirmation",
        "format": OutputFormat.CARD,
        "confidence_display": "always",
        "trace_reveal": "summary",
    },
    OutputIntent.ACT: {
        "renderer": "diff",
        "format": OutputFormat.DIFF,
        "confidence_display": "always",
        "trace_reveal": "full",
    },
    OutputIntent.DECIDE: {
        "renderer": "structured",
        "format": OutputFormat.STRUCTURED,
        "confidence_display": "threshold",
        "trace_reveal": "summary",
    },
    OutputIntent.INFORM: {
        "renderer": "text",
        "format": OutputFormat.PLAIN,
        "confidence_display": "threshold",
        "trace_reveal": "none",
    },
    OutputIntent.SILENT: {
        "renderer": "silent",
        "format": OutputFormat.SILENT,
        "confidence_display": "never",
        "trace_reveal": "none",
    },
}


# ── 路由引擎 ──

class OutputRouter:
    """输出路由引擎——根据 OutputIntent 选择渲染器并生成 RenderPlan。

    核心逻辑：
    1. 根据 intent 查询 ROUTE_TABLE 获取路由配置
    2. 在 RendererRegistry 中查找对应渲染器
    3. 如果渲染器未注册，使用 text 渲染器作为 fallback
    4. 构建 RenderPlan 返回

    Attributes:
        registry: 渲染器注册表实例。
        default_renderer: 默认渲染器名称（fallback）。
    """

    DEFAULT_RENDERER = "text"

    def __init__(self, registry: Optional[RendererRegistry] = None) -> None:
        """初始化路由引擎。

        Args:
            registry: 可选的渲染器注册表。为 None 时创建新注册表。
        """
        self.registry = registry if registry is not None else RendererRegistry()
        self._setup_default_renderers()

    def _setup_default_renderers(self) -> None:
        """注册内置渲染器。"""
        defaults = {
            "text": TextRenderer(),
            "structured": StructuredRenderer(),
            "diff": DiffRenderer(),
            "confirmation": ConfirmationRenderer(),
            "error_card": ErrorCardRenderer(),
            "silent": SilentRenderer(),
        }
        for name, renderer in defaults.items():
            if not self.registry.has(name):
                self.registry.register(name, renderer)

    def route(
        self,
        intent: OutputIntent,
        content: dict[str, Any],
        user_prefs: dict[str, Any],
    ) -> RenderPlan:
        """路由入口——根据 intent 生成 RenderPlan。

        Args:
            intent: 输出意图类型。
            content: 待渲染的内容数据。
            user_prefs: 用户偏好设置。

        Returns:
            RenderPlan，描述渲染策略。
        """
        # 查询路由表
        route_config = ROUTE_TABLE.get(intent)
        if route_config is None:
            logger.warning("No route for intent %s, using default", intent)
            route_config = ROUTE_TABLE[OutputIntent.INFORM]

        renderer_name = route_config["renderer"]

        # 从注册表查找渲染器
        renderer = self.registry.get(renderer_name)
        if renderer is None:
            logger.warning(
                "Renderer '%s' not found, falling back to '%s'",
                renderer_name,
                self.DEFAULT_RENDERER,
            )
            renderer_name = self.DEFAULT_RENDERER
            renderer = self.registry.get(renderer_name)
            if renderer is None:
                # 极端情况：连默认渲染器都没有
                logger.error("Default renderer '%s' not registered", self.DEFAULT_RENDERER)
                return RenderPlan(
                    renderer_name="none",
                    format_hint=OutputFormat.SILENT,
                    confidence_display="never",
                    trace_reveal="none",
                )

        # 让渲染器提供渲染计划参数
        plan_params = renderer.get_render_plan(intent, content)

        return RenderPlan(
            renderer_name=renderer_name,
            format_hint=plan_params.get("format_hint", route_config["format"]),
            confidence_display=plan_params.get(
                "confidence_display", route_config["confidence_display"]
            ),
            trace_reveal=plan_params.get("trace_reveal", route_config["trace_reveal"]),
        )

    def render(
        self,
        intent: OutputIntent,
        content: dict[str, Any],
        user_prefs: dict[str, Any],
        confidence: float = 1.0,
    ) -> str:
        """一站式路由+渲染——生成 RenderPlan 并执行渲染。

        Args:
            intent: 输出意图类型。
            content: 待渲染内容。
            user_prefs: 用户偏好。
            confidence: 置信度，0.0 到 1.0。

        Returns:
            渲染后的字符串。如果路由失败返回空字符串。
        """
        plan = self.route(intent, content, user_prefs)
        renderer = self.registry.get(plan.renderer_name)
        if renderer is None:
            logger.error("Renderer '%s' not available for render", plan.renderer_name)
            return ""
        return renderer.render(intent, content, user_prefs, confidence)

    def __repr__(self) -> str:
        """返回路由引擎字符串表示。"""
        return f"OutputRouter(registry={self.registry!r})"
