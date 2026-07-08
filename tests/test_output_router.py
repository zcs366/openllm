"""
Tests for IKO Output Router
=============================

覆盖路由、渲染器注册、RenderPlan 生成、渲染执行。
共 8+ 测试用例。
"""

import pytest

from openllm.iko.intent_classifier import OutputIntent
from openllm.iko.output_router import (
    BaseRenderer,
    ConfirmationRenderer,
    DiffRenderer,
    ErrorCardRenderer,
    OutputFormat,
    OutputRouter,
    RendererRegistry,
    RenderPlan,
    ROUTE_TABLE,
    SilentRenderer,
    StructuredRenderer,
    TextRenderer,
)


# ── 辅助函数 ──

def _content(text: str = "hello") -> dict:
    """构建基础 content 字典。"""
    return {"text": text}


def _prefs(**kwargs) -> dict:
    """构建 user_prefs 字典。"""
    return kwargs


# ── 路由测试 ──

class TestRouteTable:
    """路由表完整性测试。"""

    def test_all_intents_have_routes(self) -> None:
        """每种 OutputIntent 都有对应的路由配置。"""
        for intent in OutputIntent:
            assert intent in ROUTE_TABLE, f"Missing route for {intent}"

    def test_all_routes_have_required_keys(self) -> None:
        """每个路由配置包含必需字段。"""
        required_keys = {"renderer", "format", "confidence_display", "trace_reveal"}
        for intent, config in ROUTE_TABLE.items():
            missing = required_keys - set(config.keys())
            assert not missing, f"Route for {intent} missing keys: {missing}"

    def test_intent_to_renderer_mapping(self) -> None:
        """Intent → Renderer 名称映射正确。"""
        expected = {
            OutputIntent.ERROR: "error_card",
            OutputIntent.CONFIRM: "confirmation",
            OutputIntent.ACT: "diff",
            OutputIntent.DECIDE: "structured",
            OutputIntent.INFORM: "text",
            OutputIntent.SILENT: "silent",
        }
        for intent, renderer_name in expected.items():
            assert ROUTE_TABLE[intent]["renderer"] == renderer_name


class TestOutputRouter:
    """OutputRouter 路由逻辑测试。"""

    def setup_method(self) -> None:
        """每个测试前重置路由引擎。"""
        self.router = OutputRouter()

    def test_inform_routes_to_text(self) -> None:
        """INFORM → text 渲染器。"""
        plan = self.router.route(OutputIntent.INFORM, _content("hi"), _prefs())
        assert plan.renderer_name == "text"
        assert plan.format_hint == OutputFormat.PLAIN

    def test_error_routes_to_error_card(self) -> None:
        """ERROR → error_card 渲染器。"""
        plan = self.router.route(OutputIntent.ERROR, _content("fail"), _prefs())
        assert plan.renderer_name == "error_card"
        assert plan.format_hint == OutputFormat.CARD

    def test_confirm_routes_to_confirmation(self) -> None:
        """CONFIRM → confirmation 渲染器。"""
        plan = self.router.route(OutputIntent.CONFIRM, _content("sure?"), _prefs())
        assert plan.renderer_name == "confirmation"
        assert plan.format_hint == OutputFormat.CARD

    def test_act_routes_to_diff(self) -> None:
        """ACT → diff 渲染器。"""
        plan = self.router.route(OutputIntent.ACT, {"changes": []}, _prefs())
        assert plan.renderer_name == "diff"
        assert plan.format_hint == OutputFormat.DIFF

    def test_decide_routes_to_structured(self) -> None:
        """DECIDE → structured 渲染器。"""
        plan = self.router.route(OutputIntent.DECIDE, {"items": []}, _prefs())
        assert plan.renderer_name == "structured"
        assert plan.format_hint == OutputFormat.STRUCTURED

    def test_silent_routes_to_silent(self) -> None:
        """SILENT → silent 渲染器。"""
        plan = self.router.route(OutputIntent.SILENT, _content(), _prefs())
        assert plan.renderer_name == "silent"
        assert plan.format_hint == OutputFormat.SILENT

    def test_render_plan_fields_correct(self) -> None:
        """RenderPlan 字段与路由表配置一致。"""
        plan = self.router.route(OutputIntent.INFORM, _content(), _prefs())
        assert isinstance(plan, RenderPlan)
        assert isinstance(plan.renderer_name, str)
        assert isinstance(plan.format_hint, OutputFormat)
        assert isinstance(plan.confidence_display, str)
        assert isinstance(plan.trace_reveal, str)

    def test_render_inform_returns_text(self) -> None:
        """render() 对 INFORM 返回纯文本。"""
        result = self.router.render(
            OutputIntent.INFORM, _content("Hello!"), _prefs(), confidence=0.9
        )
        assert result == "Hello!"

    def test_render_error_contains_error_card(self) -> None:
        """render() 对 ERROR 返回包含错误信息的卡片。"""
        result = self.router.render(
            OutputIntent.ERROR,
            {"error": "Connection failed", "cause": "timeout", "recovery": ["retry"]},
            _prefs(),
            confidence=0.3,
        )
        assert "Connection failed" in result
        assert "timeout" in result
        assert "retry" in result

    def test_render_confirm_contains_action(self) -> None:
        """render() 对 CONFIRM 包含操作描述。"""
        result = self.router.render(
            OutputIntent.CONFIRM,
            {"action": "删除文件", "risk_level": "HIGH", "options": ["确认", "取消"]},
            _prefs(),
            confidence=0.8,
        )
        assert "删除文件" in result
        assert "高风险" in result

    def test_render_silent_returns_empty(self) -> None:
        """render() 对 SILENT 返回空字符串。"""
        result = self.router.render(OutputIntent.SILENT, _content(), _prefs())
        assert result == ""


class TestRendererRegistry:
    """RendererRegistry 注册表测试。"""

    def test_register_and_get(self) -> None:
        """注册后可通过名称获取。"""
        registry = RendererRegistry()
        renderer = TextRenderer()
        registry.register("my_text", renderer)
        assert registry.get("my_text") is renderer

    def test_get_unknown_returns_none(self) -> None:
        """获取未注册渲染器返回 None。"""
        registry = RendererRegistry()
        assert registry.get("nonexistent") is None

    def test_register_duplicate_raises(self) -> None:
        """重复注册同名渲染器抛出 ValueError。"""
        registry = RendererRegistry()
        registry.register("dup", TextRenderer())
        with pytest.raises(ValueError, match="already registered"):
            registry.register("dup", TextRenderer())

    def test_register_non_renderer_raises(self) -> None:
        """注册非 BaseRenderer 实例抛出 TypeError。"""
        registry = RendererRegistry()
        with pytest.raises(TypeError, match="BaseRenderer instance"):
            registry.register("bad", "not a renderer")

    def test_has_and_list_names(self) -> None:
        """has() 和 list_names() 正确工作。"""
        registry = RendererRegistry()
        registry.register("alpha", TextRenderer())
        registry.register("beta", SilentRenderer())
        assert registry.has("alpha")
        assert not registry.has("gamma")
        assert registry.list_names() == ["alpha", "beta"]
        assert len(registry) == 2


class TestCustomRenderer:
    """自定义渲染器集成测试。"""

    def test_register_custom_renderer(self) -> None:
        """注册自定义渲染器后可通过 OutputRouter 使用。"""

        class CustomRenderer(BaseRenderer):
            """自定义渲染器——返回固定字符串。"""

            def render(self, intent, content, user_prefs, confidence):
                return "CUSTOM_OUTPUT"

            def get_render_plan(self, intent, content):
                return {
                    "format_hint": OutputFormat.PLAIN,
                    "confidence_display": "never",
                    "trace_reveal": "none",
                }

        registry = RendererRegistry()
        registry.register("custom", CustomRenderer())
        router = OutputRouter(registry=registry)

        # 验证注册表中有自定义渲染器
        assert registry.has("custom")
        result = router.render(OutputIntent.INFORM, _content(), _prefs(), confidence=0.5)
        # INFORM 仍然路由到 text，因为 ROUTE_TABLE 没变
        assert result == "hello"


class TestRenderPlanDataclass:
    """RenderPlan 数据类测试。"""

    def test_frozen(self) -> None:
        """RenderPlan 不可变。"""
        plan = RenderPlan(
            renderer_name="text",
            format_hint=OutputFormat.PLAIN,
            confidence_display="threshold",
            trace_reveal="none",
        )
        with pytest.raises(AttributeError):
            plan.renderer_name = "changed"  # type: ignore[misc]

    def test_defaults(self) -> None:
        """RenderPlan 有合理的默认值。"""
        plan = RenderPlan(renderer_name="text", format_hint=OutputFormat.PLAIN)
        assert plan.confidence_display == "threshold"
        assert plan.trace_reveal == "summary"

    def test_equality(self) -> None:
        """相同字段的 RenderPlan 相等。"""
        p1 = RenderPlan("text", OutputFormat.PLAIN)
        p2 = RenderPlan("text", OutputFormat.PLAIN)
        assert p1 == p2


class TestDiffRendererRendering:
    """DiffRenderer 渲染细节测试。"""

    def test_diff_with_reason(self) -> None:
        """DiffRenderer 包含变更理由。"""
        router = OutputRouter()
        content = {
            "reason": "修复空指针",
            "changes": [
                {"file": "main.py", "old": "x.foo()", "new": "x.bar()"},
            ],
        }
        result = router.render(OutputIntent.ACT, content, _prefs(), confidence=0.9)
        assert "修复空指针" in result
        assert "- x.foo()" in result
        assert "+ x.bar()" in result
        assert "main.py" in result


class TestStructuredRendererRendering:
    """StructuredRenderer 渲染细节测试。"""

    def test_structured_with_table(self) -> None:
        """StructuredRenderer 渲染 Markdown 表格。"""
        router = OutputRouter()
        content = {
            "table": [
                ["名称", "值"],
                ["foo", "1"],
                ["bar", "2"],
            ]
        }
        result = router.render(OutputIntent.DECIDE, content, _prefs(), confidence=0.8)
        assert "名称" in result
        assert "foo" in result
        assert "---" in result

    def test_structured_with_items(self) -> None:
        """StructuredRenderer 渲染编号列表。"""
        router = OutputRouter()
        content = {
            "items": [
                {"label": "选项A"},
                {"label": "选项B"},
            ]
        }
        result = router.render(OutputIntent.DECIDE, content, _prefs(), confidence=0.7)
        assert "1. 选项A" in result
        assert "2. 选项B" in result
