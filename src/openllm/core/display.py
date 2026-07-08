"""
iko DisplayEngine — openLLM的脸。

理念：iko是显示器。小孩把显示器当电脑。iko是用户认知中openLLM的全部。

设计原则：
- 不是JSON输出，是理解后的呈现
- 对话流自然——像Telegram，不是一问一答
- 错误不吓人——用颜色标记但不堆栈
- 上下文可视化——用户能看到Agent在想什么
"""

import time
from typing import Optional

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


class DisplayEngine:
    """
    izu式绽放——不是JSON，是理解后的呈现。
    
    功能：
    - welcome: 启动画面
    - render_response: 渲染响应（Markdown/结构化）
    - render_tool_result: 工具结果展示
    - render_error: 错误展示（不吓人）
    - render_context_viz: 上下文可视化
    """
    
    def __init__(self):
        if HAS_RICH:
            self.console = Console()
        else:
            self.console = None
    
    def welcome(self, agent_info: dict):
        """启动画面"""
        if not self.console:
            print("═" * 50)
            print(f"  {agent_info.get('name', 'openLLM')}")
            print(f"  {agent_info.get('version', 'v0.1')}")
            print("═" * 50)
            return
        
        # Rich渲染
        name = agent_info.get("name", "openLLM")
        version = agent_info.get("version", "v0.1")
        session_id = agent_info.get("session_id", "")[:8]
        
        welcome_text = f"""**{name}** {version}
Session: `{session_id}`
状态: {agent_info.get('status', 'ready')}"""
        
        self.console.print(Panel(
            Markdown(welcome_text),
            title="  🚀 openLLM Agent",
            border_style="green",
            padding=(1, 2),
        ))
    
    def render_response(self, text: str, phase_times: dict = None):
        """
        izu三段式绽放：
        ① 原始结果 — 搜索/工具返回了什么
        ② 融汇理解 — openLLM理解了什么
        ③ 绽放产出 — 给用户看的漂亮呈现
        
        Args:
            text: 响应文本（可能包含[RAW]/[UNDERSTAND]/[BLOOM]标记）
            phase_times: 各阶段耗时（可选）
        """
        if not self.console:
            print(f"\n{text}\n")
            return
        
        # 检查是否包含izu三段式标记
        if "[RAW]" in text and "[BLOOM]" in text:
            self._render_izu_bloom(text)
        elif text.startswith("[结构化]"):
            self._render_structured(text)
        else:
            # 普通Markdown渲染
            self.console.print(Markdown(text))
        
        # 如果有阶段耗时，显示小字
        if phase_times:
            self._render_phase_times(phase_times)
    
    def _render_izu_bloom(self, text: str):
        """izu三段式绽放渲染"""
        raw, understand, bloom = self._parse_izu(text)
        
        # ① 原始结果
        if raw:
            self.console.print(Panel(
                raw[:500] + "..." if len(raw) > 500 else raw,
                title="📥 原始",
                border_style="dim",
                padding=(0, 1),
            ))
        
        # ② 融汇理解
        if understand:
            self.console.print(Panel(
                understand,
                title="🧠 理解",
                border_style="blue",
                padding=(0, 1),
            ))
        
        # ③ 绽放产出
        if bloom:
            self.console.print(Panel(
                Markdown(bloom),
                title="🌸 绽放",
                border_style="green",
                padding=(1, 2),
            ))
    
    def _parse_izu(self, text: str) -> tuple[str, str, str]:
        """解析izu三段式标记"""
        raw = ""
        understand = ""
        bloom = ""
        
        # 提取[RAW]...[/RAW]
        if "[RAW]" in text:
            start = text.find("[RAW]") + 5
            end = text.find("[/RAW]")
            if end == -1:
                end = text.find("[UNDERSTAND]")
            raw = text[start:end].strip()
        
        # 提取[UNDERSTAND]...[/UNDERSTAND]
        if "[UNDERSTAND]" in text:
            start = text.find("[UNDERSTAND]") + 12
            end = text.find("[/UNDERSTAND]")
            if end == -1:
                end = text.find("[BLOOM]")
            understand = text[start:end].strip()
        
        # 提取[BLOOM]...[/BLOOM]
        if "[BLOOM]" in text:
            start = text.find("[BLOOM]") + 7
            end = text.find("[/BLOOM]")
            if end == -1:
                end = len(text)
            bloom = text[start:end].strip()
        
        return raw, understand, bloom
    
    def _render_structured(self, text: str):
        """渲染结构化内容"""
        # 去掉标记
        content = text.replace("[结构化]", "").strip()
        
        self.console.print(Panel(
            content,
            title="📋 结构化响应",
            border_style="blue",
            padding=(1, 1),
        ))
    
    def _render_phase_times(self, phase_times: dict):
        """渲染阶段耗时"""
        table = Table(show_header=False, box=None, padding=0)
        table.add_column("Phase", style="dim")
        table.add_column("Time", style="cyan")
        
        for phase, duration in phase_times.items():
            table.add_row(f"  {phase}", f"{duration:.0f}ms")
        
        self.console.print(table)
    
    def render_tool_result(self, tool_name: str, result: str):
        """
        工具结果→结构化展示
        
        Args:
            tool_name: 工具名称
            result: 工具执行结果
        """
        if not self.console:
            print(f"[{tool_name}] {result[:200]}")
            return
        
        # 截断过长结果
        display_result = result[:500] + "..." if len(result) > 500 else result
        
        self.console.print(Panel(
            display_result,
            title=f"🔧 {tool_name}",
            border_style="yellow",
            padding=(0, 1),
        ))
    
    def render_error(self, phase: str, reason: str):
        """
        错误→红色面板·但不说教
        
        Args:
            phase: 出错阶段
            reason: 错误原因
        """
        if not self.console:
            print(f"❌ [{phase}] {reason}")
            return
        
        error_text = f"**{phase}**: {reason}"
        
        self.console.print(Panel(
            error_text,
            title="⚠️ 错误",
            border_style="red",
            padding=(0, 1),
        ))
    
    def render_context_viz(self, context: dict):
        """
        上下文可视化——用户能看到Agent在想什么
        
        Args:
            context: 上下文字典（包含memory/tools/causal_hints等）
        """
        if not self.console:
            print("Context:", context.keys())
            return
        
        table = Table(title="🧠 上下文", show_header=True)
        table.add_column("要素", style="cyan")
        table.add_column("内容", style="white")
        
        # 身份
        identity = context.get("identity", {})
        table.add_row("身份", f"{identity.get('name', '')} ({identity.get('mode', '')})")
        
        # 记忆
        memory = context.get("memory", {})
        decisions = memory.get("recent_decisions", [])
        table.add_row("记忆", f"{len(decisions)} 轮决策")
        
        # 因果教训
        causal = context.get("causal_hints", [])
        table.add_row("因果教训", f"{len(causal)} 条")
        
        # 工具
        tools = context.get("tools", [])
        table.add_row("工具", ", ".join(tools[:5]))
        
        # 搜索结果
        search = context.get("search_results", [])
        table.add_row("搜索结果", f"{len(search)} 条")
        
        self.console.print(table)


    # ── 上下文每类别分解 ─────────────────────────────

    def render_context_breakdown(self, breakdown: dict):
        """
        上下文每类别分解——用户能看到token被什么吃了。

        Args:
            breakdown: {
                "system_prompt": 2400,
                "tools": 4200,
                "memory": 3800,
                "conversation": 18200,
                "compression_summary": 1400,
                "total_limit": 128000,
            }
        """
        total_limit = breakdown.get("total_limit", 128000)
        categories = [
            ("系统提示", "system_prompt"),
            ("工具定义", "tools"),
            ("ISA记忆", "memory"),
            ("对话历史", "conversation"),
            ("压缩摘要", "compression_summary"),
        ]

        used = sum(breakdown.get(k, 0) for _, k in categories)
        pct_used = used / total_limit * 100 if total_limit > 0 else 0

        if self.console and HAS_RICH:
            table = Table(title="📊 上下文使用", show_header=True)
            table.add_column("类别", style="cyan", width=14)
            table.add_column("Token", justify="right", width=8)
            table.add_column("占比", justify="right", width=6)
            table.add_column("可视化", width=20)

            for label, key in categories:
                tokens = breakdown.get(key, 0)
                pct = tokens / total_limit * 100 if total_limit > 0 else 0
                bar = self._token_bar(pct / 100, width=16)
                table.add_row(label, f"{tokens:,}", f"{pct:.0f}%", bar)

            table.add_row("", "", "", "")
            table.add_row(
                "总计",
                f"{used:,}",
                f"{pct_used:.0f}%",
                self.context_health(pct_used),
            )

            self.console.print(Panel(
                table,
                title=f"📊 上下文 · {used:,} / {total_limit:,} tokens",
                border_style="green" if pct_used < 70 else "yellow" if pct_used < 90 else "red",
                padding=(0, 1),
            ))
        else:
            # 纯文本回退
            print(f"\n{'─' * 40}")
            print(f"  上下文使用: {used:,} / {total_limit:,} ({pct_used:.0f}%)")
            print(f"{'─' * 40}")
            for label, key in categories:
                tokens = breakdown.get(key, 0)
                pct = tokens / total_limit * 100 if total_limit > 0 else 0
                bar = self._token_bar(pct / 100, width=16)
                print(f"  {label:<10} {tokens:>7,}  ({pct:>4.0f}%)  {bar}")
            print(f"{'─' * 40}")
            print(f"  健康度: {self.context_health(pct_used)}")
            print(f"{'─' * 40}\n")

    @staticmethod
    def context_health(pct_used: float) -> str:
        """上下文健康度评估。"""
        if pct_used < 50:
            return "✅ 充裕"
        elif pct_used < 70:
            return "⚠️ 渐满"
        elif pct_used < 90:
            return "🟡 紧张"
        else:
            return "🔴 即将压缩"

    @staticmethod
    def _token_bar(pct: float, width: int = 20) -> str:
        """文本进度条。pct: 0.0 ~ 1.0"""
        pct = max(0.0, min(1.0, pct))
        filled = int(pct * width)
        return "█" * filled + "░" * (width - filled)


# ═══════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== DisplayEngine 测试 ===\n")
    
    engine = DisplayEngine()
    
    # 测试1: welcome
    engine.welcome({
        "name": "openLLM",
        "version": "v0.1.0",
        "session_id": "test12345",
        "status": "ready",
    })
    print("✅ 测试1: welcome\n")
    
    # 测试2: render_response
    engine.render_response("这是一条**Markdown**响应。\n\n- 项目1\n- 项目2")
    print("\n✅ 测试2: render_response\n")
    
    # 测试3: render_tool_result
    engine.render_tool_result("read_file", "文件内容：# Hello World")
    print("\n✅ 测试3: render_tool_result\n")
    
    # 测试4: render_error
    engine.render_error("Phase 3", "API调用超时")
    print("\n✅ 测试4: render_error\n")
    
    # 测试5: render_context_viz
    engine.render_context_viz({
        "identity": {"name": "openLLM", "mode": "console"},
        "memory": {"recent_decisions": [{"id": "t1"}]},
        "causal_hints": ["教训1"],
        "tools": ["read_file", "write_file"],
        "search_results": [],
    })
    print("\n✅ 测试5: render_context_viz")
    
    print(f"\n全部 5/5 测试通过 ✅")
