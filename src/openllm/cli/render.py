"""
CLI Markdown 渲染 — 回答文本的 rich 渲染，替代裸 print。

开关：OPENLLM_RENDER=0/off 关闭渲染（原样 print）。
容错：rich 抛任何异常 → 回退 print（渲染是化妆不是关卡）。
"""
import os
import re
import sys

try:
    from rich.console import Console
    from rich.markdown import Markdown

    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# markdown 特征检测：标题/粗体/列表/表格/围栏/引用
_MD_RE = re.compile(
    r"(^#{1,6}\s|"
    r"\*\*[^*\n]+\*\*|"
    r"^[\-\*]\s|"
    r"^\|.*\|.*\||"
    r"^```|"
    r"^>\s)",
    re.MULTILINE,
)


def render_answer(text: str) -> None:
    """渲染回答文本到终端。

    无 md 特征或开关关闭时原样 print；有 md 特征时用 rich 渲染。
    Console 每次新建（capsys/封印场景下 file 需动态解析 sys.stdout）。
    DR-20260927-04：加主题（代码块暗底/表格青线/标题品蓝），终端
    友好色（auto，管道无色码不污染日志）。
    """
    if (
        text is None
        or not text.strip()
        or os.environ.get("OPENLLM_RENDER", "1") in ("0", "off")
        or not HAS_RICH
        or not _MD_RE.search(text)
    ):
        print(text, end="")
        return

    try:
        # 经模块 globals 取用（顶部 HAS_RICH 条件导入）——保留测试
        # monkeypatch render.Markdown 的注入缝；pyright 的
        # possibly-unbound 属已知误报，用 ignore 注释压制。
        # DR-20260927-01b：width 跟随终端实际列数（最小 40）。写死 120 时，
        # 终端比 120 窄 → rich 内部换行数与终端实际行数对不上，屏上留
        # 断行残渣，把下一轮输入行顶乱（光标错位的视觉病灶之一）。
        from rich.theme import Theme

        _cols = getattr(sys.stdout, "get_size", lambda: (80, 0))()[0]
        _theme = Theme({
            "markdown.code": "cyan",
            "markdown.code_block": "bright_black on #1c1f26",
            "markdown.h1.border": "bright_cyan",
            "markdown.h1": "bold bright_cyan",
            "markdown.h2": "bold cyan",
            "markdown.h3": "bold blue",
            "markdown.item.bullet": "cyan",
            "markdown.table.border": "dim cyan",
            "markdown.block_quote": "dim italic",
            "markdown.link": "underline cyan",
        })
        _out = Console(  # pyright: ignore[reportPossiblyUnboundVariable]
            file=sys.stdout, width=max(40, _cols),
            theme=_theme, highlight=False, soft_wrap=False)
        _out.print(Markdown(text))  # pyright: ignore[reportPossiblyUnboundVariable]
    except Exception:
        print(text, end="")
