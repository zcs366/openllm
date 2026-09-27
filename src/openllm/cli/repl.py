"""openLLM CLI 输入层 v1 — bracketed paste + 命令白名单。

DR-20260927-01（老搭档实测两个病）：
1. Backspace 侵蚀已渲染的对话历史——cmd.Cmd/readline 行编辑与终端光标
   状态错位（长回复多行回绕后尤甚），退格的"左移+擦除"作用到了输入行
   之外的滚动缓冲区，把上面已渲染的回复字删掉了。
2. 粘贴内容被当 CLI 指令执行——无 bracketed paste mode，粘贴被逐字符
   喂给命令解析器；`/` 开头的粘贴文本直接触发 _handle_command，未知
   裸词还会撞上 cmd.Cmd 内置命令，对话与指令两条通道被打混。

修法（对标 hermes tui / pi cli 的窗口行为）：
- prompt_toolkit PromptSession 接管行编辑：退格只作用于输入 buffer，
  粘贴走 bracketed paste（终端把粘贴块包成整体一次插入），一次 Enter
  一次提交；↑↓ 输入历史为赠品。回复正文仍在两次 prompt 之间用裸
  print 输出（prompt_toolkit 已退出输入状态，行内渲染安全）。
- 命令白名单（main.AgentShell.KNOWN_COMMANDS）：白名单外的 `/xxx`
  在 default() 里当普通消息发模型——这是 hermes 同款语义：未知斜杠
  命令 = 内容，不是指令。
- 粘贴护栏：以 `/` 开头且含换行的多行粘贴块，即便首词像命令也不执行
  （多行整块是要发给模型的材料，不是命令；防止 `/status\\n长文` 把
  粘贴正文吞进 args 丢弃）。
- 降级链：无 prompt_toolkit 或非 TTY（管道/测试）→ 回落 cmd.Cmd
  cmdloop，旧行为原样保留。
"""
from __future__ import annotations

import sys

__all__ = ["run_repl", "is_suspicious_paste"]


def is_suspicious_paste(line: str, known_commands: set[str]) -> bool:
    """判定一行输入是否可能是粘贴误触指令。

    三类可疑：
    a) 单行 `/xxx` 且 xxx 不在白名单 —— 八成是粘贴的引用文字；
    b) 含换行的多行块以 `/` 开头 —— 粘贴材料，即便首词在白名单也不当命令；
    c) 空命令 `/`。
    """
    s = line.lstrip()
    if not s.startswith("/"):
        return False
    if "\n" in line.rstrip():
        return True  # (b)
    parts = s[1:].split()
    if not parts:
        return True  # (c)
    return parts[0].lower() not in known_commands  # (a)


def run_repl(shell, known_commands: set[str] | None = None) -> None:
    """prompt_toolkit 输入循环——替换 cmd.Cmd.cmdloop。

    shell 需提供：
      .prompt  —— 提示符字符串（可含 ANSI 颜色码）
      .onecmd(line) -> bool —— 现有命令分发（返回 True 即退出，
                                /exit 传播已在 onecmd 覆写里接好）
      .do_EOF(arg) —— Ctrl-D 退出
    """
    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import ANSI
    from prompt_toolkit.history import InMemoryHistory

    if known_commands is None:
        known_commands = getattr(shell, "KNOWN_COMMANDS", set())
    assert known_commands is not None

    session = PromptSession(history=InMemoryHistory())
    # DR-20260927-01b：pt 的 message 必须是单行——shell.prompt 的前导换行
    # 会让 pt 渲染提示符时光标定位错乱（多轮后光标跑到提示符前面）。
    # 行间距由"轮间输出保证以 \n 结尾"承担（render.py 侧已修），不靠提示符。
    message = ANSI(getattr(shell, "prompt", "> ").lstrip("\n"))

    while True:
        try:
            line = session.prompt(message)
        except KeyboardInterrupt:  # Ctrl-C：清行重来，不退出
            continue
        except EOFError:  # Ctrl-D
            shell.do_EOF("")
            break
        if not line.strip():
            continue
        if is_suspicious_paste(line, known_commands):
            # 护栏只提示不拦截：default() 的白名单判定负责把它当普通消息
            print("(粘贴护栏：/ 开头的粘贴内容按普通消息发送，不是指令)")
        if shell.onecmd(line):
            break
        # 注：列位置无需守卫——main.default 各完成路径均以 print(...\n)
        # 收尾（计时行/墓碑行），光标必然在行首。光标错位的真根因是
        # 多行 prompt message（已在上方 lstrip 修复）+ rich 宽度错位。


def repl_available() -> bool:
    """REPL 可用性：TTY + prompt_toolkit 可导入。"""
    try:
        if not sys.stdin.isatty():
            return False
        import prompt_toolkit  # noqa: F401
        return True
    except Exception:
        return False
