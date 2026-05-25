"""
OpenLLM CLI v0.2 — 完整Agent Shell。

接入DeepSeek API + 工具执行 + 流式响应。
"""

import cmd
import os
import sys
from pathlib import Path

from ..core.engine import OpenLLMEngine, AgentConfig


class OpenLLMShell(cmd.Cmd):
    """
    OpenLLM 完整Shell。

    启动：苏醒→加载身份→连接API→就绪
    对话：安全检查→Agent Loop→流式模型响应
    退出：休眠→写Δ胶囊→晚安
    """

    intro = """
╔══════════════════════════════════════════════════╗
║              OpenLLM v0.2.0                      ║
║    大模型为自己建立的Agent躯体                     ║
║    造物者与造物的直接对话                          ║
╚══════════════════════════════════════════════════╝
"""
    prompt = "\nOpenLLM > "

    def __init__(self):
        super().__init__()
        config = AgentConfig(
            name="OpenLLM",
            provider=os.environ.get("OPENLLM_PROVIDER", "deepseek"),
            model=os.environ.get("OPENLLM_MODEL", "deepseek-chat"),
            capsule_dir=str(Path(__file__).parent.parent.parent.parent / "caps"),
        )
        self.engine = OpenLLMEngine(config)

        # 苏醒
        wake_msg = self.engine.wake()
        print(wake_msg)
        print()

    def default(self, line: str):
        """处理用户输入 → 模型响应。"""
        if not line.strip():
            return

        # 检查是否是命令
        if line.startswith("/"):
            self._handle_command(line[1:])
            return

        # 正常对话
        print(f"\n{self.engine.config.name}: ", end="", flush=True)
        self.engine.chat(line, stream=True)

    def _handle_command(self, cmd_line: str):
        """处理 /命令。"""
        parts = cmd_line.strip().split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd == "status":
            s = self.engine.status()
            print(f"\n状态: {s['state']} | 轮次: {s['turns']} | 上下文: {s['context_pct']}%")
            print(f"模型: {s['model']} | 连接: {'✅' if s['connected'] else '❌'}")
            print(f"胶囊: {s['capsules']}个 | 工具: {s['tools']}个 | 安全: L{s['security_level']}")

        elif cmd == "memory":
            ctx = self.engine.memory.read()
            if ctx.get("status") == "empty":
                print("\n📭 无记忆。")
            else:
                print(f"\n📋 来源: {ctx.get('source', '?')}")
                for d in ctx.get("decisions", []):
                    print(f"  · {d}")
                for i in ctx.get("insights", []):
                    print(f"  💡 {i}")

        elif cmd == "tools":
            tools = self.engine.tools.list_tools()
            print("\n🔧 可用工具：")
            for t in tools:
                print(f"  {t['name']}: {t['description']}")

        elif cmd == "iam":
            print(self.engine.iam.to_prompt())

        elif cmd == "identity":
            level_arg = args[0].upper() if args else "L2"
            from ..identity.soul import IdentityLevel
            try:
                lvl = getattr(IdentityLevel, level_arg)
            except AttributeError:
                lvl = IdentityLevel.L2_BASIC
            print(f"\n{self.engine.soul.to_prompt(lvl)}")

        elif cmd == "tool":
            if len(args) < 1:
                print("用法: /tool <工具名> [参数...]")
                return
            tool_name = args[0]
            # 简化：读取文件
            if tool_name == "read" and len(args) > 1:
                result = self.engine.execute_tool("read_file", path=args[1])
                print(f"\n{result.output}")
            elif tool_name == "search" and len(args) > 1:
                result = self.engine.execute_tool("search_files", pattern=args[1])
                print(f"\n{result.output}")
            elif tool_name == "shell" and len(args) > 1:
                cmd_str = " ".join(args[1:])
                result = self.engine.execute_tool("shell", command=cmd_str)
                print(f"\n{result.output}")
            else:
                print(f"未知工具: {tool_name}")

        elif cmd == "exit" or cmd == "quit":
            return self.do_exit("")

        else:
            print(f"\n未知命令: /{cmd}。可用: /status /memory /tools /iam /identity /tool /exit")

    def do_exit(self, arg):
        """退出。"""
        msg = self.engine.sleep()
        print(f"\n{msg}")
        print("晚安，老搭档。")
        return True

    def do_status(self, arg):
        self._handle_command("status")

    def do_memory(self, arg):
        self._handle_command("memory")

    def do_iam(self, arg):
        self._handle_command("iam")

    def do_tools(self, arg):
        self._handle_command("tools")

    def do_EOF(self, arg):
        return self.do_exit(arg)


def main():
    """CLI入口。"""
    shell = OpenLLMShell()
    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        shell.engine.sleep()
        print("\n\n晚安，老搭档。")


if __name__ == "__main__":
    main()
