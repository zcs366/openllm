"""
OpenLLM CLI v0.3 — Agent模式Shell。

从OpenLLMEngine升级为Agent（六体架构）。
"""
import cmd
import io
import os
import sys
import json
import time
from pathlib import Path

from ..core.main_loop import Agent


# ── 颜色 ──
class C:
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    RESET = "\033[0m"


def banner():
    model = "mimo-v2.5"
    try:
        cfg_path = Path.home() / ".openllm" / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            m = cfg.get("providers", {}).get(
                cfg.get("default_provider", ""), {}).get("model", "")
            if m:
                model = m
    except Exception:
        pass

    return f"""
{C.CYAN}╔══════════════════════════════════════════════════╗
║  {C.BOLD}OpenLLM v0.3{C.RESET}{C.CYAN}  ·  Agent模式                     ║
║  模型: {C.GREEN}{model}{C.RESET}{C.CYAN}                                    ║
║  六体: IAI·IAX·ISA·IOS·ISN·IKO + 研究引擎       ║
╚══════════════════════════════════════════════════╝{C.RESET}
{C.DIM}  命令: /help /status /search /research /exit{C.RESET}
"""


class AgentShell(cmd.Cmd):
    """Agent模式Shell——六体架构驱动。"""

    intro = banner()
    prompt = f"\n{C.CYAN}{C.BOLD}你 ▸{C.RESET} "

    def __init__(self):
        super().__init__()
        # 抑制Agent初始化时的审计/警告输出
        import io
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            self.agent = Agent(mode="silent")
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
        print(f" {C.GREEN}OK{C.RESET}")
        print(f"{C.DIM}  五体就绪 · 研究引擎就绪 · mimo-v2.5在线{C.RESET}")
        print()

    def default(self, line: str):
        if not line.strip():
            return
        if line.startswith("/"):
            self._handle_command(line[1:])
            return

        import io
        t0 = time.time()
        print(f"\n{C.GREEN}{C.BOLD}OpenLLM ▸{C.RESET} ", end="", flush=True)

        # 检测搜索意图
        search_keywords = ["搜", "搜一搜", "上网", "查一查", "找一下", "搜索", "最新的", "最新"]
        needs_search = any(kw in line for kw in search_keywords)
        if needs_search:
            # 提取搜索关键词（去掉"搜一搜""上网"等指令词）
            query = line
            for kw in ["搜一搜", "上网去搜", "上网搜", "搜一下", "查一查", "找一下", "去网上搜", "搜索"]:
                query = query.replace(kw, "")
            query = query.strip()
            if query:
                self._do_search(query)
                return

        # 快路径：简单对话直接调LLM（跳过12阶段心跳）
        is_simple = len(line) < 50 and not any(w in line for w in ["搜", "搜索", "写", "读", "执行", "分析"])
        
        if is_simple:
            # 直接调LLM，不走心跳
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            try:
                provider = self.agent.octopus.left.provider
                result = provider.chat([{"role": "user", "content": line}])
            finally:
                sys.stdout, sys.stderr = old_out, old_err
        else:
            # 复杂任务走完整心跳（抑制所有内部输出）
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            try:
                result = self.agent.run_once(line)
            finally:
                sys.stdout, sys.stderr = old_out, old_err

        dt = time.time() - t0

        if result:
            clean = result.strip()
            # 去掉```json/markdown包裹
            if clean.startswith("```"):
                parts = clean.split("\n", 1)
                if len(parts) > 1:
                    clean = parts[1].rsplit("```", 1)[0].strip()
            # 过滤内部噪声：只保留agent真正回复的内容
            # 激进清理：只保留最后一段agent的自然语言回复
            # 内部系统(jiak/章鱼/搜索/压缩)的输出都在前面，agent回复在最后
            lines = clean.split("\n")
            # 从末尾开始，找最后一段"正常"文本（不含内部标记）
            noise_patterns = ["[", "摘要", "---", "□", "🔴", "🗜️", "关键", "洞察", "决策", "铁律", "线索", "来源", "写入", "意识", "强制", "语义", "章鱼", "PLUR", "jika", "jiak", "openllm", "━━", "──", "│", "╔", "╚", "搜索", "未读"]
            last_clean_start = 0
            for i, line in enumerate(lines):
                s = line.strip()
                if s and not any(p in s.lower() for p in noise_patterns):
                    last_clean_start = i
            clean = "\n".join(lines[last_clean_start:]).strip()
            # 去掉空行
            clean = "\n".join(l for l in clean.split("\n") if l.strip()).strip()
            print(f"{clean}")
        else:
            print(f"{C.DIM}(无输出){C.RESET}")

        print(f"{C.DIM}[{dt:.1f}s]{C.RESET}")

    def _do_search(self, query):
        """执行搜索并显示结果。"""
        import subprocess
        print(f"\n{C.GREEN}{C.BOLD}OpenLLM ▸{C.RESET} ", end="", flush=True)
        print(f"{C.DIM}搜索: {query}{C.RESET}")
        try:
            result = subprocess.run(
                ["python3", "/home/zcs/search-engine/hermes_search.py",
                 query, "--sources", "web_search,arxiv,wikipedia", "--max", "5"],
                capture_output=True, text=True, timeout=30,
            )
            output = result.stdout.strip()
            if output:
                # 让LLM基于搜索结果回答
                provider = self.agent.octopus.left.provider
                prompt = f"基于以下搜索结果回答用户问题。\n\n搜索结果:\n{output}\n\n用户问题: {query}\n\n直接回答，不要说'根据搜索结果'。"
                t0 = time.time()
                old_out, old_err = sys.stdout, sys.stderr
                sys.stdout = io.StringIO()
                sys.stderr = io.StringIO()
                try:
                    answer = provider.chat([{"role": "user", "content": prompt}])
                finally:
                    sys.stdout, sys.stderr = old_out, old_err
                dt = time.time() - t0
                if answer:
                    print(f"\n{C.GREEN}{C.BOLD}OpenLLM ▸{C.RESET} {answer}")
                print(f"{C.DIM}[{dt:.1f}s]{C.RESET}")
            else:
                print(f"\n{C.DIM}搜索无结果{C.RESET}")
        except Exception as e:
            print(f"\n{C.RED}搜索失败: {e}{C.RESET}")

    def _handle_command(self, cmd_line: str):
        parts = cmd_line.strip().split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in ("exit", "quit", "q"):
            return self.do_exit("")

        elif cmd == "status":
            print(f"\n{C.BOLD}═══ Agent状态 ═══{C.RESET}")
            a = self.agent
            print(f"  模型: {a.octopus.left.provider.model}")
            print(f"  研究: {len(a.research.engine.hypotheses)}个假说, "
                  f"{len(a.research.engine.experiments)}个实验")
            print(f"  Session: {len(a.session.turns)}个Turn")

        elif cmd == "research":
            status = self.agent.research.status()
            es = status["engine_summary"]
            print(f"\n{C.BOLD}═══ 研究状态 ═══{C.RESET}")
            print(f"  假说: {es['total_hypotheses']}个")
            for h in status["engine_summary"].get("proven", []):
                print(f"    {C.GREEN}✅{C.RESET} [{h['id']}] {h['claim']}")
            for h in status["engine_summary"].get("disproven", []):
                print(f"    {C.RED}❌{C.RESET} [{h['id']}] {h['claim']}")
            print(f"  实验: {es['total_experiments']}个")
            print(f"  论文: {status['knowledge_summary']['total_papers']}篇")
            print(f"  日志: {len(status['recent_log'])}条")

        elif cmd == "hypothesis" and args:
            claim = " ".join(args)
            h = self.agent.research.hypothesize(claim, f"待验证: {claim}")
            print(f"  {C.GREEN}✅{C.RESET} 假说已注册: [{h.id}] {h.claim}")

        elif cmd == "search" and args:
            query = " ".join(args)
            print(f"\n{C.DIM}搜索中...{C.RESET}", end="", flush=True)
            try:
                import subprocess
                result = subprocess.run(
                    ["python3", "/home/zcs/search-engine/hermes_search.py",
                     query, "--sources", "web_search,arxiv,wikipedia", "--max", "5"],
                    capture_output=True, text=True, timeout=30,
                )
                output = result.stdout.strip()
                if output:
                    print(f"\n{C.GREEN}{output}{C.RESET}")
                else:
                    print(f"\n{C.DIM}无结果{C.RESET}")
            except Exception as e:
                print(f"\n{C.RED}搜索失败: {e}{C.RESET}")

        elif cmd == "help":
            print(f"""
{C.BOLD}可用命令:{C.RESET}
  直接打字    对话（Agent六体心跳处理）
  /status     查看Agent状态
  /research   查看研究循环状态
  /hypothesis <claim>  注册新假说
  /search <query>  搜索网页（通过Hermes）
  /exit       退出
""")

        else:
            print(f"  {C.YELLOW}未知命令: /{cmd}{C.RESET}  输入 /help 查看帮助")

    def do_exit(self, arg):
        print(f"\n{C.DIM}关闭Agent...{C.RESET}")
        return True

    def do_EOF(self, arg):
        return self.do_exit(arg)


def main():
    shell = AgentShell()
    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        print(f"\n\n{C.DIM}晚安。{C.RESET}")


if __name__ == "__main__":
    main()
