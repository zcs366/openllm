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


def _render_reply(result) -> str:
    """把引擎返回的文本呈现给用户。空则返回 ''。

    2026-09-15 医师接骨：此处原本还有一层「噪声过滤」，逻辑是遍历所有行、把
    ``last_clean_start`` 更新为**最后一条干净行**的下标，再从那里往后切。
    后果：任何多段回复都只剩最后一段。真实模型输出实测——

        介绍一下你自己：364 字 13 行 → 126 字 2 行（丢 65%）
        你好，老搭档  ： 71 字  5 行 →   9 字 1 行（丢 87%）

    更糟的是黑名单里有「搜索／写入／洞察／来源／关键／意识」等常用词，连
    ``openLLM`` 自己名字都在内——模型正常提到这些词的行会被整行删掉。而黑名单
    里的大写项（PLUR/jika）用 ``s.lower()`` 比较，永远匹配不上，属死条目。

    内部噪声的分离是上游的职责，不是 CLI 的：
      · 心跳路径 ``agent.run_once()`` 返回前已过 ``main_loop._clean_output()``
        （反向扫描、遇内部标记即停、黑名单精准）；
      · 快路径 ``provider.chat()`` 返回的就是模型正文，本来无噪声。
    因此这一层只做一件确定的事：整段被 ``` 围栏包裹时剥掉围栏（沿用原意，
    用于清掉 ```json/markdown 包装），其余原样透传。
    """
    if result is None:
        return ""
    text = str(result).strip()
    if not text:
        return ""
    if text.startswith("```") and "\n" in text:
        body = text.split("\n", 1)[1]
        if body.rstrip().endswith("```"):
            text = body.rstrip()[:-3].strip()
    return text


class AgentShell(cmd.Cmd):
    """Agent模式Shell——六体架构驱动。"""

    intro = banner()
    prompt = f"\n{C.CYAN}{C.BOLD}你 ▸{C.RESET} "

    def __init__(self):
        super().__init__()
        # DR-20260917-04：本会话对话历史（CLI 侧账本，跨轮存活）。
        # 每轮落一对 (用户输入, 最终回复)：快路径直接拼进 messages，
        # 心跳路径经 run_once(history=) 注入 prompt。`/clear` 清空。
        self._history: list[dict] = []
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
        print(f"{C.DIM}  六体就绪 · 研究引擎就绪 · 真模型在线{C.RESET}")
        print()

    def _hist(self) -> list:
        """会话历史账本（惰性初始化）。

        容错：`AgentShell.__new__` 造的轻量替身（既有行为级钉子用例的惯用手法）
        不跑 __init__，没有 `_history`——此处懒建，避免 default() 直接 AttributeError。
        惰性而非类属性：类级可变默认值会被所有实例共享并跨会话累积。
        """
        h = getattr(self, "_history", None)
        if h is None:
            h = self._history = []
        return h

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
            # 直接调LLM，不走心跳（但必须注入身份+工具清单，否则暴露底层模型名、
            # 且模型不知道自己有工具——DR-20260917-01 品尝师首诊刀①）
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            try:
                from ..core.isa_impl import FULL_TOOLS, BASE_TOOLS, TOOL_DESCRIPTIONS
                provider = self.agent.octopus.left.provider
                # getattr容错：测试注入的假provider可能无_available属性（降级BASE_TOOLS）
                _provider_ready = bool(getattr(provider, "_available", False))
                _tools = (list(BASE_TOOLS) + list(FULL_TOOLS[len(BASE_TOOLS):])
                          if _provider_ready else list(BASE_TOOLS))
                _tool_lines = "\n".join(
                    f"- {t}: {TOOL_DESCRIPTIONS.get(t, '参数见文档')}" for t in _tools)
                _system = (
                    "你是openLLM——一个自主Agent。工具即火，火即工具。"
                    "你必须以openLLM自称，不要以任何底层模型名称（如MiMo、Qwen、GPT）自称。"
                    "\n\n你可以使用以下工具：\n" + _tool_lines +
                    "\n如果任务需要读文件、写文件、搜索或执行命令，请在回复的最后一行输出：\n"
                    'TOOL_CALLS: {"tool_calls": [{"name": "工具名", "args": {"参数": "值"}}]}\n'
                    "如果只是聊天或回答问题，不要输出TOOL_CALLS行。"
                )
                messages = [{"role": "system", "content": _system}]
                # DR-20260917-04：带上本会话历史（最近10条）——
                # 修「刚说的话就忘」：旧实现每轮 message 列表都从零新建，
                # 短跟进语（"你的看法呢？"）全在快路径，模型看不到前文。
                messages.extend(self._hist()[-10:])
                messages.append({"role": "user", "content": line})
                result = provider.chat(messages)

                # 快路径工具执行回路：模型输出TOOL_CALLS → ISN执行 → 结果回喂再答。
                # （只注入不执行 = 许诺了手却不给手，模型的自白会原样漏给用户）
                # getattr容错：测试注入的假agent/isn可能缺方法——缺任一环则跳过回路。
                from ..iai.octopus import _LeftBrain
                from ..core.models import Decision
                _extract = getattr(self.agent.octopus.left, "_extract_tool_calls", None)
                _isn_exec = getattr(getattr(self.agent, "isn", None), "execute", None)
                for _round in range(3):  # 最多3轮工具调用，防失控
                    if _extract is None or _isn_exec is None:
                        break
                    calls = _extract(result)
                    if not calls:
                        break
                    if _isn_exec is not None:
                        tool_result = _isn_exec(
                            Decision(action="execute", approved=True,
                                     reason="fast-path", tool_calls=calls))
                    else:
                        tool_result = "[工具回路不可用]"
                    messages.append({"role": "assistant", "content": result})
                    messages.append({"role": "user", "content":
                                     f"工具执行结果：\n{tool_result}\n\n"
                                     "基于以上工具结果继续回答用户。如果还需要工具，"
                                     "按同样格式输出TOOL_CALLS；否则直接给出最终回答，"
                                     "不要输出TOOL_CALLS行。"})
                    result = provider.chat(messages)
                # 清掉可能残留的TOOL_CALLS行（最后一轮仍在索要工具时）
                result = _LeftBrain._TOOLCALL_RE.sub("", result or "").strip()
            finally:
                sys.stdout, sys.stderr = old_out, old_err
        else:
            # 复杂任务走完整心跳（抑制所有内部输出）
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            try:
                result = self.agent.run_once(line, history=self._hist()[-10:])
            finally:
                sys.stdout, sys.stderr = old_out, old_err

        dt = time.time() - t0

        # 2026-09-15 医师接骨：此处原有一层「噪声过滤」，会把多段回复砍到只剩
        # 最后一段（实测丢 65%~87%）。理由与替代方案见 _render_reply 的 docstring。
        text = _render_reply(result)
        if text:
            print(text)
        else:
            print(f"{C.DIM}(无输出){C.RESET}")

        # DR-20260917-04：对话历史落账（快/慢路径通吃）。空回复不入账——
        # 否则下一轮会把「(无输出)」当成自己的上一句话。
        if text:
            self._hist().append({"role": "user", "content": line})
            self._hist().append({"role": "assistant", "content": text})

        print(f"{C.DIM}[{dt:.1f}s]{C.RESET}")

    def onecmd(self, line: str):
        """cmd.Cmd 钩子——返回真值即终止输入循环。

        2026-09-15 医师接骨：`/exit` 原先只打印「关闭Agent...」但不退出，光标又
        回到 ``你 ▸`` 继续等输入，只能 Ctrl-D 才能走。根因是 ``default()`` 丢弃了
        ``_handle_command`` 的返回值，True 传不到 ``cmdloop``。
        终止信号放在 ``onecmd`` 而不是 ``default``：typeshed 把 ``Cmd.default``
        标注为 ``-> None``，而 ``Cmd.onecmd`` 是 ``-> bool | None``。
        """
        if line.strip().lower() in ("/exit", "/quit", "/q"):
            return self.do_exit("")
        return super().onecmd(line)

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
                # DR-20260917-04：搜索路径原来是一次性直调（无 system 身份、无历史），
                # 两个病一起补：① 缺 system role → 暴露底层模型名（旧陷阱#130）
                # ② 搜完即忘 → 搜完接一句短问，前文就断了。
                _msgs = [{"role": "system", "content": (
                    "你是openLLM——一个自主Agent。"
                    "你必须以openLLM自称，不要以任何底层模型名称（如MiMo、Qwen、GPT）自称。")}]
                _msgs.extend(self._hist()[-6:])
                _msgs.append({"role": "user", "content": prompt})
                old_out, old_err = sys.stdout, sys.stderr
                sys.stdout = io.StringIO()
                sys.stderr = io.StringIO()
                try:
                    answer = provider.chat(_msgs)
                finally:
                    sys.stdout, sys.stderr = old_out, old_err
                dt = time.time() - t0
                if answer:
                    print(f"\n{C.GREEN}{C.BOLD}OpenLLM ▸{C.RESET} {answer}")
                    # 搜索结果入账，供后续轮次接住
                    self._hist().append({"role": "user", "content": query})
                    self._hist().append({"role": "assistant", "content": str(answer)})
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

        elif cmd in ("clear", "reset", "new"):
            # DR-20260917-04：清空对话历史，起一段全新对话
            n = len(self._hist())
            self._hist().clear()
            print(f"  {C.GREEN}✅{C.RESET} 对话历史已清空（{n}条）——下一轮起是新对话")

        elif cmd == "status":
            print(f"\n{C.BOLD}═══ Agent状态 ═══{C.RESET}")
            a = self.agent
            print(f"  模型: {a.octopus.left.provider.model}")
            print(f"  研究: {len(a.research.engine.hypotheses)}个假说, "
                  f"{len(a.research.engine.experiments)}个实验")
            print(f"  Session: {len(a.session.turns)}个Turn")
            print(f"  对话历史: {len(self._hist())}条（/clear 清空）")

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

        elif cmd == "context":
            a = self.agent
            p = a.octopus.left.provider
            usage = getattr(p, '_last_usage', {}) or {}
            max_ctx = getattr(a.session, 'max_context_tokens', 100000) or 100000
            total = usage.get('total_tokens', 0)
            pct = total / max_ctx * 100 if max_ctx else 0
            print(f"\n{C.BOLD}═══ 上下文状态 ═══{C.RESET}")
            print(f"  模型: {p.model} · 窗口 {max_ctx:,} token")
            print(f"  最近一次调用:")
            print(f"    输入: {usage.get('prompt_tokens',0):,} token")
            print(f"    输出: {usage.get('completion_tokens',0):,} token")
            print(f"    缓存: {usage.get('cached_tokens',0):,} token")
            print(f"  单次占用: {pct:.1f}% ({total:,}/{max_ctx:,})")

        elif cmd == "model":
            config = self._load_config()
            providers = config.get("providers", {})
            default = config.get("default_provider", "mimo")
            names = list(providers.keys())
            if not args:
                print(f"\n{C.BOLD}═══ 可用模型 ═══{C.RESET}")
                for i, (name, p) in enumerate(providers.items(), 1):
                    mark = f"{C.GREEN} ← 当前{C.RESET}" if name == default else ""
                    model = p.get("model", "?")
                    print(f"  {i}. {name}: {model}{mark}")
                print(f"\n  切换: /model <编号> 或 /model <名称>")
            else:
                target = args[0].lower()
                name = None
                # 1) 编号  2) provider名  3) 模型名模糊匹配
                if target.isdigit() and 1 <= int(target) <= len(names):
                    name = names[int(target) - 1]
                elif target in providers:
                    name = target
                else:
                    for pname, p in providers.items():
                        if target in str(p.get("model", "")).lower():
                            name = pname
                            break
                if name is None:
                    print(f"  {C.RED}未知: {target}{C.RESET}  用 /model 查看列表")
                else:
                    config["default_provider"] = name
                    self._save_config(config)
                    self._reinit_provider(name)
                    model = providers[name].get("model", "")
                    print(f"  {C.GREEN}✅ 已切换到 {name} ({model}){C.RESET}")

        elif cmd == "help":
            print(f"""
{C.BOLD}可用命令:{C.RESET}
  直接打字    对话（Agent六体心跳处理）
  /model      查看/切换模型（/model <名称> 切换）
  /context    查看上下文状态（输入/输出/缓存/占用比例）
  /status     查看Agent状态
  /research   查看研究循环状态
  /hypothesis <claim>  注册新假说
  /search <query>  搜索网页（通过Hermes）
  /exit       退出
""")

        else:
            print(f"  {C.YELLOW}未知命令: /{cmd}{C.RESET}  输入 /help 查看帮助")

    def _load_config(self) -> dict:
        cfg_path = Path.home() / ".openllm" / "config.json"
        if cfg_path.exists():
            try:
                with open(cfg_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_config(self, config: dict) -> None:
        cfg_path = Path.home() / ".openllm" / "config.json"
        with open(cfg_path, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def _reinit_provider(self, name: str) -> None:
        from openllm.core.provider_impl import LLMProvider
        self.agent.octopus.left.provider = LLMProvider(provider_name=name)
        self.agent.octopus.right.provider = LLMProvider(provider_name=name)

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
