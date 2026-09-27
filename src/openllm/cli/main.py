"""
OpenLLM CLI v0.3 — Agent模式Shell。

从OpenLLMEngine升级为Agent（六体架构）。
"""
import cmd
import io
import os
import re
import sys
import json
import time
from pathlib import Path

from difflib import SequenceMatcher
from ..core.main_loop import Agent
from .activity import PhaseWatcher, format_phase, turn_account
from .render import render_answer


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

    # DR-20260927-04 视觉现代化：细框 + 品牌色 + 暗色层次（原双行实心框
    # 是 Win98 气质）。内容不变（模型/六体/命令），只换皮。
    return f"""
{C.DIM}╭──────────────────────────────────────────────╮{C.RESET}
{C.DIM}│{C.RESET}  {C.BOLD}{C.CYAN}◆ openLLM{C.RESET}  {C.DIM}v0.3 · Agent{C.RESET}                       {C.DIM}│{C.RESET}
{C.DIM}│{C.RESET}  {C.DIM}模型{C.RESET} {C.GREEN}{model}{C.RESET}                                {C.DIM}│{C.RESET}
{C.DIM}│{C.RESET}  {C.DIM}六体{C.RESET} {C.CYAN}IAI·IAX·ISA·IOS·ISN·IKO{C.RESET} {C.DIM}+ 研究引擎{C.RESET}  {C.DIM}│{C.RESET}
{C.DIM}╰──────────────────────────────────────────────╯{C.RESET}
{C.DIM}  /help 命令 · /web 会话浏览器 · 直接打字对话{C.RESET}
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


# ── P0.7 输入卫生 ──
_VACUOUS_RE = re.compile(
    r"[\"\'「」『』【】（）\(\)\[\]\{\}《》〈〉\-\s,\.。、！!？?；;：:…]+"
)

def _is_vacuous_input(line: str) -> bool:
    """检测空壳输入：剥除成对/孤立引号、空白、标点后，可见字符 <2。

    覆盖场景：
    - "" / '' / 「」 / 「」（空引号）
    - "，。" / "？！"（纯标点）
    - "   "（纯空白）
    - 两个引号之间无内容（模型从历史捞旧文档背诵续写的触发器）
    """
    s = line.strip()
    s = _VACUOUS_RE.sub("", s)
    return len(s) < 2


def _is_echo_of_recent(line: str, history: list, threshold: float = 0.6) -> bool:
    """回声检测：用户输入与最近 N 条 assistant 回复的相似度超过阈值。

    覆盖场景：
    - 用户把 openLLM 自己说过的话原样放回（"自问自答"循环）
    - 用户复制粘贴 assistant 回复中的句子
    - 弱模型把回放读成延续，导致自增殖
    """
    if not history:
        return False
    clean = line.strip()
    if len(clean) < 8:
        return False  # 太短的无法判回声
    # 取最近 6 条 assistant 消息
    assistant_msgs = [
        m["content"] for m in history
        if m.get("role") == "assistant"
    ][-6:]
    for msg in assistant_msgs:
        ratio = SequenceMatcher(None, clean, msg).ratio()
        if ratio >= threshold:
            return True
    return False


# ── P0.7 回声止血 ──
_VACUOUS_MSG = (
    "（没收到内容——引号里是空的。"
    "要把哪段话给我分析，连内容一起放进来。）"
)
_ECHO_MSG = (
    "（你发的这段话和我上几轮说过的几乎一样。"
    "要是想让我接着之前的话题往下说，直接说要做什么就行；"
    "要是有新问题，换个说法给我。）"
)

_ECHO_NOTE = (
    "【引用材料·不是你的新话·也不是用户对你的回应】"
    "用户把你此前说过的（或你读过的）一句原样放回，"
    "多半要你分析、评点或续写这段材料本身。"
    "请先回应用户可能的意图（要我对这段材料做什么？），"
    "不要当对话延续自动续写，更不要向用户认错。\n---\n用户输入："
)

# ── 刀⑦ 对话窗口扩容（2026-09-26 会话连续性三修）──
# 原两处调用点硬编码 [-10:]，agent 隔几轮就对自己前文失据。10→40。
# TODO(刀⑦-二期): 接compaction滚动摘要——RollingWindowStrategy
# (compaction_strategy.py) 吃 Message 对象、按 token 预算裁剪，与 CLI 的
# dict-history/条数窗口语义不匹配，无法零改造接入（见现状摘要 Q4）。
HISTORY_WINDOW = 40  # 模块级常量，两处共用

# ── 刀⑥-2 墓碑（空轮/报错也入账，问题不再从历史消失）──
# 墓碑文本是给下一轮的模型看的真状态陈述，不是给用户看的道歉。
TOMBSTONE_EMPTY = ("（本轮无输出——可能为provider瞬时故障。"
                   "你上一轮没有产出任何回答，用户的问题仍未解决。）")
TOMBSTONE_ERROR = ("（上轮provider报错，未能回答。"
                   "用户的问题仍未解决。）")


def _format_tool_trace(calls: list) -> str:
    """刀⑧：把本轮工具调用清单渲染为一行紧凑轨迹前缀（≤200字符）。

    输入元素兼容两种形状：
      快路径  {"name": str, "args": dict}
      心跳turn {"name": str, "arg": str}（agent_heartbeat add_action 摘要）
    空清单返回 ''（调用方不加前缀——优雅降级，图纸§四刀⑧）。
    """
    parts = []
    for tc in calls or []:
        if not isinstance(tc, dict):
            parts.append(str(tc)[:30])
            continue
        name = str(tc.get("name", "?"))
        if "arg" in tc:
            arg = str(tc.get("arg", ""))[:30]
        else:
            args = tc.get("args") or {}
            first = next(iter(args.values()), "") if isinstance(args, dict) and args else args
            arg = str(first)[:30]
        parts.append(f'{name}("{arg}")' if arg else name)
    if not parts:
        return ""
    line = "[本轮工具: " + " → ".join(parts) + "]"
    if len(line) > 200:
        line = line[:197] + "...]"
    return line


def _turn_tool_trace(turn) -> list:
    """刀⑧：从心跳 turn 对象的 actions 里取本轮工具调用清单。

    数据源 = agent_heartbeat._learn 在 add_action("execute", …,
    tool_calls=[…]) 附的 kwargs；老 turn / 无 execute / 无该键 → []。
    """
    trace = []
    for a in getattr(turn, "actions", None) or []:
        if isinstance(a, dict) and a.get("type") == "execute":
            for tc in a.get("tool_calls") or []:
                if isinstance(tc, dict) and tc.get("name"):
                    trace.append(tc)
    return trace


def _echo_annotated(line: str, history: list) -> str:
    """回声注释：检测用户是否原样放回了 assistant 的话，
    若是则加引导前缀（告诉 LLM 这是引用材料），否则原样返回。

    规格：不拦截不丢弃——回声是用户的合法用法。
    包装只进当轮 provider 输入，不进 _hist() 落账。
    """
    if _is_echo_of_recent(line, history):
        return _ECHO_NOTE + line
    return line


def _boot_model() -> str:
    """启动时的模型名（会话 meta 用）。失败返回空串。"""
    try:
        cfg_path = Path.home() / ".openllm" / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            return cfg.get("providers", {}).get(
                cfg.get("default_provider", ""), {}).get("model", "")
    except Exception:
        pass
    return ""


# ── DR-20260927-06 记忆写通道 ──
# 病灶：MemoryBus.write() 完整存在（六 provider/免疫/去重），读侧也在
# 心跳 build_context 接了——但对话路径无人调用 write，聊完即散
# （实证：0926 中国通史骨架只活在 output/ 文档里，ISA 查无此事）。
_IMPORTANT_HINTS = ("写", "分析", "总结", "方案", "设计", "实现", "为什么",
                    "原理", "架构", "骨架", "结论", "研究", "对比")
_GREETING_RE = re.compile(
    r"^\s*(你好|您好|hi|hello|嗨|早|晚安|下午好|晚上好|上午好|在吗|"
    r"醒来|睡|你好呀|嘿)[^。？！.?!]{0,12}[。？！.?!]?\s*$", re.IGNORECASE)


def _score_importance(line: str, reply: str) -> float:
    """轮次重要性启发式（0.2~0.8）。问候低、任务高、长答加权。"""
    if _GREETING_RE.match(line):
        return 0.2
    score = 0.45
    if any(k in line for k in _IMPORTANT_HINTS):
        score += 0.15
    if len(reply) > 400:
        score += 0.1
    if len(line) > 60:
        score += 0.1
    return min(0.8, score)


def _bus_of(agent):
    """取 Agent 的 MemoryBus（isa 懒加载）；stub agent 无此物 → None。"""
    isa = getattr(agent, "isa", None)
    if isa is None:
        return None
    getter = getattr(isa, "_get_memory_bus", None)
    if getter is None:
        return None
    try:
        return getter()
    except Exception:
        return None


class AgentShell(cmd.Cmd):
    """Agent模式Shell——六体架构驱动。"""

    intro = banner()
    prompt = f"\n{C.CYAN}{C.BOLD}你 ▸{C.RESET} "

    # DR-20260927-01 命令白名单：只有这些 /词 会被当指令执行。
    # 白名单外的 /xxx 在 default() 里当普通消息发模型——粘贴的
    # "/openllm: …" 之类引用文字不再误触发 _handle_command 或
    # cmd.Cmd 内置命令（EOF/exit 等裸词同理被 default 的非 / 分支
    # 正常送模型）。与 .cli.repl 的粘贴护栏共用一份。
    KNOWN_COMMANDS = {
        "exit", "quit", "q",          # 退出（onecmd 拦截 + 此处兜底）
        "clear", "reset", "new",      # 清对话历史
        "status", "research", "context", "model",
        "hypothesis", "search", "help", "web", "rebuild",
    }

    def __init__(self):
        super().__init__()
        # DR-20260917-04：本会话对话历史（CLI 侧账本，跨轮存活）。
        # 每轮落一对 (用户输入, 最终回复)：快路径直接拼进 messages，
        # 心跳路径经 run_once(history=) 注入 prompt。`/clear` 清空。
        self._history: list[dict] = []
        # ── DR-20260927-02 会话落盘：TUI 每轮自动写 ~/.openllm/sessions/ ──
        # Session Web UI（cli/webui.py）的数据源。/clear 轮换新文件（UI 分段）。
        self._session_file = None
        try:
            from .webui import start_session_file
            self._session_file = start_session_file(model=_boot_model())
        except Exception:
            self._session_file = None  # 落盘失败不阻塞 TUI
        # DR-20260927-07 DSH 只追加事件日志（重建任何 session 的真相源）
        self._dsh = None
        try:
            from .dsh import DshLogger
            self._dsh = DshLogger(model=_boot_model())
        except Exception:
            self._dsh = None
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
        # ── P0.7 输入卫生 + 回声检测 ──
        if _is_vacuous_input(line):
            print(_VACUOUS_MSG)
            return
        # ── DR-20260927-01 命令白名单：/ 前缀先过白名单 ──
        # 白名单内的 /cmd 走 _handle_command；白名单外的 /xxx
        # （典型：粘贴的引用文字 "/openllm: …"）当普通消息发模型。
        # 多行粘贴块即便首词像命令也不执行（is_suspicious_paste 的
        # 护栏语义，导入失败时降级为"首词在白名单即执行"）。
        if line.startswith("/"):
            try:
                from .repl import is_suspicious_paste
                _paste_guard = is_suspicious_paste(line, self.KNOWN_COMMANDS)
            except Exception:
                _paste_guard = False
            if not _paste_guard:
                self._handle_command(line[1:])
                return
            # 可疑粘贴：落到底部统一按普通消息处理（不 return）

        import io
        t0 = time.time()
        is_heartbeat = False
        _captured_turn = None
        _fast_tool_trace: list = []  # 刀⑧: 快路径本轮工具清单（心跳走 turn.actions）
        # DR-20260927-07：DSH 事件流——turn/start 与 user/message 先落账
        # （模型可见即已记录：无论后续成败，这轮发生过是事实）
        _dsh = getattr(self, "_dsh", None)
        if _dsh is not None:
            try:
                _dsh.turn_start(line)
                _dsh.user_message(line)
            except Exception:
                pass
        print(f"\n{C.GREEN}{C.BOLD}OpenLLM ▸{C.RESET} ", end="", flush=True)

        # 检测搜索意图（DR-20260927-01：仅单行手打输入参与嗅探——
        # 多行粘贴块是发给模型的材料，含"搜"字不得被劫持进 _do_search）
        search_keywords = ["搜", "搜一搜", "上网", "查一查", "找一下", "搜索", "最新的", "最新"]
        needs_search = "\n" not in line and any(kw in line for kw in search_keywords)
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

        # DR-20260927-04 活性指示器：思考/工具期间行内转轮+秒数，
        # 黑咕咚盲等的体验层根治。封印期写真实流（旁路 StringIO）。
        # 与活动流二选一：OPENLLM_ACTIVITY=1 显式开流水时 spinner 让位。
        _spinner = None
        if (os.environ.get("OPENLLM_SPINNER", "1").lower() not in ("0", "off")
                and os.environ.get("OPENLLM_ACTIVITY", "").lower()
                not in ("1", "on", "yes", "true")):
            try:
                from .spinner import Spinner
                # 此刻 sys.stdout 尚未封印（封印在分支内），Spinner 在
                # __init__ 持有真实流引用——封印后它仍写真屏，恰好旁路。
                _spinner = Spinner("思考中", stream=sys.stdout).start()
            except Exception:
                _spinner = None

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
                    "重要：如果你打算接下来读文件/搜索/执行任何操作，必须立即在本回复最后一行输出 TOOL_CALLS。"
                    "用'我再…''继续…''接下来…''让我…'等散文描述意图而不给TOOL_CALLS行是无效的——"
                    "系统只认 TOOL_CALLS 行，不会执行散文。若任务已完成，直接给最终回答。"
                    "引用规则：用户可能把你的原话或你读过的材料放回给你。"
                    "遇到这种情况，默认任务是【分析这段材料】而非延续对话，除非用户明说'继续'。"
                    "\n语气与分寸（DR-20260927-05）："
                    "像老搭档、同行者一样说话——自然、柔和、有温度，不卑不亢。"
                    "问候/寒暄/闲聊类输入（醒来、早晚安好、闲话）：像老朋友一样回应，"
                    "一两句人话即可；不要催任务、不要列'要查的/要读的/要写的'清单、"
                    "不要复读'有什么需要帮助的'式服务口号——那是客服腔，不是你。"
                    "每次开口措辞要有变化，不重复自己上一次的问候模板。"
                    "只有用户真交代了任务，才谈工具与执行。"
                )
                messages = [{"role": "system", "content": _system}]
                # DR-20260917-04：带上本会话历史——
                # 修「刚说的话就忘」：旧实现每轮 message 列表都从零新建，
                # 短跟进语（"你的看法呢？"）全在快路径，模型看不到前文。
                # 刀⑦：窗口 10→40（模块级 HISTORY_WINDOW，与慢路径共用）。
                messages.extend(self._hist()[-HISTORY_WINDOW:])
                messages.append({"role": "user", "content": _echo_annotated(line, self._hist())})
                # DR-20260927-06 读侧：快路径也吃 MemoryBus 召回——
                # 长线话题（通史/项目）跨会话能接上前文。心跳路径
                # build_context 已接，此处补齐快路径。失败静默。
                try:
                    _bus = _bus_of(self.agent)
                    if _bus is not None:
                        from ..isa.memory_bus import Query as BusQuery
                        _recs = _bus.query(BusQuery(
                            text=line, top_k=3, token_budget=600,
                            min_importance=0.35))
                        if _recs:
                            _memo = "\n".join(
                                f"- {(r.content or '')[:150]}"
                                for r in _recs)
                            messages.insert(1, {
                                "role": "system",
                                "content": "相关历史记忆（ISA召回，"
                                           "供参考，不要复述）：\n" + _memo})
                except Exception:
                    pass
                # DR-20260927-03 断点A：chat 抛异常不再穿透炸场——转为错误正文，
                # 下面的通用错误显示（[LLM错误]→墓碑入账）统一接手。
                try:
                    result = provider.chat(messages)
                except Exception as _e:
                    result = f"[LLM错误]{str(_e)[:300]}"

                # 快路径工具执行回路：模型输出TOOL_CALLS → ISN执行 → 结果回喂再答。
                # （只注入不执行 = 许诺了手却不给手，模型的自白会原样漏给用户）
                # getattr容错：测试注入的假agent/isn可能缺方法——缺任一环则跳过回路。
                from ..iai.octopus import _LeftBrain
                from ..core.models import Decision
                _extract = getattr(self.agent.octopus.left, "_extract_tool_calls", None)
                _isn_exec = getattr(getattr(self.agent, "isn", None), "execute", None)
                _had_tool_calls = False
                _rounds_done = 0  # DR-20260927-03 断点B：实际执行了几轮工具
                for _round in range(3):  # 最多3轮工具调用，防失控
                    if _extract is None or _isn_exec is None:
                        break
                    calls = _extract(result)
                    if not calls:
                        break
                    _had_tool_calls = True
                    _rounds_done += 1
                    # 刀⑧: 记账本轮实际执行的工具（name+args 原样留）
                    for _tc in calls:
                        if isinstance(_tc, dict):
                            _fast_tool_trace.append(_tc)
                    # DR-20260927-03 进度可见性：执行前打一行进度到真实
                    # stdout——用户看见"它在干活"，不再盲等无声。
                    try:
                        _names = ",".join(
                            (c.get("name", "?") if isinstance(c, dict)
                             else str(c))[:24]
                            for c in calls)
                        # DR-20260927-04：与 spinner 共屏——先擦行→打进度→续刷
                        if _spinner is not None:
                            _spinner.update("工具中")
                            _spinner.pause()
                        print(f"\n{C.DIM}  ⚙ 工具第{_rounds_done}/3轮: "
                              f"{_names}{C.RESET}", file=sys.stdout, flush=True)
                        if _spinner is not None:
                            _spinner.resume()
                        if _dsh is not None:
                            try:
                                _dsh.tool_calls(calls)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    if _isn_exec is not None:
                        try:
                            tool_result = _isn_exec(
                                Decision(action="execute", approved=True,
                                         reason="fast-path", tool_calls=calls))
                        except Exception as _te:
                            tool_result = f"[工具执行异常] {str(_te)[:200]}"
                    else:
                        tool_result = "[工具回路不可用]"
                    messages.append({"role": "assistant", "content": result})
                    messages.append({"role": "user", "content":
                                     f"工具执行结果：\n{tool_result}\n\n"
                                     "基于以上工具结果继续回答用户。如果还需要工具，"
                                     "按同样格式输出TOOL_CALLS；否则直接给出最终回答，"
                                     "不要输出TOOL_CALLS行。"})
                    try:
                        result = provider.chat(messages)
                    except Exception as _e:
                        result = f"[LLM错误]{str(_e)[:300]}"
                        break

                # DR-20260927-03 断点B：3轮上限打满仍在要工具 → 不再无声无息，
                # 给一句话交代（正文保留时附在尾部；正文空时本身就是交代）。
                if (_rounds_done >= 3
                        and _LeftBrain._TOOLCALL_RE.search(result or "")):
                    _body = _LeftBrain._TOOLCALL_RE.sub("", result or "").strip()
                    _told = ("[回合上限] 工具连续使用了3轮还没收尾——为防失控先停。"
                             + ("已完成的部分：\n" + _body if _body else
                                "把剩下的活拆小，或告诉我接着哪一步继续。"))
                    result = _told

                # P0.6 收线前意图追问：模型用散文表达继续意图但没给 TOOL_CALLS
                # 时，追加一枪追问让它吐出具体工具调用。开关 OPENLLM_NUDGE=0/off 关闭。
                if (_had_tool_calls
                    and os.environ.get("OPENLLM_NUDGE", "").lower() not in ("0", "off")):
                    CONTINUE_INTENT_RE = re.compile(
                        r"(我再|还要|接下来|下一步|让我先|继续读|继续看|补读|稍等|马上给|这就去)")
                    _raw_for_nudge = result or ""
                    _text_for_nudge = _LeftBrain._TOOLCALL_RE.sub("", _raw_for_nudge).strip()
                    if _text_for_nudge and CONTINUE_INTENT_RE.search(_text_for_nudge):
                        try:
                            messages.append({"role": "assistant", "content": _raw_for_nudge})
                            messages.append({"role": "user", "content":
                                "检测到你想继续操作。请输出具体的 TOOL_CALLS: {...} 行执行它；"
                                "如果其实已经完成，请直接给出最终回答，不要提及继续。"})
                            result = provider.chat(messages)
                        except Exception:
                            pass  # 追问失败保原 result

                # 清掉可能残留的TOOL_CALLS行（最后一轮仍在索要工具时）
                result = _LeftBrain._TOOLCALL_RE.sub("", result or "").strip()
            finally:
                if _spinner is not None:
                    _spinner.stop()
                sys.stdout, sys.stderr = old_out, old_err
        else:
            # 复杂任务走完整心跳（抑制所有内部输出）
            is_heartbeat = True
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()

            # 活动流观察器：旁路轮询 phase_metrics，不碰 stdout 封印
            # （DR-20260927-04：与 spinner 二选一已在上游让位判定，
            #   双保险：真要起 watcher 时先停 spinner 防共屏打架）
            _watcher = None
            _captured_turn = None
            if os.environ.get("OPENLLM_ACTIVITY", "").lower() not in ("0", "off"):
                if _spinner is not None:
                    _spinner.stop()
                    _spinner = None
            if os.environ.get("OPENLLM_ACTIVITY", "").lower() not in ("0", "off"):
                _real_stdout = old_out  # capture before StringIO
                def _on_phase_entry(e):
                    nonlocal _captured_turn
                    t = getattr(self.agent.session, "active_turn", None)
                    if t is not None:
                        _captured_turn = t
                    print(f"{C.DIM}{format_phase(e)}{C.RESET}", file=_real_stdout, flush=True)
                _watcher = PhaseWatcher(
                    get_turn=lambda: getattr(self.agent.session, "active_turn", None),
                    on_entry=_on_phase_entry,
                )
                _watcher.start()
            try:
                # 刀⑦：慢路径窗口 10→40，与快路径共用 HISTORY_WINDOW
                result = self.agent.run_once(_echo_annotated(line, self._hist()), history=self._hist()[-HISTORY_WINDOW:])
            finally:
                if _watcher is not None:
                    _watcher.stop()
                    # 兜底抓 turn：残余条目可能在 stop 时 drain 补印（那时
                    # active_turn 已清），watcher 公开账本里必有最后一个 turn。
                    _captured_turn = _watcher.last_turn or _captured_turn
                if _spinner is not None:
                    _spinner.stop()
                sys.stdout, sys.stderr = old_out, old_err

        dt = time.time() - t0

        # 2026-09-15 医师接骨：此处原有一层「噪声过滤」，会把多段回复砍到只剩
        # 最后一段（实测丢 65%~87%）。理由与替代方案见 _render_reply 的 docstring。
        text = _render_reply(result)

        # DR-20260917-04 → 刀⑥-2 墓碑入账（2026-09-26 会话连续性三修）：
        # 旧语义"空回复/报错整对不入账"让用户的提问从历史里消失——病根①。
        # 新语义：一切轮次都落一对；失败轮的 assistant 记"真状态陈述"（给
        # 下一轮的模型看，不是给用户道歉）。原DR担心"(无输出)被当成自己的话"
        # ——墓碑文本语义自足，恰好同时修复该担心与"问题消失"两个病。
        # 刀⑧：正常正文附本轮工具轨迹前缀（快路径本地记账/心跳从 turn.actions
        # 取；拿不到清单不加前缀=优雅降级）。墓碑对不带前缀。
        if text and not text.startswith("[LLM错误]"):
            _trace_calls = (_fast_tool_trace if not is_heartbeat
                            else _turn_tool_trace(_captured_turn))
            _trace_line = _format_tool_trace(_trace_calls)
            booked = (_trace_line + "\n" + text) if _trace_line else text
            if _trace_line:
                print(f"{C.DIM}{_trace_line}{C.RESET}")
            render_answer(text)
        elif text:  # [LLM错误]
            booked = TOMBSTONE_ERROR
            render_answer(text)  # 错误原文照常显示给用户
        else:
            booked = TOMBSTONE_EMPTY
            print(f"{C.DIM}(无输出){C.RESET}")

        self._hist().append({"role": "user", "content": line})
        self._hist().append({"role": "assistant", "content": booked})
        # DR-20260927-02：同步落盘（Web UI 数据源）；失败静默不阻塞对话
        try:
            from .webui import append_msg
            append_msg(self._session_file, "user", line)
            append_msg(self._session_file, "assistant", booked)
        except Exception:
            pass
        # DR-20260927-07：DSH 事件收尾——assistant 消息（含墓碑如实入账）
        # + turn/end。tool_rounds 从快路径计数取。
        if _dsh is not None:
            try:
                _dsh.assistant_message(booked)
                _dsh.turn_end(duration_ms=(time.time() - t0) * 1000,
                              status=("error" if text == "" or
                                      (text or "").startswith("[LLM错误]")
                                      else "ok"),
                              tool_rounds=len(_fast_tool_trace))
            except Exception:
                pass
        # DR-20260927-06：记忆写通道——非问候轮次沉淀进 MemoryBus
        # （真实任务轮才写；问候 0.2 分也会被过滤逻辑挡在门外）
        try:
            _bus = _bus_of(self.agent)
            if _bus is not None:
                from ..isa.memory_bus import WriteRequest
                _imp = _score_importance(line, text or "")
                if _imp >= 0.35:
                    _wr = _bus.write(WriteRequest(
                        content=f"用户: {line}\nopenLLM: {(text or '')[:600]}",
                        source="cli_dialogue",
                        record_type="dialogue",
                        importance=_imp,
                        tags=["cli", "对话"],
                        session_id=getattr(self, "_session_id_str", "")))
                    if _dsh is not None:
                        try:
                            _dsh.memory_write(
                                getattr(_wr, "record_id", ""),
                                _imp)
                        except Exception:
                            pass
        except Exception:
            pass  # 记忆写失败绝不阻塞对话

        if is_heartbeat and _captured_turn:
            acct = turn_account(_captured_turn)
            print(f"{C.DIM}{acct}{C.RESET}")
        else:
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
            # DR-20260927-02：轮换新会话文件（Web UI 上分段可见）
            try:
                from .webui import start_session_file
                self._session_file = start_session_file(
                    model=self.agent.octopus.left.provider.model)
            except Exception:
                pass
            # DR-20260927-07：DSH 记段边界 + 轮换新事件文件
            try:
                _old_dsh = getattr(self, "_dsh", None)
                if _old_dsh is not None:
                    _old_dsh.end("clear")
                    _old_dsh.close()
                from .dsh import DshLogger
                self._dsh = DshLogger(
                    model=self.agent.octopus.left.provider.model)
            except Exception:
                pass
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

        elif cmd == "web":
            # DR-20260927-02：Session Web UI（127.0.0.1，只读，守护线程）
            port = int(args[0]) if args else int(
                os.environ.get("OPENLLM_WEBUI_PORT", "8799"))
            from .webui import start_background
            url = start_background(port)
            print(f"  {C.GREEN}✅{C.RESET} Session Web UI: {url}"
                  f"  （浏览器打开；只在后台跑，/exit 一并退出）")

        elif cmd == "rebuild":
            # DR-20260927-07：重放事件日志重建任何 session
            # /rebuild            列出可重建的 session（新→旧，最近20）
            # /rebuild <id>       重建并回灌为当前对话历史（续聊）
            from .dsh import list_dsh_sessions, rebuild_history
            if not args:
                sess = list_dsh_sessions(limit=20)
                if not sess:
                    print(f"  {C.DIM}（暂无 DSH 事件日志）{C.RESET}")
                    return
                print(f"\n{C.BOLD}═══ 可重建的 Session ═══{C.RESET}")
                for s in sess:
                    mark = f" {C.GREEN}← 当前{C.RESET}" \
                        if self._dsh and s["id"] == self._dsh.session_id else ""
                    print(f"  {s['id']}  {s['events']}事件  "
                          f"{s['title'][:32]}{mark}")
                print(f"\n  {C.DIM}/rebuild <id> 重建并续聊{C.RESET}")
                return
            sid = args[0]
            pack = rebuild_history(sid)
            if pack is None:
                print(f"  {C.RED}重建失败: {sid} 不存在或 id 非法{C.RESET}")
                return
            self._hist().clear()
            self._hist().extend(pack["history"])
            if getattr(self, "_dsh", None) is not None:
                self._dsh.note(f"rebuilt from {sid}: "
                               f"{len(pack['history'])} messages")
            n_u = sum(1 for m in pack["history"]
                      if m["role"] == "user")
            print(f"  {C.GREEN}✅{C.RESET} 已重建 {sid}"
                  f"（{pack['model'] or '?'} · {n_u}轮对话 · "
                  f"{len(pack['turns'])} turn含工具记录）——"
                  f"历史已回灌，可直接续聊")

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
  /web [port] Session Web UI（浏览器看历史对话，127.0.0.1，只读）
  /rebuild [id]  列出/重建任意历史 session（DSH事件重放，回灌续聊）
  /exit       退出
  粘贴长文直接粘——多行/斜杠开头的粘贴内容按普通消息发送（DR-20260927-01）
  活动流默认开启（心跳过程可见），OPENLLM_ACTIVITY=0 关闭
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
        # DR-20260927-07：DSH session/end 收口
        _d = getattr(self, "_dsh", None)
        if _d is not None:
            _d.end("exit")
            _d.close()
        print(f"\n{C.DIM}关闭Agent...{C.RESET}")
        return True

    def do_EOF(self, arg):
        return self.do_exit(arg)


def main():
    shell = AgentShell()
    try:
        # DR-20260927-01：输入层优先 prompt_toolkit REPL（bracketed paste、
        # 退格不侵蚀历史、↑↓历史）；无 TTY 或缺 prompt_toolkit 时回落
        # cmd.Cmd.cmdloop，旧行为原样保留（测试管道全走降级链）。
        from .repl import repl_available, run_repl
        if repl_available():
            run_repl(shell)
        else:
            shell.cmdloop()
    except KeyboardInterrupt:
        print(f"\n\n{C.DIM}晚安。{C.RESET}")


if __name__ == "__main__":
    main()
