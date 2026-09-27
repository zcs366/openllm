from .degradation_trace import trace_degradation
"""extracted from main_loop.py"""
import json, os, re, time, uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional
from .models import *
class ISN:
    """工具执行"""
    
    def __init__(self):
        # 沙箱隔离
        from .sandbox import Sandbox
        self.sandbox = Sandbox()
        # 外部工具桥接
        from .tool_bridge import ExternalToolBridge
        self.bridge = ExternalToolBridge()
        self.bridge.scan()  # 自动发现工具
        # 注册真工具
        self.tools = {
            "read_file": self._read_file,
            "write_file": self._write_file,
            "search_files": self._search_files,
            "terminal": self._terminal,
        }
        # ── 网络工具（net.py）──
        _net_ok = False
        try:
            from ..net import web_search as _ws, web_fetch as _wf
            self.tools["web_search"] = self._web_search
            self.tools["web_fetch"] = self._web_fetch
            _net_ok = True
        except Exception:
            pass  # net 不可用 → 降级不注册
        # [进化] 接入ToolRegistry的verify-before-complete机制
        try:
            from ..tools.executor import ToolRegistry, ToolResult
            self._tool_registry = ToolRegistry()
            self._tool_registry.register("write_file", self._write_file_real,
                                          description="写入文件(带验证)")
            self._tool_registry.register("terminal", self._terminal_real,
                                          description="执行Shell命令(带验证)")
            # 设置verify hook：写操作前检查沙箱+治理规则
            self._tool_registry.set_verify_hook(self._verify_before_complete)
            self._has_verify = True
        except Exception:
            self._tool_registry = None
            self._has_verify = False
        print(f"  ISN 工具执行就绪 · {len(self.tools)}个工具 · 沙箱={len(self.sandbox.allowed)}个允许路径 · 外部工具={len(self.bridge.discovered)}个 · verify={'ON' if self._has_verify else 'OFF'} · 网络={'ON' if _net_ok else 'OFF'}")
        # 加载已学习技能(血管#3: ios→isn)
        self.learned_skills: list[dict] = []
        self._load_learned_skills()
    
    def _load_learned_skills(self):
        """加载已持久化的学习技能"""
        skill_path = Path.home() / ".openllm" / "output" / "isn" / "learned_skills.jsonl"
        if skill_path.exists():
            import json as _json
            with open(skill_path) as f:
                for line in f:
                    try:
                        self.learned_skills.append(_json.loads(line))
                    except:
                        pass
            if self.learned_skills:
                print(f"  ISN 已加载 {len(self.learned_skills)} 个学习技能")
        
        # 连接3: 加载Self-Harness提案(approved状态的自动应用)
        prop_path = Path.home() / ".openllm" / "output" / "ios" / "harness_proposals.jsonl"
        if prop_path.exists():
            import json as _json
            approved = 0
            with open(prop_path) as f:
                for line in f:
                    try:
                        prop = _json.loads(line)
                        if prop.get("status") == "approved":
                            self.learned_skills.append({
                                "name": f"harness_proposal_{prop.get('mechanism', 'unknown')}",
                                "mechanism": prop.get("mechanism"),
                                "proposal": prop.get("proposal"),
                                "source": "harness_proposal",
                            })
                            approved += 1
                    except:
                        pass
            if approved:
                print(f"  ISN 已加载 {approved} 个approved的Self-Harness提案")
        
        # 连接4: 加载治理转换引擎产出的规则(P0: arXiv:2607.01087)
        gov_rules_path = Path.home() / ".openllm" / "output" / "ios" / "governance_rules.jsonl"
        if gov_rules_path.exists():
            import json as _json
            gov_count = 0
            with open(gov_rules_path) as f:
                for line in f:
                    try:
                        rule = _json.loads(line)
                        if rule.get("status") == "active":
                            self.learned_skills.append({
                                "name": f"governance_rule_{rule.get('rule_id', 'unknown')}",
                                "mechanism": rule.get("source_mechanism"),
                                "proposal": rule.get("description"),
                                "source": "governance_engine",
                                "condition": rule.get("condition"),
                                "action": rule.get("action"),
                            })
                            gov_count += 1
                    except:
                        pass
            if gov_count:
                print(f"  ISN 已加载 {gov_count} 条治理引擎规则")
    
    def execute(self, decision: Decision) -> ActionResult:
        """Phase 7: 执行"""
        if not decision.approved:
            return ActionResult(success=False, output=decision.reason)
        t0 = time.time()
        try:
            # 如果有tool_calls，执行对应工具
            output = self._execute_action(decision)
            
            # [进化] 发送skill_created信号
            try:
                from .isn_signal import skill_created
                tool_calls = getattr(decision, 'tool_calls', None) or []
                for tc in tool_calls:
                    name = tc.get("name", "unknown") if isinstance(tc, dict) else str(tc)
                    skill_created(name, "")
            except Exception:
                pass
            
            return ActionResult(success=True, output=output, duration_ms=(time.time()-t0)*1000)
        except Exception as e:
            return ActionResult(success=False, error=str(e), duration_ms=(time.time()-t0)*1000)
    
    def _execute_action(self, decision: Decision) -> str:
        """实际执行：解析decision中的tool_calls并调用对应工具"""
        # 从decision中尝试提取工具调用
        # 如果decision有tool_calls字段 → 逐个执行
        tool_calls = getattr(decision, 'tool_calls', None) or []
        
        if not tool_calls:
            # 无工具调用 → M0回显
            return f"已执行: {decision.reason}"
        
        results = []
        for tc in tool_calls:
            tool_name = tc.get("name", "") if isinstance(tc, dict) else str(tc)
            tool_args = tc.get("args", {}) if isinstance(tc, dict) else {}
            
            if tool_name in self.tools:
                try:
                    # 调用工具(传递args字典)
                    result = self.tools[tool_name](**tool_args)
                except TypeError:
                    # 如果工具不接受**kwargs，尝试单参数调用
                    result = self.tools[tool_name](str(tool_args))
                results.append(f"[{tool_name}] {result}")
            else:
                results.append(f"[未知工具] {tool_name}")
        
        return "\n".join(results) if results else f"已执行: {decision.reason}"
    
    def _read_file(self, path: str) -> str:
        """读取文件内容(沙箱检查·先解析再检查防路径穿越)"""
        # 先解析路径(防../../etc/passwd穿越)；盘符路径 I:\ 规范化为 /mnt/i/
        p = self.sandbox._normalize_path(path)
        # 再检查沙箱
        if not self.sandbox.check_path(str(p), "read"):
            return f"[沙箱拒绝] {self.sandbox.deny_reason(str(p))}"
        if not p.exists():
            return f"[错误] 文件不存在: {path}"
        if p.is_dir():
            return f"[错误] 是目录不是文件: {path}（列目录请用 terminal 的 ls）"
        if p.stat().st_size > 100000:
            return f"[错误] 文件过大: {path}"
        return p.read_text(encoding="utf-8", errors="replace")[:5000]
    
    def _write_file(self, path: str, content: str) -> str:
        """写入文件(沙箱检查·先解析再检查防路径穿越)"""
        # 先解析路径(防穿越)；盘符路径规范化，与check_path同源
        p = self.sandbox._normalize_path(path)
        # 再检查沙箱
        if not self.sandbox.check_path(str(p), "write"):
            return f"[沙箱拒绝] {self.sandbox.deny_reason(str(p))}"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"[写入成功] {path} ({len(content)}字符)"
    
    # ── DR-20260917-02：危险命令判定（词元级·命令位）──
    # 子串匹配曾误杀含"rm"的正常词（hermes/pseudo等）。词元匹配的边界：
    # "grep rm file"中rm是搜索模式不是命令，所以只查每个命令段的首词。
    _DANGEROUS_CMDS = {"rm", "sudo", "dd", "mkfs", "shred"}

    @staticmethod
    def _find_dangerous_cmd(command: str) -> Optional[str]:
        """返回命令中第一个危险命令词元；安全则返回None。
        按命令分隔符切段，只查每段首词（命令位），不误杀参数位/模式位。"""
        import shlex
        import re as _re
        # 按命令边界切段（; | && || 换行）
        _segments = _re.split(r';|\|\||\||\n|&&', command)
        for seg in _segments:
            try:
                toks = shlex.split(seg)
            except ValueError:
                toks = seg.split()
            if toks and toks[0].split("/")[-1] in ISN._DANGEROUS_CMDS:
                return toks[0]
        return None

    # ── DR-20260917-03：命令意图按「命令段」判定 ──
    # 旧实现取整条命令的首词定 op，一刀切作用于命令内所有路径，产生三类误拒：
    #   ① `cd <只读目录> && ls` —— 首词 cd 被判成写意图，只读目录被按写检查→误拒；
    #   ② `mkdir /tmp/openllm/x && ls /mnt/i/openllm/src` —— 读路径被按写检查→误拒；
    #   ③ 任何"读+写"混合命令中的只读路径，一律被按写检查→误拒。
    # 修法：命令按边界切段，逐段定意图，只检查该段内的路径。安全语义不变——
    # 写意图段（tee/cp/sed -i/python/…）的路径仍须过写白名单；重定向目标单独强查。
    _READONLY_CMDS = {
        "ls", "cat", "head", "tail", "grep", "find", "wc", "file", "stat",
        "du", "df", "ps", "pwd", "which", "date", "echo", "tree", "rg",
        "less", "uname", "env", "whoami", "id",
    }
    # 中性命令：不改文件系统内容，按读检查（cd 只换当前目录，不动内容）
    _NEUTRAL_CMDS = {"cd", "pushd", "popd", "true", ":", "export", "set", "unset"}
    _GIT_READ_SUB = {"status", "log", "diff", "show", "branch", "remote",
                     "rev-parse", "ls-files", "ls-remote"}
    _GIT_VALUE_OPTS = {"-C", "-c", "--git-dir", "--work-tree",
                       "--namespace", "--exec-path"}
    _SEG_SPLIT_RE = re.compile(r";|\|\||\||\n|&&")

    @classmethod
    def _segment_op(cls, segment: str) -> str:
        """判定单个命令段的读写意图：只读/中性命令 → read，其余 → write。

        值感知解析：git 的 -C/--git-dir 等选项后一个词元是「值」不是子命令，
        必须先跳过它再找真子命令（否则 `git -C /path status` 会被判成写意图）。
        """
        import shlex
        try:
            toks = shlex.split(segment)
        except ValueError:
            toks = segment.split()
        if not toks:
            return "read"
        verb = toks[0].split("/")[-1]
        if verb in cls._READONLY_CMDS or verb in cls._NEUTRAL_CMDS:
            return "read"
        if verb == "git":
            _skip = False
            for t in toks[1:]:
                if _skip:
                    _skip = False
                    continue
                if t.startswith("-"):
                    _skip = t in cls._GIT_VALUE_OPTS
                    continue
                return "read" if t in cls._GIT_READ_SUB else "write"
        return "write"

    def _search_files(self, pattern: str) -> str:
        """按文件名搜索。

        DR-20260917-03：搜索范围 = 沙箱可读根（读写白名单 ∪ 只读白名单）。
        旧实现只搜 ~ 且 maxdepth=4——研究资料库（/mnt/i/hermes/output）里的
        文件根本搜不到，"帮我找找资料"必然空手。范围与沙箱权限同源：能读的才搜。
        """
        import subprocess
        roots = list(self.sandbox.allowed) + list(self.sandbox.readonly)
        chunks = []
        # 已知局限（2026-09-24 放权后实测）：白名单含 589G 的 /mnt/h——全盘 find
        # 在 10s timeout 内跑不完，大根静默跳过（显式路径 read_file 不受影响，
        # 只有"盲搜全盘"受限）。待搜索走增量索引（ISA 章鱼索引）后此路退役。
        for root in roots:
            try:
                if not root.exists():
                    continue
                result = subprocess.run(
                    ["find", str(root), "-maxdepth", "6", "-name", pattern],
                    capture_output=True, text=True, timeout=10)
                if result.stdout.strip():
                    chunks.append(result.stdout.strip())
            except Exception:
                continue
        text = "\n".join(chunks)
        return text[:2000] if text else "[未找到]"

    # ── 网络工具薄包装（转调 net.py）──
    def _web_search(self, query: str = "", max_results: int = 5) -> str:
        """搜索引擎查资料。签名容错：支持 query=kwargs 或单参两条路。"""
        from ..net import web_search
        return web_search(query, max_results)

    def _web_fetch(self, url: str = "") -> str:
        """抓取网页正文。签名容错：支持 url=kwargs 或单参两条路。"""
        from ..net import web_fetch
        return web_fetch(url)
    
    def _terminal(self, command: str) -> str:
        """执行shell命令。高风险——需要沙箱检查。"""
        import subprocess
        import re
        import shlex

        # DR-20260917-02：危险命令改词元级·命令位判定（防误杀hermes等正常词，
        # 同时不放过 "grep rm x" 参数位之外任何真命令位上的rm）
        _bad = self._find_dangerous_cmd(command)
        if _bad:
            return f"[拦截] 危险命令: {_bad}"

        # DR-20260917-03：意图改逐段判定（见 _segment_op）。不再用整条命令
        # 首词一刀切——那是 cd 只读目录/混合读写命令误拒的根。
        # 重定向目标检查（> >> 写入路径必须过写白名单，防绕道写）
        for m in re.finditer(r'(?<=[\s;])>>?\s*([^\s;&|]+)', command):
            if not self.sandbox.check_path(m.group(1), "write"):
                return f"[沙箱拒绝] 重定向目标越界: {m.group(1)}"

        # 沙箱检查：按「命令段」各自的意图检查段内文件路径
        path_patterns = [
            r'(?<=\s)(/[^\s]+)',  # 绝对路径
            r'(?<=["\'])(/[^\s"\']+)(?=["\'])',  # 引号内的路径
        ]
        for _seg in self._SEG_SPLIT_RE.split(command):
            _op = self._segment_op(_seg)
            for pattern in path_patterns:
                for p in re.findall(pattern, _seg):
                    if not self.sandbox.check_path(p, _op):
                        return f"[沙箱拒绝] {self.sandbox.deny_reason(p)}"

        try:
            result = subprocess.run(command, shell=True, capture_output=True,
                                   text=True, timeout=30)
            return result.stdout[:3000] or result.stderr[:1000]
        except Exception as e:
            return f"[执行失败] {e}"
    
    # ── [进化] ToolRegistry verify hooks ──
    
    def _write_file_real(self, path: str, content: str) -> str:
        """真实写入——ToolRegistry调用此方法"""
        p = self.sandbox._normalize_path(path)
        if not self.sandbox.check_path(str(p), "write"):
            return f"[沙箱拒绝] {self.sandbox.deny_reason(str(p))}"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"[写入成功] {path} ({len(content)}字符)"
    
    def _terminal_real(self, command: str) -> str:
        """真实执行——ToolRegistry调用此方法"""
        import subprocess
        try:
            result = subprocess.run(command, shell=True, capture_output=True,
                                   text=True, timeout=30)
            return result.stdout[:3000] or result.stderr[:1000]
        except Exception as e:
            return f"[执行失败] {e}"
    
    def _verify_before_complete(self, tool_name: str, **kwargs) -> dict:
        """verify-before-complete钩子：写操作前的额外验证"""
        # 沙箱检查
        if tool_name == "write_file":
            path = kwargs.get("path", "")
            p = self.sandbox._normalize_path(path)
            if not self.sandbox.check_path(str(p), "write"):
                return {"pass": False, "reason": f"沙箱拒绝: {self.sandbox.deny_reason(str(p))}"}
        # 危险命令检查（DR-20260917-02：命令位判定，与_terminal共用_find_dangerous_cmd）
        if tool_name == "terminal":
            command = kwargs.get("command", "")
            _bad = self._find_dangerous_cmd(command)
            if _bad:
                return {"pass": False, "reason": f"危险命令: {_bad}"}
        return {"pass": True, "reason": ""}



    def list_tools(self):
        """Agent可用的工具列表(结构化)"""
        tools = []
        for name, func in self.tools.items():
            tools.append({"name": name, "type": "builtin", "description": func.__doc__ or name})
        for name, info in self.bridge.discovered.items():
            tools.append({"name": name, "type": "external", "description": info.get("description", name)})
        for skill in self.learned_skills:
            tools.append({"name": skill.get("name", "unknown"), "type": "learned", "description": skill.get("proposal", "learned")})
        return tools

    def get_tools_summary(self):
        """工具摘要(一行)"""
        tools = self.list_tools()
        by_type = {}
        for t in tools:
            by_type.setdefault(t["type"], []).append(t["name"])
        return f"{len(tools)} tools ({', '.join(f'{k}:{len(v)}' for k,v in by_type.items())})"
