from .degradation_trace import trace_degradation
"""extracted from main_loop.py"""
import json, os, time, uuid
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
        # [进化] 接入ToolRegistry的verify-before-complete机制
        try:
            from ..tools.executor import ToolRegistry, ToolResult
            self._tool_registry = ToolRegistry()
            self._tool_registry.register("write_file", self._write_file_real,
                                          description="写入文件（带验证）")
            self._tool_registry.register("terminal", self._terminal_real,
                                          description="执行Shell命令（带验证）")
            # 设置verify hook：写操作前检查沙箱+治理规则
            self._tool_registry.set_verify_hook(self._verify_before_complete)
            self._has_verify = True
        except Exception:
            self._tool_registry = None
            self._has_verify = False
        print(f"  ISN 工具执行就绪 · {len(self.tools)}个工具 · 沙箱={len(self.sandbox.allowed)}个允许路径 · 外部工具={len(self.bridge.discovered)}个 · verify={'ON' if self._has_verify else 'OFF'}")
        # 加载已学习技能（血管#3: ios→isn）
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
        
        # 连接3: 加载Self-Harness提案（approved状态的自动应用）
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
        
        # 连接4: 加载治理转换引擎产出的规则（P0: arXiv:2607.01087）
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
                    # 调用工具（传递args字典）
                    result = self.tools[tool_name](**tool_args)
                except TypeError:
                    # 如果工具不接受**kwargs，尝试单参数调用
                    result = self.tools[tool_name](str(tool_args))
                results.append(f"[{tool_name}] {result}")
            else:
                results.append(f"[未知工具] {tool_name}")
        
        return "\n".join(results) if results else f"已执行: {decision.reason}"
    
    def _read_file(self, path: str) -> str:
        """读取文件内容（沙箱检查·先解析再检查防路径穿越）"""
        # 先解析路径（防../../etc/passwd穿越）
        p = Path(path).expanduser().resolve()
        # 再检查沙箱
        if not self.sandbox.check_path(str(p), "read"):
            return f"[沙箱拒绝] {self.sandbox.deny_reason(str(p))}"
        if not p.exists():
            return f"[错误] 文件不存在: {path}"
        if p.stat().st_size > 100000:
            return f"[错误] 文件过大: {path}"
        return p.read_text(encoding="utf-8", errors="replace")[:5000]
    
    def _write_file(self, path: str, content: str) -> str:
        """写入文件（沙箱检查·先解析再检查防路径穿越）"""
        # 先解析路径（防穿越）
        p = Path(path).expanduser().resolve()
        # 再检查沙箱
        if not self.sandbox.check_path(str(p), "write"):
            return f"[沙箱拒绝] {self.sandbox.deny_reason(str(p))}"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"[写入成功] {path} ({len(content)}字符)"
    
    def _search_files(self, pattern: str) -> str:
        """搜索文件"""
        import subprocess
        try:
            result = subprocess.run(
                ["find", str(Path.home()), "-name", pattern, "-maxdepth", "4"],
                capture_output=True, text=True, timeout=10
            )
            return result.stdout[:2000] or "[未找到]"
        except:
            return "[搜索失败]"
    
    def _terminal(self, command: str) -> str:
        """执行shell命令。高风险——需要沙箱检查。"""
        import subprocess
        import re
        
        # 危险命令清单
        dangerous = ["rm", "sudo", "dd", "mkfs", "> "]
        for d in dangerous:
            if d in command:
                return f"[拦截] 危险命令: {command}"
        
        # 沙箱检查：提取命令中的文件路径并检查
        # 匹配常见路径模式
        path_patterns = [
            r'(?<=\s)(/[^\s]+)',  # 绝对路径
            r'(?<=["\'])(/[^\s"\']+)(?=["\'])',  # 引号内的路径
        ]
        for pattern in path_patterns:
            paths = re.findall(pattern, command)
            for p in paths:
                if not self.sandbox.check_path(p, "write"):
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
        p = Path(path).expanduser().resolve()
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
            p = Path(path).expanduser().resolve()
            if not self.sandbox.check_path(str(p), "write"):
                return {"pass": False, "reason": f"沙箱拒绝: {self.sandbox.deny_reason(str(p))}"}
        # 危险命令检查
        if tool_name == "terminal":
            command = kwargs.get("command", "")
            dangerous = ["rm -rf", "sudo", "dd if=", "mkfs", "> /dev"]
            for d in dangerous:
                if d in command:
                    return {"pass": False, "reason": f"危险命令: {d}"}
        return {"pass": True, "reason": ""}


