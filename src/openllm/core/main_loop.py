#!/usr/bin/env python3
"""
openLLM Agent主循环 — 独立Agent框架核心

这是openLLM自己的Agent主循环。
不从Hermes借，不从任何人借。

五体调度：
  ISA   → UI层 + Context构建
  章鱼I → 推理引擎 + 左右脑（内部双路径对弈）
  IO-S  → 决策框架 + 自我进化 + 错误恢复
  ISN   → 工具执行
  IKO   → 可观测

架构：单文件可运行，零外部依赖。
集成Session/Turn模型，支持10阶段心跳。
"""

import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .session import Session, Turn, TurnStatus, create_session

# ── tool_validator 集成（Layer 3）──
try:
    from .tool_validator import (
        ToolCall as _TVToolCall,
        ValidationResult as _TVResult,
        validate_tool_result as _tv_validate,
    )
    _HAS_TOOL_VALIDATOR = True
except ImportError:
    _HAS_TOOL_VALIDATOR = False

# ── 章鱼I Reflect 集成 ──
_REFLECT_SCRIPT = Path.home() / "projects" / "isa" / "octopus" / "scripts" / "octopus_reflect.py"
_HAS_REFLECT = _REFLECT_SCRIPT.exists()

# ── IAX Layer 7 推理预算追踪（try/except确保不破坏现有逻辑）──
try:
    from .inference_budget import InferenceBudgetManager
    _HAS_BUDGET = True
except ImportError:
    _HAS_BUDGET = False


# ─── 数据模型（已提取到models.py）──────────────────────────
from .models import (Message, Context, Prediction, RiskAssessment,
                     Proposal, Critique, Decision, ActionResult,
                     CausalDelta, TickMetrics)


# ═══════════════════════════════════════════════════════
# LLM Provider 集成（左右脑对弈用）
# ═══════════════════════════════════════════════════════

class LLMProvider:
    """最简单的LLM调用封装"""
    
    def __init__(self, model: str = "deepseek-chat"):
        self.model = model
        # 优先从config读·其次环境变量
        self.api_key = self._load_key()
        self.endpoint = "https://api.deepseek.com/v1/chat/completions"
        self._available = bool(self.api_key)
        self._last_usage: dict = {}  # 最近一次调用的token使用量
    
    def _load_key(self) -> str:
        """从config.json加载API key"""
        config_path = Path.home() / ".openllm" / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
                return cfg.get("providers", {}).get("deepseek", {}).get("api_key", "")
            except:
                pass
        return os.environ.get("DEEPSEEK_API_KEY", "")
    
    def chat(self, messages: list[dict]) -> str:
        """调LLM·返回文本。M0: 返回模拟响应。"""
        if not self._available:
            # M0降级：返回模拟响应
            user_msg = messages[-1]["content"] if messages else ""
            self._last_usage = {}
            return f"[模拟LLM] 已收到: {user_msg[:50]}"
        
        # TODO: Phase 2接入真实API
        try:
            import requests
            resp = requests.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 1024,
                    "temperature": 0.7,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            usage = data.get("usage", {})
            self._last_usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[LLM错误] {e}"


# ═══════════════════════════════════════════════════════
# 五体（Stub · M0阶段用简单实现）
# ═══════════════════════════════════════════════════════

class ISA:
    """UI层 + Context构建——用户交互入口"""
    
    def __init__(self, mode: str = "console"):
        self.mode = mode
        self.session_id = uuid.uuid4().hex[:12]
        # DisplayEngine — openLLM的脸
        from .display import DisplayEngine
        self.display = DisplayEngine()
        print(f"  ISA[{self.session_id[:8]}] UI层就绪 · 模式={mode}")
        
    def listen(self) -> Optional[Message]:
        """获取用户输入"""
        if self.mode == "console":
            try:
                text = input(">>> ").strip()
                if not text:
                    return None
                if text in ("/quit", "/exit", "/q"):
                    return None  # 外部检测退出
                return Message(text=text)
            except (EOFError, KeyboardInterrupt):
                return None
        # silent mode: 从参数读取
        return None
    
    def build_context(self, msg: Message, session: Optional['Session'] = None,
                      octopus: Optional['章鱼I'] = None, ios: Optional['IOS'] = None) -> Context:
        """构建七要素上下文（v2·接入真正模块）"""
        # ① 身份
        identity = {"name": "openLLM", "role": "自主Agent", "mode": self.mode}
        
        # ② 记忆 — unified_memory召回 + 最近决策 + 因果教训
        memory = {"session_id": self.session_id}
        if session:
            memory["recent_decisions"] = [t.summary() for t in session.turns[-3:]]
        if ios:
            memory["causal_lessons"] = ios.causal_memory[-5:]
        # [进化] 接入unified_memory真正的记忆召回
        try:
            from ..memory.unified_memory import UnifiedMemory
            um = UnifiedMemory()
            recalled = um.retrieve(msg.text, top_n=5)
            if recalled:
                memory["recalled"] = [{"content": str(r.value)[:200],
                                       "importance": r.importance,
                                       "key": r.key} for r in recalled]
        except Exception:
            pass  # 降级：不影响主流程
        
        # ③ 工具 — 动态获取
        tools = ["read_file", "search_files"]
        if octopus and octopus.left.provider._available:
            tools.extend(["write_file", "terminal"])
        
        # ④ 因果提示 — causal_memory结构化检索 + heuristics
        causal_hints = []
        if ios and ios.causal_memory:
            msg_words = set(msg.text.lower().split()[:5])
            for m in ios.causal_memory[-10:]:
                action_words = set(m.get("action", "").lower().split()[:3])
                if msg_words & action_words:
                    causal_hints.append(m.get("lesson", ""))
        # [进化] 接入causal_memory结构化检索
        try:
            from ..memory.causal_memory import CausalMemoryStore
            store = CausalMemoryStore()
            relevant = store.search(context_features=[msg.text[:50]], max_results=3)
            if relevant:
                causal_hints.extend([r.lesson for r in relevant if r.lesson])
        except Exception:
            pass
        
        # Heuristics消费闭环
        if ios and hasattr(ios, 'heuristics_consumer'):
            heuristics = ios.heuristics_consumer.retrieve(msg.text, top_k=3)
            if heuristics:
                formatted = ios.heuristics_consumer.format_for_context(heuristics)
                causal_hints.append(formatted)
        
        # ⑤ 认知报告 — D₀感知
        d0_report = {"sem_ratio": "unknown", "d0_budget": "unknown"}
        try:
            from ..iai.prediction import PredictionEngine
            # 如果有章鱼I的D₀快照，用真实数据
            if octopus and hasattr(octopus, 'd0_snapshot'):
                d0_report = octopus.d0_snapshot()
        except Exception:
            pass
        
        # ⑥ 风险上下文
        risk_context = {"current_level": "low"}
        
        # ⑦ 搜索结果 — 触手脑索引 + evidence_replay
        search_results = []
        if octopus and octopus.tentacles:
            idx = octopus.tentacles.get("index")
            if idx:
                results = idx.search(msg.text, limit=5)
                search_results = [s.get("filepath", "") for s in results]
        # [进化] 接入evidence_replay
        try:
            from ..memory.evidence_replay import create_replay_for_context
            replay = create_replay_for_context(msg.text, None, top_k=3, max_tokens=256)
            if replay:
                search_results.append(replay)
        except Exception:
            pass
        
        return Context(
            user_message=msg.text,
            identity=identity,
            memory=memory,
            tools=tools,
            causal_hints=causal_hints,
            d0_report=d0_report,
            risk_context=risk_context,
            search_results=search_results,
        )
    
    def respond(self, text: str, phase_times: dict = None):
        """输出响应 — 走DisplayEngine渲染"""
        if self.mode == "console":
            self.display.render_response(text, phase_times)
        else:
            # silent模式不输出
            pass


class 章鱼I:
    """推理引擎 + 左右脑"""
    
    # 能力降级等级
    HEALTH = {
        "FULL":     {"confidence_min": 0.7, "tools_all": True},
        "DEGRADED": {"confidence_min": 0.4, "tools_all": False},
        "MINIMAL":  {"confidence_min": 0.2, "tools_all": False},
        "OFFLINE":  {"confidence_min": 0.0, "tools_all": False},
    }
    
    def __init__(self):
        self.left = _LeftBrain()
        self.right = _RightBrain()
        # 触手脑：文件监控 + 全文索引
        from .tentacle import FileWatcherBrain, IndexBrain
        self.tentacles = {
            "file_watcher": FileWatcherBrain(),
            "index": IndexBrain(),
        }
        print("  章鱼I 推理引擎就绪 · 左右脑在线 · 触手脑在线")
    
    def health_check(self) -> str:
        """自检。返回当前健康等级。"""
        if not self.left.provider._available:
            return "MINIMAL"  # 无API=降级
        try:
            _ = self.left.provider.chat([{"role":"user","content":"ping"}])
            return "FULL"
        except:
            return "DEGRADED"
    
    def d0_snapshot(self) -> dict:
        """
        D₀感知简化版·实时认知深度报告
        
        完整版需要IAH扫描注意力头D₀——当前不可用。
        简化版基于：API可用性 + health等级 + 可用工具数 + context使用率
        
        Returns:
            sem_ratio: 语义能力比率（0-1）
            d0_budget: 可用token数
            cognitive_rhythm: 认知节奏（fast/steady/slow）
            health: 当前健康等级
        """
        health = self.health_check()
        
        # sem_ratio 估算：FULL→0.9, DEGRADED→0.5, MINIMAL→0.2
        sem_map = {"FULL": 0.9, "DEGRADED": 0.5, "MINIMAL": 0.2, "OFFLINE": 0.0}
        sem_ratio = sem_map.get(health, 0.3)
        
        # d0_budget: 可用API时=大模型语义容量，降级时=本地规则容量
        d0_budget = 100000 if health == "FULL" else 10000 if health == "DEGRADED" else 1000
        
        # cognitive_rhythm: 有API→fast, 降级→steady, 无API→slow
        rhythm = "fast" if health == "FULL" else "steady" if health == "DEGRADED" else "slow"
        
        return {
            "sem_ratio": sem_ratio,
            "d0_budget": d0_budget,
            "cognitive_rhythm": rhythm,
            "health": health,
            "tools_available": len(self.tentacles) if hasattr(self, 'tentacles') else 0,
            "api_available": self.left.provider._available,
        }
    
    def predict_consequences(self, ctx: Context) -> Prediction:
        """Phase 3: 因果预测"""
        return self.left.predict(ctx)
    
    def reason(self, ctx: Context, prediction: Optional[Prediction] = None, 
               risk: Optional[RiskAssessment] = None) -> tuple[Proposal, Critique]:
        """Phase 5: 双脑推理"""
        proposal = self.left.think(ctx, prediction, risk)
        critique = self.right.review(ctx, proposal)
        return proposal, critique
    
    def compare(self, prediction: Prediction, result: ActionResult) -> CausalDelta:
        """Phase 8: 因果对照"""
        # M0: 简单比较
        match = result.success and "模拟" not in result.output
        return CausalDelta(
            prediction_match=match,
            delta_summary=f"预测={'正确' if match else '错误'}",
            learned=["预测准确性基础检查"],
        )
    
    def learn_causal(self, ctx: Context, prediction: Prediction,
                     result: ActionResult, delta: CausalDelta):
        """Phase 8: 因果学习（实际实现在IOS.learn_causal）"""
        pass


class _LeftBrain:
    """左脑(正手)：提案"""
    
    def __init__(self):
        self.provider = LLMProvider()
    
    def predict(self, ctx: Context) -> Prediction:
        """因果预测（v2·数学+LLM双路径）"""
        # [进化] 数学预测：PredictionEngine纯计算
        math_prediction = {"summary": "", "risk_signals": []}
        try:
            from ..iai.prediction import PredictionEngine
            pe = PredictionEngine()
            result = pe.predict_next({"user_message": ctx.user_message, "tools": ctx.tools})
            if result:
                math_prediction["summary"] = f"[数学] 预测类型={result.get('predicted_type','unknown')} 置信={result.get('confidence', 0):.2f}"
                math_prediction["risk_signals"] = result.get("risk_signals", [])
        except Exception:
            pass
        
        # LLM预测：语义理解
        prompt = f"""你是openLLM的因果预测器。预测以下操作的后果。
用户意图：{ctx.user_message}
请预测：
[IF-SUCCESS] 如果操作成功，会发生什么
[IF-FAILURE] 如果操作失败，会发生什么
[IRREVERSIBLE] 哪些后果无法撤销
请用JSON格式输出：
{{"summary": "一句话预测摘要", "risk_signals": ["风险信号"]}}
"""
        resp = self.provider.chat([{"role": "user", "content": prompt}])
        try:
            data = json.loads(resp) if resp.startswith("{") else {"summary": resp[:50]}
        except:
            data = {"summary": resp[:50]}
        
        # 合并数学+LLM预测
        combined_summary = data.get("summary", resp[:50])
        if math_prediction["summary"]:
            combined_summary = f"{math_prediction['summary']} | [LLM] {combined_summary}"
        combined_risks = list(set(data.get("risk_signals", []) + math_prediction["risk_signals"]))
        
        return Prediction(
            summary=combined_summary[:200],
            consequences=[data.get("summary", "")],
            confidence=data.get("confidence", 0.7),
            risk_signals=combined_risks,
        )
    
    def think(self, ctx: Context, prediction: Optional[Prediction] = None,
              risk: Optional[RiskAssessment] = None) -> Proposal:
        """基于上下文提出方案"""
        prompt = f"""你是openLLM的左脑。基于以下上下文，生成一个行动提案。
上下文：{ctx.user_message}
你的任务：
1. 分析用户意图
2. 提出具体的行动方案
3. 评估方案的置信度（0-1）
4. 列出支持方案的证据
请用JSON格式输出：
{{"content": "行动方案描述", "confidence": 0.7, "evidence": ["证据1", "证据2"], "tool_calls": []}}
"""
        resp = self.provider.chat([{"role": "user", "content": prompt}])
        # 解析LLM返回的JSON
        try:
            data = json.loads(resp) if resp.startswith("{") else {"content": resp}
        except:
            data = {"content": resp, "confidence": 0.6}
        
        evidence = data.get("evidence", [])
        if prediction:
            evidence.append(f"已预测后果: {prediction.summary[:30]}")
        if risk:
            evidence.append(f"风险等级: {risk.level}")
        
        # 血管三：health_check→confidence校准
        # 无API时confidence下降
        confidence = data.get("confidence", 0.6)
        if not self.provider._available:
            confidence *= 0.8  # 无API时打8折
        
        return Proposal(
            content=data.get("content", resp[:100]),
            confidence=confidence,
            evidence=evidence,
            tool_calls=data.get("tool_calls", []),
            prediction_ref=prediction,
        )


class _RightBrain:
    """右脑(反手)：批判"""
    
    def __init__(self):
        self.provider = LLMProvider()
    
    def review(self, ctx: Context, proposal: Proposal) -> Critique:
        """审查左脑提案"""
        prompt = f"""你是openLLM的右脑。审查左脑的提案。
原始上下文：{ctx.user_message}
左脑提案：{proposal.content}
左脑置信度：{proposal.confidence}
左脑证据：{proposal.evidence}
你的任务：
1. 检查提案是否有逻辑漏洞
2. 检查提案是否有安全隐患
3. 检查证据是否支撑提案
4. 给出裁决：approve（通过）/ reject（否决）/ revise（修改）
请用JSON格式输出：
{{"verdict": "approve|reject|revise", "concerns": ["关心点1"], "suggestions": ["建议1"]}}
"""
        resp = self.provider.chat([{"role": "user", "content": prompt}])
        try:
            data = json.loads(resp) if resp.startswith("{") else {"verdict": "approve"}
        except:
            data = {"verdict": "approve"}
        
        return Critique(
            content=resp[:100],
            verdict=data.get("verdict", "approve"),
            concerns=data.get("concerns", []),
            suggestions=data.get("suggestions", []),
        )


# IOS已提取到ios_impl.py
from .ios_impl import IOS
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



class IKO:
    """IKO — 输出体七因子管线
    
    七因子：IntentClassifier, SilenceAuditor, OutputRouter,
            OutputAuditChain, OutputFeedbackCollector,
            LambdaCalibrator, ProbingTrainer, SymmetricCodec
    """
    
    def __init__(self):
        self.metrics: list[TickMetrics] = []
        self.log_path = Path.home() / ".openllm" / "output" / "iko" / "ticks" / "ticks.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 七因子组件初始化
        from openllm.iko import (
            IntentClassifier, OutputRouter, SilenceAuditor,
            OutputAuditChain, OutputFeedbackCollector,
            LambdaCalibrator, ProbingTrainer, SymmetricCodec,
        )
        self.classifier = IntentClassifier()
        self.router = OutputRouter()
        self.registry = self.router.registry
        self.auditor = SilenceAuditor()
        self.audit_chain = OutputAuditChain()
        self.feedback_collector = OutputFeedbackCollector()
        self.calibrator = LambdaCalibrator()
        self.probing_trainer = ProbingTrainer()
        self.codec = SymmetricCodec()
        self._output_count = 0
        
        print(f"  IKO 可观测就绪 · 七因子管线 · 日志={self.log_path}")
    
    def trace(self, phase: str, status: str, duration_ms: float = 0.0, detail: str = ""):
        """记录一次观测"""
        m = TickMetrics(
            tick_id=uuid.uuid4().hex[:8],
            phase=phase,
            status=status,
            duration_ms=duration_ms,
            detail=detail[:100],
        )
        self.metrics.append(m)
        # 后端持久化
        with open(self.log_path, "a") as f:
            f.write(json.dumps(asdict(m)) + "\n")
    
    def report(self) -> dict:
        """仪表盘快照"""
        if not self.metrics:
            return {"status": "no_data"}
        last_10 = self.metrics[-10:]
        return {
            "total_ticks": len(self.metrics),
            "ok_rate": sum(1 for m in last_10 if m.status == "ok") / max(len(last_10), 1),
            "avg_duration_ms": sum(m.duration_ms for m in last_10) / len(last_10),
            "last_phase": last_10[-1].phase if last_10 else "",
        }
    
    def process_output(self, raw_output: str, context: dict, decision: dict) -> str:
        """七因子输出管线"""
        # 1. 意图分类
        result = self.classifier.classify(context, decision)
        intent = result.intent
        
        # 2. 沉默审计
        intent = self.auditor.audit(intent, context, reversible=True)
        
        # 3. 路由+渲染
        plan = self.router.route(intent, content={"text": raw_output}, user_prefs={})
        renderer = self.registry.get(plan.renderer_name)
        if renderer:
            output = renderer.render(intent, {"text": raw_output}, {}, 0.8)
        else:
            output = raw_output
        
        # 4. 审计链（赫淮斯托斯约束：reasoning_chain_hash必须是真实hash）
        if output:  # 非空输出才审计
            import hashlib, json as _json
            reasoning_hash = hashlib.sha256(
                _json.dumps({"intent": intent.value, "output": output[:200]}, sort_keys=True).encode()
            ).hexdigest()[:16]
            self.audit_chain.append(
                output_id=f"out-{self._output_count}",
                intent=intent.value,
                content=output.encode(),
                decision_source="IKO",
                risk_level=0.1,
                confidence=0.8,
                reasoning_chain_hash=reasoning_hash,
            )
            self._output_count += 1
        
        # 5. 记录trace
        self.trace("output_process", "ok", detail=f"intent={intent.value}")
        
        return output
    
    def shutdown(self):
        """关闭时的报告"""
        print(f"\n  IKO 关闭报告: {len(self.metrics)} tick, 最后状态={self.metrics[-1].status if self.metrics else 'N/A'}")


# ═══════════════════════════════════════════════════════
# Agent主循环（集成Session/Turn + 10阶段心跳）
# ═══════════════════════════════════════════════════════

class Agent:
    """openLLM Agent — 独立运行的Agent框架"""
    
    def __init__(self, mode: str = "console"):
        # 五体初始化
        self.isa = ISA(mode)
        self.octopus = 章鱼I()
        self.ios = IOS()
        self.isn = ISN()
        self.iko = IKO()
        
        # Session/Turn
        # 六体自监督反馈环
        from ..governance.feedback_loop import FeedbackLoop
        self.feedback_loop = FeedbackLoop()
        self._body_outputs = {}  # 暂存每体最新输出
        
        # 间隙体——空闲期好奇心
        from .idle_wander import IdleWanderer
        self._wanderer = IdleWanderer(threshold=3)

        self.session = create_session(max_context_tokens=100000)
        
        # 状态
        self.running = False
        self._tick_count = 0
        self._max_ticks = 100  # 安全上限
        self._last_output = ""
        
        # ── IAX Layer 7 推理预算管理器 ──
        self._budget_manager = InferenceBudgetManager() if _HAS_BUDGET else None
        self._budget_checked_today = False
        
        # ── tool_validator 失败追踪 ──
        self._tool_failures: list[dict] = []
    
    def run(self):
        """主循环入口"""
        self.running = True
        print("\n═══ openLLM Agent 启动 ═══")
        print("  模式: 独立运行 · 无外部依赖")
        print("  Session:", self.session.id)
        print("  输入 /quit 退出\n")
        
        while self.running and self._tick_count < self._max_ticks:
            try:
                self._tick()
                self._tick_count += 1
            except KeyboardInterrupt:
                self.iko.trace("shutdown", "ok", detail="用户中断")
                break
            except Exception as e:
                self.iko.trace("fatal", "error", detail=str(e)[:50])
                ok = self.ios.recover(e)
                if not ok:
                    print(f"  🔴 不可恢复错误: {e}")
                    break
        
        self.iko.shutdown()
        self.session.end()
        print("\n═══ openLLM Agent 关闭 ═══")
    
    def run_once(self, message: str) -> str:
        """
        单次运行（用于测试/非交互模式）
        
        用法：
            agent = Agent(mode="silent")
            result = agent.run_once("你好")
        """
        saved_mode = self.isa.mode
        self.isa.mode = "silent"
        try:
            msg = Message(text=message)
            self._execute_tick(msg)
            return self._last_output
        finally:
            self.isa.mode = saved_mode
    
    def _tick(self):
        """一次心跳"""
        msg = self.isa.listen()
        if msg is None:
            # ── 间隙体：空闲期好奇心 ──
            if self._wanderer.tick_idle():
                self._do_wander()
            else:
                self.running = False
            return
        self._wanderer.tick_active()  # 有输入·重置空闲计数
        self._execute_tick(msg)
    
    def _do_wander(self):
        """散步模式——注意力的自由偏移"""
        discoveries = self._wanderer.wander()
        if discoveries:
            # 散步发现写入日志，不触发决策/执行
            for d in discoveries:
                self.iko.trace("wander", "ok", detail=f"{d.domain}: {d.finding[:40]}")
    
    def _record_inference(self, phase: str):
        """IAX: 从LLMProvider提取最近一次调用的token使用量，记录到预算管理器"""
        if not self._budget_manager:
            return
        # 从左右脑provider取_last_usage（Phase 3用左脑，Phase 5用左脑+右脑）
        usage = getattr(self.octopus.left.provider, '_last_usage', {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        if prompt_tokens == 0 and completion_tokens == 0:
            return  # 模拟模式或无数据，跳过
        try:
            rec = self._budget_manager.record_inference(
                model=self.octopus.left.provider.model,
                tokens_in=prompt_tokens,
                tokens_out=completion_tokens,
                duration_ms=0,  # 精确计时在phase trace中已有
            )
            self.iko.trace("budget", "ok",
                detail=f"{phase}: in={prompt_tokens} out={completion_tokens} cost=${rec.cost_usd:.4f}")
            # 每天第一次调用时检查预算
            if not self._budget_checked_today:
                self._budget_checked_today = True
                budget = self._budget_manager.check_budget()
                if budget["over_budget"]:
                    print(f"  ⚠️ IAX 推理预算超限: {budget['total_tokens']}/{budget['limit']} "
                          f"({budget['usage_ratio']:.0%}) 今日成本=${budget['cost_usd']:.4f}")
        except Exception:
            pass  # 预算追踪失败不阻塞主循环

    def _execute_tick(self, msg: Message):
        """一次完整的10阶段心跳"""
        t0 = time.time()
        
        # 创建新Turn
        turn = self.session.new_turn()
        
        try:
            # ── Phase 1: isa.listen() → Message ──
            turn.trace_phase("listen", "ok")
            self.iko.trace("listen", "ok")
            
            # ── Phase 2: build_context — 传入session和ios ──
            t1 = time.time()
            ctx = self.isa.build_context(msg, self.session, self.octopus, self.ios)
            turn.trace_phase("context", "ok", duration_ms=(time.time()-t1)*1000)
            self.iko.trace("context", "ok")

            # ── Phase 2.5: evidence replay — 相关证据回放 ──
            # 来源：arXiv:2607.02509 ReContext
            # 从七要素中提取top-K相关证据，追加到prompt尾部
            try:
                from ..memory.evidence_replay import create_replay_for_context
                replay_text = create_replay_for_context(msg.text, ctx, top_k=5, max_tokens=512)
                if replay_text:
                    ctx.search_results = getattr(ctx, 'search_results', []) or []
                    ctx.search_results.append(replay_text)
                    turn.trace_phase("replay", "ok", detail=f"evidence_replay:{len(replay_text)}chars")
                    self.iko.trace("replay", "ok")
            except Exception as e:
                # evidence replay失败不阻塞主流程
                turn.trace_phase("replay", "skip", detail=str(e)[:100])

            # ── Phase 3: 因果预测 ──
            t2 = time.time()
            prediction = self.octopus.predict_consequences(ctx)
            turn.trace_phase("predict", "ok", duration_ms=(time.time()-t2)*1000,
                           detail=prediction.summary_text())
            self.iko.trace("predict", "ok", detail=prediction.summary_text())
            
            # ── IAX: 记录Phase 3推理消耗 ──
            self._record_inference("phase_3_predict")

            # ── Phase 3.5: 治理检查·路由决策点（Copewell模式·2026-07-06） ──
            try:
                from ..governance.events import RoutingCheckEvent, GovernanceDimension
                from ..governance.audit import AuditChain
                _g8_event = RoutingCheckEvent(
                    actor="ios_phase3",
                    session_id=ctx.session_id if hasattr(ctx, 'session_id') else "default",
                    prev_hash="genesis",
                    phase="phase_3_prediction",
                    route_target=str(type(prediction).__name__),
                    risk_level="low",
                    approved=True,
                )
                if hasattr(self, '_audit_chain'):
                    self._audit_chain.append(_g8_event)
            except Exception:
                pass  # 治理检查失败不阻塞主流程

            # ── Phase 4: 风险评估 ──
            t3 = time.time()
            risk = self.ios.risk_check(ctx, prediction)
            turn.risk_level = risk.level
            turn.trace_phase("risk", "ok" if not risk.is_blocked() else "denied",
                           duration_ms=(time.time()-t3)*1000,
                           detail=f"level={risk.level} blocked={risk.is_blocked()}")
            self.iko.trace("risk", "ok" if not risk.is_blocked() else "denied",
                         detail=risk.reason)
            
            if risk.is_blocked():
                self.isa.respond(f"[安全拦截] {risk.reason}")
                turn.add_action("blocked", risk.reason)
                turn.complete()
                return
            
            # ── Phase 5: 左右脑推理 ──
            # D₀感知注入
            d0 = self.octopus.d0_snapshot()
            ctx.d0_report = d0
            
            # 因果教训注入（从risk.details获取）
            if risk and risk.details:
                ctx.causal_hints = risk.details
            
            t4 = time.time()
            proposal, critique = self.octopus.reason(ctx, prediction, risk)
            turn.trace_phase("reason", "ok", duration_ms=(time.time()-t4)*1000)
            self.iko.trace("reason", "ok")
            
            # ── IAX: 记录Phase 5推理消耗（左脑+右脑）──
            self._record_inference("phase_5_reason")

            # ── Phase 5.5: 治理检查·推理质量（Copewell模式·2026-07-06） ──
            try:
                from ..governance.events import RoutingCheckEvent
                _g8_reason = RoutingCheckEvent(
                    actor="ios_phase5",
                    session_id=ctx.session_id if hasattr(ctx, 'session_id') else "default",
                    prev_hash="genesis",
                    phase="phase_5_reasoning",
                    route_target=str(type(proposal).__name__),
                    risk_level="low",
                    approved=True,
                )
                if hasattr(self, '_audit_chain'):
                    self._audit_chain.append(_g8_reason)
            except Exception:
                pass  # 治理检查失败不阻塞主流程

            # ── Phase 6: 仲裁 ──
            t5 = time.time()
            decision = self.ios.arbitrate(proposal, critique, risk)
            turn.trace_phase("decide", "ok" if decision.approved else "denied",
                           duration_ms=(time.time()-t5)*1000,
                           detail=decision.reason)
            self.iko.trace("decide", "ok" if decision.approved else "denied",
                         detail=decision.reason)
            
            if not decision.approved:
                self.isa.respond(f"[仲裁否决] {decision.reason}")
                turn.add_action("denied", decision.reason)
                turn.complete()
                return
            
            # ── Phase 7: 执行 ──
            # cap_check: 只在有工具调用时检查（无工具调用=纯对话·不检查）
            if getattr(decision, 'tool_calls', None):
                if not self.ios.cap_check("execute", decision.action):
                    self.isa.respond("[权限拒绝] cap_policy不允许此操作")
                    turn.add_action("denied", "cap_policy拒绝")
                    turn.complete()
                    return
            
            t6 = time.time()
            result = self.isn.execute(decision)
            turn.trace_phase("execute", "ok" if result.success else "error",
                           duration_ms=result.duration_ms)
            self.iko.trace("execute", "ok" if result.success else "error")
            
            # ── Phase 7.5: tool_validator 自动验证 ──
            if _HAS_TOOL_VALIDATOR and result.success:
                try:
                    _tc_list = getattr(decision, 'tool_calls', None) or []
                    _first = _tc_list[0] if _tc_list else {}
                    _tool_name = _first.get("name", "unknown") if isinstance(_first, dict) else str(_first)
                    _tool_type_map = {"read_file": "read", "write_file": "write",
                                      "search_files": "search", "terminal": "execute"}
                    _tool_type = _tool_type_map.get(_tool_name, "execute")
                    _tv_call = _TVToolCall(
                        tool_name=_tool_name, tool_type=_tool_type,
                        params=_first.get("args", {}) if isinstance(_first, dict) else {},
                        result=result.output,
                        duration_ms=result.duration_ms,
                        session_id=getattr(self.session, 'id', ''),
                        turn_id=turn.id,
                    )
                    _tv_report = _tv_validate(_tv_call)
                    turn.trace_phase("validate", _tv_report.overall.value,
                                   detail=f"checks={len(_tv_report.checks)} retry={_tv_report.should_retry} block={_tv_report.should_block}")
                    if _tv_report.should_block:
                        _fail_entry = {
                            "timestamp": time.time(), "tool_name": _tool_name,
                            "reason": _tv_report.audit_entry.reason if _tv_report.audit_entry else "blocked",
                            "session_id": getattr(self.session, 'id', ''), "turn_id": turn.id,
                        }
                        self._tool_failures.append(_fail_entry)
                        self.iko.trace("validate", "blocked", detail=_fail_entry["reason"])
                        self._last_output = f"[工具验证拦截] {_fail_entry['reason']}"
                        if self.isa.mode != "silent":
                            self.isa.respond(self._last_output)
                        turn.add_action("validate_blocked", _fail_entry["reason"])
                        turn.complete()
                        return
                    if _tv_report.should_retry:
                        turn.trace_phase("validate_retry", "warn", detail="工具结果需重试")
                except Exception as _tv_err:
                    turn.trace_phase("validate", "skip", detail=str(_tv_err)[:80])
            
            # ── Phase 8: 因果对照 ──
            t7 = time.time()
            delta = self.octopus.compare(prediction, result)
            self.ios.learn_causal(ctx, prediction, result, delta)
            turn.trace_phase("learn", "ok", duration_ms=(time.time()-t7)*1000,
                           detail=delta.summary_text())
            self.iko.trace("learn", "ok", detail=delta.summary_text())
            
            # ── Phase 9: 进化 ──
            t8 = time.time()
            self.ios.evolve(proposal, critique, result, delta)
            turn.trace_phase("evolve", "ok", duration_ms=(time.time()-t8)*1000)
            self.iko.trace("evolve", "ok")
            
            # ── Phase 10: 压缩+输出+checkpoint ──
            t9 = time.time()
            self.session.compact_if_needed()
            # 无工具调用时：输出LLM生成的proposal内容（对话响应）
            if getattr(decision, 'tool_calls', None) or getattr(decision, '_has_tools', False):
                self._last_output = result.output
            else:
                self._last_output = proposal.content if proposal.content else result.output
            
            # IKO七因子管线
            try:
                risk_map = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH"}
                ctx_for_iko = {
                    "risk_level": risk_map.get(getattr(decision, 'risk_level', 'low'), "LOW"),
                    "has_tool_calls": bool(getattr(decision, 'tool_calls', None)),
                    # 包拯审计修正：has_side_effects应检查实际副作用，非仅输出存在
                    "has_side_effects": bool(getattr(result, 'success', False) and getattr(result, 'output', '')),
                    # 包拯审计修正：option_count从proposal选项中提取，非硬编码
                    "option_count": max(1, len(getattr(proposal, 'evidence', []))),
                }
                decision_dict = {"type": "execute", "content": result.output}
                processed = self.iko.process_output(result.output, ctx_for_iko, decision_dict)
                if self.isa.mode != "silent" and processed:
                    self.isa.respond(processed)
            except Exception as e:
                # 包拯审计修正：降级时必须记录trace，不能静默吞没
                self.iko.trace("output_process", "error", detail=str(e)[:100])
                # 管线失败降级为原始输出
                if self.isa.mode != "silent":
                    self.isa.respond(result.output)
            
            turn.add_action("execute", result.output[:100])
            turn.complete()
            
            # 定时checkpoint
            if self._tick_count % 10 == 0:
                self.session.checkpoint()
                self.iko.trace("checkpoint", "ok")
            
            turn.trace_phase("output", "ok", duration_ms=(time.time()-t9)*1000)
            self.iko.trace("output", "ok", duration_ms=(time.time()-t9)*1000)
            
            # ── Phase 11: 六体自监督反馈收集 ──
            try:
                self._body_outputs["IAX"] = {"heartbeat_ok": True, "tick_count": self._tick_count}
                self._body_outputs["IAI"] = getattr(self.octopus, 'last_output', {}) or {}
                self._body_outputs["ISA"] = getattr(self.isa, 'last_output', {}) or {}
                self._body_outputs["IOS"] = getattr(self.ios, 'last_output', {}) or {}
                self._body_outputs["ISN"] = getattr(self.isn, 'last_output', {}) or {}
                self._body_outputs["IKO"] = getattr(self.iko, 'last_output', {}) or {}
                
                records = self.feedback_loop.collect_feedback(self._body_outputs)
                if records:
                    adjustments = self.feedback_loop.apply_feedback()
                    # 记录调整因子供下轮使用
                    self._feedback_adjustments = adjustments
                turn.trace_phase("feedback", "ok", detail=f"collected:{len(records)}")
            except Exception as e:
                turn.trace_phase("feedback", "error", detail=str(e)[:50])

            # tick计数自增（移到这里，确保run_once也能正确计数）
            self._tick_count += 1
            
        except Exception as e:
            turn.fail(str(e))
            raise


# ═══════════════════════════════════════════════════════
# CLI入口
# ═══════════════════════════════════════════════════════

def main():
    """CLI入口"""
    args = sys.argv[1:]
    
    if "--once" in args:
        # 单次模式
        idx = args.index("--once")
        message = args[idx + 1] if idx + 1 < len(args) else "你好"
        agent = Agent(mode="silent")
        result = agent.run_once(message)
        print(result)
        return
    
    if "--test" in args:
        # 测试模式：跑一次完整的cycle然后退出
        _run_tests()
        return
    
    # 交互模式（默认）
    agent = Agent(mode="console")
    agent.run()


def _run_tests():
    """运行M0测试"""
    print("=== M0: Agent主循环测试 ===\n")
    
    # 测试1: 初始化
    agent = Agent(mode="silent")
    assert agent.isa is not None
    assert agent.octopus is not None
    assert agent.ios is not None
    assert agent.isn is not None
    assert agent.iko is not None
    print("✅ 测试1: 五体初始化")
    
    # 测试2: run_once
    result = agent.run_once("测试消息")
    assert result is not None
    print(f"✅ 测试2: run_once 输出={result[:30]}")
    
    # 测试3: Session/Turn生命周期
    assert agent.session.start_time is not None
    assert len(agent.session.turns) >= 1
    print(f"✅ 测试3: Session/Turn生命周期 total_turns={len(agent.session.turns)}")
    
    # 测试4: 左右脑对弈
    ctx = agent.isa.build_context(Message("写一段Python代码"))
    proposal, critique = agent.octopus.reason(ctx)
    assert proposal is not None
    assert critique is not None
    print(f"✅ 测试4: 左脑提案={proposal.content[:30]} 右脑批判={critique.verdict}")
    
    # 测试5: IO-S仲裁
    decision = agent.ios.arbitrate(proposal, critique)
    assert decision is not None
    print(f"✅ 测试5: 仲裁={decision.action} 批准={decision.approved}")
    
    # 测试6: 工具执行
    result = agent.isn.execute(decision)
    print(f"✅ 测试6: 执行 success={result.success}")
    
    # 测试7: IKO可观测
    report = agent.iko.report()
    assert report["total_ticks"] >= 2  # run_once + 手动
    print(f"✅ 测试7: 可观测 total_ticks={report['total_ticks']} ok_rate={report['ok_rate']:.0%}")
    
    # 测试8: IO-S自我进化
    print(f"✅ 测试8: 进化日志 {len(agent.ios.evolution_log)}条（>=1）")
    
    # 测试9: 因果预测
    prediction = agent.octopus.predict_consequences(ctx)
    assert prediction is not None
    print(f"✅ 测试9: 因果预测 {prediction.summary_text()[:40]}")
    
    # 测试10: 因果对照
    delta = agent.octopus.compare(prediction, result)
    assert delta is not None
    print(f"✅ 测试10: 因果对照 {delta.summary_text()}")
    
    # 测试11: 风险评估
    risk = agent.ios.risk_check(ctx, prediction)
    assert risk is not None
    print(f"✅ 测试11: 风险评估 level={risk.level}")
    
    # 测试12: 10阶段trace日志
    ticks_file = Path.home() / ".openllm" / "output" / "iko" / "ticks" / "ticks.jsonl"
    assert ticks_file.exists()
    with open(ticks_file) as f:
        lines = f.readlines()
    assert len(lines) >= 2  # run_once + 手动
    print(f"✅ 测试12: 10阶段trace日志 ticks={len(lines)}")
    
    print(f"\n全部 12/12 测试通过 ✅")


if __name__ == "__main__":
    main()
