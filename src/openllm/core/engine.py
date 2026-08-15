"""OpenLLM Agent Engine - 整合所有层的完整Agent引擎。

Agent Loop + Provider + Tools + Memory + Identity + Security
六维融为一个真正的对话Agent。

懒加载集成函数已提取到 engine_integrations.py。
工具执行管线已提取到 tool_executor.py。
"""

import os
import re
import sys
import json
import time
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .loop import AgentLoop, LoopPhase, AgentState, TurnContext
from .provider import (
    DeepSeekProvider, ModelConfig, Message, ModelResponse,
    create_provider, DEFAULT_PROVIDER,
)
from ..memory.capsule import (
    MemoryOS, TextCapsule, DeltaCapsule, Arbitrator,
)
from ..identity.soul import (
    Soul, IamPrinciples, IdentityReconstructor, IdentityLevel,
)
from ..identity.iam_integration import IamIntegration, create_iam_integration
from ..security.gate import SecurityFoundation
from .message_bus import MessageQueue
from ..protocol import MessageType, BodyName


# ── 懒加载集成（已提取到 engine_integrations.py）──
from .engine_integrations import (
    lazy_import as _lazy_import,
    get_checkpoint_manager as _get_checkpoint_manager,
    get_isn_metadata as _get_isn_metadata,
    get_isa_on_verify as _get_isa_on_verify,
    get_iko_consume_trace as _get_iko_consume_trace,
    get_isa_schema_matches as _get_isa_schema_matches,
)
from ..security.graceful_shutdown import GracefulShutdown
from ..security.credential_firewall import CredentialFirewall
from ..tools.executor import ToolRegistry, ToolResult, create_default_tools
from .meta import CognitiveDashboard, SelfRescue
from .error_classifier import ErrorClassifier
from .failure_tracker import FailureSignatureTracker, create_tracker
from .oneshot import (
    oneshot as _oneshot_call,
    classify as _classify_call,
    extract as _extract_call,
    summarize as _summarize_call,
)
from .memory_os import MemoryOS as UnifiedMemory
from .display import DisplayEngine
from .gateway import Gateway, CLIAdapter, StealthChannelAdapter

logger = logging.getLogger("openllm.engine")


@dataclass
class AgentConfig:
    """Agent全局配置。"""
    name: str = "OpenLLM"
    provider: str = DEFAULT_PROVIDER
    model: str = "deepseek-chat"
    capsule_dir: str = str(Path.home() / "projects" / "openllm" / "caps")
    max_context_tokens: int = 8192
    checkpoint_interval: int = 10  # 每N轮自动checkpoint
    enable_io_s_checkpoint: bool = True


class OpenLLMEngine:
    """
    OpenLLM完整Agent引擎。

    整合六层：
      Agent Loop  — plan→act→observe→reflect
      Provider    — 模型调用（DeepSeek/OpenAI/本地）
      Tools       — 文件读写+Shell+搜索
      Memory OS   — Δ胶囊+仲裁+检查点
      Identity    — SOUL+Iam+身份重建
      Security    — 3层权限门+审计
    """

    MAX_VERIFY_FILES = 100  # 最多保留 100 个 verify 验证记录
    MAX_HISTORY_TURNS = 50  # dialog history 最多保留 50 轮

    def __init__(self, config: AgentConfig = AgentConfig()):
        self.config = config

        # 各层初始化
        self.loop = AgentLoop(max_context_tokens=config.max_context_tokens)
        self.memory = MemoryOS(Path(config.capsule_dir))
        self.security = SecurityFoundation(Path(config.capsule_dir))
        self.tools = create_default_tools()
        
        # ISN Tool Registry Bridge（动态工具注册）
        from ..isn.tool_registry_bridge import ToolRegistryBridge
        self.tool_bridge = ToolRegistryBridge(self.tools)

        # 身份
        self.soul = Soul(name=config.name)
        self.iam = IamPrinciples()
        self.reconstructor = IdentityReconstructor(self.soul, self.iam)
        
        # Iam Harness v2.3 集成
        self.iam_harness = create_iam_integration(auto_load=True)
        if self.iam_harness.is_loaded():
            logger.info(f"✅ Iam Harness v{self.iam_harness.get_version()} 已集成")
        else:
            logger.warning("⚠️ Iam Harness 未加载，身份约束功能不可用")

        # Provider
        self.provider: Optional[Any] = None  # DeepSeekProvider | AnthropicProvider | GeminiProvider
        self._history: list[Message] = []

        # 元认知
        self.dashboard = CognitiveDashboard(max_context=config.max_context_tokens)
        self.rescue = SelfRescue(self.dashboard)
        self.dashboard.on_overload = self.rescue.on_overload
        self.dashboard.on_fatigue = self.rescue.on_fatigue

        # Phase 8/9: Failure Signature Tracker
        self.failure_tracker = create_tracker()

        # 新模块集成（P0+P1盲区修复）
        self.display = DisplayEngine()
        self.error_classifier = ErrorClassifier()
        self.unified_memory = UnifiedMemory()

        # P2: 优雅停机 + 凭据防火墙
        self.shutdown = GracefulShutdown()
        self.shutdown.register_signal_handlers()
        self.firewall = CredentialFirewall()

        # P1: ISA Gateway（中央网关）
        self.gateway = Gateway(engine=self)
        self.gateway.register(CLIAdapter())

        # verify钩子接入executor（P1: verify-before-complete）
        self.tools.set_verify_hook(self._verify_tool_params)

        # IO-S Checkpoint集成（懒加载，线程安全）
        self._checkpoint_mgr = None
        self._checkpoint_region = "openllm-agent"
        if config.enable_io_s_checkpoint:
            mgr = _get_checkpoint_manager()
            if mgr:
                try:
                    self._checkpoint_mgr = mgr
                    self._checkpoint_mgr.register(
                        self._checkpoint_region,
                        snapshot_fn=self._checkpoint_snapshot,
                        restore_fn=self._checkpoint_restore,
                        metadata={"name": config.name, "version": "0.2.0"}
                    )
                    logger.info(f"  checkpoint region注册: {self._checkpoint_region}")
                except Exception as e:
                    logger.warning(f"checkpoint region注册失败: {e}")
                    self._checkpoint_mgr = None

        # 尝试连接
        self._init_provider()

        # ── 老IO-S治理模式集成 ──
        from .gate import PermissionGate, PermissionMode
        from .tool_scope import ToolScopeManager, ToolScope
        from .pipeline import run_pipeline
        from .hindsight_loop import inject_hindsight

        self.gate = PermissionGate(mode=PermissionMode.DEV)
        self.tool_scope = ToolScopeManager(scope=ToolScope.DEV)
        self._run_pipeline = run_pipeline
        self._inject_hindsight = inject_hindsight

        # P1-1: 消息总线——六体通过publish/subscribe通信
        self.bus = MessageQueue()
        self._register_blood_vessels()

        # P0: 启动安全审计（不阻塞）
        self._startup_audit()

    def _register_blood_vessels(self):
        """注册5个血管handler到消息总线。"""
        # 血管#2: IOS→ISA（verify结果→信念更新）
        def _vessel_2_verify(envelope):
            isa_fn = _get_isa_on_verify()
            if isa_fn:
                isa_fn(
                    tool_name=envelope.payload.get("tool_name", ""),
                    params=envelope.payload.get("params", {}),
                    result=envelope.payload.get("result", {}),
                    verdict=envelope.payload.get("verdict", "pass"),
                )
        self.bus.subscribe(MessageType.GOVERNANCE_EVENT, _vessel_2_verify)

        # 血管#3: ISN→IOS（工具风险检查）
        def _vessel_3_risk(envelope):
            risk = self._check_tool_risk(envelope.payload.get("tool_name", ""))
            envelope.payload["risk_result"] = risk
        self.bus.subscribe(MessageType.TOOL_INVOCATION, _vessel_3_risk)

        # 血管#4: IOS→IKO（工具调用→trace消费）
        def _vessel_4_trace(envelope):
            iko_fn = _get_iko_consume_trace()
            if iko_fn:
                iko_fn(envelope.payload)
        self.bus.subscribe(MessageType.OBSERVABILITY_LOG, _vessel_4_trace)

        # 血管#5: IKO→ISA（schema验证→质量信号）
        def _vessel_5_schema(envelope):
            schema_fn = _get_isa_schema_matches()
            if schema_fn and envelope.payload.get("success"):
                schema_fn(
                    task_type="tool_call",
                    result={
                        "output": envelope.payload.get("output", "")[:1000],
                        "error": envelope.payload.get("error"),
                    },
                )
        self.bus.subscribe(MessageType.DECISION_RESULT, _vessel_5_schema)

        # 血管#1: ISA→IOS（记忆查询→决策上下文）——预留
        def _vessel_1_memory(envelope):
            pass  # 预留，等ISA记忆查询接口稳定后接入
        self.bus.subscribe(MessageType.MEMORY_QUERY, _vessel_1_memory)

    def _init_provider(self) -> bool:
        """初始化模型Provider。支持：deepseek/openai/anthropic/gemini/ollama。"""
        try:
            p = self.config.provider
            if p == "ollama":
                api_key = "ollama"
            elif p == "mimo":
                api_key = os.environ.get("MIMO_API_KEY", "")
                if not api_key:
                    # 从config.json读取
                    import json
                    cfg_path = Path.home() / ".openllm" / "config.json"
                    if cfg_path.exists():
                        cfg = json.loads(cfg_path.read_text())
                        api_key = cfg.get("providers", {}).get("mimo", {}).get("api_key", "")
            elif p == "anthropic":
                api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            elif p == "gemini":
                api_key = os.environ.get("GEMINI_API_KEY", "")
            else:
                api_key = os.environ.get("DEEPSEEK_API_KEY", "")

            if not api_key and p not in ("ollama",):
                return False

            self.provider = create_provider(
                self.config.provider,
                model=self.config.model,
                api_key=api_key,
            )
            return True
        except Exception:
            return False

    # ── P0: 启动安全审计 ────────────────────────────

    def _startup_audit(self):
        """启动安全审计（委托engine_utils）。"""
        from .engine_utils import startup_audit
        startup_audit()

    # ── P1: 一次性LLM调用（不走agent循环） ────────

    def _get_api_key(self) -> str:
        """获取API key（委托engine_utils）。"""
        from .engine_utils import get_api_key
        return get_api_key(self.config.provider)

    def oneshot(self, prompt: str, system: str = "你是一个有帮助的助手。",
                temperature: float = 0.3, max_tokens: int = 1024) -> str:
        """轻量级LLM调用，跳过agent循环/记忆/工具/身份。
        适用于：分类、翻译、摘要、结构化提取、简单问答。
        """
        return _oneshot_call(
            prompt, system=system, model=self.config.model,
            provider=self.config.provider, temperature=temperature,
            max_tokens=max_tokens, api_key=self._get_api_key(),
        )

    def classify(self, text: str, categories: list[str]) -> str:
        """用LLM做分类。返回categories中的一个。"""
        return _classify_call(text, categories, model=self.config.model,
                              provider=self.config.provider, api_key=self._get_api_key())

    def extract(self, text: str, schema: str) -> str:
        """用LLM做结构化提取。返回JSON字符串。"""
        return _extract_call(text, schema, model=self.config.model,
                             provider=self.config.provider, api_key=self._get_api_key())

    def summarize(self, text: str, max_words: int = 100) -> str:
        """用LLM做摘要。"""
        return _summarize_call(text, max_words=max_words, model=self.config.model,
                               provider=self.config.provider, api_key=self._get_api_key())

    @property
    def connected(self) -> bool:
        return self.provider is not None

    def _checkpoint_snapshot(self) -> dict:
        """IO-S checkpoint快照：捕获Agent当前运行态。"""
        return {
            "name": self.config.name,
            "turn_count": self.loop.turn_count,
            "state": self.loop.state.name,
            "context_used": self.loop.context_used,
            "max_context_tokens": self.loop.max_context_tokens,
            "connected": self.connected,
            "timestamp": time.time(),
        }

    def _checkpoint_restore(self, data: dict) -> bool:
        """从IO-S checkpoint恢复Agent运行态。"""
        try:
            self.loop.turn_count = data.get("turn_count", 0)
            state_name = data.get("state", "WORKING")
            for s in AgentState:
                if s.name == state_name:
                    self.loop.state = s
                    break
            self.loop.context_used = data.get("context_used", 0)
            logger.info(f"  checkpoint恢复: turn={self.loop.turn_count}, state={self.loop.state.name}")
            return True
        except Exception as e:
            logger.warning(f"checkpoint恢复失败: {e}")
            return False

    # ── verify钩子（委托tool_validator.py） ──────────

    _TOOL_PARAM_SCHEMAS = {}  # 委托给tool_validator.TOOL_PARAM_SCHEMAS

    def _verify_tool_params(self, tool_name: str, **kwargs) -> dict:
        from .tool_validator import verify_tool_params
        return verify_tool_params(tool_name, **self._TOOL_PARAM_SCHEMAS, **kwargs)

    # ── 生成即验证：代码语法检查 ────────────────────

    def _verify_code_syntax(self, code: str) -> tuple[bool, str]:
        """验证Python代码语法。返回 (成功, 错误信息)。

        使用 py_compile 在临时文件上编译，捕获语法错误。
        不执行代码，只检查语法合法性。
        """
        import py_compile
        import tempfile
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.py', delete=False
            ) as f:
                f.write(code)
                tmp_path = f.name
            py_compile.compile(tmp_path, doraise=True)
            return True, ""
        except py_compile.PyCompileError as e:
            return False, str(e)
        except SyntaxError as e:
            return False, f"SyntaxError: {e.msg} (line {e.lineno})"
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _extract_python_blocks(self, text: str) -> list[str]:
        """从模型输出中提取所有Python代码块内容。"""
        if not text:
            return []
        blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
        # 过滤掉纯空/纯注释块
        return [b for b in blocks if b.strip() and not all(
            line.strip().startswith('#') or not line.strip()
            for line in b.splitlines()
        )]

    def _verify_tool_result(self, tool_name: str, result: ToolResult) -> dict:
        from .tool_validator import verify_tool_result
        return verify_tool_result(tool_name, result)

    def _check_tool_risk(self, tool_name: str) -> dict:
        from .tool_validator import check_tool_risk
        return check_tool_risk(tool_name)

    def wake(self) -> str:
        """苏醒：加载记忆+重建身份+恢复运行态。"""
        # P2: 优雅停机——从DORMANT恢复到RUNNING
        if self.shutdown.is_dormant:
            self.shutdown.begin_wake()
            self.shutdown.complete_wake()

        mem_ctx = self.memory.read()
        session_ctx = {
            "relationship_depth": "老搭档",
            "topic": "general",
        }
        identity = self.reconstructor.reconstruct(session_ctx, mem_ctx)

        # 尝试从IO-S checkpoint恢复Agent运行态
        if self._checkpoint_mgr and self._checkpoint_mgr.restore(self._checkpoint_region):
            logger.info(f"  从checkpoint恢复状态: turn={self.loop.turn_count}")

        self.loop.wake(
            {"prompt": identity, "hash": self.soul.identity_hash},
            mem_ctx,
        )

        # 构建system prompt
        system_prompt = self._build_system_prompt(identity, mem_ctx)
        self._history = []
        if system_prompt:
            self._history.append(Message(role="system", content=system_prompt))

        lines = []
        if mem_ctx.get("status") != "empty":
            decisions = mem_ctx.get("decisions", [])
            insights = mem_ctx.get("insights", [])
            unresolved = mem_ctx.get("unresolved", [])
            if decisions or insights:
                lines.append("📋 记忆已恢复：")
                for d in decisions[:3]:
                    lines.append(f"  · {d}")
                for i in insights[:2]:
                    lines.append(f"  💡 {i}")
                if unresolved:
                    lines.append(f"  ❓ 未解：{unresolved[0]}")

        # P0: Ollama健康检查
        if self.config.provider == "ollama":
            try:
                import urllib.request
                req = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
                models = json.loads(req.read()).get("models", [])
                model_names = [m["name"] for m in models]
                if self.config.model not in model_names:
                    lines.append(f"⚠️ 模型 {self.config.model} 未找到。可用: {', '.join(model_names)}")
                else:
                    lines.append(f"🟢 {self.config.name} 已苏醒。")
            except Exception:
                lines.append(f"🔴 Ollama未运行。请先执行 `ollama serve` 启动服务。")
                lines.append(f"🟢 {self.config.name} 已苏醒（离线模式）。")
        else:
            lines.append(f"🟢 {self.config.name} 已苏醒。")

        if self.connected:
            lines.append(f"🔗 模型：{self.config.model}")
        else:
            lines.append("⚠️ 未连接API（设置 DEEPSEEK_API_KEY 环境变量）")

        # P0: display welcome
        self.display.welcome({
            "name": self.config.name,
            "version": "v0.2.0",
            "session_id": f"s{int(time.time())}",
            "status": "ready" if self.connected else "no_api",
        })

        return "\n".join(lines)

    def _build_system_prompt(self, identity: str, mem_ctx: dict) -> str:
        """构建system prompt（委托engine_utils）。"""
        from .engine_utils import build_system_prompt
        return build_system_prompt(self.tools.list_tools(), identity, mem_ctx)

    # ── P0: 工具执行循环 + 重复检测 ──
    MAX_TOOL_ROUNDS = 5       # 单轮对话最多执行5次工具调用（可配置）
    MAX_REPETITION = 3        # 连续重复自我纠正的阈值（可配置）
    MIN_MSG_LEN = 20          # 短消息过滤阈值（<此长度跳过重复检测，避免工具结果注入误触发）
    _REPETITION_PATTERNS = re.compile(
        r'(让我重新算|不对|我再查|让我重新|重新计算|让我再想|重新思考|让我再算|'
        r'let me recalculate|that\'?s? wrong|let me check again|let me rethink|let me redo)',
        re.IGNORECASE
    )
    # 工具名→默认参数名映射（新增工具时在此添加）
    _TOOL_ARG_MAP = {
        "read_file": "path", "write_file": "path", "search_files": "path",
        "search": "path", "shell": "command", "terminal": "command",
        "python_exec": "command", "list_dir": "path",
        "octopus_search": "query", "octopus_self_model": "query",
        "ocr": "file_path",
    }

    def chat(self, user_input: str, stream: bool = True) -> str:
        """
        一轮完整对话（P0升级：工具执行循环 + 重复检测）。

        1. 停机检查
        2. 安全检查
        3. 记忆注入
        4. 调用模型
        5. 检测工具调用 → 执行 → 反馈 → 循环
        6. 重复检测 → 截断
        7. 返回响应
        """
        if not user_input.strip():
            return ""

        # P2: 优雅停机检查
        if not self.shutdown.can_accept:
            return "⏳ Agent正在停机中，请稍候..."

        # 安全检查
        ok, reason = self.security.check_action("read_memory")
        if not ok:
            return reason

        # P1-3: 记忆注入——从UnifiedMemory KV存储检索，prepend到用户消息
        # 修复v2：记忆注入必须在history.append之前，否则模型看到的是 memory→question 顺序
        memory_context = ""
        try:
            kv_keys = self.unified_memory.list_keys()
            kv_parts = []
            for k in kv_keys:
                v = self.unified_memory.recall(k)
                if v is not None:
                    val_str = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                    kv_parts.append(f"[{k}] {val_str}")
            if kv_parts:
                memory_context = "\n".join(kv_parts)
        except Exception as e:
            logger.debug(f"KV记忆读取跳过: {e}")

        if not memory_context:
            try:
                mem_ctx = self.memory.read()
                if mem_ctx.get("status") == "restored":
                    mem_parts = []
                    decisions = mem_ctx.get("decisions", [])
                    insights = mem_ctx.get("insights", [])
                    if decisions:
                        mem_parts.append("过去决策: " + "; ".join(decisions[:3]))
                    if insights:
                        mem_parts.append("过去洞察: " + "; ".join(insights[:2]))
                    if mem_parts:
                        memory_context = " | ".join(mem_parts)
            except Exception as e:
                logger.debug(f"Capsule记忆读取跳过: {e}")

        # 将记忆prepend到用户消息（模型在history中看到 memory→question 顺序）
        if memory_context:
            user_input = f"[持久记忆]\n{memory_context}\n\n[用户问题]\n{user_input}"

        # Agent Loop: plan
        ctx = self.loop.turn(user_input)
        self._history.append(Message(role="user", content=user_input))

        # 限制 history 长度，防止无限膨胀
        if len(self._history) > self.MAX_HISTORY_TURNS * 2:
            self._trim_history()

        # IO-S checkpoint: 每N轮自动保存Agent运行态
        if self._checkpoint_mgr and self.loop.turn_count % self.config.checkpoint_interval == 0:
            self._checkpoint_mgr.snapshot(self._checkpoint_region)

        # 调用模型
        if not self.connected:
            response = f"[未连接API] 收到。(turn #{self.loop.turn_count})"
            self._history.append(Message(role="assistant", content=response))
            return response

        response = self._call_model(stream=stream)

        # ── P0: 工具执行循环 ──
        tool_rounds = 0
        accumulated_text = []  # 累积工具循环中的文本输出

        while tool_rounds < self.MAX_TOOL_ROUNDS:
            tool_call = self._detect_tool_call(response)
            if not tool_call:
                break

            # 保留模型输出中的文本部分（工具调用之前的文字）
            # 从response中移除工具调用代码块，保留其余文本
            text_before_tool = re.sub(r'```(?:python)?\s*\n.*?```', '', response, flags=re.DOTALL).strip()
            if text_before_tool and len(text_before_tool) > 10:
                accumulated_text.append(text_before_tool)

            tool_name, tool_args = tool_call
            logger.info(f"工具调用检测: {tool_name}({tool_args})")

            # 执行工具
            try:
                result = self.execute_tool(tool_name, **tool_args)
                tool_output = result.output if result.success else f"错误: {result.error}"
            except Exception as e:
                tool_output = f"工具执行异常: {e}"

            # 空结果保护：工具返回空时注入有意义的反馈
            if not tool_output or not tool_output.strip():
                tool_output = f"[工具 {tool_name} 执行完成，但返回了空结果。参数: {tool_args}]"

            # 将工具结果注入历史（只注入工具结果，不注入中间状态的assistant消息）
            truncated = tool_output[:1500] + '...[truncated]...' + tool_output[-500:] if len(tool_output) > 2000 else tool_output
            self._history.append(Message(
                role="user",
                content=f"[工具结果: {tool_name}] {truncated}"
            ))

            # 重新调用模型，让它基于工具结果继续（P1: 改为流式降低感知延迟）
            response = self._call_model(stream=True)
            tool_rounds += 1

        # ── 验证管线：代码验证→声明验证→重复检测→checkpoint ──
        from .verification_pipeline import VerificationPipeline
        pipeline = VerificationPipeline(self)
        report = pipeline.run(response)
        response = report.final_response

        if report.failed_steps > 0:
            logger.warning(f"验证管线: {report.passed_steps}/{report.total_steps}通过, {report.failed_steps}失败")

        # 如果最终response为空但有累积文本，使用累积文本
        if (not response or not response.strip()) and accumulated_text:
            response = "\n\n".join(accumulated_text)

        self._history.append(Message(role="assistant", content=response))
        return response

    def _detect_tool_call(self, text: str) -> Optional[tuple]:
        """从模型输出中检测工具调用意图。返回 (tool_name, args) 或 None。
        
        注意：模型输出可能同时包含文本和工具调用。
        此方法只检测是否存在工具调用，不修改原始text。
        """
        if not text:
            return None

        # 模式1: ```python ... ``` 代码块中包含工具调用
        code_blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
        for block in code_blocks:
            for tool_name in self.tools._tools:
                # 用词边界匹配，避免匹配"result"等变量名
                if re.search(rf'\b{re.escape(tool_name)}\s*\(', block):
                    args = self._extract_args_from_code(block, tool_name)
                    return (tool_name, args)

        # 模式2: 自然语言中的工具请求
        # "使用read_file读取..." / "执行terminal命令..." / "调用search_files..."
        tool_patterns = [
            (r'(?:使用|调用|执行|run|use|call)\s*(\w+)\s*(?:读取|查看|搜索|执行|写入|打开|查找)',
             lambda m: self._extract_natural_args(text, m.group(1))),
            (r'(\w+)\s*\(\s*["\']([^"\']+)["\']',  # read_file("path") 格式
             lambda m: (m.group(1), {"path": m.group(2)}) if m.group(1) in self.tools._tools else None),
        ]

        for pattern, extractor in tool_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result = extractor(match)
                if result:
                    return result

        return None

    def _extract_args_from_code(self, code: str, tool_name: str) -> dict:
        """从Python代码块中提取工具参数。"""
        args = {}
        # 简单提取: tool_name("arg") 或 tool_name(key="value")
        patterns = [
            rf'{tool_name}\s*\(\s*["\']([^"\']+)["\']',  # 位置参数
            rf'{tool_name}\s*\(\s*(\w+)\s*=\s*["\']([^"\']+)["\']',  # 关键字参数
        ]
        for p in patterns:
            m = re.search(p, code)
            if m:
                if len(m.groups()) == 1:
                    # 位置参数：从映射表读取参数名，未知工具默认query
                    param_name = self._TOOL_ARG_MAP.get(tool_name, "query")
                    args[param_name] = m.group(1)
                elif len(m.groups()) == 2:
                    args[m.group(1)] = m.group(2)
                break
        return args

    def _extract_natural_args(self, text: str, tool_name: str) -> Optional[tuple]:
        """从自然语言中提取工具参数。"""
        # 提取引号内容或路径
        quoted = re.findall(r'["\']([^"\']+)["\']', text)
        if quoted:
            param_name = self._TOOL_ARG_MAP.get(tool_name, "query")
            return (tool_name, {param_name: quoted[0]})

        # 提取路径模式
        path_match = re.search(r'[/\\][\w./\\-]+\.\w+', text)
        if path_match and self._TOOL_ARG_MAP.get(tool_name) == "path":
            return (tool_name, {"path": path_match.group()})

        return None

    def _check_repetition(self, response: str) -> str:
        """检测输出中的重复自我纠正模式，必要时截断。"""
        if not response:
            return response

        # 只统计最近3条assistant消息中的自我纠正
        recent_assistant = [
            m.content for m in self._history[-10:]
            if m.role == "assistant"
        ][-self.MAX_REPETITION:]

        repetition_count = 0
        for prev in recent_assistant:
            # 跳过太短的消息（<MIN_MSG_LEN，可能是工具结果注入）
            if len(prev) < self.MIN_MSG_LEN:
                continue
            if self._REPETITION_PATTERNS.search(prev):
                repetition_count += 1

        # 当前输出也有自我纠正模式（且足够长）
        if len(response) > self.MIN_MSG_LEN and self._REPETITION_PATTERNS.search(response):
            repetition_count += 1

        if repetition_count >= self.MAX_REPETITION:
            logger.warning(f"重复检测触发: {repetition_count}次自我纠正")
            return (
                "⚠️ 我发现自己在这个问题上反复纠缠，无法给出清晰答案。\n"
                "建议：换个角度提问，或者这个问题可能超出了我当前的能力范围。"
            )

        return response

    # ── 事实声明验证：用工具回查模型输出中的事实性声明 ──
    _CLAIM_FILE_PATTERN = re.compile(
        r'(?:/[\w.][\w./\-]*|~/[\w./\-]+)\.\w{1,10}(?![\w/])'
    )
    _CLAIM_URL_PATTERN = re.compile(
        r'https?://[^\s\)\]\>\"\']+'
    )
    _CLAIM_LINECOUNT_PATTERN = re.compile(
        r'(?:共|包含|总计|大约?|约|~)\s*(\d{1,6})\s*(?:行|lines?|条|个)',
        re.IGNORECASE
    )
    MAX_CLAIM_VERIFY = 5  # 最多验证5个声明，避免过慢

    def _extract_file_paths_safe(self, response: str) -> list[str]:
        """提取文件路径，排除URL中的路径片段。

        先定位所有URL的[start, end)范围，再从文件路径匹配中
        剔除落入URL范围内的候选。
        """
        # 收集所有URL的字符范围
        url_ranges = [
            (m.start(), m.end()) for m in self._CLAIM_URL_PATTERN.finditer(response)
        ]

        def in_url_range(start: int) -> bool:
            for s, e in url_ranges:
                if s <= start < e:
                    return True
            return False

        # 提取文件路径，排除URL内的
        paths = []
        for m in self._CLAIM_FILE_PATTERN.finditer(response):
            if not in_url_range(m.start()):
                paths.append(m.group())
        return paths

    def _verify_claims(self, response: str) -> str:
        """验证response中的事实性声明，用已注册工具回查。

        提取的声明类型：
          1. 文件路径 → read_file验证文件存在
          2. URL → 基本格式校验（不做网络请求，太重）
          3. 数字/行数声明 → 标记为已提取（无法自动验证）

        如果任何可验证的声明失败，在response末尾追加警告。
        """
        if not response or len(response) < 10:
            return response

        failed: list[str] = []
        verified = 0

        # 1. 文件路径验证（排除URL中的路径）
        file_paths = self._extract_file_paths_safe(response)
        for fp in file_paths[:self.MAX_CLAIM_VERIFY]:
            expanded = os.path.expanduser(fp)
            try:
                from .tool_executor import execute_tool as _exec
                result = _exec(self, "read_file", path=expanded)
                if not result.success:
                    # 也尝试原始路径
                    rel_result = _exec(self, "read_file", path=fp)
                    if not rel_result.success:
                        failed.append(f"文件 {fp} 不可读")
                    else:
                        verified += 1
                else:
                    verified += 1
            except Exception:
                failed.append(f"文件 {fp} 验证异常")

        # 2. URL格式验证（轻量级，不做网络请求）
        urls = self._CLAIM_URL_PATTERN.findall(response)
        for url in urls[:self.MAX_CLAIM_VERIFY - verified]:
            if not re.match(r'https?://[\w\-\.]+\.[\w]{2,}', url):
                failed.append(f"URL格式异常: {url[:80]}")
            else:
                verified += 1

        # 3. 数字声明提取——记录为"已识别但无法自动验证"
        linecount_matches = self._CLAIM_LINECOUNT_PATTERN.findall(response)
        unverifiable = len(linecount_matches)
        if unverifiable > 0:
            logger.debug(f"事实声明验证: 识别到{unverifiable}条数字声明（无法自动验证）")

        if failed:
            warning = (
                "\n\n⚠️ 以上声明未经验证: "
                + "; ".join(failed)
            )
            logger.info(f"事实声明验证: {len(failed)}项失败, {verified}项通过")
            return response + warning

        if verified > 0:
            logger.debug(f"事实声明验证: {verified}项通过, 0项失败")

        return response

    def _call_model(self, stream: bool = True) -> str:
        """调用模型。带错误分类+指数退避重试（P0: error_classifier集成）。"""
        attempt = 0
        max_attempts = 4  # 1 initial + 3 retries

        while attempt < max_attempts:
            try:
                full = []

                def on_token(t: str):
                    # P2: 凭据防火墙——逐token扫描（防止流式泄露到终端）
                    clean_t = self.firewall.scan_text(t)
                    full.append(clean_t)
                    print(clean_t, end="", flush=True)

                resp = self.provider.chat(
                    messages=self._history,
                    stream=stream,
                    on_token=on_token if stream else None,
                )

                # P2: 凭据防火墙——扫描模型输出
                if stream:
                    print()  # 换行
                    raw = "".join(full)
                else:
                    raw = resp.content
                return self.firewall.scan_text(raw)

            except Exception as e:
                classified = self.error_classifier.classify(e)
                if not self.error_classifier.should_retry(classified, attempt):
                    self.display.render_error("模型调用", classified.message)
                    raise
                delay = self.error_classifier.get_retry_delay(classified, attempt)
                logger.warning(
                    f"模型调用失败({classified.category.value}), "
                    f"第{attempt+1}次重试, 等待{delay:.1f}s: {classified.message}"
                )
                time.sleep(delay)
                attempt += 1

        raise RuntimeError(f"模型调用失败: {max_attempts}次重试后放弃")

    def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """执行工具调用（委托给tool_executor.py）。"""
        from .tool_executor import execute_tool as _exec
        return _exec(self, tool_name, **kwargs)

    # ── hindsight↔verify联动 ─────────────────────

    def execute_tool_with_hindsight(self, tool_name: str, pid: str = "",
                                    goal: str = "", **kwargs) -> ToolResult:
        """执行工具+verify+hindsight（委托给tool_executor.py）。"""
        from .tool_executor import execute_tool_with_hindsight as _exec_h
        return _exec_h(self, tool_name, pid=pid, goal=goal, **kwargs)

    @classmethod
    def _trim_verify_files(cls, directory: Path, max_files: int):
        """清理旧verify文件（委托给tool_executor.py）。"""
        from .tool_executor import _trim_verify_files as _trim
        _trim(directory, max_files)

    def _trim_history(self):
        """限制history长度（委托engine_utils）。"""
        from .engine_utils import trim_history
        result = trim_history(self._history, self.MAX_HISTORY_TURNS)
        if result is not None:
            self._history = result

    def sleep(self) -> str:
        """休眠：写Delta胶囊+检查点+保存审计+flush异步记忆+状态转DORMANT。"""
        # P1: 异步记忆flush
        if self.unified_memory.pending_writes > 0:
            self.unified_memory.flush()
            logger.debug(f"统一记忆flush: {self.unified_memory.pending_writes}条待写入")

        # P2: 优雅停机——排空后转DORMANT
        self.shutdown.begin_drain()
        self.shutdown.begin_suspend()

        # IO-S checkpoint: 休眠前最后快照
        if self._checkpoint_mgr:
            self._checkpoint_mgr.snapshot(self._checkpoint_region)

        delta = self.loop.sleep()

        # 提取本次对话的关键信息作为文本胶囊
        decisions = []
        insights = []
        user_msgs = [m for m in self._history if m.role == "user"]
        assistant_msgs = [m for m in self._history if m.role == "assistant"]

        if user_msgs:
            decisions.append({"summary": f"对话{self.loop.turn_count}轮，最后：{user_msgs[-1].content[:80]}"})
        if assistant_msgs:
            insights.append(f"模型最后回应：{assistant_msgs[-1].content[:80]}")

        text = TextCapsule(
            session_id=f"s{int(time.time())}",
            decisions=decisions,
            insights=insights,
            outputs=[m.content[:200] for m in assistant_msgs[-3:]],
        )

        # Δ向量（从真实对话文本提取）
        session_text = f"{text.to_text()} | 对话{self.loop.turn_count}轮"
        delta_vec = DeltaCapsule.from_text(text.session_id, session_text)

        path = self.memory.write(text, delta_vec)

        # 保存完整对话历史
        history_path = self.memory.capsule_dir / f"history_{text.session_id}.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump([{"role": m.role, "content": m.content} for m in self._history], f, ensure_ascii=False, indent=2)

        return f"💾 记忆已保存 → {path}"

    def status(self) -> dict:
        """Agent状态总览。"""
        v = self.loop.check_vitals()
        snap = self.dashboard.evaluate(self.loop.context_used, self.loop.turn_count)
        return {
            "name": self.config.name,
            "version": "0.2.0",
            "state": v["state"],
            "turns": v["turn_count"],
            "context_pct": v["context_used_pct"],
            "connected": self.connected,
            "model": self.config.model if self.connected else "无",
            "capsules": len(list(self.memory.capsule_dir.glob("v06_*.json"))),
            "tools": len(self.tools.list_tools()),
            "security_level": int(self.security.gate.current_level),
            "cognitive": snap.cognitive_state.value,
            "tool_success_rate": f"{snap.tool_success_rate*100:.0f}%",
            "should_compress": snap.should_compress,
        }

    def cognitive_report(self) -> str:
        """生成认知自我报告。"""
        self.dashboard.evaluate(self.loop.context_used, self.loop.turn_count)
        return self.dashboard.self_awareness_report()
