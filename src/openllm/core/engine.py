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
        """启动安全审计（不阻塞，不抛异常）。"""
        try:
            from ..security.startup_audit import run_startup_audit
            report = run_startup_audit()
            # 只在有警告/失败时输出
            if "❌" in report or "⚠️" in report:
                print(report)
        except Exception as e:
            logger.debug(f"启动审计跳过: {e}")

    # ── P1: 一次性LLM调用（不走agent循环） ────────

    def _get_api_key(self) -> str:
        """获取当前provider的API key。"""
        p = self.config.provider
        if p == "ollama":
            return "ollama"
        elif p == "anthropic":
            return os.environ.get("ANTHROPIC_API_KEY", "")
        elif p == "gemini":
            return os.environ.get("GEMINI_API_KEY", "")
        return os.environ.get("DEEPSEEK_API_KEY", "")

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

    # ── verify钩子 ─────────────────────────────────────

    _TOOL_PARAM_SCHEMAS = {
        "read_file": {"required": ["path"], "check": None},
        "write_file": {"required": ["path", "content"], "check": None},
        "shell": {"required": ["command"], "check": None},
        "search": {"required": ["pattern"], "check": None},
        "list_dir": {"required": [], "check": None},
        "python_exec": {"required": ["code"], "check": None},
        "octopus_search": {"required": ["query"], "check": None},
        "octopus_self_model": {"required": [], "check": None},
    }

    def _verify_tool_params(self, tool_name: str, **kwargs) -> dict:
        """工具调用前参数验证（规则引擎，零LLM）。"""
        schema = self._TOOL_PARAM_SCHEMAS.get(tool_name)
        if not schema:
            return {"pass": True, "reason": "无schema验证"}
        for req in schema["required"]:
            if req not in kwargs or kwargs[req] is None:
                return {"pass": False, "reason": f"缺少必填参数: {req}"}
            val = kwargs[req]
            if isinstance(val, str) and len(val) == 0:
                return {"pass": False, "reason": f"参数{req}为空字符串"}
        # 安全检查：Shell注入检测（正则+上下文感知）
        if tool_name == "shell":
            cmd = kwargs.get("command", "")
            # 危险模式白名单：支持变体（空格、制表符）
            dangerous_patterns = [
                (r'\brm\s+[-/][^;]*\brf\b', "rm -rf 删除操作"),
                (r'\bsudo\s', "sudo 提权操作"),
                (r'\bdestro[y5]\s+', "破坏性命令"),
                (r'>\s*/dev/(sda|sdb|sdc|nvme|mmcblk)', "磁盘写入"),
                (r'\bmkf[sz]\s', "格式化操作"),
                (r'\bdd\s+if=\s*/dev/', "dd 磁盘写入"),
                (r':\(\)\s*\{', "fork炸弹"),
                (r'\b(?:curl|wget)\s+.*?\|', "下载并执行"),
                (r'\bchmod\s+777\s', "权限放开"),
                (r'exec\s+.*[;<`]', "exec 执行注入"),
            ]
            for pattern, reason in dangerous_patterns:
                if re.search(pattern, cmd):
                    return {"pass": False, "reason": f"Shell命令包含危险模式: {reason}"}
        return {"pass": True, "reason": ""}

    def _verify_tool_result(self, tool_name: str, result: ToolResult) -> dict:
        """工具调用后结果验证。"""
        if not result.success:
            return {"pass": False, "reason": f"工具执行失败: {result.error}"}
        if not result.output:
            return {"pass": True, "reason": "成功（无输出）"}
        if len(result.output) > 100000:
            return {"pass": False, "reason": f"输出过长({len(result.output)}字节)，需截断"}
        return {"pass": True, "reason": ""}

    # ── ISN 风险检查（血管 #3: ISN→IO-S） ────────────

    RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    DEFAULT_MAX_RISK = "high"

    def _check_tool_risk(self, tool_name: str) -> dict:
        """检查工具的 ISN 风险等级。

        Returns: {"pass": bool, "reason": str, "risk_level": str}
        """
        _, meta_map = _get_isn_metadata()
        if not meta_map:
            return {"pass": True, "reason": "ISN元数据未加载", "risk_level": "unknown"}

        tool_meta = meta_map.get(tool_name)
        if not tool_meta:
            return {"pass": True, "reason": f"工具'{tool_name}'不在ISN索引中", "risk_level": "unknown"}

        risk = tool_meta.get("risk_level", "low")
        max_risk = self.DEFAULT_MAX_RISK
        if self.RISK_ORDER.get(risk, 0) > self.RISK_ORDER.get(max_risk, 0):
            return {"pass": False, "reason": f"风险等级'{risk}'超过允许上限'{max_risk}'", "risk_level": risk}

        return {"pass": True, "reason": "", "risk_level": risk}

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
        """构建完整system prompt。"""
        parts = [identity]

        # 工具描述
        tools_list = self.tools.list_tools()
        if tools_list:
            parts.append("\n## 可用工具")
            for t in tools_list:
                parts.append(f"- {t['name']}: {t['description']}")

        # 记忆
        if mem_ctx.get("status") != "empty":
            decisions = mem_ctx.get("decisions", [])
            if decisions:
                parts.append(f"\n## 上次决策\n" + "; ".join(decisions[:3]))

        return "\n".join(parts)

    def chat(self, user_input: str, stream: bool = True) -> str:
        """
        一轮完整对话。

        1. 停机检查
        2. 安全检查
        3. Agent Loop → plan
        4. 判断是否需要工具调用
        5. 调用模型（流式）
        6. 返回响应
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

        # Agent Loop: plan
        ctx = self.loop.turn(user_input)
        self._history.append(Message(role="user", content=user_input))

        # P1-3: 记忆注入——从MemoryOS检索相关记忆注入prompt
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
                    memory_injection = "[记忆上下文] " + " | ".join(mem_parts)
                    self._history.append(Message(role="system", content=memory_injection))
        except Exception as e:
            logger.debug(f"记忆注入跳过: {e}")

        # 限制 history 长度，防止无限膨胀
        if len(self._history) > self.MAX_HISTORY_TURNS * 2:
            self._trim_history()

        # IO-S checkpoint: 每N轮自动保存Agent运行态
        if self._checkpoint_mgr and self.loop.turn_count % self.config.checkpoint_interval == 0:
            self._checkpoint_mgr.snapshot(self._checkpoint_region)

        # 调用模型
        if not self.connected:
            response = f"[未连接API] 收到。(turn #{self.loop.turn_count})"
        else:
            response = self._call_model(stream=stream)

        self._history.append(Message(role="assistant", content=response))
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
        """限制 history 长度，保留 system prompt + 最近 N 轮对话。"""
        if len(self._history) <= 1:
            return
        # 保留 system prompt (索引 0) + 最近 MAX_HISTORY_TURNS 轮
        system = [self._history[0]] if self._history[0].role == "system" else []
        recent = self._history[-self.MAX_HISTORY_TURNS * 2:]  # user + assistant 各半
        old_count = len(self._history)
        self._history = system + recent
        logger.debug(f"History trimmed: {old_count} → {len(self._history)}")

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
