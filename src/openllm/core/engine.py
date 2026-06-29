"""OpenLLM Agent Engine - 整合所有层的完整Agent引擎。

Agent Loop + Provider + Tools + Memory + Identity + Security
六维融为一个真正的对话Agent。
"""

import os
import sys
import json
import time
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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
from ..security.gate import SecurityFoundation
from ..tools.executor import ToolRegistry, ToolResult, create_default_tools
from .meta import CognitiveDashboard, SelfRescue

logger = logging.getLogger("openllm.engine")

# ── IO-S Checkpoint 集成（线程安全懒加载） ───────────
_CHECKPOINT_MANAGER_INITIALIZED = False
_CHECKPOINT_MANAGER_LOCK = threading.Lock()
_CHECKPOINT_MANAGER: Optional["CheckpointManager"] = None

def _get_checkpoint_manager():
    """线程安全地获取 CheckpointManager 单例。懒加载。"""
    global _CHECKPOINT_MANAGER_INITIALIZED, _CHECKPOINT_MANAGER
    if _CHECKPOINT_MANAGER_INITIALIZED:
        return _CHECKPOINT_MANAGER
    with _CHECKPOINT_MANAGER_LOCK:
        if _CHECKPOINT_MANAGER_INITIALIZED:
            return _CHECKPOINT_MANAGER
        io_s_path = Path.home() / "io-s"
        if not io_s_path.exists():
            logger.debug("IO-S 目录不存在，跳过 checkpoint")
            _CHECKPOINT_MANAGER_INITIALIZED = True
            _CHECKPOINT_MANAGER = None
            return None
        if str(io_s_path) not in sys.path:
            sys.path.insert(0, str(io_s_path))
        try:
            from syscall.checkpoint import CheckpointManager
            _CHECKPOINT_MANAGER = CheckpointManager(interval=300, auto_start=False)
            logger.info("✅ IO-S CheckpointManager 已加载")
        except Exception as e:
            logger.warning(f"IO-S Checkpoint 不可用: {e}")
            _CHECKPOINT_MANAGER = None
        _CHECKPOINT_MANAGER_INITIALIZED = True
        return _CHECKPOINT_MANAGER

# ── ISN 元数据集成（线程安全懒加载） ────────────────
_ISN_METADATA_INITIALIZED = False
_ISN_METADATA_LOCK = threading.Lock()
_ISN_METADATA: list[dict] = []
_ISN_METADATA_MAP: dict[str, dict] = {}

def _get_isn_metadata() -> tuple[list[dict], dict[str, dict]]:
    """线程安全地获取 ISN 工具元数据。懒加载。"""
    global _ISN_METADATA_INITIALIZED, _ISN_METADATA, _ISN_METADATA_MAP
    if _ISN_METADATA_INITIALIZED:
        return _ISN_METADATA, _ISN_METADATA_MAP
    with _ISN_METADATA_LOCK:
        if _ISN_METADATA_INITIALIZED:
            return _ISN_METADATA, _ISN_METADATA_MAP
        isn_path = Path.home() / "isn"
        if not isn_path.exists():
            logger.debug("ISN 目录不存在，跳过元数据加载")
            _ISN_METADATA_INITIALIZED = True
            return _ISN_METADATA, _ISN_METADATA_MAP
        try:
            isn_parent = str(isn_path.parent)  # ~/isn 的父目录
            if isn_parent not in sys.path:
                sys.path.insert(0, isn_parent)
            from isn.router.integration import export_tool_metadata
            _ISN_METADATA = export_tool_metadata()
            _ISN_METADATA_MAP = {m["name"]: m for m in _ISN_METADATA}
            logger.info(f"✅ ISN 元数据已加载: {len(_ISN_METADATA)} 条工具")
        except Exception as e:
            logger.warning(f"ISN 元数据不可用: {e}")
            _ISN_METADATA = []
            _ISN_METADATA_MAP = {}
        _ISN_METADATA_INITIALIZED = True
        return _ISN_METADATA, _ISN_METADATA_MAP

# ── ISA 信念更新集成（线程安全懒加载） ──────────────
_ISA_BELIEF_INITIALIZED = False
_ISA_BELIEF_LOCK = threading.Lock()
_ISA_ON_VERIFY = None

def _get_isa_on_verify():
    """线程安全地获取 ISA on_verify_result 回调。懒加载。"""
    global _ISA_BELIEF_INITIALIZED, _ISA_ON_VERIFY
    if _ISA_BELIEF_INITIALIZED:
        return _ISA_ON_VERIFY
    with _ISA_BELIEF_LOCK:
        if _ISA_BELIEF_INITIALIZED:
            return _ISA_ON_VERIFY
        isa_path = Path.home() / "projects" / "isa"
        if not isa_path.exists():
            logger.debug("ISA 目录不存在，跳过信念更新")
            _ISA_BELIEF_INITIALIZED = True
            return None
        try:
            if str(isa_path) not in sys.path:
                sys.path.insert(0, str(isa_path))
            from belief_update import on_verify_result
            _ISA_ON_VERIFY = on_verify_result
            logger.info("✅ ISA belief_update 已加载")
        except Exception as e:
            logger.warning(f"ISA belief_update 不可用: {e}")
            _ISA_ON_VERIFY = None
        _ISA_BELIEF_INITIALIZED = True
        return _ISA_ON_VERIFY

# ── IKO trace消费集成（线程安全懒加载） ──────────────
_IKO_CONSUME_INITIALIZED = False
_IKO_CONSUME_LOCK = threading.Lock()
_IKO_CONSUME_TRACE = None

def _get_iko_consume_trace():
    """线程安全地获取 IKO consume_trace 回调。懒加载。"""
    global _IKO_CONSUME_INITIALIZED, _IKO_CONSUME_TRACE
    if _IKO_CONSUME_INITIALIZED:
        return _IKO_CONSUME_TRACE
    with _IKO_CONSUME_LOCK:
        if _IKO_CONSUME_INITIALIZED:
            return _IKO_CONSUME_TRACE
        iko_path = Path.home() / "projects" / "iko"
        if not iko_path.exists():
            logger.debug("IKO 目录不存在，跳过trace消费")
            _IKO_CONSUME_INITIALIZED = True
            return None
        try:
            if str(iko_path) not in sys.path:
                sys.path.insert(0, str(iko_path))
            from trace_consumer import consume_trace
            _IKO_CONSUME_TRACE = consume_trace
            logger.info("✅ IKO trace_consumer 已加载")
        except Exception as e:
            logger.warning(f"IKO trace_consumer 不可用: {e}")
            _IKO_CONSUME_TRACE = None
        _IKO_CONSUME_INITIALIZED = True
        return _IKO_CONSUME_TRACE


@dataclass
class AgentConfig:
    """Agent全局配置。"""
    name: str = "OpenLLM"
    provider: str = DEFAULT_PROVIDER
    model: str = "deepseek-chat"
    capsule_dir: str = "caps"
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

        # Provider
        self.provider: Optional[DeepSeekProvider] = None
        self._history: list[Message] = []

        # 元认知
        self.dashboard = CognitiveDashboard(max_context=config.max_context_tokens)
        self.rescue = SelfRescue(self.dashboard)
        self.dashboard.on_overload = self.rescue.on_overload
        self.dashboard.on_fatigue = self.rescue.on_fatigue

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

    def _init_provider(self) -> bool:
        """初始化模型Provider。"""
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            return False
        try:
            self.provider = create_provider(
                self.config.provider,
                model=self.config.model,
                api_key=api_key,
            )
            return True
        except Exception:
            return False

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
    }

    def _verify_tool_params(self, tool_name: str, **kwargs) -> dict:
        """工具调用前参数验证（规则引擎，零LLM）。"""
        schema = self._TOOL_PARAM_SCHEMAS.get(tool_name)
        if not schema:
            return {"pass": True, "reason": "无schema验证"}
        for req in schema["required"]:
            if req not in kwargs or kwargs[req] is None:
                return {"pass": False, "reason": f"缺少必填参数: {req}"}
            if not isinstance(kwargs[req], str) or len(str(kwargs[req])) == 0:
                return {"pass": False, "reason": f"参数{req}为空"}
        # 安全检查：Shell注入检测（正则+上下文感知）
        if tool_name == "shell":
            cmd = kwargs.get("command", "")
            # 危险模式白名单：支持变体（空格、制表符）
            import re
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
        """苏醒：加载记忆+重建身份。"""
        mem_ctx = self.memory.read()
        session_ctx = {
            "relationship_depth": "\u8001\u642d\u6863",
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
        lines.append(f"🟢 {self.config.name} 已苏醒。")
        if self.connected:
            lines.append(f"🔗 模型：{self.config.model}")
        else:
            lines.append("⚠️ 未连接API（设置 DEEPSEEK_API_KEY 环境变量）")
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

        1. 安全检查
        2. Agent Loop → plan
        3. 判断是否需要工具调用
        4. 调用模型（流式）
        5. 返回响应
        """
        if not user_input.strip():
            return ""

        # 安全检查
        ok, reason = self.security.check_action("read_memory")
        if not ok:
            return reason

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
        else:
            response = self._call_model(stream=stream)

        self._history.append(Message(role="assistant", content=response))
        return response

    def _call_model(self, stream: bool = True) -> str:
        """调用模型。流式输出到终端。"""
        full = []

        def on_token(t: str):
            full.append(t)
            print(t, end="", flush=True)

        resp = self.provider.chat(
            messages=self._history,
            stream=stream,
            on_token=on_token if stream else None,
        )

        if stream:
            print()  # 换行
            return "".join(full)
        else:
            return resp.content

    def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """执行工具调用（带安全检查+verify钩子）。"""
        ok, reason = self.security.check_action(tool_name)
        if not ok:
            return ToolResult(tool_name=tool_name, success=False, error=reason)

        # ── 血管 #3: ISN 风险检查 ──
        risk_check = self._check_tool_risk(tool_name)
        if not risk_check["pass"]:
            return ToolResult(
                tool_name=tool_name, success=False,
                error=f"ISN风险检查未通过: {risk_check['reason']}"
            )

        # ── verify钩子：调用前参数验证 ──
        param_check = self._verify_tool_params(tool_name, **kwargs)
        if not param_check["pass"]:
            return ToolResult(
                tool_name=tool_name, success=False,
                error=f"参数验证未通过: {param_check['reason']}"
            )

        result = self.tools.execute(tool_name, **kwargs)

        # ── verify钩子：调用后结果验证 ──
        result_check = self._verify_tool_result(tool_name, result)
        if not result_check["pass"]:
            logger.warning(f"结果验证未通过: {tool_name}: {result_check['reason']}")

        # ── 血管 #2: ISA 信念更新（verify → opinion 置信度） ──
        isa_on_verify = _get_isa_on_verify()
        if isa_on_verify:
            try:
                verdict = "pass" if result.success and result_check["pass"] else "fail"
                isa_on_verify(
                    tool_name=tool_name,
                    params=kwargs,
                    result={"output": result.output[:500], "error": result.error},
                    verdict=verdict,
                )
            except Exception as e:
                logger.debug(f"ISA信念更新跳过: {e}")

        # ── 血管 #4: IKO trace消费（工具调用→结构化trace） ──
        iko_consume = _get_iko_consume_trace()
        if iko_consume:
            try:
                iko_consume({
                    "type": "verify_pass" if result.success else "verify_fail",
                    "tool_name": tool_name,
                    "params": {k: str(v)[:100] for k, v in kwargs.items()},
                    "verdict": "pass" if result.success else "fail",
                    "timestamp": time.time(),
                })
            except Exception as e:
                logger.debug(f"IKO trace消费跳过: {e}")

        return result

    # ── hindsight↔verify联动 ─────────────────────

    def execute_tool_with_hindsight(self, tool_name: str, pid: str = "",
                                    goal: str = "", **kwargs) -> ToolResult:
        """执行工具 + verify验证 + hindsight经验自动提取。

        三步一体化：
          1. execute_tool()（含verify钩子）
          2. 提取verify结果
          3. 写入hindsight经验（含验证状态、负面案例标记）

        Args:
            tool_name: 工具名称
            pid: 进程ID（用于hindsight分类）
            goal: 原始目标（用于hindsight理解上下文）
            **kwargs: 工具参数
        """
        result = self.execute_tool(tool_name, **kwargs)

        # 提取verify结果
        verify_pass = result.success
        verify_reason = ""
        if result.success:
            v = self._verify_tool_result(tool_name, result)
            verify_pass = v["pass"]
            verify_reason = v["reason"]
        elif result.error:
            verify_reason = result.error

        # 如果有任务上下文，自动提取hindsight经验
        if pid and goal:
            try:
                # 加载IO-S hindsight_loop（与checkpoint相同路径）
                sys.path.insert(0, str(Path.home() / "io-s"))
                from syscall.hindsight_loop import extract_hindsight

                hindsight_data = extract_hindsight(
                    pid=pid,
                    goal=goal[:200],
                    result={
                        "success": verify_pass,
                        "execution_time": result.latency_ms / 1000.0,
                        "token_count": 0,
                    },
                    failure=verify_reason if not verify_pass else ""
                )

                # 写入verify结果到hindsight经验文件
                verify_record = {
                    "tool": tool_name,
                    "verify_pass": verify_pass,
                    "verify_reason": verify_reason,
                    "hindsight": hindsight_data,
                    "timestamp": time.time(),
                }
                hindsight_dir = Path.home() / ".io-s" / "hindsight"
                hindsight_dir.mkdir(parents=True, exist_ok=True)
                verify_path = hindsight_dir / f"verify_{pid}_{int(time.time())}.json"
                verify_path.write_text(json.dumps(verify_record, ensure_ascii=False))

                # 清理旧 verify 文件，最多保留 MAX_VERIFY_FILES 个
                self._trim_verify_files(hindsight_dir, self.MAX_VERIFY_FILES)

                logger.info(
                    f"  hindsight+verify: {pid}.{tool_name} "
                    f"{'PASS' if verify_pass else 'FAIL'}"
                )
            except Exception as e:
                logger.debug(f"hindsight提取跳过: {e}")

        return result

    @classmethod
    def _trim_verify_files(cls, directory: Path, max_files: int):
        """清理旧 verify 文件，只保留最近的 max_files 个。"""
        if not directory.exists():
            return
        files = sorted(directory.glob("verify_*.json"))
        if len(files) > max_files:
            for f in files[:-max_files]:
                f.unlink()

    def _trim_history(self):
        """限制 history 长度，保留 system prompt + 最近 N 轮对话。"""
        if len(self._history) <= 1:
            return
        # 保留 system prompt (索引 0) + 最近 MAX_HISTORY_TURNS 轮
        system = [self._history[0]] if self._history[0].role == "system" else []
        recent = self._history[-self.MAX_HISTORY_TURNS * 2:]  # user + assistant 各半
        summary = f"[已截断: 原 {len(self._history)} 条消息]"
        self._history = system + recent
        logger.debug(f"History trimmed: kept {len(self._history)}/{len(system) + len(recent)}")

    def sleep(self) -> str:
        """休眠：写Delta胶囊+检查点+保存审计。"""
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
        )

        # Δ向量（从真实对话文本提取）
        session_text = f"{text.to_text()} | 对话{self.loop.turn_count}轮"
        delta_vec = DeltaCapsule.from_text(text.session_id, session_text)

        path = self.memory.write(text, delta_vec)
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
