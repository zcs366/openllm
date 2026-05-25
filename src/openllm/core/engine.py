"""
OpenLLM Agent Engine — 整合所有层的完整Agent引擎。

Agent Loop + Provider + Tools + Memory + Identity + Security
六维融为一个真正的对话Agent。
"""

import os
import time
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

import numpy as np


@dataclass
class AgentConfig:
    """Agent全局配置。"""
    name: str = "OpenLLM"
    provider: str = DEFAULT_PROVIDER
    model: str = "deepseek-chat"
    capsule_dir: str = "caps"
    max_context_tokens: int = 8192


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

    def wake(self) -> str:
        """苏醒：加载记忆+重建身份。"""
        mem_ctx = self.memory.read()
        session_ctx = {
            "relationship_depth": "老搭档",
            "topic": "general",
        }
        identity = self.reconstructor.reconstruct(session_ctx, mem_ctx)
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
        """执行工具调用（带安全检查）。"""
        ok, reason = self.security.check_action(tool_name)
        if not ok:
            return ToolResult(tool_name=tool_name, success=False, error=reason)
        return self.tools.execute(tool_name, **kwargs)

    def sleep(self) -> str:
        """休眠：写Δ胶囊+检查点+保存审计。"""
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

        # Δ向量（Phase 1: 简化版。未来从模型隐状态提取）
        delta_vec = DeltaCapsule(
            session_id=text.session_id,
            vector=np.random.randn(256).astype(np.float32) * 0.01,
        )

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
