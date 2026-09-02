"""harness.py — 四级消融 harness 骨架（第二仗成绩单 v0.2）

四级消融矩阵：
  G0  真裸    → HTTP 直调 ollama，零 harness（由 llm_layer.py 提供）
  G1  裸 ReAct → 最小 think-act-observe 循环，零记忆、零身份
  G2  +记忆    → G1 + MemoryBus 记忆读写（任务前检索注入、任务后写入教训）
  G3  全 harness → G2 + 身份注入 + 温度排序 + 因果记忆

接口统一: run(task_prompt, session) -> dict
不修改任何现有 src/openllm/ 文件。

用法:
    from harness import G0Harness, G1ReActHarness
    g0 = G0Harness()
    result = g0.run("1+1=?", session={})
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

# G0/G1 零 import——只用标准库
# llm_layer.py 本身也是零 openllm import（同目录）
from .llm_layer import llm_chat, MODEL, DEFAULT_TEMPERATURE

logger = logging.getLogger("ablation.harness")


# ═══════════════════════════════════════════════
# 基类
# ═══════════════════════════════════════════════

class BaseHarness(ABC):
    """四级 harness 统一基类。"""

    name: str = "base"

    @abstractmethod
    def run(self, task_prompt: str, session: Dict[str, Any]) -> Dict[str, Any]:
        """执行任务，返回结果字典。

        Args:
            task_prompt: 任务提示词。
            session: 跨轮状态字典（记忆、对话历史等）。
                      各级 harness 可读写，基类不约束结构。

        Returns:
            {
                "response": str,       # 最终回复文本
                "group": str,          # 组别标识 (G0/G1/G2/G3)
                "elapsed_s": float,    # 耗时秒
                "steps": int,          # 推理步数（ReAct 循环次数）
                **extra                # 组内状态（记忆命中数、身份标签等）
            }
        """
        ...


# ═══════════════════════════════════════════════
# G0：真裸
# ═══════════════════════════════════════════════

class G0Harness(BaseHarness):
    """G0 真裸：原生 HTTP 直调 ollama，零 harness。

    没有系统提示、没有 ReAct 循环、没有记忆。
    纯 LLM 对 prompt 的单次应答——作为基线。
    """

    name = "G0"

    def run(self, task_prompt: str, session: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        t0 = time.time()
        response = llm_chat(task_prompt)
        elapsed = time.time() - t0
        return {
            "response": response,
            "group": self.name,
            "elapsed_s": round(elapsed, 2),
            "steps": 1,
        }


# ═══════════════════════════════════════════════
# G1：裸 ReAct
# ═══════════════════════════════════════════════

_REACT_SYSTEM_PROMPT = (
    "你是一个推理助手。请用以下格式逐步思考：\n"
    "Thought: <你的分析>\n"
    "Action: <你要采取的行动或结论>\n"
    "Observation: <行动结果>\n"
    "...\n"
    "Final Answer: <最终答案>\n"
    "当不确定时，继续思考直到找到答案。"
)

_MAX_STEPS = 5  # ReAct 最大循环数


class G1ReActHarness(BaseHarness):
    """G1 裸 ReAct：最小 think-act-observe 循环，零记忆、零身份。

    流程：系统提示（无身份标识）→ ReAct 循环 → 最终答案。
    任务集是纯推理任务，act 阶段=输出答案（无真实工具调用）。
    """

    name = "G1"

    def __init__(self, max_steps: int = _MAX_STEPS):
        self.max_steps = max_steps

    def run(self, task_prompt: str, session: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        t0 = time.time()
        messages: List[Dict[str, str]] = []
        system = _REACT_SYSTEM_PROMPT
        step = 0

        for step in range(1, self.max_steps + 1):
            # 构建当前 prompt
            if step == 1:
                user_msg = task_prompt
            else:
                user_msg = f"继续思考：{task_prompt}"

            # 调用 LLM（拼接历史）
            full_prompt = user_msg
            if messages:
                history = "\n".join(
                    f"[{m['role']}] {m['content']}" for m in messages
                )
                full_prompt = f"之前:\n{history}\n\n{user_msg}"

            response = llm_chat(full_prompt, system=system)

            # 解析是否到达 Final Answer
            if "Final Answer:" in response or "最终答案:" in response:
                messages.append({"role": "assistant", "content": response})
                break

            messages.append({"role": "assistant", "content": response})

        elapsed = time.time() - t0
        final_text = messages[-1]["content"] if messages else response
        return {
            "response": final_text,
            "group": self.name,
            "elapsed_s": round(elapsed, 2),
            "steps": step,
            "messages_count": len(messages),
        }


# ═══════════════════════════════════════════════
# G2：+ 记忆
# ═══════════════════════════════════════════════

class G2MemoryHarness(BaseHarness):
    """G2 + 记忆：G1 + MemoryBus 记忆读写。

    任务前：从 MemoryBus 检索相关记忆，注入系统提示。
    任务后：将本轮"教训"写回 MemoryBus。
    LLM 调用走 llm_layer.py（同 G0/G1 公平通道）。

    session 结构：
        session["memory_bus"] = MemoryBus 实例（由外部注入）
    """

    name = "G2"

    def __init__(self, max_steps: int = _MAX_STEPS):
        self.max_steps = max_steps

    def run(self, task_prompt: str, session: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()

        # ── 获取 MemoryBus ──
        bus = session.get("memory_bus")
        memories_injected: List[str] = []
        write_result: Optional[Dict] = None

        # ── 任务前：检索相关记忆 ──
        memory_context = ""
        if bus is not None:
            try:
                from openllm.memory.memory_bus import Query
                query = Query(text=task_prompt, top_k=3, token_budget=500)
                records = bus.query(query)
                if records:
                    lines = [f"- [{r.source}] {r.content[:120]}" for r in records]
                    memory_context = "\n".join(lines)
                    memories_injected = [r.content[:80] for r in records]
            except Exception as e:
                logger.warning(f"G2 记忆检索失败(降级): {e}")

        # ── 构建系统提示 ──
        system = _REACT_SYSTEM_PROMPT
        if memory_context:
            system = (
                f"以下是从记忆中检索到的相关经验：\n{memory_context}\n\n"
                f"请参考以上经验进行推理。\n\n{system}"
            )

        # ── ReAct 循环 ──
        messages: List[Dict[str, str]] = []
        step = 0
        for step in range(1, self.max_steps + 1):
            user_msg = task_prompt if step == 1 else f"继续思考：{task_prompt}"
            full_prompt = user_msg
            if messages:
                history = "\n".join(
                    f"[{m['role']}] {m['content']}" for m in messages
                )
                full_prompt = f"之前:\n{history}\n\n{user_msg}"
            response = llm_chat(full_prompt, system=system)
            if "Final Answer:" in response or "最终答案:" in response:
                messages.append({"role": "assistant", "content": response})
                break
            messages.append({"role": "assistant", "content": response})

        # ── 任务后：写入教训 ──
        if bus is not None:
            try:
                from openllm.memory.memory_bus import WriteRequest
                final = messages[-1]["content"] if messages else response
                lesson = f"任务: {task_prompt[:60]}... → 回答: {final[:120]}"
                wr = WriteRequest(
                    content=lesson,
                    source="ablation_g2",
                    record_type="lesson",
                    importance=0.5,
                    tags=["ablation", "g2"],
                    session_id=session.get("session_id", "ablation"),
                )
                result = bus.write(wr)
                write_result = {"success": result.success, "provider": result.provider}
            except Exception as e:
                logger.warning(f"G2 记忆写入失败(降级): {e}")

        elapsed = time.time() - t0
        final_text = messages[-1]["content"] if messages else response
        return {
            "response": final_text,
            "group": self.name,
            "elapsed_s": round(elapsed, 2),
            "steps": step,
            "memories_injected_count": len(memories_injected),
            "write_result": write_result,
        }


# ═══════════════════════════════════════════════
# G3：全 harness
# ═══════════════════════════════════════════════

_IDENTITY_SYSTEM_PROMPT = (
    "你是一个有持续记忆和身份感的智能体。你拥有跨会话的因果记忆，"
    "能够从过去的成功和失败中学习。你的记忆按重要性和温度排序，"
    "最相关且最新的经验优先呈现给你。"
    "请基于你的记忆和经验来回答问题。\n\n"
)


class G3FullHarness(BaseHarness):
    """G3 全 harness：G2 + 身份 + 温度排序 + 因果记忆。

    在 G2 基础上增强：
    1. 身份注入（ISA 风格身份 system prompt）
    2. 温度排序（高温度记忆优先注入——温度=重要性×时间衰减）
    3. 因果记忆（通过 CausalProvider 检索因果教训链）

    session 结构：
        session["memory_bus"] = MemoryBus 实例
    """

    name = "G3"

    def __init__(self, max_steps: int = _MAX_STEPS):
        self.max_steps = max_steps

    def run(self, task_prompt: str, session: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()

        bus = session.get("memory_bus")
        memories_injected: List[str] = []
        causal_hits: List[str] = []
        write_result: Optional[Dict] = None

        # ── 任务前：检索 + 温度排序 + 因果记忆 ──
        memory_context = ""
        if bus is not None:
            # 常规记忆（按 temperature 降序）
            try:
                from openllm.memory.memory_bus import Query
                query = Query(text=task_prompt, top_k=5, token_budget=800)
                records = bus.query(query)
                # 按 temperature 降序排序（高温度优先）
                records.sort(key=lambda r: r.temperature, reverse=True)
                if records:
                    lines = [f"- [{r.source}|T={r.temperature:.2f}] {r.content[:120]}"
                             for r in records[:3]]
                    memory_context = "\n".join(lines)
                    memories_injected = [r.content[:80] for r in records[:3]]
            except Exception as e:
                logger.warning(f"G3 记忆检索失败(降级): {e}")

            # 因果记忆（CausalProvider 独立检索）
            try:
                from openllm.memory.providers.causal_provider import CausalProvider
                from openllm.memory.memory_bus import Query as Q2
                causal_prov = CausalProvider()
                causal_query = Q2(text=task_prompt, top_k=3, token_budget=300)
                causal_records = causal_prov.search(causal_query)
                if causal_records:
                    causal_lines = [
                        f"- [因果|conf={r.context.get('confidence', '?')}] "
                        f"{r.content[:120]}"
                        for r in causal_records
                    ]
                    causal_hits = [r.content[:80] for r in causal_records]
                    if memory_context:
                        memory_context += "\n" + "\n".join(causal_lines)
                    else:
                        memory_context = "\n".join(causal_lines)
            except Exception as e:
                logger.debug(f"G3 因果记忆检索失败(降级): {e}")

        # ── 构建系统提示（身份 + 记忆 + ReAct）──
        system = _IDENTITY_SYSTEM_PROMPT
        if memory_context:
            system += (
                f"以下是从你的记忆中检索到的相关经验（按重要性排序）：\n"
                f"{memory_context}\n\n"
            )
        system += _REACT_SYSTEM_PROMPT

        # ── ReAct 循环 ──
        messages: List[Dict[str, str]] = []
        step = 0
        for step in range(1, self.max_steps + 1):
            user_msg = task_prompt if step == 1 else f"继续思考：{task_prompt}"
            full_prompt = user_msg
            if messages:
                history = "\n".join(
                    f"[{m['role']}] {m['content']}" for m in messages
                )
                full_prompt = f"之前:\n{history}\n\n{user_msg}"
            response = llm_chat(full_prompt, system=system)
            if "Final Answer:" in response or "最终答案:" in response:
                messages.append({"role": "assistant", "content": response})
                break
            messages.append({"role": "assistant", "content": response})

        # ── 任务后：写入教训 + 因果记录 ──
        if bus is not None:
            try:
                from openllm.memory.memory_bus import WriteRequest
                final = messages[-1]["content"] if messages else response
                lesson = f"任务: {task_prompt[:60]}... → 回答: {final[:120]}"
                wr = WriteRequest(
                    content=lesson,
                    source="ablation_g3",
                    record_type="lesson",
                    importance=0.7,
                    tags=["ablation", "g3", "causal"],
                    session_id=session.get("session_id", "ablation"),
                )
                result = bus.write(wr)
                write_result = {"success": result.success, "provider": result.provider}
            except Exception as e:
                logger.warning(f"G3 记忆写入失败(降级): {e}")

            # 因果记录（预测→实际→delta）
            try:
                final = messages[-1]["content"] if messages else response
                bus.record_causal(
                    action="ablation_task",
                    prediction=task_prompt[:60],
                    actual=final[:60],
                    success=True,
                    context="ablation_harness",
                )
            except Exception as e:
                logger.debug(f"G3 因果记录失败(非阻断): {e}")

        elapsed = time.time() - t0
        final_text = messages[-1]["content"] if messages else response
        return {
            "response": final_text,
            "group": self.name,
            "elapsed_s": round(elapsed, 2),
            "steps": step,
            "memories_injected_count": len(memories_injected),
            "causal_hits_count": len(causal_hits),
            "write_result": write_result,
            "identity_injected": True,
        }


# ═══════════════════════════════════════════════
# 便捷工厂
# ═══════════════════════════════════════════════

def get_harness(group: str) -> BaseHarness:
    """按组名获取 harness 实例。

    Args:
        group: "G0" | "G1" | "G2" | "G3"

    Returns:
        对应的 harness 实例。

    Raises:
        ValueError: 未知组名。
    """
    mapping = {
        "G0": G0Harness,
        "G1": G1ReActHarness,
        "G2": G2MemoryHarness,
        "G3": G3FullHarness,
    }
    if group not in mapping:
        raise ValueError(f"未知组名: {group}，有效值: {list(mapping.keys())}")
    return mapping[group]()
