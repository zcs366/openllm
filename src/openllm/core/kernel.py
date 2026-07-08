"""kernel.py — openLLM微内核

来自老IO-S kernel.py:
  - syscall注册表 + dispatch + cap_check + 标准信封
  - OPERATION_MAP语义→资源翻译
  - syscall名称→resource自动推断

搬到openLLM后：
  - 保留全部治理逻辑
  - 与openLLM的Session/Turn模型集成
  - ProcessTable用openLLM自己的AgentState
"""

import logging
import time
import uuid
from typing import Any, Callable, Optional

from .cap_policy import (
    CapPolicy, ErrCode, ok_response, error_response,
    OPERATION_MAP, SYSCALL_RESOURCE_MAP, OPENLLM_HOME,
)

logger = logging.getLogger("openllm.kernel")


class Kernel:
    """openLLM 微内核 — 零root治理 + syscall调度 + 信号路由"""

    def __init__(self, agent_id: str = "openllm-kernel"):
        self.agent_id = agent_id

        # 治理根基
        self._cap = CapPolicy()

        # syscall注册表
        self._syscall_table: dict[str, Callable] = {}

        # 心跳
        self._start_time = time.time()
        self._syscall_count = 0
        self._trace_counter = 0

        logger.info(
            f"⚙️ {agent_id} 内核启动 — "
            f"零root={'是' if self._cap.is_zero_root() else '否'}")

    # ── syscall注册 ──

    def register(self, name: str, handler: Callable):
        self._syscall_table[name] = handler

    def dispatch(self, name: str, caller_pid: str = "kernel",
                 args: Optional[dict] = None,
                 trace_id: Optional[str] = None,
                 process_cap: Optional[dict] = None) -> dict:
        """调用系统调用（含cap_check + 标准信封）。

        Returns: {"ok", "data", "error", "trace_id"}
        """
        self._syscall_count += 1
        tid = trace_id or (
            f"tr-{int(time.time()*1000)}-{self._trace_counter}")
        self._trace_counter += 1
        args = args or {}

        # 1. 检查syscall存在性
        if name not in self._syscall_table:
            return error_response(
                ErrCode.UNKNOWN_SYSCALL,
                f"未知系统调用: {name}", tid)

        # 2. cap_check（除kernel自身外都检查）
        if caller_pid != "kernel":
            resource = args.get("resource", "")
            operation = args.get("operation", "")
            # 自动推断
            if not resource or not operation:
                inferred = SYSCALL_RESOURCE_MAP.get(name)
                if inferred:
                    resource = resource or inferred[0]
                    operation = operation or inferred[1]
            # 语义→资源映射
            mapped_op = OPERATION_MAP.get(operation, operation)
            allowed, reason = self._cap.check(
                caller_pid, mapped_op, resource,
                process_cap=process_cap)
            if not allowed:
                logger.warning(
                    f"⛔ CAP_DENIED: {caller_pid} "
                    f"{operation} {resource}: {reason}")
                return error_response(
                    ErrCode.CAP_DENIED, reason, tid)

        # 3. 执行handler（过滤掉治理参数）
        handler_args = {
            k: v for k, v in args.items()
            if k not in ('resource', 'operation')}
        try:
            result = self._syscall_table[name](
                caller_pid=caller_pid, **handler_args)
            return ok_response(result, tid)
        except Exception as e:
            logger.error(f"❌ syscall {name} 异常: {e}")
            return error_response(
                ErrCode.INTERNAL_ERROR, str(e), tid)

    def list_syscalls(self) -> list[str]:
        return sorted(self._syscall_table.keys())

    # ── cap接口 ──

    def cap_check(self, caller_pid: str, operation: str,
                  resource_type: str,
                  resource_name: Optional[str] = None
                  ) -> tuple[bool, str]:
        """外部cap_check接口——供syscall handler内部调用。"""
        if caller_pid == "kernel":
            return True, ""
        return self._cap.check(
            caller_pid, operation, resource_type, resource_name)

    def get_cap_policy(self) -> dict:
        return {
            "zero_root": self._cap.is_zero_root(),
            "resources": list(
                self._cap._policy.get("resources", {}).keys()),
        }

    def get_stats(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "uptime_s": round(time.time() - self._start_time),
            "syscall_count": self._syscall_count,
            "registered_syscalls": len(self._syscall_table),
            "zero_root": self._cap.is_zero_root(),
        }
