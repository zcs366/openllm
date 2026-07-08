"""cap_policy.py — 治理根基：能力策略

来自老IO-S kernel.py的CapPolicy：
  - cap_policy.json定义谁对什么资源能做什么操作
  - 零root：加载后只读，无人可改
  - syscall用语义语言(send/recv/spawn)，cap_policy用资源语言(read/write)
  - OPERATION_MAP做翻译

搬到openLLM后：
  - 去掉IO-S路径依赖，用openLLM自己的config目录
  - 保留全部治理逻辑
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("openllm.cap_policy")

OPENLLM_HOME = Path.home() / ".openllm"


# ── 标准错误码 ──

class ErrCode:
    OK = "ok"
    UNKNOWN_SYSCALL = "unknown_syscall"
    CAP_DENIED = "cap_denied"
    RESOURCE_NOT_FOUND = "resource_not_found"
    INVALID_ARGS = "invalid_args"
    INTERNAL_ERROR = "internal_error"
    CAP_POLICY_LOCKED = "cap_policy_locked"


# ── 标准响应信封 ──

import uuid


def ok_response(data, trace_id: Optional[str] = None) -> dict:
    return {"ok": True, "data": data, "error": None,
            "trace_id": trace_id or str(uuid.uuid4())}


def error_response(code: str, message: str,
                   trace_id: Optional[str] = None) -> dict:
    return {"ok": False, "data": None,
            "error": {"code": code, "message": message},
            "trace_id": trace_id or str(uuid.uuid4())}


# ── 语义→资源映射 ──

OPERATION_MAP = {
    "send": "write",     # 发信号→写信号文件
    "recv": "read",      # 收信号→读信号文件
    "spawn": "write",    # 创建进程→写进程表
    "kill": "write",     # 杀进程→更新进程表
    "create": "write",   # 创建卡片→写卡片文件
    "list": "read",      # 列资源→读资源索引
    "read": "read",      # 直通
    "write": "write",    # 直通
}

# syscall名称 → (resource_type, default_operation)
SYSCALL_RESOURCE_MAP = {
    "signal_send":      ("signal",  "send"),
    "signal_recv":      ("signal",  "recv"),
    "signal_broadcast": ("signal",  "send"),
    "card_create":      ("card",    "create"),
    "card_read":        ("card",    "read"),
    "card_append":      ("card",    "write"),
    "card_archive":     ("card",    "write"),
    "card_split":       ("card",    "write"),
    "card_list":        ("card",    "list"),
    "card_batch_commit": ("card",   "write"),
    "agent_spawn":      ("process", "spawn"),
    "agent_kill":       ("process", "kill"),
    "agent_status":     ("process", "read"),
    "agent_list":       ("process", "list"),
    "recall_append":    ("card",    "write"),
    "recall_query":     ("card",    "read"),
}


class CapPolicy:
    """cap_policy.json 的加载器和查询接口。

    IO-S的治理根基——定义谁可以访问什么资源。
    文件一旦加载，kernel只能读不能改——零root。
    """

    def __init__(self, path: Optional[Path] = None):
        self._path = path or OPENLLM_HOME / "cap_policy.json"
        self._policy: dict = {}
        self._loaded = False
        self.load()

    def load(self):
        if not self._path.exists():
            logger.warning(f"⚠️ cap_policy.json 不存在: {self._path}")
            self._policy = self._default_policy()
            self._loaded = False
            return

        try:
            self._policy = json.loads(self._path.read_text())
            self._loaded = True
            logger.info(
                f"✅ cap_policy loaded "
                f"({len(self._policy.get('resources', {}))} 类型)")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"❌ cap_policy 加载失败: {e}")
            raise

    def _default_policy(self) -> dict:
        return {
            "_schema": "cap_policy/v1",
            "_zero_root": True,
            "resources": {},
            "boot": {"initial_cap": {
                "cards": [], "processes": [],
                "signals": [], "cap_policy": [],
            }},
            "builtin_agents": {"kernel": {"bypass_cap": True}},
        }

    def check(self, caller_pid: str, operation: str, resource_type: str,
              resource_name: Optional[str] = None,
              process_cap: Optional[dict] = None) -> tuple[bool, str]:
        """检查一个Process是否有权操作指定资源。

        Returns: (允许与否, 拒绝原因)
        """
        if caller_pid == "kernel":
            return True, ""

        # 优先检查Process自身的cap（通过继承链获得）
        if process_cap:
            perms_list = process_cap.get(resource_type, [])
            for entry in perms_list:
                perms = entry.get("perms", [])
                if operation in perms:
                    pattern = entry.get("pattern", "*")
                    if pattern == "*" or (
                            resource_name and pattern in resource_name):
                        return True, ""
            return False, (
                f"Process {caller_pid} 的cap不允许 "
                f"{operation} {resource_type}")

        # 回退到静态policy
        resources = self._policy.get("resources", {})
        res_def = resources.get(resource_type)
        if not res_def:
            return False, f"未知资源类型: {resource_type}"
        default_acl = res_def.get("default_acl", {})
        allowed_pids = default_acl.get(operation, [])
        if "kernel" in allowed_pids and caller_pid == "kernel":
            return True, ""
        if "owner" in allowed_pids:
            return True, ""
        return False, (
            f"Process {caller_pid} 没有被授权 "
            f"{operation} {resource_type}")

    def is_zero_root(self) -> bool:
        return self._policy.get("_zero_root", False)

    def get_initial_cap(self) -> dict:
        return self._policy.get("boot", {}).get("initial_cap", {})

    def __repr__(self):
        return (f"<CapPolicy loaded={self._loaded} "
                f"zero_root={self.is_zero_root()}>")
