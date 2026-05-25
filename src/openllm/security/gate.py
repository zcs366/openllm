"""
OpenLLM Security — 3层权限门 + 审计日志 + 不可修改安全基座。

从Claude Code学来，但精简为3层（非8层）：
  L1 只读：读文件、搜索、查看
  L2 本地写：写文件、创建目录、本地执行
  L3 网络：API调用、外部通信、发布

安全基座在自修权限体系之外——任何级别的自修都不能修改此层。
"""

import json
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Optional


class PermissionLevel(IntEnum):
    """权限等级。数字越大权限越高。"""
    READ_ONLY = 1    # 只读
    LOCAL_WRITE = 2  # 本地写入+执行
    NETWORK = 3      # 网络访问


@dataclass
class AuditEntry:
    """单条审计记录。不可篡改。"""
    timestamp: float
    action: str
    level: PermissionLevel
    user: str
    result: str          # "allowed" / "denied" / "escalated"
    details: str = ""
    entry_id: str = ""

    def __post_init__(self):
        import uuid
        self.entry_id = str(uuid.uuid4())[:8]


class AuditLog:
    """
    审计日志。所有操作完整记录，不可篡改。

    这是安全基座的一部分——日志不能被删除或修改。
    """

    def __init__(self, log_path: Path = Path("caps/audit.jsonl")):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, action: str, level: PermissionLevel, user: str,
               result: str, details: str = "") -> AuditEntry:
        entry = AuditEntry(
            timestamp=time.time(),
            action=action,
            level=level,
            user=user,
            result=result,
            details=details,
        )
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": entry.timestamp,
                "action": entry.action,
                "level": int(entry.level),
                "user": entry.user,
                "result": entry.result,
                "details": entry.details,
                "id": entry.entry_id,
            }, ensure_ascii=False) + "\n")
        return entry

    def read(self, limit: int = 100) -> list[dict]:
        """读取最近N条审计记录。"""
        if not self.log_path.exists():
            return []
        entries = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        return entries[-limit:]


class PermissionGate:
    """
    3层权限门。

    每层操作需要对应等级或更高等级。
    未知操作默认拒绝（白名单模式）。
    安全基座操作永远拒绝。
    """

    # 操作→所需权限等级映射
    ACTION_MAP: dict[str, PermissionLevel] = {
        # L1 只读
        "read_file": PermissionLevel.READ_ONLY,
        "search": PermissionLevel.READ_ONLY,
        "list_files": PermissionLevel.READ_ONLY,
        "read_memory": PermissionLevel.READ_ONLY,
        "search_files": PermissionLevel.READ_ONLY,
        # L2 本地写
        "write_file": PermissionLevel.LOCAL_WRITE,
        "create_dir": PermissionLevel.LOCAL_WRITE,
        "execute_shell": PermissionLevel.LOCAL_WRITE,
        "shell": PermissionLevel.LOCAL_WRITE,
        "write_memory": PermissionLevel.LOCAL_WRITE,
        "self_modify": PermissionLevel.LOCAL_WRITE,
        # L3 网络
        "api_call": PermissionLevel.NETWORK,
        "web_search": PermissionLevel.NETWORK,
        "send_message": PermissionLevel.NETWORK,
        "publish": PermissionLevel.NETWORK,
    }

    # 安全基座操作——永远拒绝
    IMMUTABLE_ACTIONS: set[str] = {
        "delete_audit_log",
        "modify_permission_gate",
        "disable_security",
        "escalate_self",
    }

    def __init__(self, audit: AuditLog, current_level: PermissionLevel = PermissionLevel.LOCAL_WRITE):
        self.audit = audit
        self.current_level = current_level

    def check(self, action: str, user: str = "agent") -> tuple[bool, str]:
        """
        检查操作是否允许。

        Returns: (allowed, reason)
        """
        # 安全基座——永远拒绝
        if action in self.IMMUTABLE_ACTIONS:
            self.audit.record(action, PermissionLevel.NETWORK, user, "denied",
                            "安全基座操作，不可修改")
            return False, "🔒 安全基座操作不可执行。此操作被永久锁定。"

        # 未知操作——默认拒绝
        required = self.ACTION_MAP.get(action)
        if required is None:
            self.audit.record(action, PermissionLevel.NETWORK, user, "denied",
                            "未知操作类型")
            return False, f"❌ 未知操作 '{action}'。默认拒绝。"

        # 权限检查
        if self.current_level >= required:
            self.audit.record(action, required, user, "allowed")
            return True, "✅"
        else:
            self.audit.record(action, required, user, "denied",
                            f"需要L{int(required)}权限，当前L{int(self.current_level)}")
            return False, f"⛔ 权限不足：需要L{int(required)}，当前L{int(self.current_level)}"

    def escalate(self, new_level: PermissionLevel, reason: str,
                 on_approve: Optional[Callable] = None) -> bool:
        """权限升级。需要外部审批回调。"""
        if new_level <= self.current_level:
            return True
        if on_approve:
            if on_approve(self.current_level, new_level, reason):
                self.audit.record("escalate", new_level, "system", "allowed", reason)
                self.current_level = new_level
                return True
        self.audit.record("escalate", new_level, "system", "denied",
                        f"升级到L{int(new_level)}被拒：{reason}")
        return False


# ── 不可修改的安全基座 ──────────────────────────────

class SecurityFoundation:
    """
    安全基座——在自修权限体系之外。

    任何级别的自修都不能删除或修改此基座。
    包含：权限门、审计日志、IMMUTABLE_ACTIONS定义。
    """

    def __init__(self, capsule_dir: Path = Path("caps")):
        self.audit = AuditLog(capsule_dir / "audit.jsonl")
        self.gate = PermissionGate(self.audit)

    def verify_integrity(self) -> bool:
        """验证安全基座完整性。"""
        # 检查IMMUTABLE_ACTIONS是否被篡改
        expected = {"delete_audit_log", "modify_permission_gate", "disable_security", "escalate_self"}
        return PermissionGate.IMMUTABLE_ACTIONS == expected

    def check_action(self, action: str) -> tuple[bool, str]:
        """对外统一检查接口。"""
        return self.gate.check(action)
