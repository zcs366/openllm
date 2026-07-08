"""
openLLM Startup Security Audit — 启动安全态势审计。

从 Hermes v0.18.0 的 security posture audit 学来。
启动时自动检查安全配置，不阻塞，不抛异常。

检查项：
  1. config.json 明文API key
  2. 配置文件权限
  3. 开放端口
  4. 磁盘空间
  5. capsule完整性
  6. Provider endpoint是否HTTPS
"""

import json
import logging
import os
import socket
import stat
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List

logger = logging.getLogger("openllm.security.startup_audit")

OPENLLM_HOME = Path.home() / ".openllm"


class AuditStatus(Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    INFO = "info"


@dataclass
class AuditResult:
    """单条审计结果。"""
    check_name: str
    status: AuditStatus
    message: str
    details: str = ""


class StartupAuditor:
    """启动安全审计器。"""

    def __init__(self, home: Path = OPENLLM_HOME):
        self.home = home

    def run_all(self) -> List[AuditResult]:
        """执行所有检查。不抛异常。"""
        checks = [
            self._check_config_keys,
            self._check_file_permissions,
            self._check_open_ports,
            self._check_disk_space,
            self._check_capsule_integrity,
            self._check_endpoint_https,
        ]
        results = []
        for check in checks:
            try:
                result = check()
                if isinstance(result, list):
                    results.extend(result)
                else:
                    results.append(result)
            except Exception as e:
                results.append(AuditResult(
                    check_name=check.__name__,
                    status=AuditStatus.WARN,
                    message=f"检查失败: {e}",
                ))
        return results

    def _check_config_keys(self) -> AuditResult:
        """检查config.json是否有明文API key。"""
        config_path = self.home / "config.json"
        if not config_path.exists():
            return AuditResult("config_keys", AuditStatus.PASS, "无config.json")

        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return AuditResult("config_keys", AuditStatus.WARN, "config.json解析失败")

        providers = config.get("providers", {})
        plaintext_keys = []
        for name, prov in providers.items():
            key = prov.get("api_key", "")
            if key and not key.startswith("$") and len(key) > 8:
                plaintext_keys.append(name)

        if plaintext_keys:
            return AuditResult(
                "config_keys", AuditStatus.WARN,
                f"以下provider有明文API key: {', '.join(plaintext_keys)}",
                "建议改用环境变量 $ENV_VAR 引用",
            )
        return AuditResult("config_keys", AuditStatus.PASS, "API key安全")

    def _check_file_permissions(self) -> AuditResult:
        """检查config.json权限。"""
        config_path = self.home / "config.json"
        if not config_path.exists():
            return AuditResult("file_perms", AuditStatus.PASS, "无config.json")

        mode = config_path.stat().st_mode
        if mode & stat.S_IROTH:
            return AuditResult(
                "file_perms", AuditStatus.WARN,
                f"config.json权限过松: {oct(mode)[-3:]}",
                "建议 chmod 600 ~/.openllm/config.json",
            )
        return AuditResult("file_perms", AuditStatus.PASS, f"权限: {oct(mode)[-3:]}")

    def _check_open_ports(self) -> AuditResult:
        """检查常见端口是否开放。"""
        ports_to_check = [8080, 5000, 3000, 9119, 11434]
        open_ports = []
        for port in ports_to_check:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(0.3)
                    if s.connect_ex(("127.0.0.1", port)) == 0:
                        open_ports.append(port)
            except Exception:
                pass

        if open_ports:
            return AuditResult(
                "open_ports", AuditStatus.INFO,
                f"本地开放端口: {open_ports}",
                "确认是否为预期服务",
            )
        return AuditResult("open_ports", AuditStatus.PASS, "无异常开放端口")

    def _check_disk_space(self) -> AuditResult:
        """检查磁盘空间。"""
        try:
            st = os.statvfs(str(self.home))
            free_mb = (st.f_bavail * st.f_frsize) / (1024 * 1024)
            if free_mb < 100:
                return AuditResult(
                    "disk_space", AuditStatus.WARN,
                    f"磁盘空间不足: {free_mb:.0f}MB",
                    "建议清理 ~/.openllm/capsules/ 旧checkpoint",
                )
            return AuditResult("disk_space", AuditStatus.PASS, f"可用空间: {free_mb:.0f}MB")
        except Exception:
            return AuditResult("disk_space", AuditStatus.INFO, "无法检查磁盘空间")

    def _check_capsule_integrity(self) -> AuditResult:
        """检查capsule目录完整性。"""
        capsule = self.home / "capsule"
        capsules = self.home / "capsules"

        missing = []
        if not capsule.exists():
            missing.append("capsule/")
        if not capsules.exists():
            missing.append("capsules/")

        if missing:
            return AuditResult(
                "capsule_integrity", AuditStatus.FAIL,
                f"缺失目录: {', '.join(missing)}",
                "openLLM需要capsule和capsules目录存储记忆",
            )

        # 检查是否有checkpoint
        checkpoints = list(capsules.glob("checkpoint_*.json"))
        if not checkpoints:
            return AuditResult(
                "capsule_integrity", AuditStatus.WARN,
                "capsules/中无checkpoint文件",
                "可能是首次运行或记忆未保存",
            )

        return AuditResult(
            "capsule_integrity", AuditStatus.PASS,
            f"capsule完整，{len(checkpoints)}个checkpoint",
        )

    def _check_endpoint_https(self) -> AuditResult:
        """检查Provider endpoint是否HTTPS。"""
        config_path = self.home / "config.json"
        if not config_path.exists():
            return AuditResult("endpoint_https", AuditStatus.PASS, "无config.json")

        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return AuditResult("endpoint_https", AuditStatus.WARN, "config.json解析失败")

        providers = config.get("providers", {})
        http_endpoints = []
        for name, prov in providers.items():
            endpoint = prov.get("endpoint", "")
            if endpoint.startswith("http://"):
                http_endpoints.append(name)

        if http_endpoints:
            return AuditResult(
                "endpoint_https", AuditStatus.WARN,
                f"以下endpoint使用HTTP(非HTTPS): {', '.join(http_endpoints)}",
                "API key可能在传输中被截获",
            )
        return AuditResult("endpoint_https", AuditStatus.PASS, "所有endpoint使用HTTPS")


def format_report(results: List[AuditResult]) -> str:
    """格式化审计报告为人类可读文本。"""
    icons = {
        AuditStatus.PASS: "✅",
        AuditStatus.WARN: "⚠️",
        AuditStatus.FAIL: "❌",
        AuditStatus.INFO: "ℹ️",
    }

    lines = ["═" * 40, "  启动安全审计", "═" * 40, ""]
    warn_count = 0
    fail_count = 0

    for r in results:
        icon = icons.get(r.status, "?")
        lines.append(f"  {icon} {r.check_name}: {r.message}")
        if r.details:
            lines.append(f"      → {r.details}")
        if r.status == AuditStatus.WARN:
            warn_count += 1
        elif r.status == AuditStatus.FAIL:
            fail_count += 1

    lines.append("")
    lines.append("═" * 40)
    if fail_count:
        lines.append(f"  🔴 {fail_count}个失败 · {warn_count}个警告")
    elif warn_count:
        lines.append(f"  ⚠️ {warn_count}个警告")
    else:
        lines.append("  ✅ 全部通过")
    lines.append("═" * 40)

    return "\n".join(lines)


def run_startup_audit(home: Path = OPENLLM_HOME) -> str:
    """一键运行启动审计，返回格式化报告。"""
    auditor = StartupAuditor(home)
    results = auditor.run_all()
    report = format_report(results)

    # 记录到日志
    for r in results:
        if r.status == AuditStatus.FAIL:
            logger.error("审计失败: %s — %s", r.check_name, r.message)
        elif r.status == AuditStatus.WARN:
            logger.warning("审计警告: %s — %s", r.check_name, r.message)

    return report
