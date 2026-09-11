"""openLLM 密钥库（keyvault）—— 把 API key 从明文 config.json 搬到系统凭据存储。

2026-09-10 定：启动审计一直在喊"config.json 有明文 API key"，今天把它兑现。

后端优先级（首个可用者胜出）：

  1. ``keyring`` 系统后端 —— macOS Keychain / Windows 凭据管理器 / Linux SecretService。
  2. ``DPAPI`` —— WSL 专用：调用 Windows 的 ``powershell.exe``，密文由 **Windows 用户主密钥**
     保护（换机器或换 Windows 用户则解不开，这正是我们要的性质）。
  3. ``file`` —— 兜底：``~/.openllm/vault/*.enc``（0600）。**不是安全存储**，只是不再把
     密钥摊在 config.json 里被备份/版本控制扫到。用它时每次都会告警。

刻意拒绝 ``keyrings.alt`` 的 PlaintextKeyring：那只是把明文换个文件放，安全性等于零，
却会让人误以为"已经进 keyring 了"。宁可降级到明说的 file 后端，也不接受假的安全。

读取顺序（``resolve_api_key``）：环境变量 → 密钥库 → config.json 明文（兼容旧配置，但会
把来源标成 insecure，供调用方告警）。
"""

from __future__ import annotations

import base64
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

SERVICE = "openllm"

# provider → 环境变量名（环境变量永远最高优先级：方便临时覆盖与 CI）
PROVIDER_ENV = {
    "mimo": ("MIMO_API_KEY",),
    "qwen": ("ALIBABA_PLAN_API_KEY", "QWEN_API_KEY"),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY",),
    "gateway": ("OPENLLM_GATEWAY_KEY",),
    "openai": ("OPENAI_API_KEY",),
}

# 这些 keyring 后端要么是明文、要么啥也不干——不接受
_REJECTED_KEYRING = {
    "keyrings.alt.file.PlaintextKeyring",
    "keyring.backends.null.Keyring",
    "keyring.backends.fail.Keyring",
}


def mask(secret: Optional[str]) -> str:
    """给报告用：只露长度和尾巴，绝不打印密钥本体。"""
    if not secret:
        return "<空>"
    if len(secret) <= 8:
        return "***"
    return f"***len={len(secret)}…{secret[-4:]}"


# ────────────────────────── 后端 ──────────────────────────


class KeyringBackend:
    """系统密钥环。在无 SecretService/无桌面会话的环境里会判为不可用。"""

    name = "keyring"

    def __init__(self) -> None:
        self._kr: Any = None
        self.secure = True
        self.detail = "不可用"

    def available(self) -> bool:
        try:
            import keyring  # noqa: F401

            kr = keyring.get_keyring()
            cls = f"{type(kr).__module__}.{type(kr).__name__}"
            if cls in _REJECTED_KEYRING:
                self.detail = f"仅有 {cls}（明文/空实现），不接受"
                return False
            # 探一次真实读写，防"挂着后端但连不上 daemon"
            probe = f"__probe__{os.getpid()}"
            kr.set_password(SERVICE, probe, "1")
            got = kr.get_password(SERVICE, probe)
            kr.delete_password(SERVICE, probe)
            if got != "1":
                self.detail = f"{cls} 读写探针失败"
                return False
            self._kr = kr
            self.detail = cls
            return True
        except Exception as e:  # 没装 / 没有 daemon / 认证失败
            self.detail = f"{type(e).__name__}: {str(e)[:80]}"
            return False

    def get(self, entry: str) -> Optional[str]:
        return self._kr.get_password(SERVICE, entry)

    def set(self, entry: str, secret: str) -> None:
        self._kr.set_password(SERVICE, entry, secret)

    def delete(self, entry: str) -> None:
        try:
            self._kr.delete_password(SERVICE, entry)
        except Exception:
            pass


class DPAPIBackend:
    """WSL 专用：借 Windows 的 DPAPI 加密，密文存 Linux 侧文件。

    为什么是它：WSL 里没有 GNOME Keyring / SecretService（装它要 sudo 且要跑 daemon），
    而 Windows 凭据体系的 DPAPI 从 WSL 可以直接调用。密钥由 **Windows 用户的登录凭据**
    派生保护——本机之外、别的 Windows 用户之下，都解不开。

    明文只走 stdin（不进 argv，不被 ps 看见）；文件只存密文。
    """

    name = "dpapi"
    secure = True

    _ENCRYPT = (
        "$ErrorActionPreference='Stop';"
        "$s = ConvertTo-SecureString ([Console]::In.ReadToEnd()) -AsPlainText -Force;"
        "[Console]::Out.Write((ConvertFrom-SecureString $s))"
    )
    _DECRYPT = (
        "$ErrorActionPreference='Stop';"
        "$s = ConvertTo-SecureString ([Console]::In.ReadToEnd());"
        "$b = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($s);"
        "try { [Console]::Out.Write([Runtime.InteropServices.Marshal]::PtrToStringBSTR($b)) }"
        "finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($b) }"
    )

    def __init__(self, vault_dir: Path) -> None:
        self.vault_dir = vault_dir
        self.detail = "不可用"

    @property
    def _ps(self) -> Optional[str]:
        return shutil.which("powershell.exe") or (
            "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
            if Path("/mnt/c/Windows").exists()
            else None
        )

    def _run(self, script: str, stdin: str) -> str:
        ps = self._ps
        if not ps:
            raise RuntimeError("找不到 powershell.exe（非 WSL 环境？）")
        proc = subprocess.run(
            [ps, "-NoProfile", "-NonInteractive", "-Command", script],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=25,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"powershell 失败: {(proc.stderr or '').strip()[:200]}")
        return proc.stdout.replace("\r", "").replace("\n", "").lstrip("\ufeff")

    def available(self) -> bool:
        if not self._ps:
            self.detail = "无 powershell.exe"
            return False
        try:
            blob = self._run(self._ENCRYPT, "probe")
            if self._run(self._DECRYPT, blob) != "probe":
                self.detail = "DPAPI 往返探针失败"
                return False
            self.detail = "Windows DPAPI（用户主密钥保护）"
            return True
        except Exception as e:
            self.detail = f"{type(e).__name__}: {str(e)[:80]}"
            return False

    def _path(self, entry: str) -> Path:
        safe = entry.replace(":", "_").replace("/", "_")
        return self.vault_dir / f"{safe}.dpapi"

    def get(self, entry: str) -> Optional[str]:
        f = self._path(entry)
        if not f.exists():
            return None
        try:
            return self._run(self._DECRYPT, f.read_text(encoding="utf-8").strip())
        except Exception as e:
            logger.warning("密钥 %s 解不开（换了机器或 Windows 用户？）: %s", entry, e)
            return None

    def set(self, entry: str, secret: str) -> None:
        if not secret:
            raise ValueError("空密钥不写入")
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        blob = self._run(self._ENCRYPT, secret)
        f = self._path(entry)
        f.write_text(blob, encoding="utf-8")
        f.chmod(0o600)

    def delete(self, entry: str) -> None:
        self._path(entry).unlink(missing_ok=True)


class FileBackend:
    """兜底：0600 文件存 base64。**不是加密**，只是把密钥挪出 config.json。"""

    name = "file"
    secure = False

    def __init__(self, vault_dir: Path) -> None:
        self.vault_dir = vault_dir
        self.detail = "明文落 0600 文件（不加密，仅兜底）"

    def available(self) -> bool:
        return True

    def _path(self, entry: str) -> Path:
        safe = entry.replace(":", "_").replace("/", "_")
        return self.vault_dir / f"{safe}.enc"

    def get(self, entry: str) -> Optional[str]:
        f = self._path(entry)
        if not f.exists():
            return None
        try:
            return base64.b64decode(f.read_text(encoding="utf-8").strip()).decode("utf-8")
        except Exception:
            return None

    def set(self, entry: str, secret: str) -> None:
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        f = self._path(entry)
        f.write_text(base64.b64encode(secret.encode("utf-8")).decode(), encoding="utf-8")
        f.chmod(0o600)

    def delete(self, entry: str) -> None:
        self._path(entry).unlink(missing_ok=True)


# ────────────────────────── 门面 ──────────────────────────


class Keyvault:
    """密钥库门面：挑后端、读写密钥。"""

    def __init__(self, home: Optional[Path] = None, backends: Optional[list] = None) -> None:
        base = Path(home) if home else Path.home()
        self.vault_dir = base / ".openllm" / "vault"
        self._backends = backends if backends is not None else [
            KeyringBackend(),
            DPAPIBackend(self.vault_dir),
            FileBackend(self.vault_dir),
        ]
        self._active: Any = None
        self._probed = False

    @property
    def backend(self) -> Any:
        if not self._probed:
            self._probed = True
            for b in self._backends:
                try:
                    if b.available():
                        self._active = b
                        break
                except Exception as e:
                    b.detail = f"{type(e).__name__}: {str(e)[:80]}"
            if self._active is None:
                self._active = FileBackend(self.vault_dir)
        return self._active

    @property
    def secure(self) -> bool:
        return bool(getattr(self.backend, "secure", False))

    def entry(self, provider: str, kind: str = "api_key") -> str:
        return f"{provider}:{kind}"

    def get(self, provider: str, kind: str = "api_key") -> Optional[str]:
        try:
            return self.backend.get(self.entry(provider, kind))
        except Exception as e:
            logger.warning("keyvault.get(%s) 失败: %s", provider, e)
            return None

    def set(self, provider: str, secret: str, kind: str = "api_key") -> None:
        if getattr(self.backend, "secure", False) is False:
            logger.warning(
                "keyvault 当前后端 %s 不安全（%s）——密钥仍是明文，只是换了位置",
                self.backend.name, getattr(self.backend, "detail", ""),
            )
        self.backend.set(self.entry(provider, kind), secret)

    def delete(self, provider: str, kind: str = "api_key") -> None:
        self.backend.delete(self.entry(provider, kind))

    def status(self) -> dict:
        return {
            "backend": self.backend.name,
            "secure": self.secure,
            "detail": getattr(self.backend, "detail", ""),
            "vault_dir": str(self.vault_dir),
            "providers": [p.name for p in self._backends if getattr(p, "secure", False)],
        }


_default: Optional[Keyvault] = None


def default_vault() -> Keyvault:
    """进程内单例（探后端要起 powershell，别每次构造都探）。

    测试/沙箱可用 ``OPENLLM_VAULT_HOME`` 重定向密钥库根目录——免得测试垃圾写进真密钥库。
    """
    global _default
    if _default is None:
        env_home = os.environ.get("OPENLLM_VAULT_HOME", "")
        _default = Keyvault(Path(env_home) if env_home else None)
    return _default


def reset_default_vault() -> None:
    """丢弃单例（测试用：改过 OPENLLM_VAULT_HOME 后重新探后端）。"""
    global _default
    _default = None


def resolve_api_key(provider: str, section: Optional[dict] = None) -> tuple[str, str]:
    """按 环境变量 → 密钥库 → config.json 明文 的顺序取密钥。

    Returns:
        (key, source)；source ∈ {env:NAME, keyvault:<backend>, config.json(plaintext), none}
    """
    for name in PROVIDER_ENV.get(provider, ()):
        v = os.environ.get(name, "")
        if v:
            return v, f"env:{name}"

    v = default_vault().get(provider)
    if v:
        return v, f"keyvault:{default_vault().backend.name}"

    if section:
        legacy = section.get("api_key", "") or ""
        if legacy and not legacy.startswith("$"):
            return legacy, "config.json(plaintext)"

    return "", "none"
