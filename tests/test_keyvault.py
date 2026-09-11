"""钥匙迁进密钥库（keyvault）——钉子。

2026-09-10：启动审计一直喊"config.json 有明文 API key"，本文件钉住"迁到哪儿了、
还能不能取到、后端到底安不安全"三件事。

两条纪律：
  ① 全部沙箱化（tmp_path / OPENLLM_VAULT_HOME），绝不把测试密钥写进真密钥库；
  ② 报告与断言里只出现 mask() 后的形态，任何 key 本体不落文字。
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openllm.security import keyvault as kv  # noqa: E402
from openllm.security.keyvault import (  # noqa: E402
    FileBackend,
    Keyvault,
    mask,
    resolve_api_key,
)

FAKE = "sk-FAKE-0123456789abcdef"  # 假密钥，仅用于测试
HAS_PS = shutil.which("powershell.exe") is not None or Path("/mnt/c/Windows").exists()


class _DeadBackend:
    """装成可用、实则不可用的后端（模拟"挂着 SecretService 但连不上 daemon"）。"""

    name = "dead"
    secure = True
    detail = ""

    def available(self) -> bool:
        return False

    def get(self, entry):  # pragma: no cover
        raise AssertionError("不可用后端不该被调用")

    set = get
    delete = get


class _StubSecureBackend:
    """内存里的安全后端（给"不安全后端要告警"这类断言做对照）。"""

    name = "stub-secure"
    secure = True
    detail = "内存桩"

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def available(self) -> bool:
        return True

    def get(self, entry: str):
        return self.store.get(entry)

    def set(self, entry: str, secret: str) -> None:
        self.store[entry] = secret

    def delete(self, entry: str) -> None:
        self.store.pop(entry, None)


def _file_vault(tmp: Path) -> Keyvault:
    return Keyvault(home=tmp, backends=[FileBackend(tmp / ".openllm" / "vault")])


# ────────────────────────── 后端选择 ──────────────────────────


def test_unavailable_backend_is_skipped(tmp_path: Path):
    """挂着的后端探不通，就往下走，不许静默降级成"假装成功"。"""
    v = Keyvault(home=tmp_path, backends=[_DeadBackend(), FileBackend(tmp_path / "vault")])
    assert v.backend.name == "file"
    assert v.secure is False


def test_file_backend_says_it_is_not_secure(tmp_path: Path):
    """file 后端必须自认不安全——它是兜底，不是安全存储。"""
    v = _file_vault(tmp_path)
    assert v.backend.name == "file"
    assert v.secure is False
    assert "不加密" in v.status()["detail"]


def test_plaintext_keyring_is_rejected_by_policy():
    """keyrings.alt 的 PlaintextKeyring 在拒收名单里（假安全比没安全更危险）。"""
    assert "keyrings.alt.file.PlaintextKeyring" in kv._REJECTED_KEYRING


# ────────────────────────── 读写 ──────────────────────────


def test_vault_roundtrip_and_delete(tmp_path: Path):
    v = _file_vault(tmp_path)
    v.set("mimo", FAKE)
    assert v.get("mimo") == FAKE
    assert v.get("qwen") is None
    v.delete("mimo")
    assert v.get("mimo") is None


def test_vault_file_is_chmod_600_and_not_plaintext(tmp_path: Path):
    v = _file_vault(tmp_path)
    v.set("mimo", FAKE)
    f = tmp_path / ".openllm" / "vault" / "mimo_api_key.enc"
    assert f.exists()
    assert stat.S_IMODE(f.stat().st_mode) == 0o600
    assert FAKE not in f.read_text(encoding="utf-8")  # base64 了，至少不裸奔


def test_mask_never_leaks_full_secret():
    m = mask(FAKE)
    assert FAKE not in m
    assert "len=" in m
    assert mask("") == "<空>"
    assert mask("short") == "***"


# ────────────────────────── 取 key 顺序 ──────────────────────────


def test_resolve_order_env_beats_vault_beats_config(tmp_path: Path, monkeypatch):
    v = _file_vault(tmp_path)
    v.set("mimo", "sk-VAULT")
    monkeypatch.setattr(kv, "default_vault", lambda: v)
    section = {"api_key": "sk-CONFIG"}

    monkeypatch.setenv("MIMO_API_KEY", "sk-ENV")
    assert resolve_api_key("mimo", section) == ("sk-ENV", "env:MIMO_API_KEY")

    monkeypatch.delenv("MIMO_API_KEY")
    assert resolve_api_key("mimo", section) == ("sk-VAULT", "keyvault:file")

    v.delete("mimo")
    assert resolve_api_key("mimo", section) == ("sk-CONFIG", "config.json(plaintext)")


def test_resolve_returns_none_when_nothing_available(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(kv, "default_vault", lambda: _file_vault(tmp_path))
    for var in kv.PROVIDER_ENV["deepseek"]:
        monkeypatch.delenv(var, raising=False)
    assert resolve_api_key("deepseek", {}) == ("", "none")


def test_resolve_ignores_env_reference_style_config(tmp_path: Path, monkeypatch):
    """config 里写 $ENV_VAR 形式的引用，不算明文 key，不该当密钥返回。"""
    monkeypatch.setattr(kv, "default_vault", lambda: _file_vault(tmp_path))
    assert resolve_api_key("mimo", {"api_key": "$MIMO_API_KEY"}) == ("", "none")


# ────────────────────────── DPAPI 真后端（本机 WSL） ──────────────────────────


@pytest.mark.skipif(not HAS_PS, reason="无 powershell.exe（非 WSL 或未开互操作）")
def test_dpapi_real_roundtrip(tmp_path: Path):
    """真加密：Windows DPAPI 往返 + 密文不含明文 + 0600。"""
    v = Keyvault(home=tmp_path, backends=[kv.DPAPIBackend(tmp_path / ".openllm" / "vault")])
    assert v.backend.name == "dpapi"
    assert v.secure is True

    v.set("mimo", FAKE)
    assert v.get("mimo") == FAKE

    f = tmp_path / ".openllm" / "vault" / "mimo_api_key.dpapi"
    blob = f.read_text(encoding="utf-8")
    assert FAKE not in blob
    assert blob.startswith("01000000d08c9ddf")  # DPAPI 用户范围 provider GUID
    assert stat.S_IMODE(f.stat().st_mode) == 0o600

    v.delete("mimo")
    assert v.get("mimo") is None


def test_default_vault_honours_env_home(tmp_path: Path, monkeypatch):
    """OPENLLM_VAULT_HOME 能把密钥库重定向到沙箱——防测试垃圾进真库。"""
    monkeypatch.setenv("OPENLLM_VAULT_HOME", str(tmp_path))
    kv.reset_default_vault()
    try:
        assert Path(kv.default_vault().vault_dir).is_relative_to(tmp_path)
    finally:
        kv.reset_default_vault()


# ────────────────────────── 启动审计认不认 ──────────────────────────


def _write_config(home: Path, providers: dict) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.json").write_text(
        json.dumps({"default_provider": "mimo", "providers": providers}, ensure_ascii=False),
        encoding="utf-8",
    )


def test_audit_warns_on_plaintext_key(tmp_path: Path):
    from openllm.security.startup_audit import AuditStatus, StartupAuditor

    _write_config(tmp_path, {"mimo": {"api_key": FAKE, "endpoint": "https://x/y"}})
    r = StartupAuditor(tmp_path)._check_config_keys()
    assert r.status == AuditStatus.WARN
    assert "明文" in r.message
    assert FAKE not in r.message  # 审计也不许复现 key 本体
    assert "migrate_keys_to_vault" in r.details


def test_audit_passes_after_migration_when_vault_secure(tmp_path: Path, monkeypatch):
    from openllm.security.startup_audit import AuditStatus, StartupAuditor

    _write_config(tmp_path, {"mimo": {"api_key_ref": "keyvault", "endpoint": "https://x/y"}})

    stub = Keyvault(home=tmp_path, backends=[_StubSecureBackend()])
    monkeypatch.setattr(kv, "default_vault", lambda: stub)
    r = StartupAuditor(tmp_path)._check_config_keys()
    assert r.status == AuditStatus.PASS
    assert "stub-secure" in r.message


def test_audit_still_warns_when_vault_backend_insecure(tmp_path: Path, monkeypatch):
    """明文清了、但后端是 file（仍是明文）——审计必须照喊，别给假安心。"""
    from openllm.security.startup_audit import AuditStatus, StartupAuditor

    _write_config(tmp_path, {"mimo": {"api_key_ref": "keyvault"}})
    monkeypatch.setattr(kv, "default_vault", lambda: _file_vault(tmp_path))
    r = StartupAuditor(tmp_path)._check_config_keys()
    assert r.status == AuditStatus.WARN
    assert "不安全" in r.message


# ────────────────────────── 引擎取 key 路径 ──────────────────────────


def test_engine_utils_env_first_then_vault(tmp_path: Path, monkeypatch):
    from openllm.core import engine_utils

    monkeypatch.setenv("HOME", str(tmp_path))  # 沙箱：_cfg_section 读不到真 config.json
    v = _file_vault(tmp_path)
    v.set("deepseek", "sk-VAULT-DS")
    monkeypatch.setattr(kv, "default_vault", lambda: v)

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ENV-DS")
    assert engine_utils.get_api_key("deepseek") == "sk-ENV-DS"

    monkeypatch.delenv("DEEPSEEK_API_KEY")
    assert engine_utils.get_api_key("deepseek") == "sk-VAULT-DS"


def test_engine_utils_ollama_needs_no_key():
    from openllm.core.engine_utils import get_api_key

    assert get_api_key("ollama") == "ollama"


def test_engine_utils_warns_on_plaintext_fallback(tmp_path: Path, monkeypatch, caplog):
    from openllm.core import engine_utils

    home = tmp_path / "home"
    (home / ".openllm").mkdir(parents=True)
    (home / ".openllm" / "config.json").write_text(
        json.dumps({"providers": {"deepseek": {"api_key": FAKE}}}), encoding="utf-8"
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(kv, "default_vault", lambda: _file_vault(tmp_path))

    with caplog.at_level("WARNING"):
        assert engine_utils.get_api_key("deepseek") == FAKE
    assert any("明文" in rec.message for rec in caplog.records)


# ────────────────────────── 迁移脚本 ──────────────────────────


def _load_migration_module():
    """scripts/ 不是包，按路径加载迁移脚本。"""
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "migrate_keys_to_vault.py"
    spec = importlib.util.spec_from_file_location("migrate_keys_to_vault", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_migration_moves_keys_and_strips_plaintext(tmp_path: Path):
    mod = _load_migration_module()
    home = tmp_path / "openllm"
    _write_config(home, {
        "mimo": {"api_key": FAKE, "endpoint": "https://x/y"},
        "qwen": {"api_key": "sk-FAKE-qwen-0001", "endpoint": "https://q/z"},
        "ollama": {"endpoint": "http://localhost:11434"},
    })
    vault = _file_vault(tmp_path)

    assert mod.migrate(home, vault=vault) == 0

    cfg = json.loads((home / "config.json").read_text(encoding="utf-8"))
    assert "api_key" not in cfg["providers"]["mimo"]
    assert cfg["providers"]["mimo"]["api_key_ref"] == "keyvault"
    assert "api_key" not in cfg["providers"]["qwen"]
    assert cfg["providers"]["mimo"]["endpoint"] == "https://x/y"  # 其它字段不许丢
    assert vault.get("mimo") == FAKE
    assert vault.get("qwen") == "sk-FAKE-qwen-0001"


def test_migration_is_idempotent(tmp_path: Path):
    mod = _load_migration_module()
    home = tmp_path / "openllm"
    _write_config(home, {"mimo": {"api_key": FAKE}})
    vault = _file_vault(tmp_path)
    assert mod.migrate(home, vault=vault) == 0
    first = (home / "config.json").read_text(encoding="utf-8")
    assert mod.migrate(home, vault=vault) == 0  # 二跑：无需迁移
    assert (home / "config.json").read_text(encoding="utf-8") == first


def test_migration_backup_is_0600(tmp_path: Path):
    mod = _load_migration_module()
    home = tmp_path / "openllm"
    _write_config(home, {"mimo": {"api_key": FAKE}})
    assert mod.migrate(home, vault=_file_vault(tmp_path)) == 0
    baks = list(home.glob("config.json.bak-*"))
    assert len(baks) == 1
    assert stat.S_IMODE(baks[0].stat().st_mode) == 0o600
    assert json.loads(baks[0].read_text(encoding="utf-8"))["providers"]["mimo"]["api_key"] == FAKE
    assert stat.S_IMODE((home / "config.json").stat().st_mode) == 0o600


def test_migration_aborts_on_readback_mismatch(tmp_path: Path):
    """读回不一致 = 没存上。此时 config.json 必须一个字没动（先写后删的底线）。"""
    mod = _load_migration_module()
    home = tmp_path / "openllm"
    _write_config(home, {"mimo": {"api_key": FAKE}})
    original = (home / "config.json").read_text(encoding="utf-8")

    class _LyingVault:
        """写入成功、读回却是别的东西——模拟密钥库半死。"""

        def __init__(self):
            self.store: dict[str, str] = {}

        def status(self):
            return {"backend": "lying", "secure": True, "detail": "桩"}

        def set(self, provider, secret):
            self.store[provider] = "something-else"

        def get(self, provider):
            return self.store.get(provider)

        def delete(self, provider):
            self.store.pop(provider, None)

    assert mod.migrate(home, vault=_LyingVault()) == 3
    assert (home / "config.json").read_text(encoding="utf-8") == original
    assert not list(home.glob("config.json.bak-*"))


def test_migration_dry_run_touches_nothing(tmp_path: Path, capsys):
    mod = _load_migration_module()
    home = tmp_path / "openllm"
    _write_config(home, {"mimo": {"api_key": FAKE}})
    vault = _file_vault(tmp_path)
    assert mod.migrate(home, dry_run=True, vault=vault) == 0
    assert vault.get("mimo") is None
    assert json.loads((home / "config.json").read_text(encoding="utf-8"))["providers"]["mimo"]["api_key"] == FAKE
    out = capsys.readouterr().out
    assert FAKE not in out  # dry-run 报告里也不许出现 key 本体


def test_plan_ignores_env_reference_and_short_junk():
    mod = _load_migration_module()
    assert mod.plan({"providers": {
        "a": {"api_key": "$MIMO_API_KEY"},
        "b": {"api_key": "short"},
        "c": {"api_key": FAKE},
        "d": {"endpoint": "x"},
    }}) == [("c", FAKE)]
