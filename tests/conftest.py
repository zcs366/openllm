"""pytest配置。"""
import os
import sys
import tempfile
from pathlib import Path

import pytest

# 确保openllm包可导入
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ── 真实 HOME 全局隔离（2026-09-16 医师接骨 · P1-2）────────────────────────
# ★ 必须在任何 openllm 模块导入**之前**执行。
#   多数落点是**模块级常量**（`Path.home() / ".openllm" / ...` 在导入期求值），
#   等到 fixture 里再 monkeypatch 就已经晚了——常量早已固化。
#
# 病（实测）：跑一次全量回归就往**用户真实记忆库**写测试数据 ——
#   ~/.openllm/memory/ 共 960 个 json，其中 216 个是 `conv-turn-N` /
#   `key-decision-N` / `测试统一记忆系统` 之类的垃圾，时间戳精确对应回归时刻。
#   conftest 此前只钉了 causal_store 与 ios_audit_dirs，**唯独没钉 HOME**。
#
# 判据（守卫见 _assert_home_isolated）：Path.home() 必须落在 openllm-pytest-home-*。
_REAL_HOME = Path(os.environ.get("HOME") or str(Path.home()))  # 打桩前先记真实 HOME
_FAKE_HOME = Path(tempfile.mkdtemp(prefix="openllm-pytest-home-"))
(_FAKE_HOME / ".openllm" / "memory").mkdir(parents=True, exist_ok=True)

# ★ 家目录下的**外部资源**一律软链保留，只把 `~/.openllm` 换成空目录。
#   本次要隔离的是 openLLM 自己的落点，**不是**剥夺测试对其它资源的访问。
#   实证（都是被这条边界卡出来的）：
#     · 给 .hermes 建空目录 → 13 个用例失败（jiak 库 / DPAPI 密钥库不可达）
#     · projects 不可达      → 3 个用例失败（test_ilm_pipeline_migration 检查
#                              /home/zcs/projects/isa/ilm 是否存在）
#   软链 = 这些路径行为与打桩前**完全一致**，同时 ~/.openllm 被完整隔离。
for _name in (".hermes", "projects", ".cache", ".local", ".config", ".io-s"):
    _src = _REAL_HOME / _name
    _dst = _FAKE_HOME / _name
    if _src.exists() and not _dst.exists():
        try:
            _dst.symlink_to(_src, target_is_directory=_src.is_dir())
        except OSError:
            pass

os.environ["HOME"] = str(_FAKE_HOME)
# ★ 打桩成「跟随环境变量」，而不是写死返回 _FAKE_HOME。
#   原因：测试的惯用手法是 `monkeypatch.setenv("HOME", tmp)` 来隔离——若 Path.home()
#   被写死，这一手法全部失效（实证：test_engine_utils_warns_on_plaintext_fallback
#   自己 setenv 到 tmp/home 写 config.json，读取端却仍拿假 HOME → 断言 '' == 'sk-FAKE'）。
#   保留 env 语义 = 默认隔离到 _FAKE_HOME，测试可自行覆盖。
Path.home = staticmethod(lambda: Path(os.environ.get("HOME") or str(_REAL_HOME)))


@pytest.fixture(autouse=True)
def _assert_home_isolated():
    """守卫：HOME 打桩若被绕过/还原，立刻失败而不是静默污染真实记忆库。

    判据放宽为「不是真实 HOME」——测试可自行 monkeypatch.setenv 到别的临时目录
    （合法隔离），但不能是用户的真实家目录。
    """
    assert Path.home() != _REAL_HOME, (
        f"HOME 隔离失效，Path.home() 回到了真实家目录 {Path.home()} —— "
        f"测试会写进用户真实记忆库，禁止继续"
    )


@pytest.fixture(autouse=True)
def _isolate_causal_store(tmp_path, monkeypatch):
    """会话级护栏：把因果库默认路径重定向到临时目录（总根因）。

    所有无参写入路径——get_causal_store()/CausalMemoryStore()（causal_memory）、
    苏醒协议（awakening）——共享同一默认路径常量。统一重定向后，
    任何测试都不会写入真实生产因果库 ~/.openllm/memory/causal。
    2026-08-24 军师亲补，源自 117+2 条测试污染隔离事故。
    """
    from openllm.isa import causal_memory
    from openllm.iax import awakening
    from openllm.core import isl_chain as isl_chain_mod

    monkeypatch.setattr(
        causal_memory, "DEFAULT_STORE_DIR", tmp_path / "causal_test"
    )
    monkeypatch.setattr(
        awakening, "DEFAULT_AWAKENING_STORE_DIR", tmp_path / "causal_test"
    )
    monkeypatch.setattr(
        isl_chain_mod, "DEFAULT_ISL_CHAIN_FILE", tmp_path / "isl_test.jsonl"
    )

    # P5：IntegrityGuardian生产路径隔离——测试不得创建~/.openllm/output/integrity/
    # 先重置单例（_guardian持有旧Path），再monkeypatch类属性
    try:
        from openllm.core import integrity_guardian as _ig_mod
        monkeypatch.setattr(_ig_mod, "_guardian", None)
        monkeypatch.setattr(
            _ig_mod.IntegrityGuardian, "BASELINE_PATH",
            tmp_path / "integrity_test" / "baseline.json",
        )
        monkeypatch.setattr(
            _ig_mod.IntegrityGuardian, "AUDIT_LOG_PATH",
            tmp_path / "integrity_test" / "audit_log.jsonl",
        )
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _isolate_ios_audit_dirs(tmp_path, monkeypatch):
    """IOS 审计默认目录隔离（2026-09-15 医师接骨，军师亲加）。

    原先 `AuditChain` / `StatefulAuditTrail` 的默认目录是
    ``~/.hermes/jiak/governance_audit/`` 且**不可重定向**（StatefulAuditTrail 甚至
    把字面量写在 __init__ 里）。于是 tests/test_governance.py 等用例把 session 文件
    以 test-001/test-002 之名**直接写进生产目录**，现场累积 1.6M+705K，且每次全量
    回归都会再加一次（2026-09-15 06:39 实测到写入）。

    与 `_isolate_causal_store` 同理：任何无参写入路径都必须能被测试钉到临时目录。
    三处接缝一起钉——DEFAULT_AUDIT_DIR 有两份（ios.audit / ios.stateful_audit），
    REASONING_AUDIT_DIR 是导入期由 DEFAULT_AUDIT_DIR 派生的，得单独钉。
    """
    from openllm.ios import audit as audit_mod
    from openllm.ios import stateful_audit as sa_mod

    sandbox = tmp_path / "ios_audit"
    monkeypatch.setattr(audit_mod, "DEFAULT_AUDIT_DIR", sandbox)
    monkeypatch.setattr(audit_mod, "REASONING_AUDIT_DIR", sandbox / "reasoning_traces")
    monkeypatch.setattr(sa_mod, "DEFAULT_AUDIT_DIR", sandbox)
