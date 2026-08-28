"""pytest配置。"""
import sys
from pathlib import Path

import pytest

# 确保openllm包可导入
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(autouse=True)
def _isolate_causal_store(tmp_path, monkeypatch):
    """会话级护栏：把因果库默认路径重定向到临时目录（总根因）。

    所有无参写入路径——get_causal_store()/CausalMemoryStore()（causal_memory）、
    苏醒协议（awakening）——共享同一默认路径常量。统一重定向后，
    任何测试都不会写入真实生产因果库 ~/.openllm/memory/causal。
    2026-08-24 军师亲补，源自 117+2 条测试污染隔离事故。
    """
    from openllm.memory import causal_memory
    from openllm.core import awakening
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
