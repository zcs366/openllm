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

    monkeypatch.setattr(
        causal_memory, "DEFAULT_STORE_DIR", tmp_path / "causal_test"
    )
    monkeypatch.setattr(
        awakening, "DEFAULT_AWAKENING_STORE_DIR", tmp_path / "causal_test"
    )
