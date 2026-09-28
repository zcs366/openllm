"""seccomp 网络面子策略测试（G2-0928，双轨第二轨）。

铁律：绝不在 pytest 进程中安装 filter——seccomp 进程终身不可逆。
真上身只走子进程（CLI 冒烟）。程序构建/布局/env 门控在进程内安全测试。
"""
import ctypes
import os
import struct
import subprocess
import sys
from pathlib import Path

from openllm.security.seccomp_net import (
    SECCOMP_RET_ALLOW,
    _AUDIT_ARCH_X86_64,
    _NR_SOCKET,
    _SECCOMP_RET_ERRNO_EPERM,
    SeccompNetLock,
    build_block_udp_program,
)

SRC = Path(__file__).resolve().parents[1] / "src"


def test_constants():
    assert _NR_SOCKET == 41            # x86_64 socket(2)
    assert _AUDIT_ARCH_X86_64 == 0xC000003E
    assert SECCOMP_RET_ALLOW == 0x7FFF0000


def test_bpf_encoding():
    """_bpf 编码 = sock_filter 布局（u16 code, u8 jt, u8 jf, u32 k，LE）。"""
    raw = build_block_udp_program()
    assert len(raw) == 12 * 8
    first = struct.unpack("<HBBI", raw[:8])
    # 第一条：LD | W | ABS（0x20），k=4（seccomp_data.arch 偏移）
    assert first == (0x20, 0, 0, 4)


def test_program_arch_guard():
    """程序必须含 arch 校验（防 32 位 shim 绕过）与 EPERM/ALLOW 两个动作。"""
    raw = build_block_udp_program()
    words = [struct.unpack("<HBBI", raw[i:i + 8]) for i in range(0, len(raw), 8)]
    # arch 校验：JEQ(0x15) k=AUDIT_ARCH_X86_64，jt=0 jf=8
    assert any(w == (0x15, 0, 8, 0xC000003E) for w in words)
    # 两个 RET 动作（BPF_RET|K = 0x06）：EPERM 与 ALLOW
    rets = [w[3] for w in words if w[0] == 0x06]
    assert _SECCOMP_RET_ERRNO_EPERM in rets
    assert SECCOMP_RET_ALLOW in rets


def test_env_gate():
    """OPENLLM_SECCOMP_NET 未设/非 block_udp → None（不安装）。"""
    import os
    saved = os.environ.pop("OPENLLM_SECCOMP_NET", None)
    try:
        assert SeccompNetLock.from_env() is None
        os.environ["OPENLLM_SECCOMP_NET"] = "block_udp"
        assert SeccompNetLock.from_env() is not None
        os.environ["OPENLLM_SECCOMP_NET"] = "yes"
        assert SeccompNetLock.from_env() is None
    finally:
        if saved is not None:
            os.environ["OPENLLM_SECCOMP_NET"] = saved


def test_cli_real_install_in_subprocess():
    """真上身（子进程隔离）：filter 安装 + UDP/EPERM + TCP 可用 + AF_UNIX 无恙。"""
    env = dict(os.environ, PYTHONPATH=str(SRC))
    r = subprocess.run(
        [sys.executable, "-m", "openllm.security.seccomp_net"],
        capture_output=True, text=True, timeout=30, env=env,
    )
    combined = r.stdout + r.stderr
    assert "VERDICT: PASS" in combined, combined
    assert "udp_blocked=True" in combined, combined


def test_engine_hook_registered():
    from openllm.core.engine import OpenLLMEngine
    assert hasattr(OpenLLMEngine, "_apply_seccomp_net_lock")
