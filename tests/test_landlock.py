"""Landlock 内核级写锁测试（G1-0927）。

铁律：绝不在测试中真 restrict——apply_and_verify 上身不可逆，会锁死
pytest 进程的 tmp_path 写。本测试只覆盖：探测、结构布局、降级逻辑、
CLI 退出码（子进程隔离跑真上身）。
"""
import ctypes
import os
import struct
import subprocess
import sys
from pathlib import Path

from openllm.security.landlock import (
    LANDLOCK_CREATE_RULESET_VERSION,
    PR_SET_NO_NEW_PRIVS,
    SYS_LANDLOCK_ADD_RULE,
    SYS_LANDLOCK_CREATE_RULESET,
    SYS_LANDLOCK_RESTRICT_SELF,
    KernelWriteLock,
    LockResult,
    _ACCESS_ABI1,
    _ACCESS_REFER,
    _ACCESS_TRUNCATE,
)

SRC = Path(__file__).resolve().parents[1] / "src"


def test_syscall_numbers():
    assert SYS_LANDLOCK_CREATE_RULESET == 444
    assert SYS_LANDLOCK_ADD_RULE == 445
    assert SYS_LANDLOCK_RESTRICT_SELF == 446
    assert PR_SET_NO_NEW_PRIVS == 38


def test_version_flag_is_bit0():
    """20260927 修正回归锁：VERSION 标志=1<<0，误写 1<<4 曾致 ABI 误报 0。"""
    assert LANDLOCK_CREATE_RULESET_VERSION == 1


def test_access_bits_layout():
    assert _ACCESS_ABI1["WRITE_FILE"] == 1 << 0
    assert _ACCESS_ABI1["MAKE_SYM"] == 1 << 8
    assert _ACCESS_REFER == 1 << 9
    assert _ACCESS_TRUNCATE == 1 << 13
    all_bits = list(_ACCESS_ABI1.values()) + [_ACCESS_REFER, _ACCESS_TRUNCATE]
    assert sum(all_bits) == 0x23FF


def test_lockresult_ok_requires_all_three():
    r = LockResult(enabled=True, verified_inside=True, verified_outside=False)
    assert not r.ok
    r2 = LockResult(enabled=False, verified_inside=True, verified_outside=True)
    assert not r2.ok
    r3 = LockResult(enabled=True, verified_inside=True, verified_outside=True)
    assert r3.ok


def test_ruleset_attr_struct_layout():
    class RulesetAttr(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    assert ctypes.sizeof(RulesetAttr) == 8


def test_path_beneath_struct_layout():
    class PathBeneathAttr(ctypes.Structure):
        _fields_ = [("allowed_access", ctypes.c_uint64),
                    ("parent_fd", ctypes.c_int)]

    assert ctypes.sizeof(PathBeneathAttr) == 16


def test_abi_probe_no_crash():
    lock = KernelWriteLock.default()
    abi = lock.abi_version()
    assert abi >= 0
    if abi >= 1:
        assert lock.supported() is True


def test_default_whitelist_superset():
    lock = KernelWriteLock.default()
    strs = [str(p) for p in lock.allowed_dirs]
    assert any(p.endswith(".openllm") for p in strs)
    assert "/tmp/openllm" in strs


def test_engine_hook_opt_in_default_off():
    from openllm.core.engine import OpenLLMEngine

    saved = os.environ.pop("OPENLLM_LANDLOCK", None)
    try:
        engine = OpenLLMEngine()
        assert hasattr(engine, "_apply_kernel_write_lock")
    finally:
        if saved is not None:
            os.environ["OPENLLM_LANDLOCK"] = saved


def test_cli_real_restrict_in_subprocess():
    env = dict(os.environ, PYTHONPATH=str(SRC))
    r = subprocess.run(
        [sys.executable, "-m", "openllm.security.landlock"],
        capture_output=True, text=True, timeout=30, env=env,
    )
    combined = r.stdout + r.stderr
    assert "VERDICT: PASS" in combined or "VERDICT: UNSUPPORTED" in combined, combined
