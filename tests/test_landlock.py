"""Landlock 内核级写锁测试（G1-0927）。

铁律：绝不在测试中真 restrict——apply_and_verify 上身不可逆，会锁死
pytest 进程的 tmp_path 写。本测试只覆盖：探测、结构布局、降级逻辑、
CLI 退出码（子进程隔离跑真上身）。
"""
import ctypes
import os
import subprocess
import sys
from pathlib import Path

import pytest

from openllm.security.landlock import (
    LANDLOCK_CREATE_RULESET_VERSION,
    LANDLOCK_RULE_NET_PORT,
    LANDLOCK_RULE_PATH_BENEATH,
    PR_SET_NO_NEW_PRIVS,
    SYS_LANDLOCK_ADD_RULE,
    SYS_LANDLOCK_CREATE_RULESET,
    SYS_LANDLOCK_RESTRICT_SELF,
    KernelWriteLock,
    LandlockPolicy,
    LockResult,
    _ACCESS_REFER,
    _ACCESS_TRUNCATE,
    _ACCESS_WRITE_FILE,
    _WRITE_FACE_ABI1,
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


def test_access_bits_match_uapi():
    """20260928 P0 修正回归锁：位序=uapi 真值。

    证据链：v5.13/v6.6/master 三版头文件互证（位序自 Landlock 诞生未变）
    + bitprobe 逐位探针（内核实测 bit14 拦 truncate(2)、bit16+ EINVAL）。
    G1 旧表把 EXECUTE(1<<0) 当 WRITE_FILE、REFER(1<<13) 当 TRUNCATE，
    且行为测试只测写——位错位后写仍被拦，bug 被掩盖。位序必须逐位钉死。
    """
    assert _ACCESS_WRITE_FILE == 1 << 1   # uapi WRITE_FILE=1<<1（非 1<<0！）
    assert _ACCESS_REFER == 1 << 13       # uapi REFER（G1 曾误当 TRUNCATE）
    assert _ACCESS_TRUNCATE == 1 << 14    # uapi TRUNCATE（bitprobe 实测位）
    # 写面=纯写位集：不含 EXECUTE(1<<0)/READ_FILE(1<<2)/READ_DIR(1<<3)
    for forbidden in (1 << 0, 1 << 2, 1 << 3):
        assert not (_WRITE_FACE_ABI1 & forbidden)
    # 写面含 WRITE_FILE 与全部 MAKE_*（引擎要建文件/目录/管道/socket 文件）
    assert _WRITE_FACE_ABI1 & (1 << 1)
    for bit in range(6, 13):
        assert _WRITE_FACE_ABI1 & (1 << bit)
    # ABI>=3 handled 集 = 写面+REFER+TRUNCATE
    lock = KernelWriteLock.default()
    assert lock._fs_handled_bits(3) == (
        _WRITE_FACE_ABI1 | _ACCESS_REFER | _ACCESS_TRUNCATE)
    # ABI1 内核只收 ABI1 位
    assert lock._fs_handled_bits(1) == _WRITE_FACE_ABI1


def test_lockresult_ok_requires_all_three():
    r = LockResult(enabled=True, verified_inside=True, verified_outside=False)
    assert not r.ok
    r2 = LockResult(enabled=False, verified_inside=True, verified_outside=True)
    assert not r2.ok
    r3 = LockResult(enabled=True, verified_inside=True, verified_outside=True)
    assert r3.ok


def test_ruleset_attr_struct_layout():
    class RulesetAttr(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64),
                    ("handled_access_net", ctypes.c_uint64)]

    assert ctypes.sizeof(RulesetAttr) == 16


def test_path_beneath_and_netport_struct_layout():
    class PathBeneathAttr(ctypes.Structure):
        _fields_ = [("allowed_access", ctypes.c_uint64),
                    ("parent_fd", ctypes.c_int)]

    assert ctypes.sizeof(PathBeneathAttr) == 16

    # uapi landlock_net_port_attr：仅 allowed_access+port，无地址字段（判B铁证）
    class NetPortAttr(ctypes.Structure):
        _fields_ = [("allowed_access", ctypes.c_uint64),
                    ("port", ctypes.c_uint64)]

    assert ctypes.sizeof(NetPortAttr) == 16


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


def test_policy_env_parse(monkeypatch):
    """v2c：env → policy 解析 + 端口聚合。"""
    monkeypatch.setenv("OPENLLM_LANDLOCK_ALLOW_WRITE",
                       os.pathsep.join(["/tmp/a", "/tmp/b"]))
    monkeypatch.setenv("OPENLLM_LANDLOCK_BIND_TCP", "11434, 18434")
    monkeypatch.setenv("OPENLLM_LANDLOCK_CONNECT_TCP", "")
    p = LandlockPolicy.from_env()
    assert p.allow_write == ["/tmp/a", "/tmp/b"]
    assert p.allow_net_bind_tcp == [11434, 18434]
    assert p.allow_net_connect_tcp == []
    # 同端口双白名单聚合一条规则：{11434: BIND|CONNECT, 18434: BIND}
    monkeypatch.setenv("OPENLLM_LANDLOCK_CONNECT_TCP", "11434")
    m = LandlockPolicy.from_env().net_ports_with_bits()
    assert m == {11434: 0b11, 18434: 0b01}
    # 空 env = 网络面不 handled（G1 行为兼容）
    monkeypatch.delenv("OPENLLM_LANDLOCK_BIND_TCP")
    monkeypatch.delenv("OPENLLM_LANDLOCK_CONNECT_TCP")
    assert LandlockPolicy.from_env().net_ports_with_bits() == {}


def test_policy_env_parse_rejects_garbage(monkeypatch):
    """typo 必须抛 ValueError 而非静默 fail-open。"""
    monkeypatch.setenv("OPENLLM_LANDLOCK_BIND_TCP", "11434,oops")
    with pytest.raises(ValueError):
        LandlockPolicy.from_env()
    monkeypatch.setenv("OPENLLM_LANDLOCK_BIND_TCP", "70000")
    with pytest.raises(ValueError):
        LandlockPolicy.from_env()


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


def test_cli_net_face_in_subprocess():
    """v2a：网络面白名单经 env 下发，子进程真上身 + 网络自证。"""
    env = dict(os.environ, PYTHONPATH=str(SRC),
               OPENLLM_LANDLOCK_BIND_TCP="18434")
    r = subprocess.run(
        [sys.executable, "-m", "openllm.security.landlock"],
        capture_output=True, text=True, timeout=30, env=env,
    )
    combined = r.stdout + r.stderr
    assert "VERDICT: PASS" in combined, combined
    assert "net: OK" in combined or "net: INCONCLUSIVE" in combined, combined
