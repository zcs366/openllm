"""
seccomp 网络面子策略 — UDP/DNS 封锁（v2 双轨制第二轨，2026-09-28）。

分工（判B + 七神终裁赫尔墨斯条款）：
- Landlock NET_PORT（security/landlock.py）：TCP bind/connect 端口白名单
  ——判B定位：TCP 端口损害控制，非网络隔离。
- 本模块（seccomp cBPF）：SOCK_DGRAM 拒绝 ——UDP/DNS 面。
  Landlock ABI 7 管不了 DGRAM，这是双轨制的轨距。
- 地址面（connect 目标 IP）：**两轨都不可表达**——cBPF 的 seccomp_data
  无指针解引用（Documentation/userspace-api/seccomp_filter.rst L284
  "contains the values of register arguments ... but does not contain
  pointers to memory"），Landlock NET_PORT 无地址字段（0928 探针判B实锤）。
  目标地址白名单留操作者环境层（防火墙/netns），如实声明，不假装覆盖。

激活：OPENLLM_SECCOMP_NET=block_udp（engine 钩子；显式 opt-in，
与 OPENLLM_LANDLOCK 互相独立）。restrict 同样进程终身不可逆。

DNS 生死宣告（诚实优先）：glibc 解析器首选 UDP 53。本策略生效后：
- 操作者环境提供 TCP DNS / 本地 DNS 代理（dnsmasq/unbound 等）→ 引擎正常解析；
- systemd-resolved 路径（nss → /run/systemd AF_UNIX varlink）不受影响
  （本策略只拦 AF_INET/6 DGRAM，不碰 AF_UNIX）；
- 都没有 → 引擎域名解析失败（直连 IP 的 endpoint 不受影响）。
这是特性不是缺陷：UDP 面收窄本就意味着"要么有 TCP DNS，要么没有 DNS"。

cBPF 程序（12 指令，x86_64 seccomp_data 布局）：
  arch 校验（AUDIT_ARCH_X86_64，防 32 位 syscall shim 绕过）→
  nr==socket(41) → domain∈{AF_INET(2), AF_INET6(10)} →
  (type & 0xff)==SOCK_DGRAM(2)（掩掉 SOCK_CLOEXEC/SOCK_NONBLOCK 标志位）→
  RET_ERRNO(EPERM)；其余 RET_ALLOW。

用法：
    python -m openllm.security.seccomp_net   # 自证：安装+验证+报告
    from openllm.security.seccomp_net import SeccompNetLock
    SeccompNetLock.block_udp().apply_and_verify()

依赖：零（纯 stdlib + ctypes）。
"""

import ctypes
import logging
import os
import socket
import struct
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("openllm.security.seccomp_net")

PR_SET_SECCOMP = 22
PR_SET_NO_NEW_PRIVS = 38
SECCOMP_MODE_FILTER = 2

# cBPF 指令编码（linux/filter.h）
BPF_LD_W_ABS = 0x20
BPF_JEQ_K = 0x15
BPF_ALU_AND_K = 0x54
BPF_RET_K = 0x06

# seccomp 返回动作
SECCOMP_RET_ALLOW = 0x7FFF0000
SECCOMP_RET_ERRNO = 0x00050000
_SECCOMP_RET_ERRNO_EPERM = SECCOMP_RET_ERRNO | 1  # EPERM

# seccomp_data 布局（x86_64）：nr@0, arch@4, instruction_pointer@8, args@[16..]
_AUDIT_ARCH_X86_64 = 0xC000003E
_NR_SOCKET = 41            # x86_64 socket(2)
_AF_INET = 2
_AF_INET6 = 10
_SOCK_DGRAM = 2


def _bpf(code: int, jt: int = 0, jf: int = 0, k: int = 0) -> bytes:
    """单条 sock_filter：{__u16 code; __u8 jt; __u8 jf; __u32 k;}（8 字节 LE）"""
    return struct.pack("<HBBI", code, jt, jf, k)


def build_block_udp_program() -> bytes:
    """block-UDP cBPF 程序（12 指令）。返回可塞进 sock_fprog 的二进制。"""
    prog = b"".join([
        # 0: A = arch
        _bpf(BPF_LD_W_ABS, k=4),
        # 1: arch != AUDIT_ARCH_X86_64 → 10（EPERM；永不 ALLOW，断 shim 绕过）
        _bpf(BPF_JEQ_K, jt=0, jf=8, k=_AUDIT_ARCH_X86_64),
        # 2: A = nr
        _bpf(BPF_LD_W_ABS, k=0),
        # 3: nr != socket(41) → 11（ALLOW）
        _bpf(BPF_JEQ_K, jt=0, jf=7, k=_NR_SOCKET),
        # 4: A = args[0] low32 = domain
        _bpf(BPF_LD_W_ABS, k=16),
        # 5: domain == AF_INET → 7（查 type）
        _bpf(BPF_JEQ_K, jt=1, jf=0, k=_AF_INET),
        # 6: domain == AF_INET6 → 7；否则 → 11
        _bpf(BPF_JEQ_K, jt=0, jf=4, k=_AF_INET6),
        # 7: A = args[1] low32 = type（含 SOCK_CLOEXEC/SOCK_NONBLOCK 高位）
        _bpf(BPF_LD_W_ABS, k=24),
        # 8: A &= 0xff（掩掉 socket type 标志位）
        _bpf(BPF_ALU_AND_K, k=0xFF),
        # 9: type != SOCK_DGRAM → 11；== → 10
        _bpf(BPF_JEQ_K, jt=0, jf=1, k=_SOCK_DGRAM),
        # 10: RET ERRNO(EPERM)
        _bpf(BPF_RET_K, k=_SECCOMP_RET_ERRNO_EPERM),
        # 11: RET ALLOW
        _bpf(BPF_RET_K, k=SECCOMP_RET_ALLOW),
    ])
    assert len(prog) == 12 * 8
    return prog


class _SockFilter(ctypes.LittleEndianStructure):
    _fields_ = [("code", ctypes.c_uint16),
                ("jt", ctypes.c_uint8),
                ("jf", ctypes.c_uint8),
                ("k", ctypes.c_uint32)]


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_uint16),
                ("filter", ctypes.POINTER(_SockFilter))]


@dataclass
class SeccompResult:
    """安装+自证结果。"""
    installed: bool
    udp_blocked: bool = False   # socket(AF_INET, DGRAM) 必须被拒
    tcp_ok: bool = False        # socket(AF_INET, STREAM) 必须仍可用
    unix_ok: bool = False       # AF_UNIX 不受影响（systemd-resolved 依赖）
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.installed and self.udp_blocked and self.tcp_ok and self.unix_ok


class SeccompNetLock:
    """seccomp cBPF 网络面子策略。安装后进程终身不可逆（同 Landlock）。"""

    def __init__(self, mode: str = "block_udp"):
        if mode != "block_udp":
            raise ValueError(f"未知模式 {mode!r}（当前支持：block_udp）")
        self.mode = mode
        self._libc = ctypes.CDLL(None, use_errno=True)
        self._libc.prctl.restype = ctypes.c_int

    @classmethod
    def from_env(cls) -> Optional["SeccompNetLock"]:
        """OPENLLM_SECCOMP_NET=block_udp → 锁实例；未设/其他值 → None（不安装）。"""
        return cls("block_udp") if os.environ.get(
            "OPENLLM_SECCOMP_NET") == "block_udp" else None

    @classmethod
    def block_udp(cls) -> "SeccompNetLock":
        return cls("block_udp")

    def apply_and_verify(self) -> SeccompResult:
        """安装 filter + 行为自证。注意：安装成功即进程终身受锁。"""
        result = SeccompResult(installed=False)
        if self._libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            result.detail = f"PR_SET_NO_NEW_PRIVS 失败 errno={ctypes.get_errno()}"
            return result

        prog = build_block_udp_program()
        buf = (ctypes.c_char * len(prog)).from_buffer_copy(prog)
        fprog = _SockFprog()
        fprog.len = len(prog) // 8  # 指令条数（8 字节/条）
        fprog.filter = ctypes.cast(buf, ctypes.POINTER(_SockFilter))
        ctypes.set_errno(0)
        ret = self._libc.prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER,
                               ctypes.byref(fprog), 0, 0)
        if ret != 0:
            err = ctypes.get_errno()
            result.detail = (
                f"PR_SET_SECCOMP 失败 errno={err}"
                + ("（内核 seccomp 被禁或参数不持）" if err in (22, 1) else ""))
            return result
        result.installed = True

        # ── 自证：DGRAM 必拒，STREAM 必通，AF_UNIX 必不受影响 ──
        try:
            socket.socket(socket.AF_INET, socket.SOCK_DGRAM).close()
            result.udp_blocked = False
            result.detail = "FAIL: socket(AF_INET, SOCK_DGRAM) 未被拒——filter 未生效（严重）"
            return result
        except OSError as exc:
            result.udp_blocked = exc.errno == 1  # EPERM
            if not result.udp_blocked:
                result.detail = f"INCONCLUSIVE: DGRAM 拒绝但 errno={exc.errno}（非EPERM）"
                return result

        try:
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).close()
            result.tcp_ok = True
        except OSError as exc:
            result.detail = f"FAIL: TCP STREAM 被误伤 errno={exc.errno}（严重）"
            return result

        try:
            socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM).close()
            result.unix_ok = True
        except OSError as exc:
            result.detail = f"FAIL: AF_UNIX 被误伤 errno={exc.errno}（systemd-resolved 会断）"
            return result

        result.detail = ("seccomp block_udp 上身且自证通过：UDP/EPERM✓ TCP可用✓ "
                         "AF_UNIX不受影响✓（DNS 需 TCP DNS 或 systemd-resolved，"
                         "地址面两轨皆不可过滤——留操作者环境层）")
        return result


def main() -> int:
    """CLI 自证入口：python -m openllm.security.seccomp_net"""
    lock = SeccompNetLock.block_udp()
    result = lock.apply_and_verify()
    print(f"installed={result.installed} udp_blocked={result.udp_blocked} "
          f"tcp_ok={result.tcp_ok} unix_ok={result.unix_ok}")
    print(result.detail)
    print(f"VERDICT: {'PASS' if result.ok else 'FAIL'}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
