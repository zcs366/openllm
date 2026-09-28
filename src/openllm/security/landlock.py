"""
Landlock 内核级沙盒锁 — 写面 + 网络面（v2 厚化，2026-09-28）。

[历史] G1（2026-09-27）：内核写锁（纯写面，opt-in OPENLLM_LANDLOCK=1）。
[P0事故] 2026-09-28 v2 施工前置探针发现：G1 的 fs 访问位表整体错位——
  把 EXECUTE(1<<0) 当 WRITE_FILE、把 REFER(1<<13) 当 TRUNCATE（真值 1<<14）。
  后果：① truncate(2) 在白名单外畅通（TRUNCATE 位从未真正生效）；
       ② EXECUTE/READ_FILE/READ_DIR 被误 handled（违反"锁写不锁读"设计
          决策1，上身即锁死白名单外一切读与执行）。
  行为测试只测写——位错位后写仍被拦（WRITE_FILE=1<<1 恰在集内），
  bug 被测试惯例掩盖。三源互证实锤：v5.13/v6.6/master 三版 uapi 头文件
  （位序自 Landlock 诞生从未变过——错在初版代码，非上游改位）+
  内核逐位二分探针（bitprobe：bit14 拦 truncate、bit16+ 本核 EINVAL）。
[v2] 按七神终裁执行序施工：v2d 诚实降级 → v2b TRUNCATE 补齐（含位表
  修正）→ v2c 策略对象（LandlockPolicy + OPENLLM_LANDLOCK_* 家族）→
  v2a 网络面 + seccomp 子策略位（security/seccomp_net.py）。
  判B实锤（NET_PORT 无地址维度，0928 探针）：端口白名单 = 全网同端口
  放行——网络面定位为"TCP 端口损害控制"，**不是网络隔离**；
  connect 目标地址在 cBPF 同样不可过滤（seccomp_data 无指针解引用，
  Documentation/userspace-api/seccomp_filter.rst L284）——地址面
  白名单在两条轨上都不可表达，如实写进警告，留操作者环境层。

设计决策（v2 增补三项，承 G1 三项）：
4. 位表以 uapi 头为准并配逐位探针回归锁：任何访问位先 probe 后信
   （克洛诺斯：逐 access-right 探测 ABI，新 scope 发布即收编）。
5. 网络面默认不 handled：policy 端口白名单为空时 handled_access_net=0，
   行为与 G1 完全一致（opt-in 内再 opt-in，不破坏既有部署）。
6. UDP/DNS 面走 seccomp 子策略位，不在本模块——Landlock ABI 7 管不了
   DGRAM，这是双轨制的轨距（见 security/seccomp_net.py）。

用法：
    python -m openllm.security.landlock          # 自证：上身+验证+报告
    from openllm.security.landlock import KernelWriteLock
    KernelWriteLock.from_env().apply_and_verify()

依赖：零（纯 stdlib + ctypes）。CUDA/第三方库无涉。
"""

import ctypes
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("openllm.security.landlock")

# x86_64 syscall 号（其他架构走 supported()=False 降级，不硬编码猜测）
SYS_LANDLOCK_CREATE_RULESET = 444
SYS_LANDLOCK_ADD_RULE = 445
SYS_LANDLOCK_RESTRICT_SELF = 446

LANDLOCK_CREATE_RULESET_VERSION = 1 << 0
LANDLOCK_RULE_PATH_BENEATH = 1
LANDLOCK_RULE_NET_PORT = 2
PR_SET_NO_NEW_PRIVS = 38

# ── fs 访问位：uapi/linux/landlock.h 真值表（v2b P0 修正）────────────
# 20260928 三源互证实锤（v5.13/v6.6/master 头文件 + bitprobe 逐位探针，
# 内核亲自交代 bit14 拦 truncate、bit16+ 本核 EINVAL）：G1 旧表整体错位
# （把 EXECUTE 当 WRITE_FILE、把 REFER 当 TRUNCATE），后果=truncate(2)
# 白名单外畅通 + 误锁读/执行。位序自 Landlock 诞生从未变过，错在初版代码。
_ACCESS_EXECUTE = 1 << 0      # ABI1
_ACCESS_WRITE_FILE = 1 << 1   # ABI1
_ACCESS_READ_FILE = 1 << 2    # ABI1
_ACCESS_READ_DIR = 1 << 3     # ABI1
_ACCESS_REMOVE_DIR = 1 << 4   # ABI1
_ACCESS_REMOVE_FILE = 1 << 5  # ABI1
_ACCESS_MAKE_CHAR = 1 << 6    # ABI1
_ACCESS_MAKE_DIR = 1 << 7     # ABI1
_ACCESS_MAKE_REG = 1 << 8     # ABI1
_ACCESS_MAKE_SOCK = 1 << 9    # ABI1
_ACCESS_MAKE_FIFO = 1 << 10   # ABI1
_ACCESS_MAKE_BLOCK = 1 << 11  # ABI1
_ACCESS_MAKE_SYM = 1 << 12    # ABI1
_ACCESS_REFER = 1 << 13       # ABI2：跨目录 link/rename
_ACCESS_TRUNCATE = 1 << 14    # ABI3：truncate(2)/ftruncate（bitprobe 实测位）
# _ACCESS_IOCTL_DEV = 1 << 15（ABI5）——有意不 handled：/dev/nvidia* ioctl
# 是推理生命线，handled 即锁死 GPU。设备面留操作者环境层（攻击树已记）。

# 写面 = 引擎自身写所需访问位；不锁读/执行（设计决策1）
_WRITE_FACE_ABI1 = (
    _ACCESS_WRITE_FILE | _ACCESS_REMOVE_DIR | _ACCESS_REMOVE_FILE
    | _ACCESS_MAKE_CHAR | _ACCESS_MAKE_DIR | _ACCESS_MAKE_REG
    | _ACCESS_MAKE_SOCK | _ACCESS_MAKE_FIFO | _ACCESS_MAKE_BLOCK
    | _ACCESS_MAKE_SYM
)

# ── net 访问位（ABI4，netport_probe 本机实测）────────────────────────
_ACCESS_NET_BIND_TCP = 1 << 0
_ACCESS_NET_CONNECT_TCP = 1 << 1


@dataclass
class LandlockPolicy:
    """沙盒策略对象（v2c）：config 可声明，engine 读取。

    allow_write：可写目录白名单（内核规则目录级）。
    allow_net_bind_tcp / allow_net_connect_tcp：TCP 端口白名单。
      空列表 = 网络面不 handled（行为与 G1 完全一致，opt-in 内再 opt-in）。
      判B警告：端口白名单语义 = "全网任意地址的同端口号"（无地址维度），
      放行 11434 即放行外网任何服务器的 11434——仅作损害控制用，非网络隔离。
    """
    allow_write: List[str] = field(default_factory=lambda: [
        str(Path.home() / ".openllm"), "/tmp/openllm"])
    allow_net_bind_tcp: List[int] = field(default_factory=list)
    allow_net_connect_tcp: List[int] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "LandlockPolicy":
        """从环境变量解析策略（v2c：OPENLLM_LANDLOCK_* 家族）。

        OPENLLM_LANDLOCK_ALLOW_WRITE   os.pathsep 分隔的目录列表
        OPENLLM_LANDLOCK_BIND_TCP      逗号分隔端口（TCP bind 白名单）
        OPENLLM_LANDLOCK_CONNECT_TCP   逗号分隔端口（TCP connect 白名单）
        语法错误抛 ValueError（调用方显式处置，防 typo 静默 fail-open）。
        """
        policy = cls()
        raw_write = os.environ.get("OPENLLM_LANDLOCK_ALLOW_WRITE", "").strip()
        if raw_write:
            policy.allow_write = [p for p in
                                  (s.strip() for s in raw_write.split(os.pathsep))
                                  if p]
        policy.allow_net_bind_tcp = cls._parse_ports(
            os.environ.get("OPENLLM_LANDLOCK_BIND_TCP", ""))
        policy.allow_net_connect_tcp = cls._parse_ports(
            os.environ.get("OPENLLM_LANDLOCK_CONNECT_TCP", ""))
        return policy

    @staticmethod
    def _parse_ports(raw: str) -> List[int]:
        ports: List[int] = []
        for tok in (s.strip() for s in raw.split(",")):
            if not tok:
                continue
            port = int(tok)  # ValueError 原样上抛
            if not 1 <= port <= 65535:
                raise ValueError(f"端口越界 [1,65535]: {port!r}")
            ports.append(port)
        return ports

    def net_ports_with_bits(self) -> Dict[int, int]:
        """按端口聚合访问位（同一端口两条白名单合并一条规则）。"""
        acc: Dict[int, int] = {}
        for p in self.allow_net_bind_tcp:
            acc[p] = acc.get(p, 0) | _ACCESS_NET_BIND_TCP
        for p in self.allow_net_connect_tcp:
            acc[p] = acc.get(p, 0) | _ACCESS_NET_CONNECT_TCP
        return acc

    def describe(self) -> str:
        net = (f"net_bind={self.allow_net_bind_tcp} "
               f"net_connect={self.allow_net_connect_tcp}"
               if (self.allow_net_bind_tcp or self.allow_net_connect_tcp)
               else "net=未handled")
        return f"write={self.allow_write} {net}"


@dataclass
class LockResult:
    """上身+自证结果。"""
    enabled: bool
    abi: int = 0
    allowed: Optional[List[str]] = None
    verified_inside: bool = False
    verified_outside: bool = False
    detail: str = ""
    canvas_dir: str = ""
    net_enabled: bool = False
    net_verified: bool = False
    net_detail: str = ""

    @property
    def ok(self) -> bool:
        return self.enabled and self.verified_inside and self.verified_outside


class KernelWriteLock:
    """Landlock 内核级写锁。锁写不锁读；一次上身，进程终身。"""

    def __init__(self, policy: Optional[LandlockPolicy] = None):
        self.policy = policy or LandlockPolicy()
        # 兼容别名（G1 时代属性，engine/测试仍按目录清单理解白名单）
        self.allowed_dirs = [Path(p).expanduser().resolve()
                             for p in self.policy.allow_write]
        self._libc = ctypes.CDLL(None, use_errno=True)
        self._syscall = self._libc.syscall
        self._syscall.restype = ctypes.c_long

    @classmethod
    def default(cls) -> "KernelWriteLock":
        return cls(LandlockPolicy())  # 默认策略（不经 env）

    @classmethod
    def from_env(cls) -> "KernelWriteLock":
        return cls(LandlockPolicy.from_env())

    # ── syscall 封装：多式回退（varargs marshalling 各环境有差，实证定式）──

    def _sys(self, nr: int, *args) -> int:
        """发 syscall，两种 marshalling 回退。返回 syscall 原值（负数=失败）。"""
        attempts = [
            # 式1：数字全显式宽度（c_long 号 + 逐参类型）
            lambda: self._syscall(ctypes.c_long(nr), *args),
            # 式2：裸 Python 值（ctypes 默认转换）
            lambda: self._syscall(nr, *args),
        ]
        last_err: Optional[int] = None
        for attempt in attempts:
            try:
                ctypes.set_errno(0)
                ret = attempt()
                if ret >= 0:
                    return ret
                last_err = ctypes.get_errno()
                # EINVAL 常见于 marshalling 错——换式重试；其他 errno 直接回
                if last_err != 22:
                    return ret
            except (ctypes.ArgumentError, TypeError, ValueError):
                continue
        return -1 if last_err is None else -last_err

    def abi_version(self) -> int:
        """查 Landlock ABI 版本。0=不可用。

        20260927 军师夜班修正：LANDLOCK_CREATE_RULESET_VERSION 标志位是 1<<0
        （官方 uapi/linux/landlock.h），早前误写 1<<4 导致内核 EINVAL、
        ABI 误报 0。实测（WSL2 6.18.33，C 直编+ctypes 双证）：flags=1 时
        返回正数=ABI 版本。
        """
        ret = self._sys(
            SYS_LANDLOCK_CREATE_RULESET,
            ctypes.c_void_p(0),
            ctypes.c_size_t(0),
            ctypes.c_uint(LANDLOCK_CREATE_RULESET_VERSION),
        )
        return ret if ret > 0 else 0

    def supported(self) -> bool:
        return self.abi_version() >= 1

    def _fs_handled_bits(self, abi: int) -> int:
        """写面 handled 位集（v2b 修正后真值）。不锁读/执行（设计决策1）。"""
        bits = _WRITE_FACE_ABI1
        if abi >= 2:
            bits |= _ACCESS_REFER
        if abi >= 3:
            bits |= _ACCESS_TRUNCATE
        return bits

    def apply_and_verify(self) -> LockResult:
        """上身+自证。注意：restrict 一旦成功不可撤销，进程终身受锁。"""
        result = LockResult(enabled=False)

        # 上身前先造验证画布（上身后再造目录本身会被锁）
        canvas_denied = tempfile.mkdtemp(prefix="landlock_verify_canvas_")
        result.canvas_dir = canvas_denied
        canvas_allowed = self.allowed_dirs[0] / "landlock_verify_ok"
        canvas_allowed.mkdir(parents=True, exist_ok=True)

        abi = self.abi_version()
        result.abi = abi
        if abi < 1:
            result.detail = f"内核不支持 Landlock（abi={abi}）——保持 Python 层检查"
            return result

        fs_handled = self._fs_handled_bits(abi)
        net_map = self.policy.net_ports_with_bits()
        net_handled = bool(net_map) and abi >= 4

        class RulesetAttr(ctypes.Structure):
            _fields_ = [("handled_access_fs", ctypes.c_uint64),
                        ("handled_access_net", ctypes.c_uint64)]

        attr = RulesetAttr(
            fs_handled,
            (_ACCESS_NET_BIND_TCP | _ACCESS_NET_CONNECT_TCP) if net_handled else 0,
        )
        # fs-only 路径传 8 字节（G1 实证定式）；双面传完整 16 字节
        # （netport_probe2 本机实证 restrict 生效）
        fd = self._sys(
            SYS_LANDLOCK_CREATE_RULESET,
            ctypes.byref(attr),
            ctypes.c_size_t(16 if net_handled else 8),
            ctypes.c_uint(0),
        )
        if fd < 0:
            result.detail = f"create_ruleset 失败 errno={-fd}"
            return result

        class PathBeneathAttr(ctypes.Structure):
            _fields_ = [("allowed_access", ctypes.c_uint64),
                        ("parent_fd", ctypes.c_int)]

        for d in self.allowed_dirs:
            if not d.is_dir():
                result.detail = f"白名单目录不存在: {d}"
                return result
            dirfd = os.open(str(d), os.O_PATH | os.O_CLOEXEC)
            try:
                rule = PathBeneathAttr(fs_handled, dirfd)
                if self._sys(
                    SYS_LANDLOCK_ADD_RULE,
                    ctypes.c_int(fd),
                    ctypes.c_uint(LANDLOCK_RULE_PATH_BENEATH),
                    ctypes.byref(rule),
                    ctypes.c_uint(0),
                ) != 0:
                    result.detail = f"add_rule({d}) 失败 errno={ctypes.get_errno()}"
                    return result
            finally:
                os.close(dirfd)

        class NetPortAttr(ctypes.Structure):
            _fields_ = [("allowed_access", ctypes.c_uint64),
                        ("port", ctypes.c_uint64)]

        for port, bits in sorted(net_map.items()):
            rule = NetPortAttr(bits, port)
            if self._sys(
                SYS_LANDLOCK_ADD_RULE,
                ctypes.c_int(fd),
                ctypes.c_uint(LANDLOCK_RULE_NET_PORT),
                ctypes.byref(rule),
                ctypes.c_uint(0),
            ) != 0:
                result.detail = f"add_rule(net:{port}) 失败 errno={ctypes.get_errno()}"
                return result

        if self._libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            result.detail = f"PR_SET_NO_NEW_PRIVS 失败 errno={ctypes.get_errno()}"
            return result

        if self._sys(SYS_LANDLOCK_RESTRICT_SELF, ctypes.c_int(fd), ctypes.c_uint(0)) != 0:
            result.detail = f"restrict_self 失败 errno={ctypes.get_errno()}"
            return result

        result.enabled = True
        result.allowed = [str(d) for d in self.allowed_dirs]

        # ── 自证：白名单内必须可写，白名单外必须 EACCES ──
        try:
            probe_ok = canvas_allowed / "probe.txt"
            probe_ok.write_text("ok", encoding="utf-8")
            probe_ok.unlink()
            result.verified_inside = True
        except OSError as exc:
            result.detail = f"白名单内写失败——上身配置有误（不可逆！立即排查）: {exc}"
            return result

        try:
            (Path(canvas_denied) / "probe.txt").write_text("x", encoding="utf-8")
            result.verified_outside = False
            result.detail = "白名单外写成功——Landlock 未生效（严重，立即报告）"
            return result
        except PermissionError:
            result.verified_outside = True
        except OSError as exc:
            # 非权限类失败也算被拦，但要如实记录
            result.verified_outside = exc.errno == 13
            if not result.verified_outside:
                result.detail = f"白名单外写异常（非EACCES）: {exc}"
                return result

        result.detail = (
            f"内核写锁上身且自证通过：ABI={abi}，"
            f"白名单内可写✓ 白名单外EACCES✓（画布 {canvas_denied} 可删）"
        )

        # ── 自证2（网络面，v2a）：仅当网络面 handled 时执行 ──
        # 判B定式（0928）：负探针用临时端口（端口0，永不在白名单），
        # 正探针 bind 白名单端口——EADDRINUSE 不再误判（有服务占用≠被拒）。
        if net_handled:
            result.net_enabled = True
            result.net_detail = self._verify_net(net_map)
            result.net_verified = result.net_detail.startswith("OK")

        return result

    def _verify_net(self, net_map: Dict[int, int]) -> str:
        """网络面行为自证（restrict 已上身，探针即终审）。返回串以 OK/FAIL/INCONCLUSIVE 开头。"""
        import errno as _errno
        import socket

        # 负探针：bind 临时端口必拒（白名单不可能含端口0）
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("0.0.0.0", 0))
            return "FAIL: bind 端口0（临时端口）成功——网络面白名单未生效（严重）"
        except OSError as exc:
            if exc.errno not in (_errno.EACCES, _errno.EPERM):
                return f"INCONCLUSIVE: bind(0) 异常 errno={exc.errno}"
        finally:
            try:
                s.close()
            except OSError:
                pass

        # 正探针：白名单端口
        port = sorted(net_map)[0]
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
            note = f"bind {port} ALLOWED✓"
        except OSError as exc:
            if exc.errno == _errno.EADDRINUSE:
                note = f"bind {port} 被占用（EADDRINUSE，规则已收无法行为复验）"
            elif exc.errno in (_errno.EACCES, _errno.EPERM):
                return f"FAIL: 白名单内 {port} bind 被拒 errno={exc.errno}（配置不符）"
            else:
                note = f"bind {port} 异常 errno={exc.errno}（如实记录）"
        finally:
            try:
                s.close()
            except OSError:
                pass

        # connect 负探针：非白名单高端口必拒（ECONNREFUSED=未被拦=严重）
        neg = 49999 if 49999 not in net_map else 49998
        c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        c.settimeout(2)
        try:
            err = c.connect_ex(("127.0.0.1", neg))
            if err in (0, _errno.ECONNREFUSED):
                return f"FAIL: connect {neg} 未被拦（ret={err}）——网络面白名单未生效（严重）"
            if err in (_errno.EACCES, _errno.EPERM):
                return (f"OK: {note}; connect {neg} EACCES✓"
                        f"（端口白名单=损害控制，非网络隔离——判B）")
            return f"INCONCLUSIVE: connect {neg} ret={err}（{note}）"
        finally:
            try:
                c.close()
            except OSError:
                pass


def main() -> int:
    """CLI 自证入口：python -m openllm.security.landlock"""
    lock = KernelWriteLock.from_env()
    abi = lock.abi_version()
    print(f"Landlock ABI: {abi}")
    print(f"策略: {lock.policy.describe()}")
    if abi < 1:
        print("VERDICT: UNSUPPORTED（内核不支持，保持 Python 层检查）")
        return 2
    result = lock.apply_and_verify()
    print(f"enabled={result.enabled} inside={result.verified_inside} "
          f"outside={result.verified_outside}")
    print(f"allowed={result.allowed}")
    if result.net_enabled:
        print(f"net: {result.net_detail}")
    print(result.detail)
    print(f"VERDICT: {'PASS' if result.ok else 'FAIL'}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
