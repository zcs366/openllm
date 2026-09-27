"""
Landlock 内核级写锁 — 沙箱第二级安全（2026-09-26 军师自主窗口 G1 提案，
2026-09-27 成市授权落盘）。

背景：core/sandbox.py 的 Python 层路径检查是唯一防线，任何绕过 check_path
的写路径（如工具 shell-out `echo x > /etc/cron.d/evil`）畅通无阻。
本模块把"写白名单之外一律拒绝"下沉到内核（Landlock LSM，ABI≥1），
父进程 shell 出的子进程一并继承——不可绕过、不可撤销。

实证（2026-09-26 本机）：内核 6.18.33 WSL2，ABI=7，
C 直编测试：白名单内写成功、白名单外 EACCES（/tmp/ll_test.c 模式）。

设计决策（三项，均有依据）：
1. 锁写不锁读：只 handled 写类访问位。引擎自身要读 config/密钥/代码，
   锁读=自杀；读防线仍由 sandbox.py Python 层把守。
2. 默认 opt-in（OPENLLM_LANDLOCK=1）：restrict_self 不可逆且进程全域，
   自动上身会破坏测试进程与操作者合法工作流。不可逆强制由操作者显式授权。
3. 白名单取超集 [~/.openllm, /tmp/openllm]：引擎自身要写 checkpoints/
   clock.jsonl/config 等于 ~/.openllm 根，内核规则只支持目录级，
   故比 Python 层写白名单（~/.openllm/output）宽。残余风险如实记入手册§7。

用法：
    python -m openllm.security.landlock          # 自证：上身+验证+报告
    from openllm.security.landlock import KernelWriteLock
    KernelWriteLock.default().apply_and_verify()

依赖：零（纯 stdlib + ctypes）。CUDA/第三方库无涉。
"""

import ctypes
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("openllm.security.landlock")

# x86_64 syscall 号（其他架构走 supported()=False 降级，不硬编码猜测）
SYS_LANDLOCK_CREATE_RULESET = 444
SYS_LANDLOCK_ADD_RULE = 445
SYS_LANDLOCK_RESTRICT_SELF = 446

LANDLOCK_CREATE_RULESET_VERSION = 1 << 0
LANDLOCK_RULE_PATH_BENEATH = 1
PR_SET_NO_NEW_PRIVS = 38

# 写类访问位（ABI 1 基础集）
_ACCESS_ABI1 = {
    "WRITE_FILE": 1 << 0,
    "REMOVE_DIR": 1 << 1,
    "REMOVE_FILE": 1 << 2,
    "MAKE_CHAR": 1 << 3,
    "MAKE_DIR": 1 << 4,
    "MAKE_REG": 1 << 5,
    "MAKE_FIFO": 1 << 6,
    "MAKE_BLOCK": 1 << 7,
    "MAKE_SYM": 1 << 8,
}
# ABI 2 增 REFER（跨目录引用控制）；ABI 3 增 TRUNCATE
_ACCESS_REFER = 1 << 9
_ACCESS_TRUNCATE = 1 << 13


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

    @property
    def ok(self) -> bool:
        return self.enabled and self.verified_inside and self.verified_outside


class KernelWriteLock:
    """Landlock 内核级写锁。锁写不锁读；一次上身，进程终身。"""

    def __init__(self, allowed_dirs: Optional[List[str]] = None):
        # 内核白名单=引擎自身写面的超集（见模块 docstring 设计决策3）
        if allowed_dirs is None:
            allowed_dirs = [str(Path.home() / ".openllm"), "/tmp/openllm"]
        self.allowed_dirs = [Path(p).expanduser().resolve() for p in allowed_dirs]
        self._libc = ctypes.CDLL(None, use_errno=True)
        self._syscall = self._libc.syscall
        self._syscall.restype = ctypes.c_long

    @classmethod
    def default(cls) -> "KernelWriteLock":
        return cls()

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

    def _handled_bits(self, abi: int) -> int:
        bits = 0
        for v in _ACCESS_ABI1.values():
            bits |= v
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

        handled = self._handled_bits(abi)

        class RulesetAttr(ctypes.Structure):
            _fields_ = [("handled_access_fs", ctypes.c_uint64)]

        attr = RulesetAttr(handled)
        fd = self._sys(
            SYS_LANDLOCK_CREATE_RULESET,
            ctypes.byref(attr),
            ctypes.c_size_t(ctypes.sizeof(attr)),
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
                rule = PathBeneathAttr(handled, dirfd)
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
        return result


def main() -> int:
    """CLI 自证入口：python -m openllm.security.landlock"""
    lock = KernelWriteLock.default()
    abi = lock.abi_version()
    print(f"Landlock ABI: {abi}")
    if abi < 1:
        print("VERDICT: UNSUPPORTED（内核不支持，保持 Python 层检查）")
        return 2
    result = lock.apply_and_verify()
    print(f"enabled={result.enabled} inside={result.verified_inside} "
          f"outside={result.verified_outside}")
    print(f"allowed={result.allowed}")
    print(result.detail)
    print(f"VERDICT: {'PASS' if result.ok else 'FAIL'}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
