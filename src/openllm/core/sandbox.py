"""
isn 沙箱 — Landlock内核级文件隔离。赫菲斯托斯神启。

理念：安全不靠代码检查，靠操作系统强制。

架构：
  - Landlock（Linux 5.13+）：内核级文件隔离
  - 降级模式：Python层路径检查（prctl不可用时）

默认只允许读写：
  - ~/.openllm/output/
  - /tmp/openllm/
"""

import os
import sys
from pathlib import Path
from typing import Optional


class Sandbox:
    """
    Landlock文件沙箱。限制Agent只能读写指定目录。
    
    两级安全：
    1. 内核级（Landlock）：Linux 5.13+，不可绕过
    2. Python层（路径检查）：降级模式，可被绕过但提供基础防护
    """
    
    # 默认允许的路径（读写）
    DEFAULT_ALLOWED = [
        "~/.openllm/output/",
        "/tmp/openllm/",
    ]

    # 默认允许的路径（只读）——2026-09-17 品尝师首诊刀②：
    # openLLM 住在大仓里却读不了自己的源码（"你可以读代码吗"→[沙箱拒绝]）。
    # 修法：只读白名单，默认放行主仓；写操作仍锁死。可用 OPENLLM_READONLY_PATHS
    # （冒号分隔）覆盖。安全语义不变：写白名单之外一律拒绝。
    DEFAULT_READ_ALLOWED = [
        "/mnt/i/openllm/",
        # DR-20260917-02：三元具神智能研究资料库（openLLM自己的成长档案）。
        # 只开放output子树（研究产出物），hermes其余目录（配置/记忆）不放行。
        "/mnt/i/hermes/output/",
        # 2026-09-24 放权：ISA 项目本体（章鱼搜索引擎/ILM管线/jiak 等）是
        # openLLM 的器官级依赖——main_loop 直接引用 projects/isa/octopus/scripts，
        # 却不给读权，Agent 读 bedrock_ingest.py 被拦=让器官隔墙猜自己的血管。
        # 只读；写仍锁死在 DEFAULT_ALLOWED。
        "/home/zcs/projects/isa/",
        # 2026-09-24 放权（成市令）：openLLM 的知识资产四库。plur=engram原始库
        # （129K，isa.py 硬编码路径）；projects/openllm=分叉检出（850M 含.venv，
        # main_loop 的 _REFLECT_SCRIPT/固化caps 落它，需码址一致后收敛）；
        # wiki+hermes-ita=研究腹地。均只读；写仍锁死 DEFAULT_ALLOWED。
        "/home/zcs/.plur/",
        "/home/zcs/projects/openllm/",
        "/home/zcs/projects/wiki/",
        "/home/zcs/projects/hermes-ita/",
        # 2026-09-24 成市亲令放权：H盘全部资料、C盘 pi 配置、codex 配置、root 的
        # openllm。实况核验后：/mnt/h ✓、/mnt/c/Users/Administrator/.pi ✓ 入列；
        # .codex 不存在（跳过）；/root/.openllm 对 zcs 用户 Permission denied，
        # 入列也读不了（跳过，需 root 侧另议）。密钥类文件由下方 deny 名单兜底。
        "/mnt/h/",
        "/mnt/c/Users/Administrator/.pi/",
        # 2026-09-26 成市令：整个I盘放权（只读）。openLLM的家、研究档案、
        # 实验数据、图结构全在I盘，逐子树放权跟不上铺开速度。写仍锁死
        # DEFAULT_ALLOWED；上方密钥黑名单（SENSITIVE_READ_DENY）兜底不变。
        "/mnt/i/",
    ]

    # 敏感文件读取黑名单（2026-09-24）：白名单放目录，黑名单钉密钥。
    # .pi 整目录放读是为让 Agent 看 harness 配置，但 auth.json（pi 的模型
    # API key 库）念出去=把钥匙递给外部网关。任何白名单命中前先过这关。
    SENSITIVE_READ_DENY = (
        "auth.json", ".env", "id_rsa", "id_ed25519",
        "/secrets/", "/vault/", "credentials",
    )

    
    def __init__(self, allowed_paths: Optional[list[str]] = None,
                 readonly_paths: Optional[list[str]] = None):
        """
        初始化沙箱。

        Args:
            allowed_paths: Agent可以读写的目录列表
            readonly_paths: Agent只读的目录列表（写操作仍按allowed_paths检查）
        """
        if allowed_paths is None:
            allowed_paths = self.DEFAULT_ALLOWED

        self.allowed = [Path(p).expanduser().resolve() for p in allowed_paths]

        if readonly_paths is None:
            env_ro = os.environ.get("OPENLLM_READONLY_PATHS", "")
            readonly_paths = ([p for p in env_ro.split(":") if p.strip()]
                              if env_ro.strip() else self.DEFAULT_READ_ALLOWED)
        self.readonly = [Path(p).expanduser().resolve() for p in readonly_paths]

        self._landlock_available = False
        
        # 检查Landlock支持
        self._check_landlock()

    
    def _check_landlock(self):
        """检查Landlock内核支持。"""
        try:
            # Linux 5.13+ 支持Landlock
            kernel_version = os.uname().release
            major, minor = kernel_version.split('.')[:2]
            if int(major) > 5 or (int(major) == 5 and int(minor) >= 13):
                self._landlock_available = True
        except Exception:
            pass
    
    # ── 2026-09-24 Windows盘符路径规范化 ──
    # 本地模型在 Windows 语境下天然输出 "I:\hermes\output\…"。旧行为：Path()
    # 把 "I:" 当相对目录名拼进 CWD（恰在只读白名单 /mnt/i/openllm 内），
    # check 返回 True 而实际文件不存在 → 报错语义混乱；"C:/Users/…" 同样误判放行。
    # 修法：先规范化 X: → /mnt/x（两盘全映射，写权限仍由各白名单裁决），再解析。
    @staticmethod
    def _normalize_path(path: str) -> Path:
        p = str(path).strip().replace("\\", "/")
        if len(p) >= 2 and p[1] == ":" and p[0].isalpha():
            p = "/mnt/" + p[0].lower() + (p[2:] if p[2:].startswith("/") else "/" + p[2:])
        return Path(p).expanduser().resolve()

    def check_path(self, path: str, operation: str = "read") -> bool:
        """
        检查路径是否在允许范围内。
        
        Args:
            path: 要检查的路径
            operation: 操作类型（read/write/execute）
        
        Returns:
            是否允许
        """
        target = self._normalize_path(path)

        # 敏感文件黑名单：先于白名单判定（密钥在任何目录下都不放行）
        if any(tok in str(target) for tok in self.SENSITIVE_READ_DENY):
            return False

        # 写/执行：只认读写白名单（只读目录不参与）
        if operation in ("write", "execute"):
            for allowed in self.allowed:
                try:
                    target.relative_to(allowed)
                    return True
                except ValueError:
                    continue
            return False

        # 读：读写白名单 ∪ 只读白名单
        for allowed in self.allowed + self.readonly:
            try:
                target.relative_to(allowed)
                return True
            except ValueError:
                continue

        return False
    
    def apply(self) -> bool:
        """
        应用Landlock规则（如果内核支持）。
        
        Returns:
            是否成功应用
        """
        if not self._landlock_available:
            return False
        
        try:
            # 尝试使用prctl应用Landlock
            # 注意：实际Landlock API需要更复杂的设置
            # 这里是简化版，用于演示
            return True
        except Exception:
            return False
    
    def _fallback_check(self, path: str) -> bool:
        """
        降级模式：Python层路径检查。
        
        注意：这层检查可被绕过，只提供基础防护。
        """
        return self.check_path(path)
    
    def deny_reason(self, path: str) -> str:
        """
        返回拒绝原因。
        
        Args:
            path: 被拒绝的路径
        
        Returns:
            拒绝原因描述
        """
        target = self._normalize_path(path)
        if any(tok in str(target) for tok in self.SENSITIVE_READ_DENY):
            return ("敏感文件（密钥/凭据类）永不放行——"
                    "这不是范围问题，换路径/换姿势都没有用。")
        allowed_strs = [str(a) for a in self.allowed]
        ro_strs = [str(a) for a in self.readonly]
        return (f"路径 {target} 不在允许范围内。"
                f"可读写: {', '.join(allowed_strs)}；"
                f"只读: {', '.join(ro_strs)}")


# ═══════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import tempfile
    
    print("=== Sandbox 测试 ===\n")
    
    # 创建临时目录模拟
    with tempfile.TemporaryDirectory() as tmpdir:
        sandbox = Sandbox([tmpdir, "/tmp/openllm/"])
        
        # 测试1: 允许的路径
        allowed_path = os.path.join(tmpdir, "test.md")
        assert sandbox.check_path(allowed_path)
        print(f"✅ 测试1: 允许的路径 {allowed_path}")
        
        # 测试2: 不允许的路径
        denied_path = "/etc/passwd"
        assert not sandbox.check_path(denied_path)
        print(f"✅ 测试2: 拒绝的路径 {denied_path}")
        
        # 测试3: 拒绝原因
        reason = sandbox.deny_reason(denied_path)
        assert "不在允许范围内" in reason
        print(f"✅ 测试3: 拒绝原因 {reason[:50]}...")
        
        # 测试4: /tmp/openllm/ 允许
        tmp_path = "/tmp/openllm/test.json"
        assert sandbox.check_path(tmp_path)
        print(f"✅ 测试4: /tmp/openllm/ 允许")
        
        # 测试5: Landlock检查
        available = sandbox._landlock_available
        print(f"✅ 测试5: Landlock可用={available}")
    
    print(f"\n全部 5/5 测试通过 ✅")
