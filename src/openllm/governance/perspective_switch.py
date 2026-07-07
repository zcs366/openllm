"""
合议流程视角切换

显示器悖论（张成市发现）：子产和IKO是同一个边界上的两面。
映射A（系统内视角）和映射B（用户外视角）是同一架构的两个投影。

视角切换不是"选哪个对"——是让系统能在两种投影间自由切换，
从而获得更完整的自我认知。

用法：
    from openllm.governance.perspective_switch import PerspectiveSwitch
    ps = PerspectiveSwitch()
    mapping = ps.get_mapping("user")  # 或 "system"
    prompt_context = ps.get_perspective_context("user")
"""
from typing import Optional


# ═══════════════════════════════════════════════════════
# 两种映射定义
# ═══════════════════════════════════════════════════════

# 映射A：系统内视角（从功能出发）
SYSTEM_PERSPECTIVE = {
    "军师": {
        "body": "IAX",
        "rationale": "军师统筹全局→IAX心跳维持全局同步",
        "supervises": "全局协调·心跳节律",
    },
    "子贡": {
        "body": "IAI",
        "rationale": "子贡外交沟通→IAI感知采集信息",
        "supervises": "信息采集·路由分发",
    },
    "萧何": {
        "body": "ISA",
        "rationale": "萧何管资源→ISA记忆管知识资源",
        "supervises": "记忆管理·路径规划",
    },
    "子产": {
        "body": "IOS",
        "rationale": "子产断该不该→IOS决策裁决",
        "supervises": "需求判断·治理决策",
    },
    "鲁班": {
        "body": "ISN",
        "rationale": "鲁班建造→ISN工具执行",
        "supervises": "工程实现·工具调用",
    },
    "韩信": {
        "body": "IKO",
        "rationale": "韩信画终局→IKO输出呈现",
        "supervises": "远景呈现·对外表达",
    },
}

# 映射B：用户外视角（从感知出发·张成市方案）
USER_PERSPECTIVE = {
    "军师": {
        "body": "IAI",
        "rationale": "军师是全局感知中心→IAI感知汇聚信息",
        "supervises": "全局感知·态势判断",
    },
    "萧何": {
        "body": "IAX",
        "rationale": "萧何维持运转→IAX心跳驱动系统",
        "supervises": "系统运转·粮道不绝",
    },
    "子贡": {
        "body": "ISA",
        "rationale": "子贡穿梭传递信息→ISA记忆流动载体",
        "supervises": "信息流动·记忆管理",
    },
    "鲁班": {
        "body": "IOS",
        "rationale": "鲁班工程判断=决策→IOS决策裁决",
        "supervises": "工程决策·资源判断",
    },
    "韩信": {
        "body": "ISN",
        "rationale": "韩信多多益善的执行→ISN承载工具调用",
        "supervises": "执行能力·目标达成",
    },
    "子产": {
        "body": "IKO",
        "rationale": "子产对用户是输出面→IKO从用户视角是输入通道",
        "supervises": "用户界面·需求呈现",
    },
}

PERSPECTIVES = {
    "system": SYSTEM_PERSPECTIVE,
    "user": USER_PERSPECTIVE,
}


class PerspectiveSwitch:
    """合议流程的视角切换器"""

    def __init__(self):
        self._current = "system"
        self._rotation_count = 0

    @property
    def current_perspective(self) -> str:
        return self._current

    def get_mapping(self, perspective: Optional[str] = None) -> dict:
        """获取指定视角的角色-六体映射"""
        p = perspective or self._current
        return PERSPECTIVES.get(p, SYSTEM_PERSPECTIVE)

    def switch_to(self, perspective: str) -> dict:
        """切换到指定视角，返回新映射"""
        if perspective not in PERSPECTIVES:
            raise ValueError(f"未知视角: {perspective}. 可选: {list(PERSPECTIVES.keys())}")
        old = self._current
        self._current = perspective
        self._rotation_count += 1
        return {
            "from": old,
            "to": perspective,
            "mapping": self.get_mapping(perspective),
            "rotation": self._rotation_count,
        }

    def auto_rotate(self) -> dict:
        """自动轮换视角（每次合议调用一次）"""
        next_p = "user" if self._current == "system" else "system"
        return self.switch_to(next_p)

    def get_perspective_context(self, perspective: Optional[str] = None) -> str:
        """生成注入到合议prompt中的视角上下文"""
        mapping = self.get_mapping(perspective)
        p = perspective or self._current

        lines = [f"当前合议视角: {p}视角"]
        if p == "user":
            lines.append("(显示器悖论：从用户角度看，子产是输出面，IKO是输入通道)")
        else:
            lines.append("(从系统内部功能出发的角色分配)")

        lines.append("")
        lines.append("角色-六体映射:")
        for role, info in mapping.items():
            lines.append(f"  {role} → {info['body']} ({info['rationale']})")

        return "\n".join(lines)

    def get_diff(self) -> dict:
        """对比两种视角的差异"""
        diff = {}
        for role in SYSTEM_PERSPECTIVE:
            s_body = SYSTEM_PERSPECTIVE[role]["body"]
            u_body = USER_PERSPECTIVE[role]["body"]
            diff[role] = {
                "system": s_body,
                "user": u_body,
                "same": s_body == u_body,
            }
        return diff
