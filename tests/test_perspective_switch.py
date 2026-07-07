#!/usr/bin/env python3
"""Tests for perspective_switch.py"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/projects/openllm/src"))

from openllm.governance.perspective_switch import (
    PerspectiveSwitch, SYSTEM_PERSPECTIVE, USER_PERSPECTIVE, PERSPECTIVES
)

ok = 0; fail = 0
def check(n, c):
    global ok, fail
    if c: ok += 1; print(f"  OK {n}")
    else: fail += 1; print(f"  FAIL {n}")

print("[1] Perspective mappings exist")
check("system映射有6角色", len(SYSTEM_PERSPECTIVE) == 6)
check("user映射有6角色", len(USER_PERSPECTIVE) == 6)
check("两个映射都有body/rationale/supervises", all(
    all(k in v for k in ("body", "rationale", "supervises"))
    for v in SYSTEM_PERSPECTIVE.values()
))

print("\n[2] PerspectiveSwitch defaults")
ps = PerspectiveSwitch()
check("默认视角=system", ps.current_perspective == "system")
check("get_mapping返回system映射", ps.get_mapping()["军师"]["body"] == "IAX")

print("\n[3] Switch to user perspective")
result = ps.switch_to("user")
check("切换后视角=user", ps.current_perspective == "user")
check("切换返回from=system", result["from"] == "system")
check("切换返回to=user", result["to"] == "user")
check("user映射军师→IAI", ps.get_mapping()["军师"]["body"] == "IAI")
check("user映射子产→IKO", ps.get_mapping()["子产"]["body"] == "IKO")

print("\n[4] Auto rotate")
result2 = ps.auto_rotate()
check("轮换后回到system", ps.current_perspective == "system")
check("轮换计数=2", ps._rotation_count == 2)

print("\n[5] Perspective context generation")
ctx = ps.get_perspective_context("user")
check("context含user视角", "user视角" in ctx)
check("context含显示器悖论", "显示器悖论" in ctx)
check("context含角色映射", "军师 → IAI" in ctx)

print("\n[6] Diff between perspectives")
diff = ps.get_diff()
check("军师在两视角不同", not diff["军师"]["same"])
check("子产在两视角不同", not diff["子产"]["same"])
# 鲁班在两个映射中不同（ISN vs IOS）
check("鲁班在两视角不同", not diff["鲁班"]["same"])

print("\n[7] Invalid perspective raises")
try:
    ps.switch_to("invalid")
    check("无效视角应报错", False)
except ValueError:
    check("无效视角报ValueError", True)

print(f"\nRESULT: {ok}/{ok+fail} passed")
if fail > 0: sys.exit(1)
