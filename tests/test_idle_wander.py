#!/usr/bin/env python3
"""Tests for idle_wander.py — 间隙体空闲期协议"""
import sys, os, time
sys.path.insert(0, os.path.expanduser("~/projects/openllm/src"))

from openllm.core.idle_wander import IdleWanderer, WanderDiscovery, WANDER_DOMAINS

ok = 0; fail = 0
def check(name, cond):
    global ok, fail
    if cond: ok += 1; print(f"  OK {name}")
    else: fail += 1; print(f"  FAIL {name}")

print("[1] IdleWanderer creation")
w = IdleWanderer(threshold=3)
check("threshold=3", w.threshold == 3)
check("初始idle_count=0", w._idle_count == 0)
check("初始无散步", w._total_wanders == 0)

print("\n[2] tick_idle counting")
check("tick1不触发", not w.tick_idle())
check("tick2不触发", not w.tick_idle())
check("tick3触发散步", w.tick_idle())
check("idle_count=3", w._idle_count == 3)

print("\n[3] tick_active resets")
w2 = IdleWanderer(threshold=3)
w2.tick_idle(); w2.tick_idle()
w2.tick_active()
check("活跃后重置为0", w2._idle_count == 0)
check("重置后不触发", not w2.tick_idle())

print("\n[4] wander execution")
w3 = IdleWanderer(threshold=1)
w3.tick_idle()
discoveries = w3.wander()
check("散步返回发现列表", isinstance(discoveries, list))
check("发现数量>0", len(discoveries) > 0)
check("总散步次数=1", w3._total_wanders == 1)
check("散步后idle_count重置", w3._idle_count == 0)

print("\n[5] WanderDiscovery structure")
d = discoveries[0]
check("有domain", bool(d.domain))
check("有query", bool(d.query))
check("有finding", bool(d.finding))
check("有relevance", 0 <= d.relevance <= 1)
check("source=idle_wander", d.source == "idle_wander")
entry = d.to_memory_entry()
check("to_memory_entry有content", "content" in entry)
check("to_memory_entry有source", entry["source"] == "idle_wander")

print("\n[6] Domain coverage")
visited = set()
for _ in range(5):
    w4 = IdleWanderer(threshold=1)
    w4.tick_idle()
    ds = w4.wander()
    for d2 in ds:
        visited.add(d2.domain)
check("散步覆盖多个领域", len(visited) >= 3)

print("\n[7] Should act on discovery")
d_high = WanderDiscovery("test", "q", "f", relevance=0.9)
d_low = WanderDiscovery("test", "q", "f", relevance=0.3)
check("高相关性→应行动", w.should_act_on_discovery(d_high))
check("低相关性→不行动", not w.should_act_on_discovery(d_low))

print("\n[8] Status report")
status = w3.get_status()
check("status有mode", "mode" in status)
check("status有total_wanders", "total_wanders" in status)
check("status有visited_domains", "visited_domains" in status)

print("\n[9] Recent discoveries")
recent = w3.get_recent_discoveries(n=2)
check("recent是list", isinstance(recent, list))
check("recent条目有content", "content" in recent[0] if recent else False)

print("\n[10] All domains eventually visited")
w5 = IdleWanderer(threshold=1)
all_visited = set()
for _ in range(20):
    w5.tick_idle()
    ds = w5.wander()
    for d3 in ds:
        all_visited.add(d3.domain)
    if len(all_visited) >= len(WANDER_DOMAINS):
        break
check("全部领域被访问", len(all_visited) >= len(WANDER_DOMAINS))

print(f"\nRESULT: {ok}/{ok+fail} passed")
if fail > 0: sys.exit(1)
