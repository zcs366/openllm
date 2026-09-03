#!/usr/bin/env python3
"""comm_demo.py — AgentComm 通信栈端到端演示（ISA 通信体战役·落地证明）

场景：
  1. 六体注册进通信网（Registry）
  2. IAI 感知到洞察 → 可靠 ask ISN（裁决 → 会话 → ACK → 回复 → 闭环 CLOSED）
  3. IKO 广播语义路由——只发给对"模型训练"感兴趣的体（P0-2）
  4. 黑名单裁决拦截演示（P0-1）
  5. 温度脉冲 × 打扰预算（P1-4 × M2）

跑：python3 examples/comm_demo.py
"""
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openllm.comm import Registry, CommAgent
from openllm.iai.session import SessionManager, SessionStatus
from openllm.iai.event_bus import EventBus
from openllm.governance.adjudication import TransmissionAdjudicator


def banner(title: str) -> None:
    print(f"\n── {title} ─{'─' * max(0, 50 - len(title))}")


def main() -> int:
    demo_dir = Path(tempfile.mkdtemp(prefix="comm-demo-"))
    print("═══ AgentComm 多 agent 可靠通信演示 ═══")
    print(f"demo 目录: {demo_dir}")

    # 基础设施（裁决 + 会话 + 事件总线共享）
    sessions = SessionManager(sessions_dir=demo_dir / "sessions")
    bus = EventBus(log_dir=demo_dir / "events")
    adj = TransmissionAdjudicator(log_dir=demo_dir / "adj")
    adj.set_daily_budget("张成市", 2)  # 张成市每日最多 2 次打扰型唤醒
    reg = Registry(sessions=sessions, bus=bus, adjudicator=adj)

    # 通信事件观察者（bus 可见性：谁和谁通信全程可感知）
    comm_events = []
    bus.subscribe(lambda e: comm_events.append(e.type), source_filter="SESSION")
    def watch_comm(e):
        pass  # comm.* 事件由各 agent source 发出，单独收集
    for src in ("IAI", "ISA", "ISN", "IKO", "军师"):
        bus.subscribe(lambda e, src=src: comm_events.append(f"{src}:{e.type}"),
                      source_filter=src)

    # 六体注册
    iai = CommAgent("IAI", registry=reg, interests=["模型训练", "语义路由"])
    isa = CommAgent("ISA", registry=reg, interests=["记忆检索"])
    isn = CommAgent("ISN", registry=reg, interests=["工具调用", "技能"])
    ios = CommAgent("IO-S", registry=reg, interests=["治理"])
    iko = CommAgent("IKO", registry=reg, interests=["输出排版"])
    junshi = CommAgent("军师", registry=reg, interests=["一切"])

    # ISN 能回答问题
    isn.on_message(lambda agent, msg: f"[ISN] 收到'{msg.body[:30]}'——已执行并返回结果")
    isa.on_message(lambda agent, msg: f"[ISA] 记忆检索: {msg.body[:30]} → 3 条相关")

    # ── 场景 2：IAI → ISN 可靠 ask ──
    banner("场景1 · IAI → ISN 可靠会话（裁决→ACK→回复→闭环）")
    r = iai.send("ISN", "帮我在技能库查一下语义路由相关技能", kind="ask", importance=0.6)
    s0 = sessions.get(r.session_id)
    print(f"send ok={r.ok} 会话={r.session_id} 状态={s0.status.name if s0 else '?'}")
    isn.run_once()
    s1 = sessions.get(r.session_id)
    print(f"ISN 处理后  状态={s1.status.name if s1 else '?'}")
    iai.run_once()
    s2 = sessions.get(r.session_id)
    print(f"IAI 收回复  状态={s2.status.name if s2 else '?'}   ← 闭环 CLOSED")

    # ── 场景 2：ISA → IAI ask（反向）──
    banner("场景1b · ISA → IAI 反向会话")
    iai.on_message(lambda agent, msg: f"[IAI] 感知确认: {msg.body[:30]}")
    r2 = isa.send("IAI", "最近有没有新的感知数据需要入库", kind="ask")
    iai.run_once()
    isa.run_once()
    s = sessions.get(r2.session_id)
    print(f"ISA→IAI ask 会话终态: {s.status.name if s else '?'}")

    # ── 场景 3：语义广播找人 ──
    banner("场景2 · IKO 语义广播（只发给兴趣匹配者）")
    got = {aid: [] for aid in ("IAI", "ISA", "ISN", "IO-S")}
    for aid in got:
        a = reg.get(aid)
        if a:
            a.on_message(lambda ag, m, aid=aid: got[aid].append(m.body))
    results = iko.broadcast("模型训练本周进展：LoRA 微调 7B 损失降 0.8", semantic=True)
    for res in results:
        target = reg.get(res.message.to_id)
        if target:
            target.run_once()
    print(f"广播送达 {len(results)} 个体: {[x.message.to_id for x in results if x.ok and x.message is not None]}")
    for aid, msgs in got.items():
        if msgs:
            print(f"  {aid} 收到: {msgs[0][:36]}…")

    # ── 场景 4：黑名单裁决拦截 ──
    banner("场景3 · 裁决拦截（黑名单）")
    adj2_blocked = TransmissionAdjudicator(log_dir=demo_dir / "adj2")
    adj2_blocked.block_pair("IO-S", "ISN")
    reg2 = Registry(sessions=SessionManager(sessions_dir=demo_dir / "s2"),
                    bus=bus, adjudicator=adj2_blocked)
    ios2 = CommAgent("IO-S", registry=reg2)
    isn2 = CommAgent("ISN", registry=reg2)
    bad = ios2.send("ISN", "立刻停止当前任务", kind="tell")
    print(f"IO-S→ISN 被黑名单拦截: ok={bad.ok} verdict={bad.verdict_action} reason={bad.reason[:40]}")

    # ── 场景 5：温度脉冲 × 打扰预算 ──
    banner("场景4 · 温度脉冲（军师想到你 × 打扰预算 2/日）")
    from openllm.iko.pulse import PulseEngine, TimeWindow
    pulse_eng = PulseEngine(log_dir=demo_dir / "pulses", recipient="张成市",
                            adjudicator=adj, time_window=TimeWindow(0, 24),
                            randomize_kind=False,
                            context={"context": "通信体系统"})
    for i in range(3):
        p = pulse_eng.tick()
        print(f"  脉冲 {i + 1}: {'投递 ✓ ' + p.message if p else '拦截（预算/裁决）✗'}")

    # ── 统计 ──
    banner("通信统计")
    for aid in ("IAI", "ISA", "ISN", "军师"):
        a = reg.get(aid)
        if a:
            st = a.stats()
            print(f"  {aid}: sent={st['sent']} received={st['received']} "
                  f"replied={st['replied']} blocked={st['blocked']}")
    print(f"\n会话总数: {sessions.get_stats()['total']} "
          f"活跃: {sessions.get_stats()['active']}")
    print(f"事件总线观察: 捕获 {len(comm_events)} 个 comm.* 事件")
    seen = sorted(set(comm_events))
    print(f"  事件类型: {seen[:8]}{'…' if len(seen) > 8 else ''}")
    print("\n✅ 端到端演示完成——多 agent 可靠通信系统落地可用")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
