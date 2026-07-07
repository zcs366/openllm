#!/usr/bin/env python3
"""六体自监督压力测试"""
import sys, os, tempfile, json, shutil
sys.path.insert(0, os.path.expanduser("~/projects/openllm/src"))

from openllm.governance.feedback_loop import FeedbackLoop, FeedbackStore
from dataclasses import asdict

tmpdir = tempfile.mkdtemp()
ok = 0; fail = 0
def check(n, c):
    global ok, fail
    if c: ok += 1; print(f"  OK {n}")
    else: fail += 1; print(f"  FAIL {n}")

normal_outputs = {
    "IAX": {"heartbeat_ok": True, "tick_count": 1},
    "IAI": {"results": [{"score": 0.8}]},
    "ISA": {"memory_count": 10, "session_count": 3},
    "IOS": {"decisions_made": 5},
    "ISN": {"tools_called": 10, "error_rate": 0.05},
    "IKO": {"output_length": 500, "format_ok": True},
}

print("=" * 50)
print("六体自监督压力测试")
print("=" * 50)

# 基线
print("\n[基线] 正常状态")
fl = FeedbackLoop(store_path=os.path.join(tmpdir, "base.jsonl"))
records = fl.collect_feedback(normal_outputs)
check("基线: 6条反馈", len(records) == 6)
report = fl.get_health_report()
check("基线: 全部healthy", all(b["status"] == "healthy" for b in report["bodies"].values()))

# 故障1: IAX心跳停止
print("\n[故障1] IAX心跳停止")
fl1 = FeedbackLoop(store_path=os.path.join(tmpdir, "f1.jsonl"))
outputs1 = dict(normal_outputs); outputs1["IAX"] = {"heartbeat_ok": False}
recs1 = fl1.collect_feedback(outputs1)
# FeedbackRecord用属性访问
iax = [r for r in recs1 if r.target_body == "IAX"]
check("故障1: IAX被评估", len(iax) > 0)
if iax:
    check("故障1: 分数<0.5", iax[0].score < 0.5)
    check("故障1: 检测心跳异常", any("心跳" in i for i in iax[0].issues))

# 故障2: IAI感知失真
print("\n[故障2] IAI感知失真")
fl2 = FeedbackLoop(store_path=os.path.join(tmpdir, "f2.jsonl"))
outputs2 = dict(normal_outputs); outputs2["IAI"] = {}
recs2 = fl2.collect_feedback(outputs2)
iai = [r for r in recs2 if r.target_body == "IAI"]
check("故障2: IAI被评估", len(iai) > 0)
if iai:
    check("故障2: 分数<0.5", iai[0].score < 0.5)

# 故障3: ISA记忆丢失
print("\n[故障3] ISA记忆丢失")
fl3 = FeedbackLoop(store_path=os.path.join(tmpdir, "f3.jsonl"))
outputs3 = dict(normal_outputs); outputs3["ISA"] = {"memory_count": 0}
recs3 = fl3.collect_feedback(outputs3)
isa = [r for r in recs3 if r.target_body == "ISA"]
check("故障3: ISA被评估", len(isa) > 0)
if isa:
    check("故障3: 分数<0.5", isa[0].score < 0.5)
    check("故障3: 检测记忆为空", any("记忆" in i for i in isa[0].issues))

# 故障4: ISN工具错误率飙升
print("\n[故障4] ISN工具错误率飙升")
fl4 = FeedbackLoop(store_path=os.path.join(tmpdir, "f4.jsonl"))
outputs4 = dict(normal_outputs); outputs4["ISN"] = {"tools_called": 10, "error_rate": 0.5}
recs4 = fl4.collect_feedback(outputs4)
isn = [r for r in recs4 if r.target_body == "ISN"]
check("故障4: ISN被评估", len(isn) > 0)
if isn:
    check("故障4: 分数<0.5", isn[0].score < 0.5)
    check("故障4: 检测错误率", any("错误率" in i for i in isn[0].issues))

# 故障5: IOS决策偏移
print("\n[故障5] IOS决策偏移")
fl5 = FeedbackLoop(store_path=os.path.join(tmpdir, "f5.jsonl"))
outputs5 = dict(normal_outputs); outputs5["IOS"] = {}
recs5 = fl5.collect_feedback(outputs5)
ios = [r for r in recs5 if r.target_body == "IOS"]
check("故障5: IOS被评估", len(ios) > 0)

# 故障6: IKO输出偏离
print("\n[故障6] IKO输出偏离")
fl6 = FeedbackLoop(store_path=os.path.join(tmpdir, "f6.jsonl"))
outputs6 = dict(normal_outputs); outputs6["IKO"] = {}
recs6 = fl6.collect_feedback(outputs6)
iko = [r for r in recs6 if r.target_body == "IKO"]
check("故障6: IKO被评估", len(iko) > 0)

# Hash链完整性
print("\n[审计] Hash链完整性")
check("正常链完整", fl.store.verify_chain())
for i, fpath in enumerate([
    os.path.join(tmpdir, "f1.jsonl"),
    os.path.join(tmpdir, "f2.jsonl"),
    os.path.join(tmpdir, "f3.jsonl"),
    os.path.join(tmpdir, "f4.jsonl"),
], start=1):
    check(f"故障{i}链完整", FeedbackStore(fpath).verify_chain())

# apply_feedback
print("\n[反馈] apply_feedback调整因子")
fl_adj = FeedbackLoop(store_path=os.path.join(tmpdir, "adj.jsonl"))
for _ in range(3):
    fl_adj.collect_feedback({
        "IAX": {"heartbeat_ok": False},
        "IAI": {"results": [{"score": 0.8}]},
        "ISA": {"memory_count": 10},
        "IOS": {"decisions_made": 5},
        "ISN": {"tools_called": 10, "error_rate": 0.05},
        "IKO": {"output_length": 500},
    })
adj = fl_adj.apply_feedback()
check("IAX调整因子<0.7", adj.get("IAX", 1.0) < 0.7)
check("其他体>0.7", adj.get("ISA", 0) > 0.7)

shutil.rmtree(tmpdir, ignore_errors=True)
print(f"\n{'='*50}")
print(f"RESULT: {ok}/{ok+fail} passed")
if fail > 0: sys.exit(1)
