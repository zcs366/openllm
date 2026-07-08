#!/usr/bin/env python3
"""Reliability Engine 测试"""
import sys
sys.path.insert(0, 'src')
from openllm.governance.reliability import (
    ReliabilityEngine, ObservabilityDetector, RepairabilityEngine,
    EvolvabilityEngine, InfraAwarenessMonitor,
    ReliabilityDimension, Severity, RepairAction,
    ReliabilityEvent, RepairResult,
)

# T1
obs = ObservabilityDetector()
e1 = obs.check_stateful_audit(cumulative_score=0.8, threshold=0.77)
assert e1 and e1.severity == Severity.MEDIUM
e2 = obs.check_stateful_audit(cumulative_score=2.0, threshold=0.77)
assert e2.severity == Severity.HIGH
print('T1 OK: observability detection')

# T2
repair = RepairabilityEngine()
action = repair.decide_action(e1)
result = repair.execute_repair(e1)
assert result.success
print(f'T2 OK: repairability action={action.value}')

# T3
evo = EvolvabilityEngine()
for i in range(4):
    ev = ReliabilityEvent(event_id=f't{i}', dimension=ReliabilityDimension.OBSERVABILITY,
                          severity=Severity.HIGH, source='tool_call', description=f'fail{i}')
    rr = RepairResult(success=True, action=RepairAction.FALLBACK, detail=f'fb{i}', lesson_learned='tool unavailable')
    evo.record_lesson(ev, rr)
assert evo.get_patterns().get('tool_call:high', 0) >= 3
assert len(evo.get_proposals()) > 0
print('T3 OK: evolvability patterns+proposals')

# T4
infra = InfraAwarenessMonitor()
s1 = infra.take_snapshot(active_sessions=5, critical_events_24h=0)
assert s1.overall_health == 'healthy'
s2 = infra.take_snapshot(critical_events_24h=2)
assert s2.overall_health == 'critical'
print('T4 OK: infra awareness healthy->critical')

# T5
engine = ReliabilityEngine()
ev = ReliabilityEvent(event_id='e1', dimension=ReliabilityDimension.OBSERVABILITY,
                      severity=Severity.MEDIUM, source='main_loop', description='timeout')
rr = engine.process_event(ev)
assert rr.success
report = engine.get_health_report()
assert report['observability']['total_events'] >= 1
print(f'T5 OK: unified engine events={report["observability"]["total_events"]}')

# T6
engine2 = ReliabilityEngine()
count = 0
for sev in Severity:
    for src in ['tool_call', 'main_loop', 'stateful_audit']:
        ev = ReliabilityEvent(event_id=f'x-{sev.value}-{src}', dimension=ReliabilityDimension.OBSERVABILITY,
                              severity=sev, source=src, description='test')
        rr = engine2.process_event(ev)
        assert rr.action is not None
        count += 1
print(f'T6 OK: strategy table {count} cases')

print('ALL 6 TESTS PASSED')
