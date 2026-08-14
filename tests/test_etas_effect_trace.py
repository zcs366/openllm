"""
test_etas_effect_trace.py — ETAS三件套测试
==========================================

覆盖：
  T1: EffectRow — ToolRegistry effects字段
  T2: ActionTrace — TraceEvent + ActionTraceStore
  T3: TraceTransparentExecute — executor trace记录
  T4: TraceSpec — spec编译 + monitor检查
"""

import os
import sys
import tempfile
import time
import unittest

# 确保项目根目录在sys.path中
_project_root = os.path.join(os.path.dirname(__file__), "..", "src")
if _project_root not in sys.path:
    sys.path.insert(0, os.path.abspath(_project_root))

from openllm.core.action_trace import (
    TraceEvent, EventPhase, DenialCause, ActionTraceStore,
    make_trace_event,
)
from openllm.core.trace_spec import (
    TraceSpec, TraceMonitor, TemporalConstraint,
    spec_approval_before, spec_dry_run_only, spec_read_only,
)
from openllm.tools.executor import ToolRegistry, ToolResult, create_default_tools


# ══════════════════════════════════════════════════════
# T2: ActionTrace 测试
# ══════════════════════════════════════════════════════

class TestTraceEvent(unittest.TestCase):
    """TraceEvent数据结构测试。"""
    
    def test_create_event(self):
        e = make_trace_event(EventPhase.REQUEST, "File.write", params={"path": "/tmp/x"})
        self.assertEqual(e.phase, EventPhase.REQUEST)
        self.assertEqual(e.action_name, "File.write")
        self.assertIn("path", e.params)
        self.assertTrue(e.event_id)
    
    def test_event_immutable(self):
        e = make_trace_event(EventPhase.COMMIT, "shell")
        with self.assertRaises(AttributeError):
            e.phase = EventPhase.DENIED  # frozen=True
    
    def test_event_to_dict(self):
        e = make_trace_event(
            EventPhase.DENIED, "Email.send",
            denial_cause=DenialCause.POLICY,
            denial_reason="策略拒绝",
        )
        d = e.to_dict()
        self.assertEqual(d["phase"], "denied")
        self.assertEqual(d["denial_cause"], "policy")
        self.assertEqual(d["denial_reason"], "策略拒绝")


class TestActionTraceStore(unittest.TestCase):
    """ActionTraceStore存储测试。"""
    
    def test_append_and_size(self):
        store = ActionTraceStore()
        self.assertEqual(store.size, 0)
        store.append(make_trace_event(EventPhase.REQUEST, "a"))
        store.append(make_trace_event(EventPhase.COMMIT, "a"))
        self.assertEqual(store.size, 2)
    
    def test_filter_by_action(self):
        store = ActionTraceStore()
        store.append(make_trace_event(EventPhase.REQUEST, "File.write"))
        store.append(make_trace_event(EventPhase.COMMIT, "File.write"))
        store.append(make_trace_event(EventPhase.REQUEST, "shell"))
        results = store.filter_by_action("File.write")
        self.assertEqual(len(results), 2)
    
    def test_filter_by_phase(self):
        store = ActionTraceStore()
        store.append(make_trace_event(EventPhase.REQUEST, "a"))
        store.append(make_trace_event(EventPhase.COMMIT, "a"))
        store.append(make_trace_event(EventPhase.DENIED, "b"))
        self.assertEqual(len(store.filter_by_phase(EventPhase.REQUEST)), 1)
        self.assertEqual(len(store.filter_by_phase(EventPhase.COMMIT)), 1)
        self.assertEqual(len(store.filter_by_phase(EventPhase.DENIED)), 1)
    
    def test_has_request_and_commit(self):
        store = ActionTraceStore()
        self.assertFalse(store.has_request("a"))
        self.assertFalse(store.has_commit("a"))
        store.append(make_trace_event(EventPhase.REQUEST, "a"))
        self.assertTrue(store.has_request("a"))
        self.assertFalse(store.has_commit("a"))
        store.append(make_trace_event(EventPhase.COMMIT, "a"))
        self.assertTrue(store.has_commit("a"))
    
    def test_request_commit_pairs(self):
        store = ActionTraceStore()
        store.append(make_trace_event(EventPhase.REQUEST, "File.write"))
        store.append(make_trace_event(EventPhase.COMMIT, "File.write"))
        pairs = store.request_commit_pairs()
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][0].phase, EventPhase.REQUEST)
        self.assertEqual(pairs[0][1].phase, EventPhase.COMMIT)
    
    def test_to_summary(self):
        store = ActionTraceStore()
        store.append(make_trace_event(EventPhase.REQUEST, "a"))
        store.append(make_trace_event(EventPhase.COMMIT, "a"))
        store.append(make_trace_event(EventPhase.DENIED, "b"))
        summary = store.to_summary()
        self.assertEqual(summary["total_events"], 3)
        self.assertEqual(summary["requests"], 1)
        self.assertEqual(summary["commits"], 1)
        self.assertEqual(summary["denials"], 1)
    
    def test_max_events_aging(self):
        store = ActionTraceStore(max_events=10)
        for i in range(15):
            store.append(make_trace_event(EventPhase.REQUEST, f"action_{i}"))
        # 应该老化了，不超过max_events
        self.assertLessEqual(store.size, 10)
    
    def test_clear(self):
        store = ActionTraceStore()
        store.append(make_trace_event(EventPhase.REQUEST, "a"))
        store.clear()
        self.assertEqual(store.size, 0)


# ══════════════════════════════════════════════════════
# T1: EffectRow 测试
# ══════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════
# T3: TraceTransparentExecute 测试
# ══════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════
# T4: TraceSpec 测试
# ══════════════════════════════════════════════════════

class TestTraceSpec(unittest.TestCase):
    """TraceSpec编译+monitor检查测试。"""
    
    def test_deny_spec(self):
        spec = spec_dry_run_only()
        monitor = spec.compile()
        
        trace = ActionTraceStore()
        trace.append(make_trace_event(EventPhase.REQUEST, "File.write"))
        trace.append(make_trace_event(EventPhase.COMMIT, "File.write"))
        
        passed, reason = monitor.check_event(trace.events[1], trace)
        self.assertFalse(passed)
        self.assertIn("策略拒绝", reason)
    
    def test_allow_spec(self):
        spec = TraceSpec("AllowRead").allow("Memory.read")
        monitor = spec.compile()
        
        trace = ActionTraceStore()
        event = make_trace_event(EventPhase.COMMIT, "Memory.read")
        trace.append(event)
        
        passed, reason = monitor.check_event(event, trace)
        self.assertTrue(passed)
    
    def test_temporal_constraint_pass(self):
        spec = (TraceSpec("PublishPolicy")
                .allow("Approval.request")
                .allow("File.write")
                .require_before("Approval.request", "File.write"))
        monitor = spec.compile()
        
        trace = ActionTraceStore()
        # approval先于write
        trace.append(make_trace_event(EventPhase.COMMIT, "Approval.request"))
        trace.append(make_trace_event(EventPhase.COMMIT, "File.write"))
        
        passed, reason = monitor.check_event(trace.events[1], trace)
        self.assertTrue(passed)
    
    def test_temporal_constraint_fail(self):
        spec = (TraceSpec("PublishPolicy")
                .allow("Approval.request")
                .allow("File.write")
                .require_before("Approval.request", "File.write"))
        monitor = spec.compile()
        
        trace = ActionTraceStore()
        # write没有先approval → 违规
        trace.append(make_trace_event(EventPhase.COMMIT, "File.write"))
        
        passed, reason = monitor.check_event(trace.events[0], trace)
        self.assertFalse(passed)
        self.assertIn("时序违规", reason)
    
    def test_approval_before_template(self):
        spec = spec_approval_before("File.write")
        monitor = spec.compile()
        
        # 正常路径：approval → write
        trace1 = ActionTraceStore()
        trace1.append(make_trace_event(EventPhase.COMMIT, "Approval.request"))
        trace1.append(make_trace_event(EventPhase.COMMIT, "File.write"))
        passed, _ = monitor.check_event(trace1.events[1], trace1)
        self.assertTrue(passed)
        
        # 违规路径：write没有approval
        trace2 = ActionTraceStore()
        trace2.append(make_trace_event(EventPhase.COMMIT, "File.write"))
        passed, reason = monitor.check_event(trace2.events[0], trace2)
        self.assertFalse(passed)
    
    def test_request_events_not_checked(self):
        """request事件不触发deny检查——只有commit才检查。"""
        spec = spec_dry_run_only()
        monitor = spec.compile()
        
        trace = ActionTraceStore()
        event = make_trace_event(EventPhase.REQUEST, "File.write")
        trace.append(event)
        
        passed, _ = monitor.check_event(event, trace)
        self.assertTrue(passed)  # request不检查
    
    def test_check_full_trace(self):
        spec = (TraceSpec("Test")
                .deny("Shell.exec")
                .allow("File.write"))
        monitor = spec.compile()
        
        trace = ActionTraceStore()
        trace.append(make_trace_event(EventPhase.COMMIT, "File.write"))
        trace.append(make_trace_event(EventPhase.COMMIT, "Shell.exec"))
        trace.append(make_trace_event(EventPhase.COMMIT, "File.write"))
        
        results = monitor.check_full_trace(trace)
        self.assertEqual(len(results), 3)
        self.assertIsNone(results[0][1])   # File.write: pass
        self.assertIsNotNone(results[1][1])  # Shell.exec: denied
        self.assertIsNone(results[2][1])   # File.write: pass


if __name__ == "__main__":
    unittest.main()
