"""test_etas_persist.py — ActionTrace持久化测试"""
import os
import sys
import json
import tempfile
import time
import unittest
from pathlib import Path

_project_root = os.path.join(os.path.dirname(__file__), "..", "src")
if _project_root not in sys.path:
    sys.path.insert(0, os.path.abspath(_project_root))

from openllm.core.action_trace import (
    TraceEvent, EventPhase, DenialCause, ActionTraceStore,
    SecurityFilter, ActionTraceWriter, make_trace_event,
)


class TestSecurityFilter(unittest.TestCase):
    """T1: SecurityFilter测试。"""
    
    def test_filter_password(self):
        result = SecurityFilter.filter_params({"password": "secret123"})
        self.assertEqual(result["password"], "[FILTERED]")
    
    def test_filter_token(self):
        result = SecurityFilter.filter_params({"token": "abc123"})
        self.assertEqual(result["token"], "[FILTERED]")
    
    def test_filter_api_key(self):
        result = SecurityFilter.filter_params({"api_key": "sk-abcdefghijklmnop"})
        self.assertEqual(result["api_key"], "[FILTERED]")
    
    def test_filter_bearer(self):
        result = SecurityFilter.filter_params({"auth": "Bearer eyJhbGciOiJIUzI1NiJ9"})
        self.assertEqual(result["auth"], "[FILTERED]")
    
    def test_preserve_normal_params(self):
        result = SecurityFilter.filter_params({"path": "/tmp/test", "limit": 10})
        self.assertEqual(result["path"], "/tmp/test")
        self.assertEqual(result["limit"], 10)
    
    def test_recursive_filter(self):
        result = SecurityFilter.filter_params({"config": {"password": "x", "name": "test"}})
        self.assertEqual(result["config"]["password"], "[FILTERED]")
        self.assertEqual(result["config"]["name"], "test")
    
    def test_empty_params(self):
        self.assertEqual(SecurityFilter.filter_params({}), {})
        self.assertEqual(SecurityFilter.filter_params(None), {})
    
    def test_filter_private_key(self):
        result = SecurityFilter.filter_params({"key": "-----BEGIN RSA PRIVATE KEY-----"})
        self.assertEqual(result["key"], "[FILTERED]")


class TestActionTraceWriter(unittest.TestCase):
    """T2: ActionTraceWriter持久化测试。"""
    
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)
    
    def test_append_creates_file(self):
        writer = ActionTraceWriter("test_sid", storage_dir=Path(self._tmpdir))
        event = make_trace_event(EventPhase.REQUEST, "read_file", params={"path": "/tmp/x"})
        writer.append(event)
        writer.close()
        
        path = Path(self._tmpdir) / "action_trace.jsonl"
        self.assertTrue(path.exists())
        lines = path.read_text().strip().split("\n")
        self.assertEqual(len(lines), 1)
        
        entry = json.loads(lines[0])
        self.assertEqual(entry["v"], 1)
        self.assertEqual(entry["sid"], "test_sid")
        self.assertEqual(entry["phase"], "request")
        self.assertEqual(entry["action"], "read_file")
    
    def test_append_filters_sensitive(self):
        writer = ActionTraceWriter("test_sid", storage_dir=Path(self._tmpdir))
        event = make_trace_event(EventPhase.REQUEST, "shell", params={"command": "ls", "password": "secret"})
        writer.append(event)
        writer.close()
        
        path = Path(self._tmpdir) / "action_trace.jsonl"
        entry = json.loads(path.read_text().strip())
        self.assertEqual(entry["params"]["password"], "[FILTERED]")
        self.assertEqual(entry["params"]["command"], "ls")
    
    def test_multiple_appends(self):
        writer = ActionTraceWriter("test_sid", storage_dir=Path(self._tmpdir))
        for i in range(5):
            writer.append(make_trace_event(EventPhase.COMMIT, f"tool_{i}"))
        writer.close()
        
        path = Path(self._tmpdir) / "action_trace.jsonl"
        lines = path.read_text().strip().split("\n")
        self.assertEqual(len(lines), 5)
    
    def test_rotation(self):
        writer = ActionTraceWriter("test_sid", storage_dir=Path(self._tmpdir), max_file_size=100)
        # 写入足够数据触发归档
        for i in range(20):
            writer.append(make_trace_event(EventPhase.COMMIT, "tool", params={"data": "x" * 10}))
        writer.close()
        
        # 应该有归档文件
        files = list(Path(self._tmpdir).glob("*.arc"))
        self.assertGreater(len(files), 0)
    
    def test_load_recent(self):
        writer = ActionTraceWriter("test_sid", storage_dir=Path(self._tmpdir))
        for i in range(3):
            writer.append(make_trace_event(EventPhase.COMMIT, f"tool_{i}"))
        writer.close()
        
        entries = ActionTraceWriter.load_recent(n=2, storage_dir=Path(self._tmpdir))
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[-1]["action"], "tool_2")
    
    def test_load_recent_empty(self):
        entries = ActionTraceWriter.load_recent(n=10, storage_dir=Path(self._tmpdir))
        self.assertEqual(entries, [])


class TestActionTraceStorePersist(unittest.TestCase):
    """T3: ActionTraceStore集成测试。"""
    
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)
    
    def test_store_with_writer(self):
        writer = ActionTraceWriter("test_sid", storage_dir=Path(self._tmpdir))
        store = ActionTraceStore(session_id="test_sid", writer=writer)
        
        store.append(make_trace_event(EventPhase.REQUEST, "read_file"))
        store.append(make_trace_event(EventPhase.COMMIT, "read_file"))
        
        self.assertEqual(store.size, 2)
        
        # 验证已持久化
        writer.close()
        entries = ActionTraceWriter.load_recent(n=10, storage_dir=Path(self._tmpdir))
        self.assertEqual(len(entries), 2)
    
    def test_store_without_writer(self):
        store = ActionTraceStore()
        store.append(make_trace_event(EventPhase.REQUEST, "test"))
        self.assertEqual(store.size, 1)  # 纯内存模式仍工作


if __name__ == "__main__":
    unittest.main()
