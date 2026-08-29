"""
测试 AutoCausalWriter — 自动因果记忆写入
- 测试1: record() 写入后 CausalMemoryStore 能检索到
- 测试2: success=True 时 delta_magnitude=0
- 测试3: success=False 时 delta_magnitude > 0
- 测试4: write_log.jsonl 有审计记录
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openllm.memory.auto_causal_writer import AutoCausalWriter


# ═══════════════════════════════════════════════════
# 测试1: record() 写入后能读取
# ═══════════════════════════════════════════════════
def test_record_writes_file():
    """record() 写入后，文件应存在且可被 CausalMemoryStore 加载。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = AutoCausalWriter(store_dir=Path(tmpdir))
        result = writer.record(
            action="pip install torch",
            prediction="会成功",
            actual="安装成功",
            success=True,
            context="dependency install",
        )
        
        # 验证返回值
        assert result["memory_id"].startswith("auto-"), f"memory_id format wrong: {result['memory_id']}"
        assert result["actual_success"] is True
        assert result["delta_magnitude"] == 0.0
        
        # 验证文件存在
        files = list(Path(tmpdir).glob("auto-*.json"))
        assert len(files) == 1, f"Expected 1 file, got {len(files)}"
        
        # 验证文件内容可读
        with open(files[0], encoding="utf-8") as f:
            data = json.load(f)
        assert data["action_signature"] == "pip install torch"
        # DR-20260829-02: lesson升级为因果结构（"X 因采取了Y策略而成功"/"X 因Z而失败，教训：W"）
        assert "因" in data["lesson"] and data["lesson"].endswith(("成功", ")")) or "教训" in data["lesson"]
        
        print(f"✅ 测试1 PASS: record writes {files[0].name}, content verified")


# ═══════════════════════════════════════════════════
# 测试2: success=True 时 delta_magnitude=0
# ═══════════════════════════════════════════════════
def test_success_zero_delta():
    """success=True → delta_magnitude 应为 0。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = AutoCausalWriter(store_dir=Path(tmpdir))
        result = writer.record(
            action="test action",
            prediction="predicted",
            actual="actual",
            success=True,
        )
        assert result["delta_magnitude"] == 0.0, \
            f"success=True should have delta=0, got {result['delta_magnitude']}"
        print(f"✅ 测试2 PASS: success=True → delta=0")


# ═══════════════════════════════════════════════════
# 测试3: success=False 时 delta_magnitude > 0
# ═══════════════════════════════════════════════════
def test_failure_positive_delta():
    """success=False → delta_magnitude 应 > 0。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = AutoCausalWriter(store_dir=Path(tmpdir))
        result = writer.record(
            action="risky operation",
            prediction="will work",
            actual="crashed with error",
            success=False,
        )
        assert result["delta_magnitude"] > 0.0, \
            f"success=False should have delta>0, got {result['delta_magnitude']}"
        # delta_magnitude = min(1.0, len("will work -> crashed with error") / 100)
        expected_delta = min(1.0, len("will work -> crashed with error") / 100.0)
        assert abs(result["delta_magnitude"] - expected_delta) < 0.001, \
            f"delta_mismatch: expected {expected_delta}, got {result['delta_magnitude']}"
        print(f"✅ 测试3 PASS: success=False → delta={result['delta_magnitude']:.4f}")


# ═══════════════════════════════════════════════════
# 测试4: write_log.jsonl 有审计记录
# ═══════════════════════════════════════════════════
def test_write_log():
    """写入后 write_log.jsonl 应有对应的审计记录。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = AutoCausalWriter(store_dir=Path(tmpdir))
        writer.record(
            action="audit test",
            prediction="pred",
            actual="act",
            success=False,
        )
        
        log_path = Path(tmpdir) / "write_log.jsonl"
        assert log_path.exists(), f"write_log.jsonl should exist at {log_path}"
        
        with open(log_path, encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        
        assert len(lines) == 1, f"Expected 1 log entry, got {len(lines)}"
        assert lines[0]["action"] == "audit test"
        assert lines[0]["success"] is False
        assert "delta_magnitude" in lines[0]
        
        print(f"✅ 测试4 PASS: write_log.jsonl has 1 audit entry")


# ═══════════════════════════════════════════════════
# 运行
# ═══════════════════════════════════════════════════
if __name__ == "__main__":
    test_record_writes_file()
    test_success_zero_delta()
    test_failure_positive_delta()
    test_write_log()
    print(f"\n🎉 全部 4/4 测试通过！")
