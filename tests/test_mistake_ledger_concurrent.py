"""并发写入测试——验证文件锁在多进程场景下的安全性"""

import os
import sys
import tempfile
import multiprocessing
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from openllm.memory.mistake_ledger import MistakeLedger


def _writer(path, agent_id, count):
    """子进程写入函数"""
    ledger = MistakeLedger(path)
    for i in range(count):
        ledger.append(
            what=f"来自{agent_id}的错误{i}",
            why=f"测试并发写入",
            agent=agent_id,
        )


def test_concurrent_writes():
    """测试：多进程并发写入不丢数据"""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name

    try:
        num_processes = 4
        writes_per_process = 20
        expected_total = num_processes * writes_per_process

        # 启动多个进程并发写入
        processes = []
        for i in range(num_processes):
            p = multiprocessing.Process(
                target=_writer, args=(path, f"agent_{i}", writes_per_process)
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join(timeout=10)

        # 验证总行数
        ledger = MistakeLedger(path)
        total = ledger.count()
        assert total == expected_total, f"期望{expected_total}条，实际{total}条（丢失{expected_total - total}条）"

        # 验证所有agent的数据都存在
        for i in range(num_processes):
            results = ledger.query(agent=f"agent_{i}", limit=writes_per_process)
            assert len(results) == writes_per_process, \
                f"agent_{i}: 期望{writes_per_process}条，实际{len(results)}条"

        print(f"✅ test_concurrent_writes passed ({num_processes}进程×{writes_per_process}次={expected_total}条，零丢失)")
    finally:
        os.unlink(path)


if __name__ == "__main__":
    print("=== 并发写入测试 ===\n")
    test_concurrent_writes()
