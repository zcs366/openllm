"""
MemoryOS — 统一记忆层。

封装Δ胶囊/jika/RECALL/MEUSER 的读写入口。
ISA是唯一写入口，iai消费。

三层记忆：
  热记忆：当前Session的上下文
  温记忆：近期决策、因果教训
  冷记忆：历史归档、长期知识

温度公式：T = imp × e^(-λt) + heat
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import math


@dataclass
class MemoryEntry:
    """记忆条目。"""
    key: str
    value: dict
    importance: float = 0.5  # 重要性 0-1
    heat: float = 0.0        # 热度（被访问次数调制）
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: int = 86400         # 生存时间（秒），默认24小时
    
    @property
    def is_expired(self) -> bool:
        """是否过期。"""
        return time.time() - self.created_at > self.ttl
    
    def temperature(self, decay_lambda: float = 0.01, causal_delta: float = 0.0) -> float:
        """
        计算记忆温度。
        
        T = imp × e^(-λt) + heat + causal_delta × 3.0
        
        因果效应参与遗忘决策：causal_delta越大=教训越深=温度越高。
        
        Args:
            decay_lambda: 衰减系数
            causal_delta: 因果效应权重(0-1)，默认0不影响现有调用
        """
        t = time.time() - self.last_accessed
        base = self.importance * math.exp(-decay_lambda * t)
        # 因果效应参与遗忘决策：causal_delta × CAUSAL_BONUS
        return base + self.heat + causal_delta * 3.0


class MemoryOS:
    """
    统一记忆层。
    
    封装Δ胶囊/jika/RECALL/MEUSER 的读写入口。
    ISA是唯一写入口，iai消费。
    """
    
    def __init__(self, memory_dir: Optional[str] = None, async_write: bool = True):
        self.dir = Path(memory_dir or Path.home() / ".openllm" / "output" / "memory")
        self.dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, MemoryEntry] = {}
        self._async_write = async_write
        self._write_queue: list[MemoryEntry] = []  # 异步写入队列
        self._write_errors: list[tuple[str, Exception]] = []
        self._load_all()
    
    def _load_all(self):
        """从磁盘加载所有记忆到缓存。"""
        for fpath in self.dir.glob("*.json"):
            try:
                with open(fpath, encoding="utf-8") as f:
                    data = json.load(f)
                entry = MemoryEntry(**data)
                self._cache[entry.key] = entry
            except Exception:
                continue
    
    def _save(self, entry: MemoryEntry):
        """保存单条记忆到磁盘。支持异步队列。"""
        if self._async_write:
            self._write_queue.append(entry)
            self._flush_queue()
        else:
            self._write_to_disk(entry)

    def _write_to_disk(self, entry: MemoryEntry):
        """实际写入磁盘。"""
        fpath = self.dir / f"{entry.key}.json"
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump({
                "key": entry.key,
                "value": entry.value,
                "importance": entry.importance,
                "heat": entry.heat,
                "created_at": entry.created_at,
                "last_accessed": entry.last_accessed,
                "access_count": entry.access_count,
                "ttl": entry.ttl,
            }, f, ensure_ascii=False, indent=2)

    def _flush_queue(self):
        """处理写入队列。失败不阻塞，记录错误（最多100条）。"""
        while self._write_queue:
            entry = self._write_queue.pop(0)
            try:
                self._write_to_disk(entry)
            except Exception as e:
                if len(self._write_errors) < 100:
                    self._write_errors.append((entry.key, e))

    def flush(self):
        """手动刷新所有待写入记忆。退出时调用。"""
        self._async_write = False
        self._flush_queue()

    @property
    def pending_writes(self) -> int:
        """队列中待写入的数量。"""
        return len(self._write_queue)
    
    def remember(self, key: str, value: dict, importance: float = 0.5, 
                 ttl: int = 86400, heat: float = 0.0):
        """
        存入记忆。isa是唯一写入口。
        
        Args:
            key: 记忆键（唯一标识）
            value: 记忆值（JSON可序列化）
            importance: 重要性 0-1
            ttl: 生存时间（秒）
            heat: 热度
        """
        entry = MemoryEntry(
            key=key,
            value=value,
            importance=importance,
            heat=heat,
            ttl=ttl,
        )
        self._cache[key] = entry
        self._save(entry)
    
    def recall(self, key: str) -> Optional[dict]:
        """
        检索记忆。iai消费。
        
        Returns:
            记忆值，或None
        """
        entry = self._cache.get(key)
        if entry is None:
            return None
        
        # 检查过期
        if entry.is_expired:
            self.forget(key)
            return None
        
        # 更新访问信息
        entry.last_accessed = time.time()
        entry.access_count += 1
        entry.heat += 0.1  # 每次访问增加热度
        self._save(entry)
        
        return entry.value
    
    def forget(self, key: str):
        """
        主动遗忘。衰减到零=自然遗忘。
        """
        if key in self._cache:
            del self._cache[key]
        
        fpath = self.dir / f"{key}.json"
        if fpath.exists():
            fpath.unlink()
    
    def temperature(self, key: str, decay_lambda: float = 0.01) -> float:
        """
        查询记忆温度。T=imp×e^(-λt)+heat
        
        Returns:
            温度值
        """
        entry = self._cache.get(key)
        if entry is None:
            return 0.0
        return entry.temperature(decay_lambda)
    
    def list_keys(self) -> list[str]:
        """列出所有记忆键。"""
        return list(self._cache.keys())
    
    def count(self) -> int:
        """记忆总数。"""
        return len(self._cache)
    
    def cleanup_expired(self) -> int:
        """清理过期记忆。返回清理数量。"""
        expired = [k for k, v in self._cache.items() if v.is_expired]
        for k in expired:
            self.forget(k)
        return len(expired)


# ═══════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import tempfile
    
    print("=== MemoryOS 测试 ===\n")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        mem = MemoryOS(tmpdir)
        
        # 测试1: remember
        mem.remember("test_key", {"message": "hello"}, importance=0.8)
        assert mem.count() == 1
        print(f"✅ 测试1: remember count={mem.count()}")
        
        # 测试2: recall
        value = mem.recall("test_key")
        assert value == {"message": "hello"}
        print(f"✅ 测试2: recall value={value}")
        
        # 测试3: temperature
        temp = mem.temperature("test_key")
        assert temp > 0
        print(f"✅ 测试3: temperature={temp:.4f}")
        
        # 测试4: forget
        mem.forget("test_key")
        value = mem.recall("test_key")
        assert value is None
        print(f"✅ 测试4: forget后recall={value}")
        
        # 测试5: list_keys
        mem.remember("k1", {"a": 1})
        mem.remember("k2", {"b": 2})
        keys = mem.list_keys()
        assert len(keys) == 2
        print(f"✅ 测试5: list_keys={keys}")
        
        # 测试6: cleanup_expired
        mem.remember("expire_test", {"c": 3}, ttl=0)  # 立即过期
        cleaned = mem.cleanup_expired()
        assert cleaned >= 1
        print(f"✅ 测试6: cleanup_expired={cleaned}")
    
    print(f"\n全部 6/6 测试通过 ✅")
