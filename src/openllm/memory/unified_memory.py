"""
统一记忆系统 — 将Δ胶囊与MemoryOS统一为三层记忆架构

三层记忆：
1. 热记忆（Hot Memory）：当前Session的上下文，高速读写
2. 温记忆（Warm Memory）：近期决策、因果教训，中等速度
3. 冷记忆（Cold Memory）：历史归档、长期知识，低速但大容量

集成：
- Δ胶囊：语义向量记忆，跨会话恢复身份
- MemoryOS：文本胶囊记忆，可审计可读
- 温度衰减函数：T = imp × e^(-λt) + heat

用法：
    from openllm.memory.unified_memory import UnifiedMemory
    
    memory = UnifiedMemory()
    
    # 存储记忆
    memory.store(
        key="决策-001",
        value={"decision": "使用Python 3.12", "reason": "性能更好"},
        importance=0.8,
        layer="warm",
    )
    
    # 检索记忆
    results = memory.retrieve("Python版本选择", layer="warm", top_n=5)
    
    # 跨会话恢复
    memory.checkpoint()
    memory.restore()
"""

import json
import math
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List
import hashlib

logger = logging.getLogger("openllm.unified_memory")

# ── 路径配置 ──
MEMORY_HOME = Path.home() / ".openllm" / "memory"
HOT_MEMORY_DIR = MEMORY_HOME / "hot"
WARM_MEMORY_DIR = MEMORY_HOME / "warm"
COLD_MEMORY_DIR = MEMORY_HOME / "cold"
CAPSULE_DIR = MEMORY_HOME / "capsules"


# ── 数据结构 ──

@dataclass
class MemoryEntry:
    """记忆条目（四问元数据版）"""
    key: str
    value: Dict[str, Any]
    importance: float = 0.5  # 重要性 (0.0-1.0)
    heat: float = 0.0        # 热度（被访问次数调制）
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: int = 86400         # 生存时间（秒），默认24小时
    layer: str = "warm"      # 记忆层 (hot/warm/cold)
    tags: List[str] = field(default_factory=list)
    
    # 四问元数据
    why: str = ""              # 为什么这条记忆重要
    when_forget: str = ""      # 什么时候应该遗忘
    how_correct: str = ""      # 记错了怎么修正
    
    @property
    def is_expired(self) -> bool:
        """是否过期"""
        return time.time() - self.created_at > self.ttl
    
    def temperature(self, decay_lambda: Optional[float] = None) -> float:
        """
        计算记忆温度（增强版）。
        
        T = imp × e^(-λ_eff × t) + heat_adj
        
        λ_eff = λ_base × type_factor
        heat_adj = heat × (1 + log(1 + access_count))
        
        Args:
            decay_lambda: 衰减系数（None=自动根据标签选择）
        """
        t = time.time() - self.last_accessed
        
        # 动态λ：根据标签选择基础衰减系数
        if decay_lambda is None:
            decay_lambda = self._get_lambda_by_tags()
        
        # 酒神双杯：访问次数调制heat
        heat_adj = self.heat * (1 + math.log(1 + self.access_count))
        
        base = self.importance * math.exp(-decay_lambda * t)
        return base + heat_adj
    
    def _get_lambda_by_tags(self) -> float:
        """根据标签选择λ：insight慢衰减，noise快衰减"""
        tag_set = set(t.lower() for t in self.tags)
        if "insight" in tag_set or "决策" in tag_set:
            return 0.001  # 洞察/决策：几乎永存
        elif "fact" in tag_set or "事实" in tag_set:
            return 0.01   # 事实：正常衰减
        elif "noise" in tag_set or "噪声" in tag_set:
            return 0.05   # 噪声：快速衰减
        elif "habit" in tag_set or "习惯" in tag_set:
            return 0.005  # 习惯：慢衰减
        else:
            return 0.01   # 默认：正常衰减
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "key": self.key,
            "value": self.value,
            "importance": self.importance,
            "heat": self.heat,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "ttl": self.ttl,
            "layer": self.layer,
            "tags": self.tags,
            "why": self.why,
            "when_forget": self.when_forget,
            "how_correct": self.how_correct,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """从字典创建"""
        return cls(**data)


# ── 统一记忆系统 ──

class UnifiedMemory:
    """
    统一记忆系统。
    
    将Δ胶囊与MemoryOS统一为三层记忆架构。
    """
    
    def __init__(self, memory_dir: Optional[Path] = None):
        """
        初始化统一记忆系统。
        
        Args:
            memory_dir: 记忆目录（默认~/.openllm/memory）
        """
        self.memory_dir = memory_dir or MEMORY_HOME
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建三层记忆目录
        HOT_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        WARM_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        COLD_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        CAPSULE_DIR.mkdir(parents=True, exist_ok=True)
        
        # 内存缓存
        self._hot_cache: Dict[str, MemoryEntry] = {}
        self._warm_cache: Dict[str, MemoryEntry] = {}
        self._cold_cache: Dict[str, MemoryEntry] = {}
        
        # 加载现有记忆
        self._load_all_memories()
        
        logger.info("✅ 统一记忆系统初始化完成")
    
    def _load_all_memories(self):
        """加载所有记忆到缓存"""
        self._load_layer("hot", HOT_MEMORY_DIR, self._hot_cache)
        self._load_layer("warm", WARM_MEMORY_DIR, self._warm_cache)
        self._load_layer("cold", COLD_MEMORY_DIR, self._cold_cache)
    
    def _load_layer(self, layer: str, directory: Path, cache: Dict[str, MemoryEntry]):
        """加载指定层的记忆"""
        for file_path in directory.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                entry = MemoryEntry.from_dict(data)
                cache[entry.key] = entry
            except Exception as e:
                logger.warning(f"加载记忆失败 {file_path}: {e}")
    
    def store(
        self,
        key: str,
        value: Dict[str, Any],
        importance: float = 0.5,
        layer: str = "warm",
        tags: List[str] = None,
        ttl: int = 86400,
        why: str = "",
        when_forget: str = "",
        how_correct: str = "",
    ) -> MemoryEntry:
        """
        存储记忆。
        
        Args:
            key: 记忆键
            value: 记忆值
            importance: 重要性 (0.0-1.0)
            layer: 记忆层 (hot/warm/cold)
            tags: 标签列表
            ttl: 生存时间（秒）
            why: 为什么这条记忆重要
            when_forget: 什么时候应该遗忘
            how_correct: 记错了怎么修正
            
        Returns:
            MemoryEntry: 记忆条目
        """
        entry = MemoryEntry(
            key=key,
            value=value,
            importance=importance,
            layer=layer,
            tags=tags or [],
            ttl=ttl,
            why=why,
            when_forget=when_forget,
            how_correct=how_correct,
        )
        
        # 根据层选择缓存和目录
        if layer == "hot":
            cache = self._hot_cache
            directory = HOT_MEMORY_DIR
        elif layer == "cold":
            cache = self._cold_cache
            directory = COLD_MEMORY_DIR
        else:  # warm
            cache = self._warm_cache
            directory = WARM_MEMORY_DIR
        
        # 更新缓存
        cache[key] = entry
        
        # 持久化到磁盘
        self._save_to_disk(entry, directory)
        
        logger.debug(f"存储记忆: {key} ({layer})")
        return entry
    
    def retrieve(
        self,
        query: str,
        layer: str = None,
        top_n: int = 5,
        min_importance: float = 0.0,
    ) -> List[MemoryEntry]:
        """
        检索记忆。
        
        Args:
            query: 查询字符串
            layer: 指定层（None表示搜索所有层）
            top_n: 返回数量
            min_importance: 最小重要性
            
        Returns:
            List[MemoryEntry]: 记忆条目列表
        """
        results = []
        
        # 搜索指定层或所有层
        caches = []
        if layer is None or layer == "hot":
            caches.append(self._hot_cache)
        if layer is None or layer == "warm":
            caches.append(self._warm_cache)
        if layer is None or layer == "cold":
            caches.append(self._cold_cache)
        
        for cache in caches:
            for entry in cache.values():
                # 检查重要性
                if entry.importance < min_importance:
                    continue
                
                # 检查是否过期
                if entry.is_expired:
                    continue
                
                # 简单的关键词匹配（后续可升级为语义匹配）
                if self._matches_query(entry, query):
                    results.append(entry)
        
        # 按温度排序
        results.sort(key=lambda e: e.temperature(), reverse=True)
        
        return results[:top_n]
    
    def _matches_query(self, entry: MemoryEntry, query: str) -> bool:
        """检查记忆是否匹配查询（支持关键词拆分+字符级匹配）"""
        query_lower = query.lower()
        
        # 移除常见停用词和标点
        stop_words = {"的", "是什么", "怎么", "如何", "？", "？", "是", "在", "有", "和", "与", "或", "了", "吗", "呢"}
        keywords = [w for w in query_lower.split() if w not in stop_words and len(w) > 1]
        
        # 如果没有有效关键词，用原始查询
        if not keywords:
            keywords = [query_lower]
        
        # 检查键（任一关键词匹配即可）
        key_lower = entry.key.lower()
        for kw in keywords:
            if kw in key_lower:
                return True
            # 字符级匹配：检查关键词中的字符是否都在键中
            if len(kw) > 2:
                chars_in_key = sum(1 for c in kw if c in key_lower)
                if chars_in_key >= len(kw) * 0.5:  # 50%字符匹配
                    return True
        
        # 检查值
        value_str = json.dumps(entry.value, ensure_ascii=False).lower()
        for kw in keywords:
            if kw in value_str:
                return True
        
        # 检查标签
        for tag in entry.tags:
            tag_lower = tag.lower()
            for kw in keywords:
                if kw in tag_lower:
                    return True
        
        return False
    
    def get(self, key: str, layer: str = None) -> Optional[MemoryEntry]:
        """
        获取指定键的记忆。
        
        Args:
            key: 记忆键
            layer: 指定层（None表示搜索所有层）
            
        Returns:
            Optional[MemoryEntry]: 记忆条目
        """
        # 搜索指定层或所有层
        caches = []
        if layer is None or layer == "hot":
            caches.append(self._hot_cache)
        if layer is None or layer == "warm":
            caches.append(self._warm_cache)
        if layer is None or layer == "cold":
            caches.append(self._cold_cache)
        
        for cache in caches:
            if key in cache:
                entry = cache[key]
                # 更新访问信息
                entry.last_accessed = time.time()
                entry.access_count += 1
                return entry
        
        return None
    
    def delete(self, key: str, layer: str = None) -> bool:
        """
        删除指定键的记忆。
        
        Args:
            key: 记忆键
            layer: 指定层（None表示删除所有层）
            
        Returns:
            bool: 是否删除成功
        """
        deleted = False
        
        # 删除指定层或所有层
        caches = []
        if layer is None or layer == "hot":
            caches.append(("hot", self._hot_cache, HOT_MEMORY_DIR))
        if layer is None or layer == "warm":
            caches.append(("warm", self._warm_cache, WARM_MEMORY_DIR))
        if layer is None or layer == "cold":
            caches.append(("cold", self._cold_cache, COLD_MEMORY_DIR))
        
        for layer_name, cache, directory in caches:
            if key in cache:
                del cache[key]
                # 删除磁盘文件
                file_path = directory / f"{self._safe_filename(key)}.json"
                if file_path.exists():
                    file_path.unlink()
                deleted = True
                logger.debug(f"删除记忆: {key} ({layer_name})")
        
        return deleted
    
    def _safe_filename(self, key: str) -> str:
        """生成安全的文件名"""
        # 使用MD5哈希避免文件名问题
        return hashlib.md5(key.encode("utf-8")).hexdigest()
    
    def _save_to_disk(self, entry: MemoryEntry, directory: Path):
        """保存记忆到磁盘"""
        file_path = directory / f"{self._safe_filename(entry.key)}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(entry.to_dict(), f, ensure_ascii=False, indent=2)
    
    def checkpoint(self):
        """
        创建检查点。
        
        将当前记忆状态保存到Δ胶囊。
        """
        try:
            # 收集所有记忆
            all_memories = []
            for cache in [self._hot_cache, self._warm_cache, self._cold_cache]:
                all_memories.extend(cache.values())
            
            # 创建检查点数据
            checkpoint_data = {
                "timestamp": time.time(),
                "memory_count": len(all_memories),
                "memories": [m.to_dict() for m in all_memories],
            }
            
            # 保存到检查点文件
            checkpoint_file = CAPSULE_DIR / f"checkpoint_{int(time.time())}.json"
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"创建检查点: {checkpoint_file}")
            
        except Exception as e:
            logger.error(f"创建检查点失败: {e}")
    
    def restore(self, checkpoint_file: Path = None):
        """
        从检查点恢复。
        
        Args:
            checkpoint_file: 检查点文件（默认使用最新的）
        """
        try:
            # 如果没有指定检查点文件，使用最新的
            if checkpoint_file is None:
                checkpoint_files = sorted(CAPSULE_DIR.glob("checkpoint_*.json"), reverse=True)
                if not checkpoint_files:
                    logger.warning("没有找到检查点文件")
                    return
                checkpoint_file = checkpoint_files[0]
            
            # 读取检查点数据
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
            
            # 恢复记忆
            memories_data = checkpoint_data.get("memories", [])
            for memory_data in memories_data:
                entry = MemoryEntry.from_dict(memory_data)
                self.store(
                    key=entry.key,
                    value=entry.value,
                    importance=entry.importance,
                    layer=entry.layer,
                    tags=entry.tags,
                    ttl=entry.ttl,
                )
            
            logger.info(f"从检查点恢复: {checkpoint_file}")
            
        except Exception as e:
            logger.error(f"从检查点恢复失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取记忆统计信息。
        
        Returns:
            Dict: 统计信息
        """
        return {
            "hot_memory_count": len(self._hot_cache),
            "warm_memory_count": len(self._warm_cache),
            "cold_memory_count": len(self._cold_cache),
            "total_memory_count": (
                len(self._hot_cache) +
                len(self._warm_cache) +
                len(self._cold_cache)
            ),
            "memory_directory": str(self.memory_dir),
        }
    
    def cleanup_expired(self):
        """清理过期的记忆"""
        expired_keys = []
        
        for cache in [self._hot_cache, self._warm_cache, self._cold_cache]:
            for key, entry in cache.items():
                if entry.is_expired:
                    expired_keys.append(key)
        
        for key in expired_keys:
            self.delete(key)
        
        logger.info(f"清理过期记忆: {len(expired_keys)} 条")


# ── 便捷函数 ──

def create_unified_memory(memory_dir: Optional[Path] = None) -> UnifiedMemory:
    """
    创建UnifiedMemory实例。
    
    Args:
        memory_dir: 记忆目录
        
    Returns:
        UnifiedMemory: 实例
    """
    return UnifiedMemory(memory_dir=memory_dir)


def quick_store(
    key: str,
    value: Dict[str, Any],
    importance: float = 0.5,
    layer: str = "warm",
) -> MemoryEntry:
    """
    便捷函数：快速存储记忆。
    
    Args:
        key: 记忆键
        value: 记忆值
        importance: 重要性
        layer: 记忆层
        
    Returns:
        MemoryEntry: 记忆条目
    """
    memory = create_unified_memory()
    return memory.store(key, value, importance, layer)


def quick_retrieve(
    query: str,
    layer: str = None,
    top_n: int = 5,
) -> List[MemoryEntry]:
    """
    便捷函数：快速检索记忆。
    
    Args:
        query: 查询字符串
        layer: 指定层
        top_n: 返回数量
        
    Returns:
        List[MemoryEntry]: 记忆条目列表
    """
    memory = create_unified_memory()
    return memory.retrieve(query, layer, top_n)