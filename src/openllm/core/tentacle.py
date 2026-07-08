"""
iai 触手脑 — 神经末端。八柱国之前两个脑。

触手脑 = 具有局部自治能力的神经末端
每个触手脑：
  - 独立感知（扫描/索引/搜索）
  - 独立存储（快照/索引）
  - 轻量自治（不依赖中心调度）

通信协议（雅典娜神启）：
  - L0自治级：触手脑自己决策，不上报
  - L1通道级：上报给大脑，等待确认
  - 上报/确认/自治权限 三要素缺一不可

FileWatcherBrain: 文件脑 — 监控目录变化·L0自治·变化→自动索引
IndexBrain: 索引脑 — 全文搜索。记忆：RECALL/session/docs
"""

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from collections import defaultdict


# ═══════════════════════════════════════════════════════
# 触手脑通信协议
# ═══════════════════════════════════════════════════════

@dataclass
class TentacleReport:
    """触手脑→大脑的上报"""
    brain_name: str        # 哪个触手脑
    discovery_type: str    # new_file / modified / pattern_found / index_built
    importance: float      # 0-1 重要性
    data: dict             # 上报数据
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False  # 是否已确认


class TentacleBrain:
    """
    触手脑基类 — 所有触手脑的共同行为
    
    通信协议：
    - L0自治级：auto_decide() — 不需要上报大脑
    - L1通道级：report() — 上报给大脑，等待确认
    """
    
    def __init__(self, name: str):
        self.name = name
        self.pending_reports: list[TentacleReport] = []
        self.acknowledged_reports: list[TentacleReport] = []
    
    def report(self, discovery_type: str, importance: float, data: dict) -> TentacleReport:
        """
        上报给大脑·L1通道级
        
        Args:
            discovery_type: 发现类型（new_file/modified/pattern_found等）
            importance: 重要性 0-1
            data: 上报数据
        
        Returns:
            上报对象
        """
        report = TentacleReport(
            brain_name=self.name,
            discovery_type=discovery_type,
            importance=importance,
            data=data,
        )
        self.pending_reports.append(report)
        return report
    
    def auto_decide(self, data: dict) -> bool:
        """
        L0自治级决策·不需要上报大脑
        
        子类覆写此方法实现自治逻辑。
        返回True表示自治处理完毕，不需要上报。
        """
        return True
    
    def ack(self, report: TentacleReport):
        """确认上报已处理"""
        report.acknowledged = True
        if report in self.pending_reports:
            self.pending_reports.remove(report)
        self.acknowledged_reports.append(report)
    
    def get_pending(self) -> list[TentacleReport]:
        """获取待处理上报"""
        return self.pending_reports
    
    def summary(self) -> dict:
        """状态摘要"""
        return {
            "name": self.name,
            "pending": len(self.pending_reports),
            "acknowledged": len(self.acknowledged_reports),
        }


# ═══════════════════════════════════════════════════════
# 文件脑
# ═══════════════════════════════════════════════════════

class FileWatcherBrain(TentacleBrain):
    """
    文件脑 — 监控目录变化·L0自治·变化→自动索引
    
    功能：
    - 扫描目录变化（新增/修改/删除）
    - 维护文件快照（mtime对比）
    - 变化→自动触发索引更新
    - L0自治：小变化自己处理，大变化上报
    """
    
    def __init__(self, watch_dir: str = "~/.openllm/output"):
        super().__init__("file_watcher")
        self.dir = Path(watch_dir).expanduser()
        self.snapshot: dict[str, float] = {}  # {path: mtime}
        self._indexed_files: set[str] = set()
        
        # 首次扫描建立基线
        if self.dir.exists():
            self.scan()
    
    def scan(self) -> list[dict]:
        """
        扫描变化·返回变更文件列表
        
        Returns:
            变更描述列表 [{"type": "new_file", "name": "...", "path": "..."}]
        """
        changes = []
        
        if not self.dir.exists():
            return changes
        
        current_files: set[str] = set()
        
        for f in self.dir.rglob("*"):
            if f.is_file():
                path_str = str(f)
                current_files.add(path_str)
                
                try:
                    mtime = f.stat().st_mtime
                except OSError:
                    continue
                
                if path_str not in self.snapshot:
                    # 新增文件
                    change = {"type": "new_file", "name": f.name, "path": path_str}
                    changes.append(change)
                    self._indexed_files.add(path_str)
                    # L0自治：小文件自己处理，大变化上报
                    if self.auto_decide(change):
                        self.report("new_file", 0.5, change)
                elif self.snapshot[path_str] != mtime:
                    # 修改文件
                    change = {"type": "modified", "name": f.name, "path": path_str}
                    changes.append(change)
                    self._indexed_files.add(path_str)
                    if self.auto_decide(change):
                        self.report("modified", 0.3, change)
                
                self.snapshot[path_str] = mtime
        
        # 检测删除
        deleted = set(self.snapshot.keys()) - current_files
        for path_str in deleted:
            change = {"type": "deleted", "name": Path(path_str).name, "path": path_str}
            changes.append(change)
            self._indexed_files.discard(path_str)
            self.report("deleted", 0.4, change)
        
        return changes
    
    def auto_decide(self, data: dict) -> bool:
        """L0自治：小变化自己处理"""
        # 默认：所有变化都上报（保守策略）
        return True
    
    def has_changes(self) -> bool:
        """是否有未处理的变化。"""
        changes = self.scan()
        return len(changes) > 0
    
    def summary(self) -> dict:
        """文件脑状态摘要。"""
        base = super().summary()
        base.update({
            "watch_dir": str(self.dir),
            "snapshot_count": len(self.snapshot),
            "indexed_count": len(self._indexed_files),
        })
        return base


# ═══════════════════════════════════════════════════════
# 索引脑
# ═══════════════════════════════════════════════════════

class IndexBrain(TentacleBrain):
    """
    索引脑 — 全文搜索。记忆：RECALL/session/docs
    
    功能：
    - 建立文件内容索引（倒排索引）
    - 支持关键词搜索
    - 支持正则搜索
    """
    
    def __init__(self, index_dir: str = "~/.openllm/output"):
        super().__init__("index")
        self.dir = Path(index_dir).expanduser()
        # 倒排索引：{keyword: [filepath]}
        self.index: dict[str, list[str]] = defaultdict(list)
        # 文件内容缓存：{filepath: content}
        self._content_cache: dict[str, str] = {}
    
    def build(self, force: bool = False) -> int:
        """
        构建索引。
        
        Args:
            force: 是否强制重建（忽略缓存）
        
        Returns:
            索引文件数
        """
        if not self.dir.exists():
            return 0
        
        if not force and self.index:
            return len(self._content_cache)
        
        # 清空旧索引
        self.index.clear()
        self._content_cache.clear()
        
        # 扫描并索引
        indexed_count = 0
        for f in self.dir.rglob("*.md"):  # 索引markdown文件
            if f.is_file():
                try:
                    content = f.read_text(encoding="utf-8", errors="replace")
                    self._content_cache[str(f)] = content
                    self._index_file(str(f), content)
                    indexed_count += 1
                except Exception:
                    continue
        
        # 也索引jsonl文件
        for f in self.dir.rglob("*.jsonl"):
            if f.is_file():
                try:
                    content = f.read_text(encoding="utf-8", errors="replace")
                    self._content_cache[str(f)] = content
                    self._index_file(str(f), content)
                    indexed_count += 1
                except Exception:
                    continue
        
        # 上报索引构建完成
        if indexed_count > 0:
            self.report("index_built", 0.6, {"files_indexed": indexed_count})
        
        return indexed_count
    
    def _index_file(self, filepath: str, content: str):
        """索引单个文件内容。"""
        # 简单分词：按空格和标点分割
        words = re.findall(r'\w+', content.lower())
        
        # 去重并建立倒排索引
        seen_words = set()
        for word in words:
            if word not in seen_words and len(word) > 1:  # 跳过单字符
                self.index[word].append(filepath)
                seen_words.add(word)
    
    def search(self, query: str, limit: int = 10) -> list[dict]:
        """
        搜索索引。
        
        Args:
            query: 搜索关键词
            limit: 返回结果数
        
        Returns:
            搜索结果列表 [{"filepath": str, "snippet": str}]
        """
        query_lower = query.lower()
        query_words = re.findall(r'\w+', query_lower)
        
        # 统计每个文件的匹配分数
        scores: dict[str, int] = defaultdict(int)
        
        for word in query_words:
            if word in self.index:
                for filepath in self.index[word]:
                    scores[filepath] += 1
        
        # 按分数排序
        sorted_files = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # 生成结果
        results = []
        for filepath, score in sorted_files[:limit]:
            # 提取片段
            content = self._content_cache.get(filepath, "")
            snippet = self._extract_snippet(content, query_words)
            
            results.append({
                "filepath": filepath,
                "score": score,
                "snippet": snippet,
            })
        
        return results
    
    def _extract_snippet(self, content: str, query_words: list[str], 
                         context_len: int = 100) -> str:
        """提取搜索片段。"""
        if not content:
            return ""
        
        content_lower = content.lower()
        
        # 找到第一个匹配位置
        for word in query_words:
            pos = content_lower.find(word)
            if pos != -1:
                start = max(0, pos - context_len)
                end = min(len(content), pos + context_len)
                snippet = content[start:end]
                # 添加省略号
                if start > 0:
                    snippet = "..." + snippet
                if end < len(content):
                    snippet = snippet + "..."
                return snippet
        
        return content[:200]
    
    def summary(self) -> dict:
        """索引脑状态摘要。"""
        base = super().summary()
        base.update({
            "index_dir": str(self.dir),
            "terms_count": len(self.index),
            "files_cached": len(self._content_cache),
        })
        return base


# ═══════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import tempfile
    
    print("=== 触手脑通信协议测试 ===\n")
    
    # 测试1: TentacleBrain基类
    brain = TentacleBrain("test")
    report = brain.report("test_event", 0.5, {"key": "value"})
    assert report.brain_name == "test"
    assert len(brain.pending_reports) == 1
    print(f"✅ 测试1: TentacleBrain上报")
    
    # 测试2: ack确认
    brain.ack(report)
    assert report.acknowledged == True
    assert len(brain.pending_reports) == 0
    assert len(brain.acknowledged_reports) == 1
    print(f"✅ 测试2: ack确认")
    
    # 测试3: FileWatcherBrain上报
    with tempfile.TemporaryDirectory() as tmpdir:
        watcher = FileWatcherBrain(tmpdir)
        test_file = Path(tmpdir) / "test.md"
        test_file.write_text("hello")
        changes = watcher.scan()
        assert len(changes) >= 1
        assert len(watcher.pending_reports) >= 1
        print(f"✅ 测试3: FileWatcherBrain上报 changes={len(changes)}")
    
    # 测试4: IndexBrain上报
    with tempfile.TemporaryDirectory() as tmpdir:
        indexer = IndexBrain(tmpdir)
        test_file = Path(tmpdir) / "readme.md"
        test_file.write_text("# Test\nHello world")
        count = indexer.build()
        assert count == 1
        assert len(indexer.pending_reports) >= 1
        print(f"✅ 测试4: IndexBrain上报 files={count}")
    
    print(f"\n全部 4/4 测试通过 ✅")
