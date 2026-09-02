"""
corpus_writer.py — 清洗后数据写入语料库
SQLite索引 + JSONL全文
"""
import sqlite3
import json
import os
import time
from typing import List, Optional
from ..base import ILMDocument


class CorpusWriter:
    """ILM语料库写入器"""
    
    def __init__(self, db_path: str = None, jsonl_path: str = None):
        """
        Args:
            db_path: SQLite数据库路径（默认~/projects/isa/ilm/corpus.db）
            jsonl_path: JSONL全文路径（默认~/projects/isa/ilm/corpus.jsonl）
        """
        if db_path is None:
            db_path = os.environ.get("ILM_CORPUS_DB", os.path.expanduser("~/projects/isa/ilm/corpus.db"))
        if jsonl_path is None:
            jsonl_path = os.environ.get("ILM_CORPUS_JSONL", os.path.expanduser("~/projects/isa/ilm/corpus.jsonl"))
        
        self.db_path = db_path
        self.jsonl_path = jsonl_path
        self._init_db()
    
    def _init_db(self):
        """初始化SQLite数据库"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                source_path TEXT,
                content TEXT NOT NULL,
                doc_type TEXT NOT NULL,
                domain TEXT DEFAULT '',
                quality_score REAL DEFAULT 0.0,
                timestamp REAL DEFAULT 0.0,
                content_hash TEXT DEFAULT '',
                source_ref TEXT DEFAULT '',
                metadata TEXT DEFAULT '{}',
                created_at REAL DEFAULT 0.0
            )
        """)
        # 向后兼容：给已有表加source_ref列
        try:
            conn.execute("ALTER TABLE documents ADD COLUMN source_ref TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass  # 列已存在
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_source ON documents(source)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_doc_type ON documents(doc_type)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_quality ON documents(quality_score DESC)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON documents(timestamp DESC)
        """)
        conn.commit()
        conn.close()
    
    def write_batch(self, documents: List[ILMDocument]) -> dict:
        """
        批量写入文档
        
        Returns:
            {"written": int, "skipped": int, "errors": int}
        """
        stats = {"written": 0, "skipped": 0, "errors": 0}
        
        # SQLite批量写入
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for doc in documents:
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO documents 
                    (doc_id, source, source_path, content, doc_type, domain,
                     quality_score, timestamp, content_hash, source_ref, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    doc.doc_id,
                    doc.source,
                    doc.source_path,
                    doc.content,
                    doc.doc_type,
                    doc.domain,
                    doc.quality_score,
                    doc.timestamp,
                    doc.content_hash,
                    doc.source_ref,
                    json.dumps(doc.metadata, ensure_ascii=False),
                    time.time(),
                ))
                stats["written"] += 1
            except Exception as e:
                stats["errors"] += 1
                print(f"[corpus_writer] Error writing {doc.doc_id}: {e}")
        
        conn.commit()
        conn.close()
        
        # JSONL追加写入
        try:
            with open(self.jsonl_path, 'a', encoding='utf-8') as f:
                for doc in documents:
                    f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"[corpus_writer] JSONL write error: {e}")
            stats["errors"] += 1
        
        print(f"[corpus_writer] Written: {stats['written']}, Skipped: {stats['skipped']}, Errors: {stats['errors']}")
        return stats
    
    def get_stats(self) -> dict:
        """获取语料库统计"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        cursor.execute("SELECT COUNT(*) FROM documents")
        stats["total"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT doc_type, COUNT(*) FROM documents GROUP BY doc_type")
        stats["by_type"] = dict(cursor.fetchall())
        
        cursor.execute("SELECT source, COUNT(*) FROM documents GROUP BY source")
        stats["by_source"] = dict(cursor.fetchall())
        
        cursor.execute("SELECT AVG(quality_score) FROM documents")
        stats["avg_quality"] = cursor.fetchone()[0] or 0.0
        
        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM documents")
        row = cursor.fetchone()
        stats["time_range"] = {"min": row[0], "max": row[1]}
        
        conn.close()
        return stats
    
    def search(self, query: str, limit: int = 10) -> List[dict]:
        """简单文本搜索（FTS5未建索引时用LIKE）"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT doc_id, source, doc_type, content, quality_score
            FROM documents
            WHERE content LIKE ?
            ORDER BY quality_score DESC
            LIMIT ?
        """, (f"%{query}%", limit))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results


if __name__ == "__main__":
    writer = CorpusWriter()
    stats = writer.get_stats()
    print("=== Corpus Stats ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
