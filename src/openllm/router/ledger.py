"""
JSONL账本 - append-only记录每次路由执行
"""
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


class Ledger:
    """JSONL append-only账本"""
    
    def __init__(self, ledger_path: Optional[Path] = None):
        if ledger_path is None:
            ledger_path = Path.home() / '.openllm' / 'router' / 'ledger.jsonl'
        
        self.ledger_path = ledger_path
        self._ensure_directory()
    
    def _ensure_directory(self):
        """确保账本目录存在"""
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    
    def append(self, record: Dict[str, Any]):
        """
        追加一条记录
        
        record必须包含：
        - timestamp: 时间戳
        - provider: 执行器名
        - model: 模型名
        - prompt_tokens: 提示token数
        - completion_tokens: 完成token数
        - latency_ms: 延迟毫秒
        - budget: 预算档
        - result: ok|fallback:xx|fail
        - query_preview: query前80字符
        """
        # 确保必要字段
        required_fields = ['timestamp', 'provider', 'model', 'result']
        for field in required_fields:
            if field not in record:
                raise ValueError(f"缺少必要字段: {field}")
        
        # 安全处理：确保不记录敏感内容
        safe_record = self._sanitize_record(record)
        
        # 写入JSONL
        with open(self.ledger_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(safe_record, ensure_ascii=False) + '\n')
    
    def _sanitize_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """清理记录，确保不包含敏感内容"""
        safe = {}
        
        for key, value in record.items():
            # 不记录key相关内容
            if 'key' in key.lower() and 'token' not in key.lower():
                continue
            
            # query_preview最多80字符
            if key == 'query_preview' and isinstance(value, str):
                value = value[:80]
            
            safe[key] = value
        
        return safe
    
    def read_records(self, last_n: Optional[int] = None) -> list:
        """
        读取账本记录
        
        Args:
            last_n: 只读取最近N条，None则读取全部
        
        Returns:
            记录列表（按时间正序）
        """
        if not self.ledger_path.exists():
            return []
        
        records = []
        with open(self.ledger_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        records.append(record)
                    except json.JSONDecodeError:
                        continue
        
        if last_n is not None:
            records = records[-last_n:]
        
        return records
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取账本统计

        Returns:
            包含总次数、各执行器统计、成功率、分场景统计（v0.5）
        """
        records = self.read_records()

        if not records:
            return {
                'total': 0,
                'providers': {},
                'scenarios': {},
                'success_rate': 0.0
            }

        stats = {
            'total': len(records),
            'providers': {},
            'scenarios': {},  # v0.5: 按scenario分组统计
            'success_count': 0,
            'fail_count': 0,
            'fallback_count': 0,
            'total_prompt_tokens': 0,
            'total_completion_tokens': 0,
            'avg_latency_ms': 0
        }

        total_latency = 0

        for record in records:
            provider = record.get('provider', 'unknown')
            result = record.get('result', 'unknown')
            scenario = record.get('scenario', 'interactive')  # v0旧记录无scenario字段，归入interactive

            # v0.5: 分场景统计（量/延迟/成功）
            if scenario not in stats['scenarios']:
                stats['scenarios'][scenario] = {'count': 0, 'success': 0, 'total_latency': 0}
            s = stats['scenarios'][scenario]
            s['count'] += 1

            # 统计各执行器
            if provider not in stats['providers']:
                stats['providers'][provider] = {'count': 0, 'success': 0}
            stats['providers'][provider]['count'] += 1

            # 统计结果
            if result == 'ok':
                stats['success_count'] += 1
                stats['providers'][provider]['success'] += 1
                s['success'] += 1
            elif result.startswith('fallback:'):
                stats['fallback_count'] += 1
            else:
                stats['fail_count'] += 1

            # 统计tokens
            stats['total_prompt_tokens'] += record.get('prompt_tokens', 0)
            stats['total_completion_tokens'] += record.get('completion_tokens', 0)

            # 统计延迟
            latency = record.get('latency_ms', 0)
            total_latency += latency
            s['total_latency'] += latency

        # 计算平均值
        if stats['total'] > 0:
            stats['success_rate'] = stats['success_count'] / stats['total']
            stats['avg_latency_ms'] = total_latency // stats['total']
        for s in stats['scenarios'].values():
            s['avg_latency_ms'] = s['total_latency'] // s['count'] if s['count'] > 0 else 0

        return stats
    
    def clear(self):
        """清空账本（谨慎使用）"""
        if self.ledger_path.exists():
            self.ledger_path.unlink()