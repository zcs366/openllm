"""
执行器注册表 - 从providers.json加载，解析key来源
"""
import json
import os
import sqlite3
from pathlib import Path


def _env_with_hermes_fallback(name: str) -> str:
    """查环境变量，查不到回退~/.hermes/.env（逐行解析，不打印值）"""
    v = os.environ.get(name, '')
    if v:
        return v
    env_file = Path.home() / '.hermes' / '.env'
    try:
        for line in env_file.read_text(encoding='utf-8', errors='ignore').splitlines():
            if line.startswith(name + '='):
                return line.split('=', 1)[1].strip()
    except OSError:
        pass
    return ''
from typing import Any, Dict, List, Optional
import sys

# 添加openllm src到path以访问keyvault
sys.path.insert(0, '/mnt/i/openllm/src')


class Provider:
    """单个执行器配置"""
    
    def __init__(self, config: Dict[str, Any]):
        self.name: str = config['name']
        self.endpoint: str = config['endpoint']
        self.protocol: str = config.get('protocol', 'openai')
        self.model: str = config.get('model', '')
        self.models: List[str] = config.get('models', [])
        self.key_source: str = config.get('key_source', '')
        self.cost_tier: int = config.get('cost_tier', 0)
        self.quality_tier: int = config.get('quality_tier', 1)
        self.capabilities: List[str] = config.get('capabilities', [])
        self.notes: str = config.get('notes', '')
        self.sqlite_query: str = config.get('sqlite_query', '')
        self.sqlite_db: str = config.get('sqlite_db', '')
        self._key: Optional[str] = None
    
    @property
    def available_models(self) -> List[str]:
        """返回所有可用模型"""
        if self.models:
            return self.models
        return [self.model] if self.model else []
    
    def get_key(self) -> Optional[str]:
        """获取API key，不打印明文"""
        if self._key is not None:
            return self._key
        
        if not self.key_source:
            self._key = ''
            return self._key
        
        try:
            if self.key_source.startswith('env:'):
                # 环境变量（os.environ查不到时回退~/.hermes/.env）
                env_name = self.key_source[4:]
                self._key = _env_with_hermes_fallback(env_name)
            
            elif self.key_source.startswith('file:'):
                # 文件读取
                file_path = Path(self.key_source[5:]).expanduser()
                if file_path.exists():
                    self._key = file_path.read_text().strip()
                else:
                    self._key = ''
            
            elif self.key_source == 'sqlite:runtime':
                # SQLite运行时提取（new-api专用）
                self._key = self._extract_key_from_sqlite()
            
            elif self.key_source.startswith('vault:'):
                # keyvault提取
                parts = self.key_source.split(':')
                if len(parts) >= 3:
                    vault_name = parts[1]
                    key_name = parts[2]
                    self._key = self._extract_key_from_vault(vault_name, key_name)
                else:
                    self._key = ''
            
            else:
                self._key = ''
        
        except Exception:
            self._key = ''
        
        return self._key
    
    def _extract_key_from_sqlite(self) -> str:
        """从SQLite运行时提取key（只读）"""
        if not self.sqlite_db or not self.sqlite_query:
            return ''
        
        db_path = Path(self.sqlite_db)
        if not db_path.exists():
            return ''
        
        try:
            # 只读连接
            conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
            cursor = conn.cursor()
            cursor.execute(self.sqlite_query)
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0]:
                return result[0]
            return ''
        except Exception:
            return ''
    
    def _extract_key_from_vault(self, vault_name: str, key_name: str) -> str:
        """从openllm keyvault提取key"""
        try:
            from openllm.security.keyvault import default_vault
            vault = default_vault()
            return vault.get(key_name) or ''
        except Exception:
            return ''
    
    def resolve_endpoint(self) -> str:
        """解析endpoint（支持环境变量，os.environ查不到时回退~/.hermes/.env）"""
        if self.endpoint.startswith('$'):
            env_name = self.endpoint[1:]
            resolved = _env_with_hermes_fallback(env_name)
            return resolved if resolved else self.endpoint
        return self.endpoint


class Registry:
    """执行器注册表"""
    
    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = Path(__file__).parent / 'providers.json'
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.providers: Dict[str, Provider] = {}
        for p_config in config.get('providers', []):
            provider = Provider(p_config)
            self.providers[provider.name] = provider
    
    def get_provider(self, name: str) -> Optional[Provider]:
        """获取指定执行器"""
        return self.providers.get(name)
    
    def get_all_providers(self) -> List[Provider]:
        """获取所有执行器"""
        return list(self.providers.values())
    
    def get_providers_by_tier(self, tier: int, tier_type: str = 'quality') -> List[Provider]:
        """按质量/成本档筛选执行器"""
        if tier_type == 'quality':
            return [p for p in self.providers.values() if p.quality_tier == tier]
        elif tier_type == 'cost':
            return [p for p in self.providers.values() if p.cost_tier == tier]
        return []
    
    def get_provider_for_model(self, model: str) -> Optional[Provider]:
        """根据模型名找到对应执行器"""
        for provider in self.providers.values():
            if model in provider.available_models:
                return provider
        return None


# 全局注册表实例
_registry: Optional[Registry] = None


def get_registry() -> Registry:
    """获取全局注册表实例"""
    global _registry
    if _registry is None:
        _registry = Registry()
    return _registry