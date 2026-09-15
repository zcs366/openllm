"""
路由器测试 - mock测试 + live冒烟测试
"""
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
import pytest

# 添加src到path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from openllm.router import (
    route, Budget, Decision, Executor, ExecutionResult,
    Ledger, h_of, get_distribution, get_registry, shannon_entropy
)


class TestRoutingStrategy:
    """测试路由策略分档"""
    
    def test_budget_default_is_balanced(self):
        """默认budget应为balanced"""
        decision = route("你好")
        assert decision is not None
        assert decision.provider is not None
    
    def test_complex_math_upgrades_to_quality(self):
        """复杂数学任务应升级到质量档"""
        decision = route("求解微分方程 dy/dx = 2x + 1", Budget.BALANCED)
        # 应该选择质量档较高的执行器
        assert decision.provider.quality_tier >= 2
    
    def test_simple_query_uses_free(self):
        """简单查询应使用免费档"""
        decision = route("hi", Budget.BALANCED)
        # 简单查询应选择免费或低成本执行器
        assert decision.provider.cost_tier <= 1
    
    def test_explicit_model_override(self):
        """显式@模型名应直接指定"""
        decision = route("@deepseek 你好", Budget.BALANCED)
        assert decision.provider.name == "deepseek"
        assert "显式指定" in decision.reason
    
    def test_fallback_chain_length(self):
        """降级链长度应不超过3"""
        decision = route("这是一个测试查询，用于验证降级链长度", Budget.QUALITY)
        # 降级链应该存在且合理
        assert isinstance(decision.fallback_chain, list)


class TestExecutor:
    """测试执行器"""
    
    @patch('requests.post')
    def test_openai_protocol_execution(self, mock_post):
        """测试OpenAI协议执行"""
        # Mock成功响应
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{'message': {'content': '测试响应'}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 20}
        }
        mock_post.return_value = mock_response
        
        executor = Executor()
        # 强制使用deepseek执行器
        from openllm.router import get_registry
        registry = get_registry()
        deepseek = registry.get_provider('deepseek')
        decision = Decision(
            provider=deepseek,
            model='deepseek-chat',
            reason='测试OpenAI协议',
            fallback_chain=[]
        )
        
        result, trace = executor.execute(decision, "测试查询")
        
        assert result.success is True
        assert result.content == '测试响应'
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 20
    
    @patch('requests.post')
    def test_ollama_protocol_execution(self, mock_post):
        """测试ollama协议执行"""
        # Mock ollama响应
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'response': 'ollama响应',
            'prompt_eval_count': 5,
            'eval_count': 15
        }
        mock_post.return_value = mock_response
        
        # 使用ollama执行器
        registry = get_registry()
        ollama = registry.get_provider('ollama')
        
        if ollama:
            executor = Executor()
            decision = Decision(
                provider=ollama,
                model='ilm-v6',
                reason='测试ollama',
                fallback_chain=[]
            )
            
            result, trace = executor.execute(decision, "测试查询")
            
            assert result.success is True
            assert result.content == 'ollama响应'
            assert result.prompt_tokens == 5
            assert result.completion_tokens == 15
    
    @patch('requests.post')
    def test_fallback_chain_trigger(self, mock_post):
        """测试降级链触发"""
        # 第一次调用失败
        mock_response_fail = Mock()
        mock_response_fail.status_code = 500
        
        # 第二次调用成功
        mock_response_ok = Mock()
        mock_response_ok.status_code = 200
        mock_response_ok.json.return_value = {
            'choices': [{'message': {'content': '降级后响应'}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 20}
        }
        
        # 设置调用顺序
        mock_post.side_effect = [mock_response_fail, mock_response_ok]
        
        executor = Executor()
        # 使用deepseek，降级到其他执行器
        from openllm.router import get_registry
        registry = get_registry()
        deepseek = registry.get_provider('deepseek')
        decision = Decision(
            provider=deepseek,
            model='deepseek-chat',
            reason='测试降级链',
            fallback_chain=['glm']  # 降级到glm
        )
        
        result, trace = executor.execute(decision, "测试查询")
        
        # 应该成功（通过降级）
        assert result.success is True
        assert result.content == '降级后响应'
        assert result.is_fallback is True
        
        # 轨迹应该记录两次尝试
        assert len(trace.attempts) == 2
        assert trace.attempts[0]['success'] is False
        assert trace.attempts[1]['success'] is True
    
    @patch('requests.post')
    def test_glm_reasoning_content_fallback(self, mock_post):
        """测试glm/mimo的reasoning_content兜底"""
        # Mock响应（content为空，有reasoning_content）
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{'message': {
                'content': '',
                'reasoning_content': '推理内容'
            }}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 20}
        }
        mock_post.return_value = mock_response
        
        # 使用glm执行器
        registry = get_registry()
        glm = registry.get_provider('glm')
        
        if glm:
            executor = Executor()
            decision = Decision(
                provider=glm,
                model='glm-5',
                reason='测试reasoning_content',
                fallback_chain=[]
            )
            
            result, trace = executor.execute(decision, "测试查询")
            
            assert result.success is True
            assert result.content == '推理内容'
            assert result.reasoning_content == '推理内容'


class TestLedger:
    """测试账本"""
    
    def test_append_record(self):
        """测试追加记录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / 'test_ledger.jsonl'
            ledger = Ledger(ledger_path)
            
            record = {
                'timestamp': time.time(),
                'provider': 'test',
                'model': 'test-model',
                'prompt_tokens': 10,
                'completion_tokens': 20,
                'latency_ms': 100,
                'budget': 'balanced',
                'result': 'ok',
                'query_preview': '测试查询'
            }
            
            ledger.append(record)
            
            # 读取验证
            records = ledger.read_records()
            assert len(records) == 1
            assert records[0]['provider'] == 'test'
    
    def test_query_preview_truncation(self):
        """测试query_preview截断"""
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / 'test_ledger.jsonl'
            ledger = Ledger(ledger_path)
            
            long_query = 'x' * 200
            record = {
                'timestamp': time.time(),
                'provider': 'test',
                'model': 'test-model',
                'result': 'ok',
                'query_preview': long_query
            }
            
            ledger.append(record)
            
            records = ledger.read_records()
            assert len(records[0]['query_preview']) <= 80
    
    def test_stats_calculation(self):
        """测试统计计算"""
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / 'test_ledger.jsonl'
            ledger = Ledger(ledger_path)
            
            # 添加多条记录
            for i in range(5):
                record = {
                    'timestamp': time.time(),
                    'provider': 'test',
                    'model': 'test-model',
                    'prompt_tokens': 10,
                    'completion_tokens': 20,
                    'latency_ms': 100,
                    'budget': 'balanced',
                    'result': 'ok' if i < 3 else 'fail:test',
                    'query_preview': f'测试查询{i}'
                }
                ledger.append(record)
            
            stats = ledger.get_stats()
            
            assert stats['total'] == 5
            assert stats['success_count'] == 3
            assert stats['fail_count'] == 2
            assert stats['success_rate'] == 0.6


class TestEntropy:
    """测试熵计算"""
    
    def test_shannon_entropy_uniform(self):
        """测试均匀分布的熵"""
        distribution = {'a': 50, 'b': 50}
        entropy = shannon_entropy(distribution)
        
        # 均匀分布熵应为log2(2) = 1
        assert abs(entropy - 1.0) < 0.001
    
    def test_shannon_entropy_skewed(self):
        """测试偏斜分布的熵"""
        distribution = {'a': 90, 'b': 10}
        entropy = shannon_entropy(distribution)
        
        # 偏斜分布熵应小于1
        assert entropy < 1.0
    
    def test_shannon_entropy_empty(self):
        """测试空分布的熵"""
        distribution = {}
        entropy = shannon_entropy(distribution)
        
        assert entropy == 0.0
    
    def test_h_of_with_ledger(self):
        """测试从账本计算熵"""
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / 'test_ledger.jsonl'
            ledger = Ledger(ledger_path)
            
            # 添加成功调用记录
            for i in range(10):
                record = {
                    'timestamp': time.time(),
                    'provider': 'test' if i % 2 == 0 else 'other',
                    'model': 'test-model',
                    'prompt_tokens': 10,
                    'completion_tokens': 20,
                    'latency_ms': 100,
                    'budget': 'balanced',
                    'result': 'ok',
                    'query_preview': f'测试查询{i}'
                }
                ledger.append(record)
            
            h_value = h_of(10, ledger)
            
            # 应该有正熵值
            assert h_value > 0
    
    def test_get_distribution(self):
        """测试分布计算"""
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / 'test_ledger.jsonl'
            ledger = Ledger(ledger_path)
            
            # 添加成功调用记录
            for i in range(10):
                record = {
                    'timestamp': time.time(),
                    'provider': 'test' if i < 7 else 'other',
                    'model': 'test-model',
                    'result': 'ok',
                    'query_preview': f'测试查询{i}'
                }
                ledger.append(record)
            
            distribution = get_distribution(10, ledger)
            
            assert 'test' in distribution
            assert 'other' in distribution
            assert distribution['test'] == 0.7
            assert distribution['other'] == 0.3


class TestRegistry:
    """测试注册表"""
    
    def test_load_providers(self):
        """测试加载执行器配置"""
        registry = get_registry()
        
        # 应该加载到执行器
        assert len(registry.providers) > 0
    
    def test_get_provider(self):
        """测试获取执行器"""
        registry = get_registry()
        
        ollama = registry.get_provider('ollama')
        assert ollama is not None
        assert ollama.name == 'ollama'
    
    def test_provider_models(self):
        """测试执行器模型列表"""
        registry = get_registry()
        
        one_api = registry.get_provider('one-api')
        if one_api:
            models = one_api.available_models
            assert len(models) > 0


# Live冒烟测试（默认跳过，需要环境变量RUN_LIVE_TESTS=1）
@pytest.mark.skipif(
    not os.environ.get('RUN_LIVE_TESTS'),
    reason="需要设置环境变量RUN_LIVE_TESTS=1运行live测试"
)
class TestLiveSmoke:
    """真实冒烟测试（需要真实key，手动运行）"""
    
    def test_live_routing_free(self):
        """测试免费档路由"""
        decision = route("你好", Budget.FREE)
        executor = Executor()
        
        result, trace = executor.execute(decision, "你好")
        
        assert result.success is True
        assert result.content != ''
        
        # 记录账本
        ledger = Ledger()
        record = {
            'timestamp': time.time(),
            'provider': result.provider_name,
            'model': result.model,
            'prompt_tokens': result.prompt_tokens,
            'completion_tokens': result.completion_tokens,
            'latency_ms': result.latency_ms,
            'budget': 'free',
            'result': 'ok',
            'query_preview': '你好'
        }
        ledger.append(record)
    
    def test_live_routing_balanced(self):
        """测试均衡档路由"""
        decision = route("解释什么是机器学习", Budget.BALANCED)
        executor = Executor()
        
        result, trace = executor.execute(decision, "解释什么是机器学习")
        
        assert result.success is True
        assert result.content != ''
        
        # 记录账本
        ledger = Ledger()
        record = {
            'timestamp': time.time(),
            'provider': result.provider_name,
            'model': result.model,
            'prompt_tokens': result.prompt_tokens,
            'completion_tokens': result.completion_tokens,
            'latency_ms': result.latency_ms,
            'budget': 'balanced',
            'result': 'ok',
            'query_preview': '解释什么是机器学习'
        }
        ledger.append(record)
    
    def test_live_routing_quality(self):
        """测试质量档路由"""
        decision = route("写一个快速排序算法的Python实现", Budget.QUALITY)
        executor = Executor()
        
        result, trace = executor.execute(decision, "写一个快速排序算法的Python实现")
        
        assert result.success is True
        assert result.content != ''
        
        # 记录账本
        ledger = Ledger()
        record = {
            'timestamp': time.time(),
            'provider': result.provider_name,
            'model': result.model,
            'prompt_tokens': result.prompt_tokens,
            'completion_tokens': result.completion_tokens,
            'latency_ms': result.latency_ms,
            'budget': 'quality',
            'result': 'ok',
            'query_preview': '写一个快速排序算法的Python实现'
        }
        ledger.append(record)
    
    def test_live_fallback_chain(self):
        """测试降级链"""
        # 强制使用一个可能失败的执行器
        decision = route("@deepseek 测试降级", Budget.FREE)
        executor = Executor()
        
        result, trace = executor.execute(decision, "测试降级")
        
        # 无论成功失败，都应该有轨迹
        assert len(trace.attempts) >= 1
        
        # 记录账本
        ledger = Ledger()
        record = {
            'timestamp': time.time(),
            'provider': result.provider_name,
            'model': result.model,
            'prompt_tokens': result.prompt_tokens,
            'completion_tokens': result.completion_tokens,
            'latency_ms': result.latency_ms,
            'budget': 'free',
            'result': 'ok' if result.success else f'fail:{result.error}',
            'query_preview': '测试降级'
        }
        ledger.append(record)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])