"""
统一执行器 - 支持OpenAI/ollama双协议，带降级链
"""
import time
import json
import requests
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from .registry import Provider
from .router import Decision


@dataclass
class ExecutionResult:
    """执行结果"""
    success: bool
    content: str
    provider_name: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    error: Optional[str] = None
    is_fallback: bool = False
    reasoning_content: Optional[str] = None  # glm/mimo的reasoning_content


@dataclass
class ExecutionTrace:
    """执行轨迹（降级链记录）"""
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    final_result: Optional[ExecutionResult] = None
    
    def add_attempt(self, provider_name: str, model: str, success: bool, error: Optional[str] = None):
        """添加一次尝试记录"""
        self.attempts.append({
            'provider': provider_name,
            'model': model,
            'success': success,
            'error': error,
            'timestamp': time.time()
        })


class Executor:
    """统一执行器"""
    
    def __init__(self, timeout: int = 45):
        self.timeout = timeout
    
    def execute(self, decision: Decision, query: str, 
                temperature: float = 0.7, max_tokens: int = 2000) -> Tuple[ExecutionResult, ExecutionTrace]:
        """
        执行请求，带降级链
        
        Returns:
            (ExecutionResult, ExecutionTrace)
        """
        trace = ExecutionTrace()
        
        # 构建候选列表：主选 + 降级链
        candidates = [(decision.provider, decision.model)] + [
            (None, name) for name in decision.fallback_chain
        ]
        
        for provider, model_name in candidates:
            if provider is None:
                # 从注册表获取降级执行器
                from .registry import get_registry
                registry = get_registry()
                provider = registry.get_provider(model_name)
                if provider is None:
                    continue
                # v0.5: 降级时也尊重场景池——优先选场景内模型
                from .router import Scenario
                try:
                    scn = Scenario(decision.scenario)
                except ValueError:
                    scn = Scenario.INTERACTIVE
                scn_models = provider.batch_models() if scn == Scenario.BATCH else provider.interactive_models()
                model_name = scn_models[0] if scn_models else (
                    provider.available_models[0] if provider.available_models else provider.model
                )
                is_fallback = True
            else:
                is_fallback = False
            
            try:
                start_time = time.time()
                
                if provider.protocol == 'ollama':
                    result = self._execute_ollama(provider, model_name, query, temperature, max_tokens)
                else:
                    result = self._execute_openai(provider, model_name, query, temperature, max_tokens)
                
                latency_ms = int((time.time() - start_time) * 1000)
                result.latency_ms = latency_ms
                result.is_fallback = is_fallback
                
                trace.add_attempt(provider.name, model_name, True)
                trace.final_result = result
                
                return result, trace
            
            except Exception as e:
                error_msg = str(e)
                trace.add_attempt(provider.name, model_name, False, error_msg)
                continue
        
        # 所有尝试都失败
        final_result = ExecutionResult(
            success=False,
            content='',
            provider_name=decision.provider.name,
            model=decision.model,
            error="所有执行器均失败",
            latency_ms=0
        )
        trace.final_result = final_result
        
        return final_result, trace
    
    def _execute_openai(self, provider: Provider, model: str, query: str,
                       temperature: float, max_tokens: int) -> ExecutionResult:
        """执行OpenAI协议请求"""
        endpoint = provider.resolve_endpoint()
        
        # 处理mimo的完整chat URL（不拼接/chat/completions）
        if provider.name == 'mimo':
            url = endpoint
        else:
            url = f"{endpoint}/chat/completions"
        
        headers = {
            'Content-Type': 'application/json'
        }

        # v0.5: 注册表声明的额外头（如openrouter的HTTP-Referer/X-Title）
        headers.update(provider.extra_headers)

        # 本地路由自给自足守卫：local=true执行器禁止走OpenAI协议外呼
        if provider.local and not endpoint.startswith(('http://127.0.0.1', 'http://localhost')):
            raise ValueError(f"执行器{provider.name}标记local=true但endpoint非本地，禁止外呼")

        # 获取API key
        api_key = provider.get_key()
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        
        payload = {
            'model': model,
            'messages': [{'role': 'user', 'content': query}],
            'temperature': temperature,
            'max_tokens': max_tokens
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        
        if response.status_code == 429:
            raise Exception("Rate limited (429)")
        elif response.status_code >= 500:
            raise Exception(f"Server error ({response.status_code})")
        elif response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.text[:200]}")
        
        data = response.json()
        
        # 提取内容和usage
        content = ''
        reasoning_content = None
        
        if 'choices' in data and len(data['choices']) > 0:
            choice = data['choices'][0]
            message = choice.get('message', {})
            content = message.get('content', '')
            
            # glm/mimo的reasoning_content兜底
            if not content and 'reasoning_content' in message:
                reasoning_content = message['reasoning_content']
                content = reasoning_content
        
        # 提取usage
        usage = data.get('usage', {})
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        
        return ExecutionResult(
            success=True,
            content=content,
            provider_name=provider.name,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_content=reasoning_content
        )
    
    def _execute_ollama(self, provider: Provider, model: str, query: str,
                       temperature: float, max_tokens: int) -> ExecutionResult:
        """执行ollama原生协议（/api/generate）"""
        endpoint = provider.resolve_endpoint()
        url = f"{endpoint}/api/generate"
        
        payload = {
            'model': model,
            'prompt': query,
            'stream': False,
            'options': {
                'temperature': temperature,
                'num_predict': max_tokens
            }
        }
        
        response = requests.post(url, json=payload, timeout=self.timeout)
        
        if response.status_code != 200:
            raise Exception(f"Ollama HTTP {response.status_code}: {response.text[:200]}")
        
        data = response.json()
        
        # 提取内容和usage
        content = data.get('response', '')
        
        # ollama的usage格式
        prompt_tokens = data.get('prompt_eval_count', 0)
        completion_tokens = data.get('eval_count', 0)
        
        return ExecutionResult(
            success=True,
            content=content,
            provider_name=provider.name,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )