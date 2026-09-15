"""
路由器CLI - 一行用：python -m openllm.router "问题"
"""
import argparse
import sys
from typing import Optional
from .router import route, Budget
from .executor import Executor
from .ledger import Ledger
from .entropy import h_of, get_distribution


def format_budget(budget_str: str) -> Budget:
    """解析budget字符串"""
    try:
        return Budget(budget_str.lower())
    except ValueError:
        print(f"错误: 无效的budget '{budget_str}'，可选值: free, balanced, quality")
        sys.exit(1)


def print_decision(decision):
    """打印路由决策"""
    print(f"\n{'='*60}")
    print(f"路由决定: {decision.provider.name}")
    print(f"模型: {decision.model}")
    print(f"理由: {decision.reason}")
    print(f"降级链: {' -> '.join(decision.fallback_chain) if decision.fallback_chain else '无'}")
    print(f"{'='*60}\n")


def print_result(result):
    """打印执行结果"""
    if result.success:
        print(f"\n--- 回答 ---\n")
        print(result.content)
        print(f"\n--- Token统计 ---")
        print(f"输入tokens: {result.prompt_tokens}")
        print(f"输出tokens: {result.completion_tokens}")
        print(f"总tokens: {result.prompt_tokens + result.completion_tokens}")
        print(f"延迟: {result.latency_ms}ms")
    else:
        print(f"\n执行失败: {result.error}")
        print(f"尝试的执行器: {result.provider_name}")
        print(f"模型: {result.model}")


def print_stats(ledger: Ledger):
    """打印账本统计"""
    stats = ledger.get_stats()
    
    print(f"\n{'='*60}")
    print(f"路由器账本统计")
    print(f"{'='*60}")
    print(f"总调用次数: {stats['total']}")
    print(f"成功次数: {stats['success_count']}")
    print(f"降级次数: {stats['fallback_count']}")
    print(f"失败次数: {stats['fail_count']}")
    print(f"成功率: {stats['success_rate']:.2%}")
    print(f"总输入tokens: {stats['total_prompt_tokens']}")
    print(f"总输出tokens: {stats['total_completion_tokens']}")
    print(f"平均延迟: {stats['avg_latency_ms']}ms")
    
    print(f"\n--- 各执行器统计 ---")
    for provider, p_stats in stats['providers'].items():
        success_rate = p_stats['success'] / p_stats['count'] if p_stats['count'] > 0 else 0
        print(f"  {provider}: {p_stats['count']}次调用, 成功率{success_rate:.2%}")
    
    # Shannon熵
    h_value = h_of(100, ledger)
    distribution = get_distribution(100, ledger)
    
    print(f"\n--- 路由分布熵 ---")
    print(f"H(r) = {h_value:.4f} bits")
    print(f"分布: {distribution}")
    print(f"{'='*60}\n")


def main():
    """CLI主函数"""
    parser = argparse.ArgumentParser(
        description='openLLM路由器 - 智能选择执行器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m openllm.router "你好"
  python -m openllm.router --budget quality "写一个快速排序算法"
  python -m openllm.router --stats
  python -m openllm.router @deepseek "解释量子计算"
        """
    )
    
    parser.add_argument('query', nargs='?', help='要处理的查询')
    parser.add_argument('--budget', '-b', 
                       choices=['free', 'balanced', 'quality'],
                       default='balanced',
                       help='预算档位 (默认: balanced)')
    parser.add_argument('--stats', '-s', 
                       action='store_true',
                       help='显示账本统计')
    parser.add_argument('--temperature', '-t',
                       type=float, default=0.7,
                       help='温度参数 (默认: 0.7)')
    parser.add_argument('--max-tokens', '-m',
                       type=int, default=2000,
                       help='最大输出tokens (默认: 2000)')
    
    args = parser.parse_args()
    
    # 初始化组件
    ledger = Ledger()
    executor = Executor()
    
    # 显示统计
    if args.stats:
        print_stats(ledger)
        return
    
    # 必须提供查询
    if not args.query:
        parser.print_help()
        sys.exit(1)
    
    # 路由决策
    budget = format_budget(args.budget)
    decision = route(args.query, budget)
    
    print_decision(decision)
    
    # 执行请求
    result, trace = executor.execute(decision, args.query, 
                                    temperature=args.temperature,
                                    max_tokens=args.max_tokens)
    
    # 打印结果
    print_result(result)
    
    # 记录账本
    record = {
        'timestamp': __import__('time').time(),
        'provider': result.provider_name,
        'model': result.model,
        'prompt_tokens': result.prompt_tokens,
        'completion_tokens': result.completion_tokens,
        'latency_ms': result.latency_ms,
        'budget': budget.value,
        'result': 'ok' if result.success else f'fail:{result.error}',
        'query_preview': args.query[:80]
    }
    
    # 标记降级
    if result.is_fallback:
        record['result'] = f'fallback:{trace.attempts[-2]["provider"]}' if len(trace.attempts) > 1 else 'fallback:unknown'
    
    ledger.append(record)


if __name__ == '__main__':
    main()