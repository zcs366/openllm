"""
openLLM路由器包 - 智能选择执行器，失败自动降级，全程记账
"""
from .router import route, Budget, Decision
from .executor import Executor, ExecutionResult, ExecutionTrace
from .registry import Registry, Provider, get_registry
from .ledger import Ledger
from .entropy import h_of, get_distribution, shannon_entropy

__version__ = '0.1.0'
__all__ = [
    'route',
    'Budget',
    'Decision',
    'Executor',
    'ExecutionResult',
    'ExecutionTrace',
    'Registry',
    'Provider',
    'get_registry',
    'Ledger',
    'h_of',
    'get_distribution',
    'shannon_entropy'
]