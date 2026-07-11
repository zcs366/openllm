"""
engine_integrations.py — 懒加载集成函数

从engine.py提取的5个线程安全懒加载函数+globals。
每个函数负责加载外部模块的回调（ISA/ISN/IKO/Checkpoint），
不存在就返回None（降级模式）。
"""
import importlib.util
import logging
import threading
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("openllm.engine.integrations")


# ═══ 通用懒加载 ═══

def lazy_import(module_path: Path, module_name: str):
    """从指定路径懒加载模块，不污染sys.path。"""
    if not module_path.exists():
        return None
    try:
        if module_path.is_dir():
            spec = importlib.util.spec_from_file_location(
                module_name, module_path / "__init__.py")
        else:
            spec = importlib.util.spec_from_file_location(
                module_name, module_path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


# ═══ IO-S Checkpoint 集成 ═══

_CHECKPOINT_MANAGER_INITIALIZED = False
_CHECKPOINT_MANAGER_LOCK = threading.Lock()
_CHECKPOINT_MANAGER: Any = None


def get_checkpoint_manager():
    """线程安全地获取 IO-S CheckpointManager。懒加载。"""
    global _CHECKPOINT_MANAGER_INITIALIZED, _CHECKPOINT_MANAGER
    if _CHECKPOINT_MANAGER_INITIALIZED:
        return _CHECKPOINT_MANAGER
    with _CHECKPOINT_MANAGER_LOCK:
        if _CHECKPOINT_MANAGER_INITIALIZED:
            return _CHECKPOINT_MANAGER
        io_s_path = Path.home() / "projects" / "io-s"
        if not io_s_path.exists():
            _CHECKPOINT_MANAGER_INITIALIZED = True
            return None
        try:
            _io_s_mod = lazy_import(io_s_path / "syscall" / "checkpoint", "io_s_checkpoint")
            if _io_s_mod and hasattr(_io_s_mod, 'CheckpointManager'):
                _CHECKPOINT_MANAGER = _io_s_mod.CheckpointManager()
            logger.info("✅ IO-S CheckpointManager 已加载")
        except Exception as e:
            logger.warning(f"IO-S CheckpointManager 不可用: {e}")
            _CHECKPOINT_MANAGER = None
        _CHECKPOINT_MANAGER_INITIALIZED = True
        return _CHECKPOINT_MANAGER


# ═══ ISN 工具元数据集成 ═══

_ISN_METADATA_INITIALIZED = False
_ISN_METADATA_LOCK = threading.Lock()
_ISN_METADATA: list = []
_ISN_METADATA_MAP: dict = {}


def get_isn_metadata() -> tuple[list[dict], dict[str, dict]]:
    """线程安全地获取 ISN 工具元数据。懒加载。"""
    global _ISN_METADATA_INITIALIZED, _ISN_METADATA, _ISN_METADATA_MAP
    if _ISN_METADATA_INITIALIZED:
        return _ISN_METADATA, _ISN_METADATA_MAP
    with _ISN_METADATA_LOCK:
        if _ISN_METADATA_INITIALIZED:
            return _ISN_METADATA, _ISN_METADATA_MAP
        isn_path = Path.home() / "isn"
        if not isn_path.exists():
            logger.debug("ISN 目录不存在，跳过元数据加载")
            _ISN_METADATA_INITIALIZED = True
            return _ISN_METADATA, _ISN_METADATA_MAP
        try:
            _isn_mod = lazy_import(isn_path / "router" / "integration", "isn_integration")
            export_tool_metadata = getattr(_isn_mod, 'export_tool_metadata', None) if _isn_mod else None
            if export_tool_metadata:
                _ISN_METADATA = export_tool_metadata()
            _ISN_METADATA_MAP = {m["name"]: m for m in _ISN_METADATA}
            logger.info(f"✅ ISN 元数据已加载: {len(_ISN_METADATA)} 条工具")
        except Exception as e:
            logger.warning(f"ISN 元数据不可用: {e}")
            _ISN_METADATA = []
            _ISN_METADATA_MAP = {}
        _ISN_METADATA_INITIALIZED = True
        return _ISN_METADATA, _ISN_METADATA_MAP


# ═══ ISA 信念更新集成 ═══

_ISA_BELIEF_INITIALIZED = False
_ISA_BELIEF_LOCK = threading.Lock()
_ISA_ON_VERIFY = None


def get_isa_on_verify():
    """线程安全地获取 ISA on_verify_result 回调。懒加载。"""
    global _ISA_BELIEF_INITIALIZED, _ISA_ON_VERIFY
    if _ISA_BELIEF_INITIALIZED:
        return _ISA_ON_VERIFY
    with _ISA_BELIEF_LOCK:
        if _ISA_BELIEF_INITIALIZED:
            return _ISA_ON_VERIFY
        isa_path = Path.home() / "projects" / "isa"
        if not isa_path.exists():
            logger.debug("ISA 目录不存在，跳过信念更新")
            _ISA_BELIEF_INITIALIZED = True
            return None
        try:
            _isa_mod = lazy_import(isa_path / "belief_update", "isa_belief_update")
            if _isa_mod and hasattr(_isa_mod, 'on_verify_result'):
                _ISA_ON_VERIFY = _isa_mod.on_verify_result
            else:
                _ISA_ON_VERIFY = None
            logger.info("✅ ISA belief_update 已加载")
        except Exception as e:
            logger.warning(f"ISA belief_update 不可用: {e}")
            _ISA_ON_VERIFY = None
        _ISA_BELIEF_INITIALIZED = True
        return _ISA_ON_VERIFY


# ═══ IKO trace消费集成 ═══

_IKO_CONSUME_INITIALIZED = False
_IKO_CONSUME_LOCK = threading.Lock()
_IKO_CONSUME_TRACE = None


def get_iko_consume_trace():
    """线程安全地获取 IKO consume_trace 回调。懒加载。"""
    global _IKO_CONSUME_INITIALIZED, _IKO_CONSUME_TRACE
    if _IKO_CONSUME_INITIALIZED:
        return _IKO_CONSUME_TRACE
    with _IKO_CONSUME_LOCK:
        if _IKO_CONSUME_INITIALIZED:
            return _IKO_CONSUME_TRACE
        iko_path = Path.home() / "projects" / "iko"
        if not iko_path.exists():
            logger.debug("IKO 目录不存在，跳过trace消费")
            _IKO_CONSUME_INITIALIZED = True
            return None
        try:
            _iko_mod = lazy_import(iko_path / "trace_consumer", "iko_trace_consumer")
            if _iko_mod and hasattr(_iko_mod, 'consume_trace'):
                _IKO_CONSUME_TRACE = _iko_mod.consume_trace
            else:
                _IKO_CONSUME_TRACE = None
            logger.info("✅ IKO trace_consumer 已加载")
        except Exception as e:
            logger.warning(f"IKO trace_consumer 不可用: {e}")
            _IKO_CONSUME_TRACE = None
        _IKO_CONSUME_INITIALIZED = True
        return _IKO_CONSUME_TRACE


# ═══ ISA opinion_manager 集成（血管#5: IKO→ISA） ═══

_ISA_OPINION_INITIALIZED = False
_ISA_OPINION_LOCK = threading.Lock()
_ISA_SCHEMA_MATCHES = None


def get_isa_schema_matches():
    """线程安全地获取 ISA schema_matches 回调。懒加载。"""
    global _ISA_OPINION_INITIALIZED, _ISA_SCHEMA_MATCHES
    if _ISA_OPINION_INITIALIZED:
        return _ISA_SCHEMA_MATCHES
    with _ISA_OPINION_LOCK:
        if _ISA_OPINION_INITIALIZED:
            return _ISA_SCHEMA_MATCHES
        isa_path = Path.home() / "projects" / "isa"
        if not isa_path.exists():
            _ISA_OPINION_INITIALIZED = True
            return None
        try:
            _isa_mod2 = lazy_import(isa_path / "opinion_manager", "isa_opinion_manager")
            if _isa_mod2 and hasattr(_isa_mod2, 'schema_matches'):
                _ISA_SCHEMA_MATCHES = _isa_mod2.schema_matches
            else:
                _ISA_SCHEMA_MATCHES = None
            logger.info("✅ ISA opinion_manager.schema_matches 已加载")
        except Exception as e:
            logger.warning(f"ISA opinion_manager 不可用: {e}")
            _ISA_SCHEMA_MATCHES = None
        _ISA_OPINION_INITIALIZED = True
        return _ISA_SCHEMA_MATCHES
