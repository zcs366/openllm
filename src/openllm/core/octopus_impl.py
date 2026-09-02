"""core/octopus_impl.py — 薄转发壳（向后兼容）。

章鱼I已物理搬入 openllm.iai.octopus（IAI融合·Step 2·2026-09-02）。
此文件保留所有旧import路径：from openllm.core.octopus_impl import X 仍然有效。

为避免循环导入（iai.octopus→core→core.octopus_impl→iai.octopus），
所有re-export均使用延迟导入——首次访问时才触发。
"""


def __getattr__(name):
    """延迟导入：首次访问章鱼I等符号时才从iai.octopus加载。"""
    _LAZY_IMPORTS = {
        "章鱼I", "_LeftBrain", "_RightBrain",
        "_format_relative_time", "_FAST_PATH_MAX_LEN",
        "_TOOL_KEYWORDS", "trace_degradation",
    }
    if name in _LAZY_IMPORTS:
        from openllm.iai.octopus import (
            章鱼I, _LeftBrain, _RightBrain,
            _format_relative_time, _FAST_PATH_MAX_LEN,
            _TOOL_KEYWORDS, trace_degradation,
        )
        # 缓存到模块命名空间，下次直接命中
        globals().update({
            "章鱼I": 章鱼I, "_LeftBrain": _LeftBrain, "_RightBrain": _RightBrain,
            "_format_relative_time": _format_relative_time,
            "_FAST_PATH_MAX_LEN": _FAST_PATH_MAX_LEN,
            "_TOOL_KEYWORDS": _TOOL_KEYWORDS,
            "trace_degradation": trace_degradation,
        })
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "章鱼I", "_LeftBrain", "_RightBrain",
    "_format_relative_time",
    "_FAST_PATH_MAX_LEN", "_TOOL_KEYWORDS",
    "trace_degradation",
]
