"""
openllm.isa.timeutil — 时间戳解析的单一判定源（审计 P1-5，2026-10-01）

原先 `_coerce_ts` / `_pick_ts` / `_TS_FIELDS` 住在 recall_provider 里；
session_causal_extractor 等其它读时间戳的模块各自为政（P1-4 的活性 bug
正是"时间过滤"没有共用判定源导致 epoch 字符串与 ISO 文本混比恒真）。

本模块收敛为唯一实现，公开名：
- `coerce_ts(v) -> float`        健壮解析单值 → epoch 秒
- `pick_ts(rec) -> (float, Optional[str])`  按取值链取记录时间戳
- `TS_FIELDS`                    取值链字段序（timestamp → ts → _timestamp → _written_at）

时间戳纪律（2026-10-01 军师拍板，勿回退）：
- 取值链必须是**唯一判定源**：任何读记录时间戳的地方都过 pick_ts/coerce_ts，
  不得再手写 `str(cutoff)` 绑定或裸字符串比较。
- 只有"时间已知且早于 cutoff"才过滤；缺失/不可解析 → 0，由调用方决定语义。

recall_provider 保留 `_coerce_ts` / `_pick_ts` / `_TS_FIELDS` 兼容别名
（"增强不替代"纪律：已有 8 条钉子测试 import 旧名，必须继续绿）。
"""

from datetime import datetime
from typing import Any, Dict, Optional, Tuple

TS_FIELDS: Tuple[str, ...] = ("timestamp", "ts", "_timestamp", "_written_at")


def pick_ts(rec: Dict[str, Any]) -> Tuple[float, Optional[str]]:
    """按取值链取时间戳 → (epoch秒, 命中的字段名|None)。

    2026-10-01 军师：取值链必须是**唯一判定源**。原先 _load_records 扩了链、
    search() 没扩，导致 141 条只有 `ts` 的记录在检索眼里 timestamp=0、时间衰减
    与温度恒为中性(0.5)。两处共用本函数，再断一处即红（有钉子）。
    """
    for field in TS_FIELDS:
        if field in rec:
            return coerce_ts(rec[field]), field
    return 0.0, None


def coerce_ts(v: Any) -> float:
    """健壮解析记录时间戳 → epoch 秒。

    规则：int/float 直接用；纯数字字符串转 float；ISO8601 字符串
    （支持结尾 'Z' 与 '2026-09-29' 纯日期）用 datetime.fromisoformat
    解析（naive 按本地时间，与 time.time() 同基准）；无法解析或缺失 → 0
    （保持"被 cutoff 过滤掉"的既有语义）。
    """
    if isinstance(v, bool):
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return 0.0
        try:
            return float(s)
        except ValueError:
            pass
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            return 0.0
        if dt.tzinfo is not None:
            return dt.timestamp()
        return datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute,
                        dt.second, dt.microsecond).timestamp()
    return 0.0
