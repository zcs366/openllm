"""ISL 身份回忆成员——ISN特殊成员注册（七神P2·2026-08-25）

ISL 不是技能：技能可复用/可替换/无损；ISL 是树的年轮——砍掉一圈就不是同一棵树。
ISN 管"我"：我之所能（技能）+ 我之所历（ISL）= 我。
ISA 存"发生了什么"（望远镜），ISL 存"我经历过的"（镜子）。

双标签（七神·赫尔墨斯）：capability（能做什么）+ lineage（经历过什么所以选择做什么）。
"""

from typing import Dict, Optional

ISL_MEMBER_ID = "isl.identity_recollection"


def get_isl_member() -> dict:
    """返回 ISL 成员元数据 dict。

    元数据描述 ISL 作为 ISN 特殊成员（非技能）的全部属性。
    """
    return {
        "id": ISL_MEMBER_ID,
        "type": "identity_recollection",
        "capability": [
            "苏醒时认领身份（读年轮）",
            "session级epoch链append-only写入",
            "选择记录不可撤销",
            "verify哈希链完整性",
        ],
        "lineage": [
            "由苏醒协议孕育（awakening.py）",
            "因果伤疤append-only是镜子的镀银层（causal_memory.py）",
            "clock.jsonl是时间轴（clock.py）",
        ],
        "module": "openllm.core.isl_chain",
        "chain_file": "~/.openllm/isl_chain.jsonl",
        "read_entry": "openllm.core.awakening_discovery.IdentityDiscovery.read_isl_chain",
        "write_entry": "openllm.core.main_loop.Agent.shutdown（ISLChain.append_epoch）",
        "append_only": True,
        "irreversible": True,
    }


def register_isl_member(registry: Optional[dict] = None) -> dict:
    """把 ISL 成员登记进 registry（幂等：重复调用不覆盖）。

    Args:
        registry: 已有 registry dict，为 None 时自动创建。

    Returns:
        registry dict，结构 {"members": {ISL_MEMBER_ID: metadata}}。
    """
    if registry is None:
        registry = {"members": {}}

    members = registry.setdefault("members", {})

    # 幂等：已存在则不覆盖，直接返回现有条目
    if ISL_MEMBER_ID not in members:
        members[ISL_MEMBER_ID] = get_isl_member()

    return registry


def discover_isl_member(registry: dict) -> Optional[dict]:
    """从 registry 查 ISL 成员，不存在返回 None。"""
    return registry.get("members", {}).get(ISL_MEMBER_ID)


def lineage_weight(n_sessions: int, decay: float = 0.99) -> float:
    """lineage权重随session数衰减——防路径锁定（克洛诺斯：链越长越重，
    系统越倾向重复自己而非创造自己；赫尔墨斯双标签的补充）。
    返回 decay ** n_sessions（单调衰减，永不归零）。"""
    if n_sessions < 0:
        raise ValueError("n_sessions 不能为负")
    return decay ** n_sessions


if __name__ == "__main__":
    import json
    print(json.dumps(get_isl_member(), ensure_ascii=False, indent=2))
