"""ISL 身份回忆成员注册测试（T1-T5）

验收：PYTHONPATH=src python3 -m pytest tests/test_isl_member.py -q
"""

from openllm.isn.isl_member import ISL_MEMBER_ID, get_isl_member


class TestISLMember:
    """ISL 身份回忆成员注册模块测试。"""

    def test_T1_metadata_complete(self):
        """T1 元数据完整：含全部字段，capability 和 lineage 均为非空列表。"""
        member = get_isl_member()
        # 必需字段
        for key in ("id", "type", "capability", "lineage", "module",
                     "chain_file", "read_entry", "write_entry",
                     "append_only", "irreversible"):
            assert key in member, f"缺少字段: {key}"
        # 双标签非空列表
        assert isinstance(member["capability"], list) and len(member["capability"]) >= 3
        assert isinstance(member["lineage"], list) and len(member["lineage"]) >= 3

    def test_T2_type_special(self):
        """T2 类型特殊：type=identity_recollection, append_only=True, irreversible=True。"""
        member = get_isl_member()
        assert member["type"] == "identity_recollection"
        assert member["append_only"] is True
        assert member["irreversible"] is True

    def test_T3_register_idempotent(self):
        """T3 注册幂等：两次注册，members 只有1个 ISL 条目。"""
        from openllm.isn.isl_member import register_isl_member

        registry = register_isl_member()           # 第1次
        first_entry = registry["members"][ISL_MEMBER_ID]
        registry = register_isl_member(registry)   # 第2次（幂等）
        second_entry = registry["members"][ISL_MEMBER_ID]

        assert len(registry["members"]) == 1
        assert first_entry is second_entry          # 同一对象，未被覆盖

    def test_T4_discover_hit_miss(self):
        """T4 discover 命中/未命中：注册后命中，空 registry 未命中。"""
        from openllm.isn.isl_member import register_isl_member, discover_isl_member

        # 已注册 → 命中
        registry = register_isl_member()
        found = discover_isl_member(registry)
        assert found is not None
        assert found["id"] == ISL_MEMBER_ID

        # 空 registry → 未命中
        assert discover_isl_member({"members": {}}) is None
        assert discover_isl_member({}) is None

    def test_T5_importable(self):
        """T5 模块可导入：ISL_MEMBER_ID 和 get_isl_member 可从模块导入。"""
        from openllm.isn.isl_member import ISL_MEMBER_ID, get_isl_member

        assert ISL_MEMBER_ID == "isl.identity_recollection"
        member = get_isl_member()
        assert member["id"] == ISL_MEMBER_ID
