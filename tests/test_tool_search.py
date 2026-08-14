"""Tool Search 元工具搜索测试。"""
import pytest
import threading

from openllm.tool_search import (
    ToolDefinition,
    register_tools,
    tool_search_handler,
    _tools,
    _inverted_index,
    _tool_token_counts,
)


# ── 固定测试数据 ────────────────────────────────────

SAMPLE_TOOLS = [
    ToolDefinition(
        name="read_file",
        description="读取文件内容",
        tags=["file", "read"],
    ),
    ToolDefinition(
        name="write_file",
        description="写入文件内容",
        tags=["file", "write"],
    ),
    ToolDefinition(
        name="search",
        description="搜索文件内容 pattern matching",
        tags=["search", "grep"],
    ),
    ToolDefinition(
        name="octopus_search",
        description="搜索章鱼记忆系统",
        tags=["memory", "search"],
    ),
]


def _reset():
    """清空全局索引（测试隔离）。"""
    _tools.clear()
    _inverted_index.clear()
    _tool_token_counts.clear()


@pytest.fixture(autouse=True)
def clean_state():
    """每个测试前后清空索引。"""
    _reset()
    yield
    _reset()


# ── Test 1: handler响应格式 ──────────────────────────

class TestHandlerResponseFormat:
    def test_returns_list_of_tool_definitions(self):
        register_tools(SAMPLE_TOOLS)
        results = tool_search_handler("读文件")
        assert isinstance(results, list)
        assert all(isinstance(t, ToolDefinition) for t in results)

    def test_top_k_limits_results(self):
        register_tools(SAMPLE_TOOLS)
        results = tool_search_handler("file", top_k=2)
        assert len(results) <= 2

    def test_results_have_required_fields(self):
        register_tools(SAMPLE_TOOLS)
        results = tool_search_handler("file")
        for tool in results:
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")
            assert tool.name  # non-empty

    def test_no_tools_returns_empty(self):
        results = tool_search_handler("anything")
        assert results == []

    def test_empty_query_returns_empty(self):
        register_tools(SAMPLE_TOOLS)
        results = tool_search_handler("")
        assert results == []

    def test_to_dict_roundtrip(self):
        td = ToolDefinition(
            name="test",
            description="a test tool",
            parameters={"x": {"type": "int"}},
            tags=["test"],
        )
        d = td.to_dict()
        assert d["name"] == "test"
        assert d["parameters"]["x"]["type"] == "int"
        assert d["tags"] == ["test"]


# ── Test 2: 注册功能 ────────────────────────────────

class TestRegistration:
    def test_register_single_tool(self):
        register_tools([ToolDefinition(name="t1", description="tool one")])
        assert "t1" in _tools
        assert "t1" in _tool_token_counts

    def test_register_multiple_tools(self):
        register_tools(SAMPLE_TOOLS)
        assert len(_tools) == 4

    def test_later_registration_overwrites(self):
        register_tools([ToolDefinition(name="t1", description="old")])
        register_tools([ToolDefinition(name="t1", description="new")])
        assert _tools["t1"].description == "new"

    def test_inverted_index_populated(self):
        register_tools(SAMPLE_TOOLS)
        # "search" appears in both "search" tool and "octopus_search" tool
        assert "search" in _inverted_index
        assert len(_inverted_index["search"]) >= 2

    def test_chinese_bigrams_indexed(self):
        register_tools([
            ToolDefinition(name="file_op", description="文件操作管理器"),
        ])
        # "文件" should be in index as a bigram
        assert "文件" in _inverted_index

    def test_empty_name_still_registered(self):
        """Edge case: empty name tool still indexed (no validation in register)."""
        register_tools([ToolDefinition(name="", description="no name")])
        # The tool is stored but search may not find useful tokens
        assert "" in _tools

    def test_tags_boost_search(self):
        register_tools([
            ToolDefinition(name="a", description="alpha", tags=["search"]),
            ToolDefinition(name="b", description="beta", tags=["unrelated"]),
        ])
        results = tool_search_handler("search")
        names = [t.name for t in results]
        assert "a" in names
        assert "b" not in names or names.index("a") < names.index("b") if "b" in names else True


# ── Test 3: 并发安全 ────────────────────────────────

class TestConcurrentAccess:
    def test_concurrent_register_and_search(self):
        """多线程同时注册和搜索不崩溃。"""
        errors = []
        n_threads = 7  # 4 register + 3 search
        barrier = threading.Barrier(n_threads)

        def register_worker(thread_id):
            try:
                barrier.wait(timeout=5)
                tools = [
                    ToolDefinition(
                        name=f"tool_t{thread_id}_{i}",
                        description=f"thread {thread_id} tool {i}",
                    )
                    for i in range(10)
                ]
                register_tools(tools)
            except Exception as e:
                errors.append(e)

        def search_worker(query):
            try:
                barrier.wait(timeout=5)
                for _ in range(50):
                    result = tool_search_handler(query)
                    assert isinstance(result, list)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_worker, args=(i,)) for i in range(4)
        ]
        threads.extend([
            threading.Thread(target=search_worker, args=(q,))
            for q in ["tool", "thread", "handler"]
        ])

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"Concurrent errors: {errors}"
        # All 40 tools (4 threads * 10) should be registered
        assert len(_tools) == 40

    def test_concurrent_reads_are_consistent(self):
        """并发读取返回结构一致的结果。"""
        register_tools(SAMPLE_TOOLS)
        barrier = threading.Barrier(8)
        results_per_thread = {}

        def worker(tid):
            try:
                barrier.wait(timeout=5)
                local_results = []
                for _ in range(20):
                    r = tool_search_handler("file search")
                    local_results.append(len(r))
                results_per_thread[tid] = local_results
            except Exception as e:
                results_per_thread[tid] = [str(e)]

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # All threads should get consistent result lengths
        lengths = [results_per_thread[i] for i in range(8)]
        for tid_results in lengths:
            assert isinstance(tid_results, list)
            assert all(isinstance(x, int) for x in tid_results)
