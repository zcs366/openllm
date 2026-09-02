"""
test_ilm_pipeline_migration.py — 验证ILM数据管线从isa/ilm搬迁到openllm.iai.ilm后import链完整

验证项：
1. 所有模块import成功（16个模块）
2. base.py核心类可实例化
3. format_converter核心函数可调用（不依赖真实DB）
4. filters/dedup/mixer/writer子模块核心函数可调用（不依赖真实DB）
5. __init__.py包级导出完整
6. 路径参数化验证（env var override）
"""
import os
import sys
import hashlib
import tempfile
import json
import pytest

# 确保src/在sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_src_dir = os.path.join(_project_root, "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)


# ═══════════════════════════════════════════════════════
# 1. Import链完整性（16个模块逐个验证）
# ═══════════════════════════════════════════════════════

class TestImportChain:
    """验证所有模块可以通过openllm.iai.ilm包路径import"""

    def test_import_base(self):
        from openllm.iai.ilm.base import ILMDocument, DocType, SourceType
        assert ILMDocument is not None
        assert DocType.RELATION.value == "relation"

    def test_import_format_converter(self):
        from openllm.iai.ilm.format_converter import FormatConverter, _quality_score
        assert FormatConverter is not None
        assert callable(_quality_score)

    def test_import_sources(self):
        from openllm.iai.ilm.sources.session_extractor import extract_sessions, get_session_stats
        from openllm.iai.ilm.sources.recall_extractor import extract_recall
        from openllm.iai.ilm.sources.jiak_extractor import extract_jiak
        from openllm.iai.ilm.sources.output_extractor import extract_output
        assert callable(extract_sessions)
        assert callable(extract_recall)
        assert callable(extract_jiak)
        assert callable(extract_output)

    def test_import_filters(self):
        from openllm.iai.ilm.filters.heuristic_filter import filter_batch, filter_document
        from openllm.iai.ilm.filters.pii_filter import filter_pii
        from openllm.iai.ilm.filters.classifier import classify, classify_with_confidence
        assert callable(filter_batch)
        assert callable(filter_pii)
        assert callable(classify)

    def test_import_dedup(self):
        from openllm.iai.ilm.dedup.exact_dedup import exact_dedup
        from openllm.iai.ilm.dedup.minhash_dedup import dedup_documents
        assert callable(exact_dedup)
        assert callable(dedup_documents)

    def test_import_mixer(self):
        from openllm.iai.ilm.mixer.domain_mixer import mix_documents, get_domain_stats
        assert callable(mix_documents)
        assert callable(get_domain_stats)

    def test_import_writers(self):
        from openllm.iai.ilm.writers.corpus_writer import CorpusWriter
        from openllm.iai.ilm.writers.jiak_writer import write_to_jiak, _build_card
        assert CorpusWriter is not None
        assert callable(write_to_jiak)

    def test_import_pipeline(self):
        from openllm.iai.ilm.ilm_pipeline import run_pipeline
        assert callable(run_pipeline)

    def test_package_init_exports(self):
        from openllm.iai.ilm import (
            ILMDocument, DocType, SourceType,
            FormatConverter, format_converter_main,
            run_pipeline,
        )
        assert ILMDocument is not None
        assert callable(run_pipeline)


# ═══════════════════════════════════════════════════════
# 2. base.py核心类可实例化
# ═══════════════════════════════════════════════════════

class TestBaseClasses:
    """验证base.py的数据模型可正确实例化"""

    def test_ilmdocument_creation(self):
        from openllm.iai.ilm.base import ILMDocument, DocType, SourceType
        doc = ILMDocument(
            source=SourceType.SESSION.value,
            source_path="test.db",
            content="这是一条测试消息，用于验证ILMDocument可以正确创建。",
            doc_type=DocType.RELATION.value,
            timestamp=1700000000.0,
            metadata={"session_id": "test123"},
            source_ref="state.db:test123:msg1",
        )
        assert doc.doc_id  # 自动生成
        assert doc.content_hash  # 自动生成
        assert doc.source == "session"
        assert doc.doc_type == "relation"

    def test_ilmdocument_serialization(self):
        from openllm.iai.ilm.base import ILMDocument
        doc = ILMDocument(
            source="session", source_path="test.db",
            content="测试序列化功能", doc_type="relation",
            timestamp=1700000000.0, source_ref="test:ref",
        )
        d = doc.to_dict()
        assert isinstance(d, dict)
        assert d["content"] == "测试序列化功能"

        doc2 = ILMDocument.from_dict(d)
        assert doc2.content == doc.content
        assert doc2.doc_id == doc.doc_id

    def test_doctype_values(self):
        from openllm.iai.ilm.base import DocType
        assert DocType.RELATION.value == "relation"
        assert DocType.TECHNICAL.value == "technical"
        assert DocType.ENGINEERING.value == "engineering"
        assert DocType.KNOWLEDGE.value == "knowledge"


# ═══════════════════════════════════════════════════════
# 3. format_converter核心函数可调用（不依赖真实DB）
# ═══════════════════════════════════════════════════════

class TestFormatConverterLogic:
    """验证format_converter的纯逻辑函数可独立运行"""

    def test_quality_score(self):
        from openllm.iai.ilm.format_converter import _quality_score
        # 短文本低分
        assert _quality_score("hi") < 0.3
        # 有代码块加分
        q = _quality_score("这是一个很长的回答。\n```python\nprint('hello')\n```\n")
        assert q >= 0.6
        # 有结构化内容加分
        q2 = _quality_score("根据分析，核心发现：\n1. 第一点\n2. 第二点\n3. 第三点\n")
        assert q2 >= 0.4  # numbered list gets base score

    def test_correction_pattern(self):
        from openllm.iai.ilm.format_converter import CORRECTION_PATTERN
        assert CORRECTION_PATTERN.search("不对，应该是这样")
        assert CORRECTION_PATTERN.search("错了，重来")
        assert CORRECTION_PATTERN.search("理解错用户意图了")
        assert not CORRECTION_PATTERN.search("今天天气真好")

    def test_cron_filter(self):
        from openllm.iai.ilm.format_converter import _is_cron
        assert _is_cron("[IMPORTANT: You are running as a scheduled cron job...")
        assert not _is_cron("这是一条普通消息")

    def test_auto_message_filter(self):
        from openllm.iai.ilm.format_converter import _is_auto_message
        assert _is_auto_message("[ASYNC] delegation started")
        assert _is_auto_message("[OUT-OF-BAND] user message")
        assert not _is_auto_message("这是真人输入的消息")

    def test_tool_noise_filter(self):
        from openllm.iai.ilm.format_converter import _is_tool_noise
        assert _is_tool_noise("调用工具结果：...")
        assert _is_tool_noise("Tool execution completed")
        assert not _is_tool_noise("请帮我分析这段代码")

    def test_trivial_instruction_filter(self):
        from openllm.iai.ilm.format_converter import TRIVIAL_INSTRUCTION
        assert TRIVIAL_INSTRUCTION.match("好的")
        assert TRIVIAL_INSTRUCTION.match("ok")
        assert TRIVIAL_INSTRUCTION.match("继续")
        assert not TRIVIAL_INSTRUCTION.match("请帮我写一个Python脚本")


# ═══════════════════════════════════════════════════════
# 4. filters/dedup/mixer子模块核心函数可调用
# ═══════════════════════════════════════════════════════

class TestSubmoduleFunctions:
    """验证子模块核心函数可独立调用（使用小样本数据）"""

    def _make_docs(self):
        from openllm.iai.ilm.base import ILMDocument, DocType
        return [
            ILMDocument(
                source="session", source_path="test.db",
                content="用户纠正了模型的回答，这是关系信号。" * 5,
                doc_type=DocType.RELATION.value,
                timestamp=1700000000.0, source_ref="test:1",
            ),
            ILMDocument(
                source="session", source_path="test.db",
                content="架构设计需要考虑扩展性和性能平衡。" * 5,
                doc_type=DocType.TECHNICAL.value,
                timestamp=1700000001.0, source_ref="test:2",
            ),
            ILMDocument(
                source="session", source_path="test.db",
                content="P0任务已完成，测试通过，可以部署上线。" * 5,
                doc_type=DocType.ENGINEERING.value,
                timestamp=1700000002.0, source_ref="test:3",
            ),
        ]

    def test_exact_dedup(self):
        from openllm.iai.ilm.dedup.exact_dedup import exact_dedup
        docs = self._make_docs()
        # 加一条完全重复的
        docs.append(docs[0].__class__(
            source="session", source_path="test.db",
            content=docs[0].content,  # 完全相同
            doc_type=docs[0].doc_type,
            timestamp=1700000010.0, source_ref="test:dup",
        ))
        result = exact_dedup(docs)
        assert len(result) == 3  # 去重后3条

    def test_minhash_dedup(self):
        from openllm.iai.ilm.dedup.minhash_dedup import dedup_documents
        docs = self._make_docs()
        result = dedup_documents(docs, threshold=0.9)
        # 三条内容不同，不应被去重
        assert len(result) >= 3

    def test_heuristic_filter(self):
        from openllm.iai.ilm.filters.heuristic_filter import filter_batch
        docs = self._make_docs()
        filtered, stats = filter_batch(docs, min_len=10)
        assert stats["total"] == 3
        assert stats["passed"] >= 2  # 至少2条通过

    def test_pii_filter(self):
        from openllm.iai.ilm.filters.pii_filter import filter_pii
        from openllm.iai.ilm.base import ILMDocument, DocType
        doc = ILMDocument(
            source="session", source_path="test.db",
            content="请联系张老师 13812345678 获取详情",
            doc_type=DocType.RELATION.value,
            timestamp=1700000000.0, source_ref="test:pii",
        )
        filtered, stats = filter_pii([doc])
        assert stats["pii_masked"] >= 1
        assert "13812345678" not in filtered[0].content

    def test_domain_mixer(self):
        from openllm.iai.ilm.mixer.domain_mixer import mix_documents
        docs = self._make_docs()
        mixed = mix_documents(docs)
        assert len(mixed) >= 2  # ratio control may trim minority types

    def test_classifier(self):
        from openllm.iai.ilm.filters.classifier import classify
        assert classify("不对，你应该这样做。记住以后都要这样做。") == "relation"
        assert classify("架构设计需要考虑D0注意力机制") == "technical"


# ═══════════════════════════════════════════════════════
# 5. 路径参数化验证
# ═══════════════════════════════════════════════════════

class TestPathParameterization:
    """验证路径可通过环境变量覆盖"""

    def test_env_var_override_format_converter(self):
        """format_converter的main()接受env var"""
        os.environ["ILM_STATE_DB"] = "/tmp/test_state.db"
        os.environ["ILM_CORPUS_DB"] = "/tmp/test_corpus.db"
        os.environ["ILM_TRAIN_DATA"] = "/tmp/test_train_data"
        try:
            # 验证env var被读取（不实际运行，只验证默认值）
            from openllm.iai.ilm.format_converter import FormatConverter
            # FormatConverter.__init__直接接受参数，env var在argparse层
            import openllm.iai.ilm.format_converter as fc_mod
            # 读取源码确认env var存在
            src = open(fc_mod.__file__).read()
            assert "ILM_STATE_DB" in src
            assert "ILM_CORPUS_DB" in src
            assert "ILM_TRAIN_DATA" in src
        finally:
            del os.environ["ILM_STATE_DB"]
            del os.environ["ILM_CORPUS_DB"]
            del os.environ["ILM_TRAIN_DATA"]

    def test_env_var_override_pipeline(self):
        """ilm_pipeline.py的PATHS支持env var"""
        import openllm.iai.ilm.ilm_pipeline as pipeline_mod
        src = open(pipeline_mod.__file__).read()
        assert "ILM_STATE_DB" in src
        assert "ILM_RECALL_JSONL" in src
        assert "ILM_JIAK_CARDS" in src

    def test_env_var_override_corpus_writer(self):
        """corpus_writer.py支持env var"""
        import openllm.iai.ilm.writers.corpus_writer as cw_mod
        src = open(cw_mod.__file__).read()
        assert "ILM_CORPUS_DB" in src
        assert "ILM_CORPUS_JSONL" in src


# ═══════════════════════════════════════════════════════
# 6. 原仓文件未被修改验证
# ═══════════════════════════════════════════════════════

class TestOriginalRepoUntouched:
    """验证原仓~/projects/isa/ilm/的文件未被修改"""

    def test_original_base_py_exists(self):
        original = os.path.expanduser("~/projects/isa/ilm/base.py")
        assert os.path.exists(original), "原仓base.py被删除了！"

    def test_original_has_old_style_imports(self):
        """原仓仍然使用from base import（不是from ..base import）"""
        original = os.path.expanduser("~/projects/isa/ilm/sources/session_extractor.py")
        with open(original) as f:
            content = f.read()
        assert "from base import" in content, "原仓session_extractor.py的import被意外修改了！"

    def test_original_format_converter_exists(self):
        original = os.path.expanduser("~/projects/isa/ilm/format_converter.py")
        assert os.path.exists(original)


# ═══════════════════════════════════════════════════════
# 7. 核心端到端：用小样本构造ILMDocument→过滤→去重→配比
# ═══════════════════════════════════════════════════════

class TestEndToEndMiniPipeline:
    """用纯内存小数据跑一遍核心管线（不依赖DB）"""

    def test_mini_pipeline(self):
        from openllm.iai.ilm.base import ILMDocument, DocType
        from openllm.iai.ilm.filters.heuristic_filter import filter_batch
        from openllm.iai.ilm.filters.pii_filter import filter_pii
        from openllm.iai.ilm.dedup.exact_dedup import exact_dedup
        from openllm.iai.ilm.dedup.minhash_dedup import dedup_documents
        from openllm.iai.ilm.mixer.domain_mixer import mix_documents

        # 构造10条测试文档
        docs = []
        contents = [
            ("不对，你说的不对。", DocType.RELATION.value, "session"),
            ("架构设计需要考虑扩展性", DocType.TECHNICAL.value, "session"),
            ("P0任务已完成交付", DocType.ENGINEERING.value, "session"),
            ("Transformer注意力机制的D0维度测量", DocType.TECHNICAL.value, "output"),
            ("记住以后不要这样做", DocType.RELATION.value, "session"),
            ("算法优化需要考虑性能瓶颈", DocType.TECHNICAL.value, "output"),
            ("测试通过，部署上线", DocType.ENGINEERING.value, "session"),
            ("请帮我分析13912345678这个号码", DocType.RELATION.value, "session"),
            ("论文发现了一个新方法", DocType.TECHNICAL.value, "output"),
            ("优先级调整为P1", DocType.ENGINEERING.value, "session"),
        ]
        for i, (content, doc_type, source) in enumerate(contents):
            docs.append(ILMDocument(
                source=source, source_path=f"test:{i}",
                content=content, doc_type=doc_type,
                timestamp=1700000000.0 + i,
                source_ref=f"test:doc:{i}",
            ))

        # Phase 1: 精确去重
        exact = exact_dedup(docs)
        assert len(exact) == 10  # 无重复

        # Phase 2: MinHash去重
        deduped = dedup_documents(exact, threshold=0.9)
        assert len(deduped) >= 8

        # Phase 3: 过滤
        filtered, fstats = filter_batch(deduped, min_len=5)
        assert fstats["passed"] >= 5

        # Phase 4: PII脱敏
        pii_filtered, pii_stats = filter_pii(filtered)

        # Phase 5: 配比
        mixed = mix_documents(pii_filtered)
        assert len(mixed) >= 3

        # 验证配比后各类型都有
        types = set(d.doc_type for d in mixed)
        assert len(types) >= 2  # 至少2种类型
