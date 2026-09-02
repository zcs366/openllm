"""ILM (Inferred Lifetime Memory) data pipeline.

Migrated from ~/projects/isa/ilm/ into openLLM's iai package.
Pure-stdlib data processing: sources → filters → dedup → mixer → writers.

Environment variables for path configuration:
    ILM_STATE_DB          — Hermes state.db path
    ILM_CORPUS_DB         — corpus.db path
    ILM_CORPUS_JSONL      — corpus.jsonl path
    ILM_TRAIN_DATA        — training data output directory
    ILM_RECALL_JSONL      — RECALL.jsonl path
    ILM_JIAK_CARDS        — jiak cards directory
    ILM_HERMES_OUTPUT     — hermes/output path
    ILM_I_OUTPUT          — I: drive output path
    ILM_INCREMENTAL_STATE — incremental state file path
"""

from openllm.iai.ilm.base import ILMDocument, DocType, SourceType
from openllm.iai.ilm.format_converter import FormatConverter, main as format_converter_main
from openllm.iai.ilm.ilm_pipeline import run_pipeline

__all__ = [
    "ILMDocument",
    "DocType",
    "SourceType",
    "FormatConverter",
    "format_converter_main",
    "run_pipeline",
]
