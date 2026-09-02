"""
ilm_pipeline.py — ILM总调度器
支持全数据源 + 增量模式
"""
import os
import sys
import time
import argparse
import hashlib



from .base import ILMDocument
from .sources.session_extractor import extract_sessions, get_session_stats
from .sources.recall_extractor import extract_recall
from .sources.jiak_extractor import extract_jiak
from .sources.output_extractor import extract_output
from .filters.heuristic_filter import filter_batch
from .filters.pii_filter import filter_pii
from .dedup.exact_dedup import exact_dedup
from .dedup.minhash_dedup import dedup_documents
from .mixer.domain_mixer import mix_documents, get_domain_stats
from .writers.corpus_writer import CorpusWriter

# 数据源路径
PATHS = {
    "state_db": os.environ.get("ILM_STATE_DB", os.path.expanduser("~/.hermes/state.db")),
    "recall_jsonl": os.environ.get("ILM_RECALL_JSONL", os.path.expanduser("~/.hermes/jiak/RECALL.jsonl")),
    "jiak_cards": os.environ.get("ILM_JIAK_CARDS", os.path.expanduser("~/.hermes/jiak/cards")),
    "hermes_output": os.environ.get("ILM_HERMES_OUTPUT", os.path.expanduser("~/hermes/output")),
    "i_hermes_output": os.environ.get("ILM_I_OUTPUT", os.path.expanduser("/mnt/i/hermes/output")),
}

# 增量状态文件
INCREMENTAL_STATE = os.environ.get('ILM_INCREMENTAL_STATE', os.path.expanduser('~/projects/isa/ilm/.incremental_state.json'))


def _load_incremental_state() -> dict:
    """加载增量清洗状态（上次处理到哪条）"""
    import json
    if os.path.exists(INCREMENTAL_STATE):
        with open(INCREMENTAL_STATE) as f:
            return json.load(f)
    return {}


def _save_incremental_state(state: dict):
    """保存增量清洗状态"""
    import json
    with open(INCREMENTAL_STATE, 'w') as f:
        json.dump(state, f, indent=2)



def run_pipeline(sources: list = None,
                 max_docs: int = 0,
                 dedup_threshold: float = 0.7,
                 min_content_len: int = 10,
                 verbose: bool = True,
                 incremental: bool = False) -> dict:
    """
    运行ILM清洗管线
    
    铁律：
    - 增量模式：只处理新增，不做全量重建
    - source_ref永远填充
    - 三类数据三套规则
    """
    start_time = time.time()
    report = {"start_time": start_time, "sources": {}, "pipeline": {}, "final": {}}
    
    if sources is None:
        sources = ["session"]
    
    # 增量状态
    state = _load_incremental_state() if incremental else {}
    seen_hashes = set(state.get("seen_hashes", []))
    
    # ========== Phase 1: 提取 ==========
    all_docs = []
    
    if "session" in sources:
        db = PATHS["state_db"]
        if os.path.exists(db):
            stats = get_session_stats(db)
            report["sources"]["session"] = stats
            if verbose:
                print(f"\n📦 Session: {stats.get('total_messages',0)} msgs ({stats.get('messages_ge_10chars',0)} ≥10ch)")
            docs = list(extract_sessions(db, min_content_len=min_content_len, max_docs=max_docs))
            all_docs.extend(docs)
            if verbose:
                print(f"  → {len(docs)} docs")
    
    if "recall" in sources:
        path = PATHS["recall_jsonl"]
        if os.path.exists(path):
            docs = list(extract_recall(path, min_content_len=min_content_len, max_docs=max_docs))
            all_docs.extend(docs)
            if verbose:
                print(f"📦 RECALL: → {len(docs)} docs")
    
    if "jiak" in sources:
        path = PATHS["jiak_cards"]
        if os.path.exists(path):
            docs = list(extract_jiak(path, min_content_len=min_content_len, max_docs=max_docs))
            all_docs.extend(docs)
            if verbose:
                print(f"📦 jiak: → {len(docs)} docs")
    
    if "output" in sources:
        dirs = [PATHS["hermes_output"]]
        if os.path.exists(PATHS["i_hermes_output"]):
            dirs.append(PATHS["i_hermes_output"])
        docs = list(extract_output(dirs, min_content_len=min_content_len, max_docs=max_docs))
        all_docs.extend(docs)
        if verbose:
            print(f"📦 output: → {len(docs)} docs")
    
    # 增量过滤：跳过已处理的
    if incremental and seen_hashes:
        before = len(all_docs)
        all_docs = [d for d in all_docs if d.content_hash not in seen_hashes]
        if verbose:
            print(f"\n🔄 Incremental: {before} → {len(all_docs)} new docs")
    
    report["pipeline"]["extracted"] = len(all_docs)
    if verbose:
        print(f"\n📊 Total: {len(all_docs)}")
        if all_docs:
            print(f"  Types: {get_domain_stats(all_docs)}")
    
    # ========== Phase 2: 去重（先去重，不对重复数据做两次过滤） ==========
    # SFT数据构建铁律：先去重→再过滤→再配比
    exact_deduped = exact_dedup(all_docs)
    deduped = dedup_documents(exact_deduped, threshold=dedup_threshold)
    report["pipeline"]["deduped"] = len(deduped)
    if verbose:
        print(f"\n🗑️ Dedup: {len(all_docs)} → {len(exact_deduped)} (exact) → {len(deduped)} (minhash)")

    # ========== Phase 3: 过滤 ==========
    filtered, fstats = filter_batch(deduped, min_len=min_content_len)
    report["pipeline"]["filtered"] = fstats
    if verbose:
        print(f"\n🔍 Filter: {fstats['passed']}/{fstats['total']} passed")

    # ========== Phase 3.5: PII脱敏 ==========
    filtered, pii_stats = filter_pii(filtered)
    report["pipeline"]["pii"] = pii_stats

    # ========== Phase 4: 配比 ==========
    mixed = mix_documents(filtered)
    report["pipeline"]["mixed"] = len(mixed)
    if verbose:
        print(f"\n⚖️ Mixed: {len(mixed)} ({get_domain_stats(mixed)})")
    
    # ========== Phase 5: 写入 ==========
    writer = CorpusWriter()
    wstats = writer.write_batch(mixed)
    report["pipeline"]["written"] = wstats
    
    # 更新增量状态
    if incremental:
        for doc in mixed:
            seen_hashes.add(doc.content_hash)
        state["seen_hashes"] = list(seen_hashes)[-50000:]  # 保留最近5万条
        state["last_run"] = time.time()
        _save_incremental_state(state)
    
    corpus_stats = writer.get_stats()
    report["final"] = corpus_stats
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"\n✅ Done in {elapsed:.1f}s | Corpus: {corpus_stats.get('total',0)} docs")
    
    report["elapsed"] = elapsed
    return report


def main():
    parser = argparse.ArgumentParser(description="ILM Data Cleaning Pipeline")
    parser.add_argument("--sources", nargs="+", default=["session"],
                        choices=["session", "recall", "jiak", "output", "all"],
                        help="Data sources")
    parser.add_argument("--max-docs", type=int, default=0,
                        help="Max docs per source (0=unlimited)")
    parser.add_argument("--dedup-threshold", type=float, default=0.7)
    parser.add_argument("--min-len", type=int, default=10)
    parser.add_argument("--incremental", action="store_true",
                        help="Incremental mode: only process new data")
    parser.add_argument("--quiet", action="store_true")
    
    args = parser.parse_args()
    if "all" in args.sources:
        args.sources = ["session", "recall", "jiak", "output"]
    
    report = run_pipeline(
        sources=args.sources,
        max_docs=args.max_docs,
        dedup_threshold=args.dedup_threshold,
        min_content_len=args.min_len,
        verbose=not args.quiet,
        incremental=args.incremental,
    )
    
    import json
    rp = os.path.expanduser("~/projects/isa/ilm/pipeline_report.json")
    with open(rp, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"📄 Report: {rp}")


if __name__ == "__main__":
    main()
