"""OpenLLM octopus search — v4: unified internal + external + blackbox + MAB routing"""
import json, subprocess, sys, time
from pathlib import Path

JIAK_DIR = Path.home() / ".hermes" / "jiak"
ISA_CARDS = Path.home() / ".hermes" / "isa" / "brain" / "cards"
RECALL_PATH = JIAK_DIR / "RECALL.jsonl"
CARDS_DIR = JIAK_DIR / "cards"
HELPER = str(Path.home() / ".hermes" / "plugins" / "isa_memory" / "_web_helper.py")

# ─── v4: 搜索引擎路径 ───
SEARCH_ENGINE = str(Path.home() / "search-engine")
sys.path.insert(0, SEARCH_ENGINE)

# ─── v4: 延迟加载v4模块 ───
_blackbox = None
_backend_caps = None
_mab_router = None


def _ensure_v4():
    """延迟加载v4模块，失败不影响基本搜索。"""
    global _blackbox, _backend_caps, _mab_router
    if _blackbox is None:
        try:
            from search_blackbox import record_search as _bb
            _blackbox = _bb
        except ImportError:
            _blackbox = False
    if _backend_caps is None:
        try:
            import backend_caps as _bc
            _backend_caps = _bc
        except ImportError:
            _backend_caps = False
    if _mab_router is None:
        try:
            from intent_routing_mab import IntentRouter
            _mab_router = IntentRouter()
        except ImportError:
            _mab_router = False


# ─── 内部搜索（RECALL + jiak + ISA）───

def _search_recall(query, limit=5):
    results = []
    if not RECALL_PATH.exists():
        return results
    q = query.lower()
    for line in RECALL_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            c = rec.get("content", "")
            if q in c.lower():
                results.append({"source": "RECALL", "type": rec.get("type", "?"),
                                "content": c[:300], "ts": rec.get("timestamp", "")})
        except json.JSONDecodeError:
            continue
    return results[:limit]


def _search_jiak(query, limit=5):
    results = []
    if not CARDS_DIR.exists():
        return results
    q = query.lower()
    for cf in CARDS_DIR.glob("*.json"):
        try:
            card = json.loads(cf.read_text(encoding="utf-8"))
            if q in json.dumps(card, ensure_ascii=False).lower():
                kw = card.get("keywords", [])
                hits = sum(1 for k in kw if k.lower() in q)
                results.append({"source": "jiak", "card_id": card.get("card_id", cf.stem),
                                "title": card.get("title", ""), "summary": card.get("summary", "")[:200],
                                "relevance": hits})
        except Exception:
            continue
    results.sort(key=lambda x: x["relevance"], reverse=True)
    return results[:limit]


def _search_isa(query, limit=5):
    results = []
    if not ISA_CARDS.exists():
        return results
    q = query.lower()
    for cf in ISA_CARDS.glob("*.json"):
        try:
            card = json.loads(cf.read_text(encoding="utf-8"))
            if q in json.dumps(card, ensure_ascii=False).lower():
                results.append({"source": "isa", "card_id": card.get("card_id", cf.stem),
                                "title": card.get("title", ""), "summary": card.get("summary", "")[:200]})
        except Exception:
            continue
    return results[:limit]


def _search_web(query, limit=3):
    """外部web搜索（通过hermes_tools）。"""
    Path(HELPER).write_text(
        'import sys,json\n'
        'try:\n'
        '  from hermes_tools import web_search\n'
        '  r=web_search(sys.argv[1],limit=int(sys.argv[2]))\n'
        '  [print(json.dumps(x,ensure_ascii=False)) for x in r.get("data",{}).get("web",[])]\n'
        'except:pass\n'
    )
    results = []
    try:
        r = subprocess.run([sys.executable, HELPER, query, str(limit)],
                           capture_output=True, text=True, timeout=15)
        for line in r.stdout.strip().splitlines():
            if line.strip():
                item = json.loads(line)
                results.append({"source": "web", "title": item.get("title", ""),
                                "url": item.get("url", ""), "desc": item.get("description", "")[:200]})
    except Exception:
        pass
    return results[:limit]


# ─── v4: 带黑匣子的外部搜索 ───

def _search_web_tracked(query, limit=3, backend_name="web_search"):
    """外部web搜索，带黑匣子记录。"""
    _ensure_v4()
    t0 = time.time()
    results = _search_web(query, limit)
    ms = (time.time() - t0) * 1000
    if _blackbox:
        _blackbox(backend=backend_name, query=query, success=len(results) > 0,
                  latency_ms=ms, n_results=len(results))
    return results


# ─── 主搜索函数 ───

def tool_octopus_search(query, limit=5, external=True):
    """Unified search: internal memory + external web. v4: blackbox + MAB routing."""
    _ensure_v4()
    all_r = _search_recall(query, limit) + _search_jiak(query, limit) + _search_isa(query, limit)
    if external:
        all_r.extend(_search_web_tracked(query, limit))

    if not all_r:
        return "No results for '{}'".format(query)

    out = "Octopus search '{}': {} results\n\n".format(query, len(all_r))
    for i, r in enumerate(all_r[:limit * 2], 1):
        s = r["source"]
        if s == "RECALL":
            out += "{}. [RECALL:{}] {}...\n".format(i, r["type"], r["content"][:150])
        elif s == "jiak":
            out += "{}. [card:{}] {} - {}\n".format(i, r["card_id"], r["title"], r["summary"][:120])
        elif s == "isa":
            out += "{}. [ISA:{}] {} - {}\n".format(i, r["card_id"], r["title"], r["summary"][:120])
        elif s == "web":
            out += "{}. [web] {}\n   {}\n".format(i, r["title"], r["url"])
        out += "\n"
    return out


def tool_octopus_self_model():
    """查看章鱼自省状态。"""
    try:
        import importlib.util as _ilu
        _sm_path = Path.home() / ".hermes" / "octopus" / "scripts" / "self_model.py"
        _spec = _ilu.spec_from_file_location("self_model", _sm_path)
        if _spec and _spec.loader:
            _mod = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            self_model = _mod.self_model
            return json.dumps(self_model(), indent=2, ensure_ascii=False, default=str)
        return "self_model.py not found at {}".format(_sm_path)
    except Exception as e:
        return "self_model error: {}".format(e)


# ─── v4: 新增工具 ───

def tool_octopus_search_stats(days=7):
    """查看搜索黑匣子统计——哪些后端好用、哪些经常挂。"""
    _ensure_v4()
    if not _blackbox:
        return "search_blackbox module not available"
    from search_blackbox import get_stats
    stats = get_stats(days=days)
    if stats["total"] == 0:
        return "No search records in last {} days".format(days)

    out = "=== Search Stats ({} days) ===\n".format(days)
    out += "Total: {} | Success: {:.0%} | Avg latency: {:.0f}ms\n\n".format(
        stats["total"], stats["success_rate"], stats["avg_ms"])
    out += "Backend breakdown:\n"
    for be, s in sorted(stats["by_backend"].items(), key=lambda x: x[1]["total"], reverse=True):
        out += "  {}: {} hits | {:.0%} success | {:.0f}ms avg\n".format(
            be, s["total"], s["rate"], s["avg_ms"])
    return out


def tool_octopus_route(query, n=3):
    """根据查询内容推荐最优搜索后端（MAB策略）。"""
    _ensure_v4()
    if not _backend_caps:
        return "backend_caps module not available"

    from backend_caps import classify_query, get_best_backends
    qt = classify_query(query)
    backends = get_best_backends(query, n=n)

    out = "Query type: {}\nRecommended backends (top {}):\n".format(qt, n)
    for i, be in enumerate(backends, 1):
        caps = _backend_caps.BACKEND_CAPS[be]
        score = caps.match_score(qt)
        out += "  {}. {} (score={:.2f})\n".format(i, be, score)

    if _mab_router:
        mab_backends = _mab_router.select_backends(query, n=n)
        out += "\nMAB selection: {}\n".format(mab_backends)
        summary = _mab_router.get_stats_summary()
        if summary:
            out += "MAB stats:\n"
            for be, s in list(summary.items())[:5]:
                out += "  {}: pulls={} reward={:.2f}\n".format(be, s["pulls"], s["avg_reward"])

    return out


def tool_octopus_health():
    """章鱼搜索系统健康检查——所有后端可达性。"""
    _ensure_v4()
    out = "=== Octopus v4 Health Check ===\n\n"

    # 黑匣子状态
    bb_path = Path.home() / ".hermes" / "search_blackbox" / "search_log.jsonl"
    if bb_path.exists():
        lines = bb_path.read_text().strip().splitlines()
        out += "Blackbox: {} records\n".format(len(lines))
    else:
        out += "Blackbox: no records yet\n"

    # 后端能力声明
    if _backend_caps:
        out += "Backend caps: {} backends declared\n".format(len(_backend_caps.BACKEND_CAPS))
    else:
        out += "Backend caps: NOT LOADED\n"

    # MAB路由
    if _mab_router:
        summary = _mab_router.get_stats_summary()
        out += "MAB router: {} backends tracked\n".format(len(summary))
    else:
        out += "MAB router: NOT LOADED\n"

    # jiak卡片
    cards = list(CARDS_DIR.glob("*.json")) if CARDS_DIR.exists() else []
    out += "Jiak cards: {}\n".format(len(cards))

    # RECALL
    if RECALL_PATH.exists():
        lines = RECALL_PATH.read_text().strip().splitlines()
        out += "RECALL entries: {}\n".format(len(lines))
    else:
        out += "RECALL: empty\n"

    return out


if __name__ == "__main__":
    print(tool_octopus_search("ISA memory system"))
