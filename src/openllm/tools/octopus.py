"""OpenLLM octopus search - unified internal + external"""
import json, subprocess, sys
from pathlib import Path

JIAK_DIR = Path.home() / ".hermes" / "jiak"
ISA_CARDS = Path.home() / ".hermes" / "isa" / "brain" / "cards"
RECALL_PATH = JIAK_DIR / "RECALL.jsonl"
CARDS_DIR = JIAK_DIR / "cards"
HELPER = str(Path.home() / ".hermes" / "plugins" / "isa_memory" / "_web_helper.py")


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


def _search_internal(query, limit=5):
    r = _search_recall(query, limit) + _search_jiak(query, limit) + _search_isa(query, limit)
    return r[:limit * 2]


def _search_web(query, limit=3):
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


def tool_octopus_search(query, limit=5, external=True):
    """Unified search: internal memory + external web."""
    all_r = _search_recall(query, limit) + _search_jiak(query, limit) + _search_isa(query, limit)
    if external:
        all_r.extend(_search_web(query, limit))

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
    try:
        import importlib.util as _ilu
        _sm_path = Path.home() / ".hermes" / "octopus" / "scripts" / "self_model.py"
        _spec = _ilu.spec_from_file_location("self_model", _sm_path)
        if _spec and _spec.loader:
            _mod = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            self_model = _mod.self_model
        return json.dumps(self_model(), indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        return "self_model error: {}".format(e)


if __name__ == "__main__":
    print(tool_octopus_search("ISA memory system"))
