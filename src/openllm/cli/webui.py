"""openLLM Session Web UI — DR-20260927-02。

背景：老搭档 2026-09-26 令 openLLM（TUI 自主）做 session web ui，未交付——
查实：写白名单内零 HTML 落盘，仅 ~/.openllm/output/iai/checkpoints/ 两个
薄 checkpoint（226B，只有 turn_count 无对话正文）。根因：Session 模型
（core/session.py）只存元数据快照，从不持久化对话内容——身体里没这器官。
军师接续完成。

架构（零依赖，stdlib only，与 landlock 哲学一致）：

1. 落盘（由 cli/main.py AgentShell 调用）：
   ~/.openllm/sessions/<id>.jsonl，逐行 JSON：
     {"type":"meta","session_id":...,"started_at":...,"model":...}
     {"type":"msg","role":"user|assistant","content":...,"ts":...}
   id = 启动时间戳 + 短随机（可读）；TUI /clear 轮换新 id（UI 上可见分段）。
   中途断电最多丢半行（load 容错跳过坏行）。

2. 服务（http.server ThreadingHTTPServer，仅绑 127.0.0.1）：
   GET /                   单页（内嵌 HTML/JS，无 CDN，离线可用）
   GET /api/sessions       会话列表（新→旧，title=首条用户消息）
   GET /api/session/<id>   会话消息流
   安全：session id 白名单正则（防路径穿越）；只读服务，无 POST。

3. 入口：
   · TUI 内 /web [port]  —— 守护线程起服务，打印 URL（不断输入循环）
   · python -m openllm.cli.webui [port] —— 独立前台进程（看历史会话）
   端口：OPENLLM_WEBUI_PORT 或 8799。
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

__all__ = [
    "sessions_dir", "start_session_file", "append_msg", "list_sessions",
    "load_session", "serve", "start_background", "BASE_URL",
]

PORT_DEFAULT = 8799
_SID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")  # 防路径穿越
_server = None          # 模块级单例（/web 幂等）
_server_lock = threading.Lock()


def sessions_dir() -> Path:
    """会话落盘目录（测试可用 OPENLLM_SESSIONS_DIR 覆写）。"""
    d = Path(os.environ.get(
        "OPENLLM_SESSIONS_DIR",
        str(Path.home() / ".openllm" / "sessions")))
    d.mkdir(parents=True, exist_ok=True)
    return d


def new_session_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]


def start_session_file(model: str = "") -> Path:
    """新会话：写 meta 行，返回会话文件路径。"""
    sid = new_session_id()
    path = sessions_dir() / f"{sid}.jsonl"
    meta = {"type": "meta", "session_id": sid,
            "started_at": time.time(), "model": model}
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    return path


def append_msg(path, role: str, content: str) -> None:
    """追加一条消息（user/assistant）。path 为 None 时静默跳过
    （测试用 __new__ 裸壳无会话文件的容错）。"""
    if path is None:
        return
    rec = {"type": "msg", "role": role, "content": str(content),
           "ts": time.time()}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def list_sessions() -> list:
    """会话列表（新→旧）。title=首条用户消息前40字；坏行/坏文件跳过。"""
    out = []
    for p in sorted(sessions_dir().glob("*.jsonl"),
                    key=lambda x: x.stat().st_mtime, reverse=True):
        meta, title = {}, ""
        try:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # 断电半行容错
                    if rec.get("type") == "meta":
                        meta = rec
                    elif (rec.get("type") == "msg"
                          and rec.get("role") == "user" and not title):
                        title = str(rec.get("content", ""))[:40]
            if not meta:
                continue
            ts = meta.get("started_at", p.stat().st_mtime)
            out.append({
                "id": meta.get("session_id", p.stem),
                "title": title or "（无用户消息）",
                "meta": time.strftime("%m-%d %H:%M", time.localtime(ts))
                        + f" · {meta.get('model', '?')}",
                "started_at": ts,
            })
        except OSError:
            continue
    return out


def load_session(sid: str):
    """读会话消息流。非法 id / 不存在 → None（HTTP 404）。"""
    if not _SID_RE.match(sid or ""):
        return None
    p = sessions_dir() / f"{sid}.jsonl"
    if not p.exists():
        return None
    msgs = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") == "msg":
                msgs.append({"role": rec.get("role", "?"),
                             "content": rec.get("content", ""),
                             "ts": rec.get("ts")})
    return msgs


# ── HTTP 层 ──

_PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>openLLM · Session Web UI</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--fg:#d7dae0;--dim:#7a8190;
--accent:#4ec9b0;--user:#569cd6}
*{box-sizing:border-box}
body{margin:0;display:flex;height:100vh;background:var(--bg);color:var(--fg);
font:14px/1.6 "Cascadia Code",ui-monospace,Consolas,monospace}
#side{width:280px;background:var(--panel);overflow-y:auto;
border-right:1px solid #262b36;flex-shrink:0}
#side h1{font-size:13px;color:var(--dim);padding:14px;margin:0;
letter-spacing:2px;border-bottom:1px solid #262b36}
.sess{padding:10px 14px;cursor:pointer;border-left:3px solid transparent}
.sess:hover,.sess.on{background:#1d222c}
.sess.on{border-left-color:var(--accent)}
.sess .t{font-size:13px;white-space:nowrap;overflow:hidden;
text-overflow:ellipsis}
.sess .m{font-size:11px;color:var(--dim)}
#main{flex:1;overflow-y:auto;padding:24px 32px}
.msg{max-width:860px;margin:0 auto 18px;padding:10px 16px;border-radius:8px;
white-space:pre-wrap;word-break:break-word}
.msg.user{background:#12233a;border:1px solid #1d3a5f}
.msg.assistant{background:var(--panel);border:1px solid #262b36}
.msg .who{font-size:11px;color:var(--dim);margin-bottom:4px}
.msg.user .who{color:var(--user)}.msg.assistant .who{color:var(--accent)}
#empty{color:var(--dim);text-align:center;margin-top:40vh}
</style></head><body>
<nav id="side"><h1>openLLM · SESSIONS</h1><div id="list"></div></nav>
<main id="main"><div id="empty">← 选择会话查看对话流<br><br>
新对话在 TUI 每轮自动落盘；/clear 开新段</div></main>
<script>
const $=s=>document.querySelector(s);let cur=null;
function fmt(ts){if(!ts)return"";const d=new Date(ts*1000);
return (d.getMonth()+1)+"/"+d.getDate()+" "+
String(d.getHours()).padStart(2,"0")+":"+String(d.getMinutes()).padStart(2,"0");}
async function refresh(){try{
const r=await fetch("/api/sessions");const ss=await r.json();
const el=$("#list");el.innerHTML="";
for(const s of ss){const d=document.createElement("div");
d.className="sess"+(s.id===cur?" on":"");
const t=document.createElement("div");t.className="t";t.textContent=s.title;
const m=document.createElement("div");m.className="m";m.textContent=s.meta;
d.append(t,m);d.onclick=()=>open_(s.id);el.append(d);}}catch(e){}}
async function open_(id){cur=id;refresh();
const r=await fetch("/api/session/"+id);if(!r.ok)return;
const msgs=(await r.json()).messages;const main=$("#main");main.innerHTML="";
if(!msgs.length){const e=document.createElement("div");e.id="empty";
e.textContent="（空会话）";main.append(e);return;}
for(const m of msgs){const d=document.createElement("div");
d.className="msg "+m.role;
const w=document.createElement("div");w.className="who";
w.textContent=m.role==="user"?"你":("openLLM · "+fmt(m.ts));
const c=document.createElement("div");c.className="body";
c.textContent=m.content;d.append(w,c);main.append(d);}
main.scrollTop=main.scrollHeight;}
refresh();setInterval(refresh,5000);
</script></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args, **kwargs):  # 静默——TUI 同进程时防刷屏
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, _PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if self.path == "/api/sessions":
            body = json.dumps(list_sessions(), ensure_ascii=False).encode()
            self._send(200, body, "application/json; charset=utf-8")
            return
        m = re.match(r"^/api/session/([^/]+)$", self.path)
        if m:
            msgs = load_session(m.group(1))
            if msgs is None:
                self._send(404, b'{"error":"not found"}',
                           "application/json")
                return
            body = json.dumps({"id": m.group(1), "messages": msgs},
                              ensure_ascii=False).encode()
            self._send(200, body, "application/json; charset=utf-8")
            return
        self._send(404, b"not found", "text/plain")


def BASE_URL(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def serve(port: int = PORT_DEFAULT):
    """阻塞式起服务（独立进程入口）。"""
    port = int(port)
    print(f"openLLM Session Web UI → {BASE_URL(port)}  (Ctrl-C 退出)")
    ThreadingHTTPServer(("127.0.0.1", port), _Handler).serve_forever()


def start_background(port: int = PORT_DEFAULT) -> str:
    """幂等起守护线程服务（TUI /web 用），返回 URL。"""
    global _server
    with _server_lock:
        if _server is not None:
            return BASE_URL(_server.server_address[1])
        port = int(port)
        try:
            _server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
        except OSError:
            # 端口被占 → 让内核给一个空闲口
            _server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        t = threading.Thread(target=_server.serve_forever, daemon=True)
        t.start()
        return BASE_URL(_server.server_address[1])


if __name__ == "__main__":
    import sys
    serve(int(sys.argv[1]) if len(sys.argv) > 1 else PORT_DEFAULT)
