"""
openLLM↔Hermes 通信桥接层。

设计原则：
- openLLM是独立Agent，Hermes是BIOS
- 两者通过HTTP API通信，不互相import
- 双向：openLLM可调Hermes工具，Hermes可触发openLLM任务

Hermes→openLLM: POST /api/task  (Hermes发起任务给openLLM)
openLLM→Hermes: POST /api/hermes/hermes_search  (openLLM调Hermes搜索)
openLLM→Hermes: POST /api/hermes/hermes_web_extract  (openLLM调Hermes提取)
"""

import json
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler


# ── 数据模型 ──

@dataclass
class BridgeMessage:
    """通信消息。"""
    source: str          # "hermes" 或 "openllm"
    target: str          # "hermes" 或 "openllm"
    action: str          # 操作名
    payload: dict = field(default_factory=dict)
    message_id: str = ""
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.message_id:
            self.message_id = f"msg_{int(self.timestamp*1000)}"

    def to_json(self) -> str:
        return json.dumps({
            "source": self.source, "target": self.target,
            "action": self.action, "payload": self.payload,
            "message_id": self.message_id, "timestamp": self.timestamp,
        }, ensure_ascii=False)

    @classmethod
    def from_json(cls, data: str) -> "BridgeMessage":
        d = json.loads(data)
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k) or k in ("source","target","action","payload","message_id","timestamp")})


@dataclass
class BridgeResponse:
    """通信响应。"""
    ok: bool
    data: Any = None
    error: str = ""
    message_id: str = ""

    def to_json(self) -> str:
        return json.dumps({
            "ok": self.ok, "data": self.data,
            "error": self.error, "message_id": self.message_id,
        }, ensure_ascii=False, default=str)


# ── Hermes客户端（openLLM调Hermes） ──

class HermesClient:
    """
    openLLM调Hermes的客户端。

    两种调用方式：
    1. HTTP模式：通过Hermes Web UI API（需要Hermes server运行）
    2. 文件模式：写信号文件到共享目录（Hermes通过cron消费）
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8648",
                 signal_dir: Optional[Path] = None):
        self.base_url = base_url.rstrip("/")
        self.signal_dir = signal_dir or Path.home() / ".openllm" / "signals"
        self.signal_dir.mkdir(parents=True, exist_ok=True)

    def search(self, query: str, sources: str = "web_search") -> dict:
        """调Hermes搜索。"""
        return self._call_hermes("hermes_search", {
            "query": query, "sources": sources
        })

    def web_extract(self, urls: list[str]) -> dict:
        """调Hermes网页提取。"""
        return self._call_hermes("hermes_web_extract", {"urls": urls})

    def send_signal(self, action: str, payload: dict) -> str:
        """文件模式：写信号文件到共享目录。"""
        msg = BridgeMessage(source="openllm", target="hermes",
                           action=action, payload=payload)
        signal_path = self.signal_dir / f"{msg.message_id}.json"
        signal_path.write_text(msg.to_json())
        return msg.message_id

    def _call_hermes(self, action: str, payload: dict) -> dict:
        """HTTP模式调Hermes API。"""
        try:
            import requests
            resp = requests.post(
                f"{self.base_url}/api/action",
                json={"action": action, "payload": payload},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            # 降级到文件模式
            msg_id = self.send_signal(action, payload)
            return {"ok": False, "error": str(e),
                    "fallback": "signal_file", "signal_id": msg_id}


# ── Hermes服务端（Hermes调openLLM） ──

class OpenLLMService:
    """
    openLLM对外暴露的HTTP API服务。

    端点：
    - POST /api/task: 接收Hermes发来的任务
    - GET /api/status: 返回openLLM状态
    - GET /api/research: 返回研究循环状态
    """

    def __init__(self, agent=None, port: int = 8199):
        self.agent = agent  # openLLM Agent实例
        self.port = port
        self._results: dict[str, Any] = {}

    def start(self, background: bool = True) -> Optional[int]:
        """启动HTTP服务。"""
        service = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path == "/api/task":
                    content_len = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(content_len)
                    data = json.loads(body)
                    # 异步执行任务
                    task_id = f"task_{int(time.time()*1000)}"
                    result = service._handle_task(data)
                    service._results[task_id] = result
                    resp = BridgeResponse(ok=True, data={"task_id": task_id, "result": result})
                else:
                    resp = BridgeResponse(ok=False, error=f"Unknown path: {self.path}")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(resp.to_json().encode())

            def do_GET(self):
                if self.path == "/api/status":
                    data = {"status": "running", "agent": bool(service.agent)}
                    resp = BridgeResponse(ok=True, data=data)
                elif self.path == "/api/research":
                    data = service.agent.research.status() if service.agent else {}
                    resp = BridgeResponse(ok=True, data=data)
                else:
                    resp = BridgeResponse(ok=False, error=f"Unknown path: {self.path}")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(resp.to_json().encode())

            def log_message(self, format, *args):
                pass  # 静默

        server = HTTPServer(("127.0.0.1", self.port), Handler)
        if background:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            return self.port
        else:
            server.serve_forever()

    def _handle_task(self, data: dict) -> dict:
        """处理Hermes发来的任务。"""
        action = data.get("action", "")
        payload = data.get("payload", {})

        if action == "run_once":
            message = payload.get("message", "")
            result = self.agent.run_once(message) if self.agent else ""
            return {"output": result}

        elif action == "research_status":
            return self.agent.research.status() if self.agent else {}

        elif action == "research_add_hypothesis":
            h = self.agent.research.hypothesize(
                payload["claim"], payload["prediction"])
            return h.dict()

        return {"error": f"Unknown action: {action}"}


# ── 信号消费器（Hermes cron消费openLLM信号） ──

class SignalConsumer:
    """
    消费openLLM写入共享目录的信号文件。
    配合Hermes cron job使用。
    """

    def __init__(self, signal_dir: Optional[Path] = None):
        self.signal_dir = signal_dir or Path.home() / ".openllm" / "signals"
        self._handlers: dict[str, Callable] = {}

    def register(self, action: str, handler: Callable):
        """注册信号处理器。"""
        self._handlers[action] = handler

    def consume(self) -> list[dict]:
        """消费所有未处理的信号。"""
        results = []
        for signal_file in sorted(self.signal_dir.glob("*.json")):
            try:
                msg = BridgeMessage.from_json(signal_file.read_text())
                handler = self._handlers.get(msg.action)
                if handler:
                    result = handler(msg.payload)
                    results.append({"signal_id": msg.message_id,
                                   "action": msg.action, "result": result})
                else:
                    results.append({"signal_id": msg.message_id,
                                   "action": msg.action, "result": "no handler"})
                # 处理完移入consumed目录
                consumed_dir = self.signal_dir / "consumed"
                consumed_dir.mkdir(exist_ok=True)
                signal_file.rename(consumed_dir / signal_file.name)
            except Exception as e:
                results.append({"signal_id": signal_file.stem,
                               "error": str(e)})
        return results
