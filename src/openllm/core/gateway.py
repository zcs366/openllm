"""
openLLM Gateway — ISA中央网关。

架构吸收：
  OpenClaw — 插件系统热加载channel adapter，每个平台dedicated adapter
  Hermes   — 单gateway进程，session key路由，跨channel共享记忆
  张成市    — "输入即记忆"，"做伪装者/潜入者，不靠大厂API"

三层分工（与OpenClaw同构）：
  ChannelAdapter — 薄适配器（格式转换+隐身协议）
  Gateway        — 路由层（排队+session key+上下文管理）
  Engine         — 智能层（LLM+工具+记忆）

设计原则：
  1. 不依赖大厂官方API——用反向协议+浏览器自动化+cookie认证
  2. 做潜入者/伪装者——像元宝一样无缝接入微信/钉钉/飞书
  3. 插件式添加频道——继承ChannelAdapter，3个方法搞定
  4. Session key统一路由——跨channel共享记忆和身份
  5. 输入即记忆——每条消息写入MemoryOS
"""

import time
import uuid
import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("openllm.gateway")


# ── 消息标准格式 ────────────────────────────────────

class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ChannelMessage:
    """
    从任何频道进入的标准消息格式。

    Session key格式（吸收Hermes）：
      agent:main:{channel}:{scope}:{user_id}
      例: agent:main:telegram:private:123456789
    """
    content: str
    channel: str           # 频道标识: "cli" / "telegram" / "wechat" / "dingtalk"
    user_id: str = ""      # 发送者ID（频道内唯一）
    session_id: str = ""   # 跨频道会话ID
    message_id: str = ""   # 消息ID（去重用）
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)  # 频道特定元数据

    def __post_init__(self):
        if not self.message_id:
            self.message_id = str(uuid.uuid4())[:8]
        if not self.session_id:
            self.session_id = f"agent:main:{self.channel}:private:{self.user_id}"

    @property
    def scope(self) -> str:
        """从session_id提取scope。"""
        parts = self.session_id.split(":")
        return parts[3] if len(parts) >= 4 else "private"


# ── 频道适配器基类 ──────────────────────────────────

class ChannelAdapter:
    """
    频道适配器基类。

    每个平台继承此类，实现3个方法：
    - receive(): 平台消息 → ChannelMessage
    - send(): 响应 → 平台格式并发送
    - health(): 健康检查

    添加新频道的流程：
    1. 继承ChannelAdapter
    2. 实现receive/send/health
    3. gateway.register(MyAdapter())

    不做的事（交给Gateway/Engine）：
    - 不做消息排队
    - 不做LLM调用
    - 不做记忆写入
    """

    def __init__(self, channel_name: str, stealth: bool = False):
        """
        Args:
            channel_name: 频道标识（如 "telegram", "wechat", "dingtalk"）
            stealth: 是否为隐身模式（潜入者/伪装者）
        """
        self.channel_name = channel_name
        self.stealth = stealth
        self._gateway: Optional["Gateway"] = None
        self._authenticated = False

    def bind(self, gateway: "Gateway"):
        """绑定到Gateway。"""
        self._gateway = gateway

    def send(self, channel_message: ChannelMessage, response: str) -> bool:
        """
        将响应发送到频道。

        子类必须实现：将response转为平台格式并发送。

        Returns: 发送成功与否
        """
        raise NotImplementedError

    def receive(self, raw_data: Any) -> Optional[ChannelMessage]:
        """
        将平台原始数据转为ChannelMessage。

        子类必须实现：解析平台特定格式。

        Returns: ChannelMessage 或 None（无效消息）
        """
        raise NotImplementedError

    def health(self) -> dict:
        """
        健康检查。子类可选实现。

        Returns: {"status": "ok"/"error"/"auth_required", "detail": "..."}
        """
        return {"status": "ok", "detail": "base adapter"}

    def on_message(self, raw_data: Any):
        """
        收到平台消息时的入口。

        标准流程：receive() → Gateway.enqueue()
        """
        msg = self.receive(raw_data)
        if msg and self._gateway:
            self._gateway.enqueue(msg)


# ── 潜入者适配器基类 ────────────────────────────────

class StealthChannelAdapter(ChannelAdapter):
    """
    潜入者/伪装者适配器基类。

    核心思路：不用大厂官方API（要审批），用反向协议+浏览器自动化+cookie认证。
    像淘宝元宝一样，以"普通用户"身份接入微信/钉钉/飞书。

    技术方案（按优先级）：
    1. 反向协议（最轻量，如微信web协议）
    2. 浏览器自动化（headless Chrome + stealth）
    3. Cookie注入（从已登录浏览器提取cookie）
    4. 桌面Hook（注入消息接收钩子）

    安全原则：
    - 频率限制：不超过正常用户行为
    - 消息过滤：只处理@提及或关键词触发
    - 账号隔离：用小号而非主号
    """

    def __init__(self, channel_name: str):
        super().__init__(channel_name, stealth=True)
        self._cookie_path: Optional[Path] = None
        self._rate_limit_ms: int = 1000  # 最小消息间隔
        self._last_message_time: float = 0.0

    def _check_rate_limit(self) -> bool:
        """频率限制检查。"""
        now = time.time()
        if (now - self._last_message_time) * 1000 < self._rate_limit_ms:
            logger.warning(f"频率限制: {self.channel_name} 消息过快")
            return False
        self._last_message_time = now
        return True

    def _load_cookies(self) -> dict:
        """从文件加载cookie。"""
        if not self._cookie_path or not self._cookie_path.exists():
            return {}
        try:
            import json
            return json.loads(self._cookie_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Cookie加载失败: {e}")
            return {}

    def _save_cookies(self, cookies: dict):
        """保存cookie到文件。"""
        if not self._cookie_path:
            return
        try:
            import json
            self._cookie_path.parent.mkdir(parents=True, exist_ok=True)
            self._cookie_path.write_text(
                json.dumps(cookies, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"Cookie保存失败: {e}")

    def health(self) -> dict:
        cookies = self._load_cookies()
        return {
            "status": "ok" if cookies else "auth_required",
            "detail": f"stealth adapter, cookies={'loaded' if cookies else 'missing'}",
        }


# ── CLI适配器 ───────────────────────────────────────

class CLIAdapter(ChannelAdapter):
    """终端CLI适配器。最简单的频道——直接print/stdin。"""

    def __init__(self):
        super().__init__("cli", stealth=False)

    def send(self, channel_message: ChannelMessage, response: str) -> bool:
        print(response)
        return True

    def receive(self, raw_data: Any) -> Optional[ChannelMessage]:
        if isinstance(raw_data, str) and raw_data.strip():
            return ChannelMessage(
                content=raw_data.strip(),
                channel="cli",
                user_id="local",
            )
        return None


# ── Webhook适配器（通用HTTP接入） ──────────────────

class WebhookAdapter(ChannelAdapter):
    """
    通用Webhook适配器。

    用途：任何支持HTTP回调的平台（如Telegram Bot API、Discord Webhook）。
    也适用于自建频道——提供一个HTTP端点，平台POST消息过来。

    使用方式：
        adapter = WebhookAdapter("mybot", port=8080)
        adapter.on_receive = my_callback  # 收到消息时的回调
        gateway.register(adapter)
    """

    def __init__(self, channel_name: str, port: int = 8080):
        super().__init__(channel_name, stealth=False)
        self.port = port
        self.on_receive: Optional[Callable] = None
        self._buffer: list = []

    def send(self, channel_message: ChannelMessage, response: str) -> bool:
        """Webhook响应：写入buffer供HTTP端点返回。"""
        self._buffer.append({
            "message_id": channel_message.message_id,
            "response": response,
            "timestamp": time.time(),
        })
        return True

    def receive(self, raw_data: Any) -> Optional[ChannelMessage]:
        if isinstance(raw_data, dict):
            return ChannelMessage(
                content=raw_data.get("text", ""),
                channel=self.channel_name,
                user_id=raw_data.get("user_id", "unknown"),
                metadata=raw_data,
            )
        return None

    def pop_response(self) -> Optional[dict]:
        """弹出一条待发送响应（供HTTP端点消费）。"""
        return self._buffer.pop(0) if self._buffer else None


# ── Gateway 核心 ────────────────────────────────────

class Gateway:
    """
    ISA中央网关。

    吸收 OpenClaw（插件式adapter）+ Hermes（session key路由）。
    张成市设计："输入即记忆，频道是薄适配器，ISA是中央网关。"

    核心职责：
    1. 插件式频道注册（3方法搞定）
    2. Session key统一路由（跨channel共享记忆）
    3. 消息排队 + 上下文窗口管理
    4. 输入即记忆（每条消息写入MemoryOS）
    5. 响应路由回正确频道

    不做的事：
    - 不做LLM调用（交给Engine）
    - 不做格式转换（交给ChannelAdapter）
    - 不做记忆检索（交给MemoryOS）
    """

    def __init__(self, engine=None, max_context_messages: int = 100):
        self._engine = engine
        self._adapters: dict[str, ChannelAdapter] = {}
        self._sessions: dict[str, list] = {}  # session_key → [ChannelMessage]
        self._queue: list = []
        self._max_context = max_context_messages
        self._lock = threading.Lock()
        self._processing = False
        self._total_processed = 0

    # ── 频道注册（插件式，吸收OpenClaw） ────────────

    def register(self, adapter: ChannelAdapter):
        """
        注册频道适配器。一行搞定。

        用法：
            gateway.register(CLIAdapter())
            gateway.register(TelegramAdapter(token="xxx"))
            gateway.register(WeChatStealthAdapter(cookie_path="..."))
        """
        adapter.bind(self)
        self._adapters[adapter.channel_name] = adapter
        mode = " [隐身]" if adapter.stealth else ""
        logger.info(f"频道注册: {adapter.channel_name}{mode}")

    def unregister(self, channel: str):
        """注销频道。"""
        self._adapters.pop(channel, None)

    def get_adapter(self, channel: str) -> Optional[ChannelAdapter]:
        return self._adapters.get(channel)

    @property
    def channels(self) -> list[str]:
        return list(self._adapters.keys())

    @property
    def stealth_channels(self) -> list[str]:
        """已注册的隐身频道。"""
        return [c for c, a in self._adapters.items() if a.stealth]

    # ── 消息队列 ────────────────────────────────────

    def enqueue(self, message: ChannelMessage):
        """
        消息入队。
        输入即记忆：同时写入MemoryOS。
        """
        with self._lock:
            self._queue.append(message)

        # 输入即记忆
        if self._engine and hasattr(self._engine, "unified_memory"):
            self._engine.unified_memory.remember(
                key=f"input_{message.message_id}",
                value={
                    "channel": message.channel,
                    "session": message.session_id,
                    "content": message.content[:500],
                    "timestamp": message.timestamp,
                },
                importance=0.3,
            )

    def enqueue_text(self, text: str, channel: str = "cli",
                     user_id: str = "local") -> ChannelMessage:
        """快捷入队。"""
        msg = ChannelMessage(content=text, channel=channel, user_id=user_id)
        self.enqueue(msg)
        return msg

    @property
    def queue_size(self) -> int:
        with self._lock:
            return len(self._queue)

    # ── Session路由（吸收Hermes） ───────────────────

    def _get_session(self, message: ChannelMessage) -> list:
        """获取或创建session（基于session_id）。"""
        key = message.session_id
        if key not in self._sessions:
            self._sessions[key] = []
        return self._sessions[key]

    def get_session_history(self, session_id: str) -> list:
        """获取session历史。"""
        return self._sessions.get(session_id, [])

    def clear_session(self, session_id: str):
        """清空session。"""
        self._sessions.pop(session_id, None)

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)

    # ── 处理循环 ────────────────────────────────────

    def process_next(self) -> Optional[str]:
        """
        处理队列中的下一条消息。

        Returns: 响应文本，或None
        """
        with self._lock:
            if self._processing or not self._queue:
                return None
            msg = self._queue.pop(0)
            self._processing = True

        # 注意：锁已释放，LLM调用可能耗时较长。
        # _processing标记确保同一时间只处理一条消息。
        try:
            # Session路由
            session = self._get_session(msg)
            session.append(msg)

            # 调用Engine
            if self._engine:
                response = self._engine.chat(msg.content)
            else:
                response = f"[Gateway无Engine] 收到: {msg.content}"

            # 记录响应到session
            from .provider import Message
            session.append(Message(role="assistant", content=response))

            # 路由响应回频道
            adapter = self._adapters.get(msg.channel)
            if adapter:
                adapter.send(msg, response)

            self._total_processed += 1
            return response

        except Exception as e:
            logger.error(f"处理消息失败: {e}")
            return f"[Gateway错误] {e}"

        finally:
            with self._lock:
                self._processing = False

    def process_all(self) -> list[str]:
        """处理队列中所有消息。"""
        responses = []
        while True:
            r = self.process_next()
            if r is None:
                break
            responses.append(r)
        return responses

    # ── 状态 ────────────────────────────────────────

    def status(self) -> dict:
        return {
            "channels": self.channels,
            "stealth_channels": self.stealth_channels,
            "queue_size": self.queue_size,
            "active_sessions": self.active_sessions,
            "total_processed": self._total_processed,
            "processing": self._processing,
        }

    def health_check(self) -> dict:
        """所有频道健康检查。"""
        return {
            ch: adapter.health()
            for ch, adapter in self._adapters.items()
        }

    def __repr__(self) -> str:
        return (
            f"Gateway(channels={len(self._adapters)}, "
            f"stealth={len(self.stealth_channels)}, "
            f"queue={self.queue_size}, sessions={self.active_sessions})"
        )
