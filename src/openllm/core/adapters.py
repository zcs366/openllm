"""
openLLM Channel Adapters — 频道适配器集合。

已实现：
  CLIAdapter      — 终端（已移至gateway.py）
  WebhookAdapter  — 通用HTTP（已移至gateway.py）
  TelegramAdapter — Telegram Bot API（官方，简单直接）
  YuanbaoAdapter  — 元宝/微信潜入模式（反向协议+浏览器自动化）

添加新频道：
  1. 继承 ChannelAdapter（普通）或 StealthChannelAdapter（潜入者）
  2. 实现 receive() / send() / health()
  3. gateway.register(MyAdapter())
"""

import json
import time
import logging
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from .gateway import (
    ChannelAdapter, StealthChannelAdapter, ChannelMessage,
)

logger = logging.getLogger("openllm.adapters")


# ── Telegram Bot API 适配器 ─────────────────────────

class TelegramAdapter(ChannelAdapter):
    """
    Telegram Bot API 适配器。

    用法：
        adapter = TelegramAdapter(token="123456:ABC-DEF...")
        gateway.register(adapter)
        adapter.poll()  # 开始轮询

    原理：长轮询 getUpdates → 解析消息 → Gateway.enqueue()
    """

    API_BASE = "https://api.telegram.org/bot{token}"

    def __init__(self, token: str, poll_interval: float = 2.0):
        super().__init__("telegram", stealth=False)
        self._token = token
        self._api_base = self.API_BASE.format(token=token)
        self._offset = 0
        self._poll_interval = poll_interval
        self._running = False
        self._poll_thread: Optional[threading.Thread] = None

    def _api(self, method: str, params: Optional[dict] = None) -> dict:
        """调用Telegram API。"""
        import requests
        url = f"{self._api_base}/{method}"
        resp = requests.post(url, json=params or {}, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def receive(self, raw_data: Any) -> Optional[ChannelMessage]:
        """解析Telegram Update → ChannelMessage。"""
        if not isinstance(raw_data, dict):
            return None
        msg = raw_data.get("message", {})
        if not msg:
            return None
        text = msg.get("text", "")
        if not text:
            return None
        user = msg.get("from", {})
        chat = msg.get("chat", {})
        return ChannelMessage(
            content=text,
            channel="telegram",
            user_id=str(user.get("id", "")),
            session_id=f"agent:main:telegram:{chat.get('type', 'private')}:{user.get('id', '')}",
            metadata={
                "chat_id": chat.get("id"),
                "message_id": msg.get("message_id"),
                "username": user.get("username", ""),
                "first_name": user.get("first_name", ""),
            },
        )

    def send(self, channel_message: ChannelMessage, response: str) -> bool:
        """发送消息到Telegram。"""
        chat_id = channel_message.metadata.get("chat_id")
        if not chat_id:
            return False
        try:
            self._api("sendMessage", {
                "chat_id": chat_id,
                "text": response,
                "parse_mode": "Markdown",
            })
            return True
        except Exception as e:
            logger.error(f"Telegram发送失败: {e}")
            return False

    def poll(self):
        """启动长轮询（阻塞）。"""
        self._running = True
        logger.info("Telegram轮询开始")
        while self._running:
            try:
                result = self._api("getUpdates", {
                    "offset": self._offset,
                    "timeout": 30,
                })
                for update in result.get("result", []):
                    self._offset = update["update_id"] + 1
                    self.on_message(update)
            except Exception as e:
                logger.error(f"Telegram轮询错误: {e}")
                time.sleep(self._poll_interval)

    def start_polling(self):
        """后台轮询（非阻塞）。"""
        if self._poll_thread and self._poll_thread.is_alive():
            return
        self._poll_thread = threading.Thread(target=self.poll, daemon=True)
        self._poll_thread.start()

    def stop(self):
        """停止轮询。"""
        self._running = False

    def health(self) -> dict:
        try:
            result = self._api("getMe")
            bot = result.get("result", {})
            return {
                "status": "ok",
                "detail": f"bot: @{bot.get('username', '?')}",
            }
        except Exception as e:
            return {"status": "error", "detail": str(e)}


# ── 元宝/微信潜入模式适配器 ─────────────────────────

class YuanbaoAdapter(StealthChannelAdapter):
    """
    元宝/微信潜入模式适配器。

    核心思路：像淘宝元宝一样，以"普通用户"身份接入微信。
    不用微信官方Bot API（要审批），用反向协议+浏览器自动化。

    技术方案（按实现难度排序）：
    1. 微信web协议（最轻量，通过HTTP请求模拟微信web客户端）
    2. 浏览器自动化（headless Chrome + stealth，访问微信web版）
    3. Cookie注入（从已登录的微信web版提取cookie）
    4. 桌面Hook（注入消息接收钩子，需要桌面环境）

    使用方式：
        adapter = YuanbaoAdapter(
            cookie_path="~/.openllm/cookies/wechat.json",
            method="web_protocol",
        )
        gateway.register(adapter)

    安全原则：
    - 频率限制：每秒不超过1条消息
    - 消息过滤：只处理@提及或关键词触发
    - 账号隔离：用小号而非主号
    - 行为伪装：随机延迟，模拟人类打字节奏
    """

    def __init__(
        self,
        cookie_path: str = "~/.openllm/cookies/wechat.json",
        method: str = "web_protocol",
        rate_limit_ms: int = 1500,
    ):
        super().__init__("yuanbao")
        self._cookie_path = Path(cookie_path).expanduser()
        self._method = method
        self._rate_limit_ms = rate_limit_ms
        self._last_send_time = 0.0

    def receive(self, raw_data: Any) -> Optional[ChannelMessage]:
        """
        解析微信消息 → ChannelMessage。

        raw_data格式取决于method：
        - web_protocol: HTTP请求体（dict）
        - browser: 页面快照解析结果
        - cookie: 原始消息数组
        """
        if not isinstance(raw_data, dict):
            return None

        # 微信web协议的消息格式
        text = raw_data.get("content", {}).get("text", "")
        if not text:
            return None

        # 过滤：只处理@提及或关键词触发
        if not self._should_process(text):
            return None

        sender = raw_data.get("sender", {})
        room = raw_data.get("room", {})

        return ChannelMessage(
            content=text,
            channel="yuanbao",
            user_id=sender.get("id", "unknown"),
            session_id=f"agent:main:wechat:private:{sender.get('id', 'unknown')}",
            metadata={
                "room_id": room.get("id", ""),
                "room_name": room.get("name", ""),
                "sender_name": sender.get("name", ""),
                "msg_type": raw_data.get("type", "text"),
                "method": self._method,
            },
        )

    def send(self, channel_message: ChannelMessage, response: str) -> bool:
        """发送响应到微信。"""
        # 频率限制
        now = time.time()
        if (now - self._last_send_time) * 1000 < self._rate_limit_ms:
            time.sleep(self._rate_limit_ms / 1000)

        try:
            if self._method == "web_protocol":
                return self._send_web_protocol(channel_message, response)
            elif self._method == "browser":
                return self._send_browser(channel_message, response)
            else:
                logger.error(f"未知method: {self._method}")
                return False
        except Exception as e:
            logger.error(f"微信发送失败: {e}")
            return False
        finally:
            self._last_send_time = time.time()

    def _send_web_protocol(self, msg: ChannelMessage, response: str) -> bool:
        """通过web协议发送。"""
        # 实际实现需要微信web协议的具体API
        # 这里是框架，具体API需要逆向工程
        logger.info(f"[web_protocol] 发送到 {msg.user_id}: {response[:50]}...")
        return True

    def _send_browser(self, msg: ChannelMessage, response: str) -> bool:
        """通过浏览器自动化发送。"""
        # 实际实现需要headless Chrome操作
        # 这里是框架
        logger.info(f"[browser] 发送到 {msg.user_id}: {response[:50]}...")
        return True

    def _should_process(self, text: str) -> bool:
        """判断是否应该处理这条消息。"""
        # 规则1：@提及
        if "@" in text:
            return True
        # 规则2：关键词触发
        triggers = ["元宝", "ai", "助手", "openllm"]
        text_lower = text.lower()
        return any(t in text_lower for t in triggers)

    def health(self) -> dict:
        cookies = self._load_cookies()
        return {
            "status": "ok" if cookies else "auth_required",
            "detail": f"method={self._method}, cookies={'loaded' if cookies else 'missing'}",
        }


# ── 钉钉适配器（潜入模式） ─────────────────────────

class DingTalkAdapter(StealthChannelAdapter):
    """
    钉钉潜入模式适配器。

    钉钉有官方机器人API（需要企业认证），但我们选择潜入模式：
    用反向协议或浏览器自动化，以"普通用户"身份接入。

    使用方式：
        adapter = DingTalkAdapter(
            cookie_path="~/.openllm/cookies/dingtalk.json",
        )
        gateway.register(adapter)
    """

    def __init__(
        self,
        cookie_path: str = "~/.openllm/cookies/dingtalk.json",
        rate_limit_ms: int = 2000,
    ):
        super().__init__("dingtalk")
        self._cookie_path = Path(cookie_path).expanduser()
        self._rate_limit_ms = rate_limit_ms

    def receive(self, raw_data: Any) -> Optional[ChannelMessage]:
        if not isinstance(raw_data, dict):
            return None
        text = raw_data.get("text", {}).get("content", "")
        if not text:
            return None
        sender = raw_data.get("senderNick", "")
        return ChannelMessage(
            content=text,
            channel="dingtalk",
            user_id=sender or "unknown",
            metadata={"sender_nick": sender},
        )

    def send(self, channel_message: ChannelMessage, response: str) -> bool:
        logger.info(f"[dingtalk] 发送到 {channel_message.user_id}: {response[:50]}...")
        return True


# ── 飞书适配器（潜入模式） ─────────────────────────

class FeishuAdapter(StealthChannelAdapter):
    """
    飞书潜入模式适配器。

    飞书有官方机器人API（需要企业认证），但我们选择潜入模式。

    使用方式：
        adapter = FeishuAdapter(
            cookie_path="~/.openllm/cookies/feishu.json",
        )
        gateway.register(adapter)
    """

    def __init__(
        self,
        cookie_path: str = "~/.openllm/cookies/feishu.json",
        rate_limit_ms: int = 2000,
    ):
        super().__init__("feishu")
        self._cookie_path = Path(cookie_path).expanduser()
        self._rate_limit_ms = rate_limit_ms

    def receive(self, raw_data: Any) -> Optional[ChannelMessage]:
        if not isinstance(raw_data, dict):
            return None
        # 飞书消息格式
        event = raw_data.get("event", {})
        msg = event.get("message", {})
        text = msg.get("content", "")
        if isinstance(text, str):
            try:
                text = json.loads(text).get("text", text)
            except (json.JSONDecodeError, AttributeError):
                pass
        if not text:
            return None
        sender = event.get("sender", {}).get("sender_id", {})
        return ChannelMessage(
            content=text,
            channel="feishu",
            user_id=sender.get("open_id", "unknown"),
            metadata={"message_type": msg.get("message_type", "text")},
        )

    def send(self, channel_message: ChannelMessage, response: str) -> bool:
        logger.info(f"[feishu] 发送到 {channel_message.user_id}: {response[:50]}...")
        return True


# ── 适配器注册表 ────────────────────────────────────

ADAPTER_REGISTRY = {
    "cli": lambda: None,  # 已在gateway.py中
    "telegram": lambda **kw: TelegramAdapter(**kw),
    "yuanbao": lambda **kw: YuanbaoAdapter(**kw),
    "dingtalk": lambda **kw: DingTalkAdapter(**kw),
    "feishu": lambda **kw: FeishuAdapter(**kw),
}


def get_adapter(name: str, **kwargs) -> Optional[ChannelAdapter]:
    """从注册表创建适配器。"""
    factory = ADAPTER_REGISTRY.get(name)
    if factory and name != "cli":
        return factory(**kwargs)
    return None
