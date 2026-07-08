"""
openLLM Credential Firewall — API key不暴露给LLM，域名白名单。

从 Hermes v0.18.0 的 credential isolation 学来。
三道防线：
  1. 输出过滤：工具/模型输出中的key被替换为 [REDACTED]
  2. 上下文净化：确保key不进入LLM对话上下文
  3. 域名白名单：网络请求只允许白名单域名
"""

import re
import logging
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger("openllm.credential_firewall")

# ── 常见API key正则 ──────────────────────────────────

_KEY_PATTERNS = [
    (re.compile(r'sk-[a-zA-Z0-9]{20,}'), 'OpenAI/DeepSeek key'),
    (re.compile(r'AIza[a-zA-Z0-9_-]{35}'), 'Google API key'),
    (re.compile(r'AKIA[A-Z0-9]{16}'), 'AWS access key'),
    (re.compile(r'ghp_[a-zA-Z0-9]{36}'), 'GitHub PAT'),
    (re.compile(r'xox[bpsa]-[a-zA-Z0-9-]+'), 'Slack token'),
    (re.compile(r'Bearer\s+[a-zA-Z0-9._-]{20,}'), 'Bearer token'),
    (re.compile(r'(?i)api[_-]?key\s*[:=]\s*["\']?[a-zA-Z0-9._-]{16,}'), 'Generic API key'),
]


class CredentialFirewall:
    """
    凭据防火墙。

    用法：
        fw = CredentialFirewall()

        # 过滤工具输出
        clean_output = fw.scan_text(raw_output)

        # 检测文本中是否有key
        if fw.has_credentials(text):
            logger.warning("检测到凭据！")

        # 检查域名
        ok, reason = fw.check_domain("https://api.deepseek.com/v1")
    """

    DEFAULT_WHITELIST = {
        'api.deepseek.com',
        'api.openai.com',
        'api.anthropic.com',
        'generativelanguage.googleapis.com',
        'localhost',
        '127.0.0.1',
    }

    def __init__(self, whitelist: Optional[set[str]] = None):
        self.whitelist = whitelist or self.DEFAULT_WHITELIST.copy()
        self._redact_count = 0
        self._blocked_domains: list[str] = []
        self._max_blocked_domains = 1000  # 防止无限增长

    def scan_text(self, text: str) -> str:
        """
        扫描文本，替换所有凭据为 [REDACTED:xxxx...]。
        返回净化后的文本。原始文本不变。
        """
        if not text:
            return text

        redacted = text
        for pattern, key_type in _KEY_PATTERNS:
            matches = pattern.findall(redacted)
            if matches:
                for m in matches:
                    # 保留前4字符用于调试
                    display = m[:min(8, len(m))]
                    replacement = f"[REDACTED:{display}...]"
                    redacted = redacted.replace(m, replacement)
                    self._redact_count += 1
                    logger.warning(f"凭据拦截({key_type}): {display}...")

        return redacted

    def has_credentials(self, text: str) -> bool:
        """检测文本中是否包含凭据。不修改文本。"""
        for pattern, _ in _KEY_PATTERNS:
            if pattern.search(text):
                return True
        return False

    def check_domain(self, url: str) -> tuple[bool, str]:
        """
        检查URL域名是否在白名单中。

        Returns:
            (allowed, reason)
        """
        try:
            parsed = urlparse(url)
            host = parsed.hostname or ""
            if host in self.whitelist:
                return True, ""
            msg = f"域名 '{host}' 不在白名单中"
            if len(self._blocked_domains) < self._max_blocked_domains:
                self._blocked_domains.append(host)
            return False, msg
        except Exception as e:
            return False, f"URL解析失败: {e}"

    def add_domain(self, domain: str):
        """动态添加域名到白名单。"""
        self.whitelist.add(domain)
        logger.info(f"白名单新增: {domain}")

    def remove_domain(self, domain: str):
        """从白名单移除域名。"""
        self.whitelist.discard(domain)
        logger.info(f"白名单移除: {domain}")

    @property
    def stats(self) -> dict:
        """防火墙统计。"""
        return {
            "redact_count": self._redact_count,
            "blocked_domains": len(self._blocked_domains),
            "whitelist_size": len(self.whitelist),
        }
