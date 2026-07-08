"""
openLLM Error Classifier — 类型化错误分类 + 智能重试策略。

从 Hermes v0.18.0 的 typed send-error classification 学来。
5种错误类型，每种有独立的重试策略：

  RATE_LIMIT   → 等几秒重试（指数退避）
  AUTH_EXPIRED → 刷新token后重试
  NETWORK      → 换endpoint或重试
  PERMISSION   → 不重试，报错
  QUOTA        → 换provider
"""

import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger("openllm.error_classifier")


class ErrorCategory(Enum):
    """错误分类。"""
    RATE_LIMIT = "rate_limit"
    AUTH_EXPIRED = "auth_expired"
    NETWORK = "network"
    PERMISSION = "permission"
    QUOTA = "quota"
    UNKNOWN = "unknown"


@dataclass
class ClassifiedError:
    """分类后的错误。"""
    category: ErrorCategory
    original_error: Optional[Exception] = None
    message: str = ""
    retry_after_seconds: float = 0.0
    max_retries: int = 0
    status_code: int = 0

    @property
    def is_retryable(self) -> bool:
        return self.max_retries > 0


# ── 默认策略 ────────────────────────────────────────

_STRATEGIES = {
    ErrorCategory.RATE_LIMIT:   {"retry_after": 5.0,  "max_retries": 3},
    ErrorCategory.AUTH_EXPIRED: {"retry_after": 0.0,  "max_retries": 1},
    ErrorCategory.NETWORK:      {"retry_after": 2.0,  "max_retries": 3},
    ErrorCategory.PERMISSION:   {"retry_after": 0.0,  "max_retries": 0},
    ErrorCategory.QUOTA:        {"retry_after": 0.0,  "max_retries": 0},
    ErrorCategory.UNKNOWN:      {"retry_after": 10.0, "max_retries": 1},
}


def _make_classified(category: ErrorCategory, error=None, msg="", code=0) -> ClassifiedError:
    s = _STRATEGIES[category]
    return ClassifiedError(
        category=category,
        original_error=error,
        message=msg or str(error) if error else "",
        retry_after_seconds=s["retry_after"],
        max_retries=s["max_retries"],
        status_code=code,
    )


# ── 分类器 ──────────────────────────────────────────

class ErrorClassifier:
    """分类LLM API调用错误，决定重试策略。"""

    def classify(self, error: Exception) -> ClassifiedError:
        """从异常对象分类。"""
        error_str = str(error).lower()
        cls_name = type(error).__name__.lower()

        # 网络层错误
        if isinstance(error, (ConnectionError, TimeoutError, OSError)):
            return _make_classified(ErrorCategory.NETWORK, error)

        if "connection" in error_str or "timeout" in error_str or "timed out" in error_str:
            return _make_classified(ErrorCategory.NETWORK, error)

        # HTTP 状态码（从requests库的异常中提取）
        status_code = getattr(error, "status_code", None) or getattr(error, "response", None)
        if status_code and hasattr(status_code, "status_code"):
            status_code = status_code.status_code
        if isinstance(status_code, int) and status_code > 0:
            return self.classify_response(status_code, str(error))

        # 文本匹配
        if "rate limit" in error_str or "429" in error_str or "too many requests" in error_str:
            return _make_classified(ErrorCategory.RATE_LIMIT, error, code=429)

        if any(k in error_str for k in ("unauthorized", "401", "token expired", "invalid api key")):
            return _make_classified(ErrorCategory.AUTH_EXPIRED, error, code=401)

        if any(k in error_str for k in ("forbidden", "403", "access denied")):
            return _make_classified(ErrorCategory.PERMISSION, error, code=403)

        if any(k in error_str for k in ("quota", "402", "insufficient", "billing")):
            return _make_classified(ErrorCategory.QUOTA, error, code=402)

        return _make_classified(ErrorCategory.UNKNOWN, error)

    def classify_response(self, status_code: int, body: str) -> ClassifiedError:
        """从HTTP响应分类。"""
        body_lower = body.lower()

        if status_code == 429 or "rate limit" in body_lower:
            return _make_classified(ErrorCategory.RATE_LIMIT, msg=body[:200], code=429)

        if status_code == 401:
            return _make_classified(ErrorCategory.AUTH_EXPIRED, msg=body[:200], code=401)

        if status_code == 403:
            return _make_classified(ErrorCategory.PERMISSION, msg=body[:200], code=403)

        if status_code == 402 or "quota" in body_lower:
            return _make_classified(ErrorCategory.QUOTA, msg=body[:200], code=402)

        if status_code in (502, 503, 504):
            return _make_classified(ErrorCategory.NETWORK, msg=body[:200], code=status_code)

        return _make_classified(ErrorCategory.UNKNOWN, msg=body[:200], code=status_code)

    def should_retry(self, classified: ClassifiedError, attempt: int) -> bool:
        """是否应该重试。attempt从0开始。"""
        if not classified.is_retryable:
            return False
        return attempt < classified.max_retries

    def get_retry_delay(self, classified: ClassifiedError, attempt: int) -> float:
        """计算重试延迟（指数退避+抖动）。"""
        base = classified.retry_after_seconds
        if base <= 0:
            base = 1.0
        delay = base * (2 ** attempt)
        jitter = random.uniform(0.5, 1.5)
        return min(delay * jitter, 60.0)  # 最长60秒
