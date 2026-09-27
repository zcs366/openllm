"""
OpenLLM 网络层 — ISN 的手。

提供 SSRF 防护、HTTP GET、搜索、网页抓取。
所有网络能力归此模块——今后扩展只改这里。

安全设计：
  - SLD 域名黑名单：阻止 localhost/内网/metadata 地址
  - allow_redirects=False：防重定向绕过 SSRF 闸
  - max_bytes 截断：防 OOM
  - subprocess 调用搜索引擎，隔离执行环境
"""
import re
import sys
import requests as _requests  # hard dependency (already in venv)
from urllib.parse import urlparse

# ── SLD 域名黑名单（SSRF 防线） ──
_BLOCKED_HOST_RE = re.compile(
    r'^(?:'
    r'localhost'                   # localhost
    r'|127\.\d+\.\d+\.\d+'       # 127.0.0.0/8
    r'|0\.0\.0\.0'                # 0.0.0.0
    r'|10\.\d+\.\d+\.\d+'        # 10.0.0.0/8
    r'|192\.168\.\d+\.\d+'       # 192.168.0.0/16
    r'|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+'  # 172.16.0.0/12
    r'|::1'                       # IPv6 loopback
    r'|169\.254\.\d+\.\d+'       # link-local metadata (169.254.169.254)
    r'|169\.254\.\d+\.\d+\.\d+'  # link-local metadata extended
    r')$',
    re.IGNORECASE,
)


def is_blocked_host(url: str) -> bool:
    """检查 URL 的主机名是否属于内网/元数据地址（SSRF 防线）。

    拦截：localhost, 127.x, 0.0.0.0, 10.x, 192.168.x, 172.16-31.x,
          ::1, 169.254.x（metadata）。
    """
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
    except Exception:
        return True  # 解析失败 = 拦截

    # 字面 IP 判断
    if _BLOCKED_HOST_RE.match(host):
        return True
    # 非 IP、非纯域名——逐段检查（如 [::1]:8080 的方括号场景）
    return False


def http_get(url: str, timeout: int = 15, max_bytes: int = 400_000) -> dict:
    """HTTP GET 请求（SSRF 防护版）。

    返回 dict:
      ok: bool
      status: int (0=网络错误)
      headers_content_type: str
      text: str (截断到 max_bytes)
      error: str (失败时)
    """
    if is_blocked_host(url):
        return {
            "ok": False, "status": 0, "headers_content_type": "",
            "text": "", "error": "该地址属于本机/内网，不开放",
        }

    try:
        resp = _requests.get(
            url,
            timeout=timeout,
            allow_redirects=False,   # 防重定向绕过 SSRF 闸
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) "
                     "Chrome/120.0.0.0 Safari/537.36"},
        )
        ct = resp.headers.get("Content-Type", "")
        text = resp.text[:max_bytes] if hasattr(resp, "text") else ""
        return {
            "ok": resp.status_code < 400,
            "status": resp.status_code,
            "headers_content_type": ct,
            "text": text,
            "error": None if resp.status_code < 400 else f"HTTP {resp.status_code}",
        }
    except _requests.exceptions.Timeout:
        return {"ok": False, "status": 0, "headers_content_type": "",
                "text": "", "error": f"请求超时 ({timeout}s)"}
    except _requests.exceptions.ConnectionError as e:
        return {"ok": False, "status": 0, "headers_content_type": "",
                "text": "", "error": f"连接失败: {e}"}
    except Exception as e:
        return {"ok": False, "status": 0, "headers_content_type": "",
                "text": "", "error": f"请求失败: {e}"}


# ── 搜索引擎路径 ──
_SEARCH_ENGINE = "/home/zcs/search-engine/hermes_search.py"


def web_search(query: str, max_results: int = 5) -> str:
    """调用本地搜索引擎查资料。

    使用 hermes_search.py CLI，超时 60s。
    引擎不存在/超时/非零退出 → 返回 "[搜索失败] 原因"（人话）。
    """
    import subprocess as _subprocess

    try:
        # 确定 Python 解释器
        python_exe = sys.executable or "python3"
        cmd = [
            python_exe, _SEARCH_ENGINE, query,
            "--sources", "web_search,cnscrape",
            "--max", str(max_results),
        ]
        result = _subprocess.run(
            cmd, capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()[:4000]
        elif result.returncode != 0:
            stderr_hint = (result.stderr or "").strip()[:200]
            return f"[搜索失败] 引擎退出码 {result.returncode}: {stderr_hint or '未知错误'}"
        else:
            return "[搜索失败] 引擎返回空结果"
    except FileNotFoundError:
        return f"[搜索失败] 搜索引擎不存在: {_SEARCH_ENGINE}"
    except _subprocess.TimeoutExpired:
        return "[搜索失败] 搜索超时 (60s)"
    except Exception as e:
        return f"[搜索失败] {e}"


def web_fetch(url: str) -> str:
    """抓取网页正文（SSRF 防护 + 粗提正文）。

    - SLD 拦下 → "[网络拒绝] 该地址属于本机/内网，不开放"
    - text/html → 去 script/style/标签，压缩空白
    - 非 html → 直接返回截断文本
    """
    if is_blocked_host(url):
        return "[网络拒绝] 该地址属于本机/内网，不开放"

    result = http_get(url, timeout=15, max_bytes=400_000)
    if not result["ok"]:
        return f"[网络失败] {result.get('error', '未知错误')}"

    text = result["text"]
    ct = result["headers_content_type"]

    # 粗提 HTML 正文
    if "text/html" in ct or text.lstrip().startswith("<"):
        # 去 script/style 标签及内容
        text = re.sub(r'<script\b[^>]*>.*?</script>', '', text, flags=re.S | re.I)
        text = re.sub(r'<style\b[^>]*>.*?</style>', '', text, flags=re.S | re.I)
        # 去所有 HTML 标签
        text = re.sub(r'<[^>]+>', ' ', text)
        # 压缩空白
        text = re.sub(r'\s+', ' ', text).strip()

    return text[:4000] if text else "[空内容]"
