"""网络工具刀测试 — net.py + ISN/ISA 注册面 + octopus 同源验证。

T1: SLD 域名黑名单全覆盖
T2: http_get 重定向拦截（allow_redirects=False）
T3: web_fetch SLD 闸拦截
T4: 真网络冒烟（baidu 可用则跑，无网则 skip）
T5: web_search 引擎缺失降级
T6: ISN 注册面（tools 字典 + FULL_TOOLS + TOOL_DESCRIPTIONS 键域）
T7: octopus 同源验证（无第二份 _tool_desc 字面字典）
"""
import inspect
from unittest.mock import MagicMock, patch

import pytest


# ═══════════════════════════════════════════════════════
# T1: SLD 域名黑名单
# ═══════════════════════════════════════════════════════

class TestSLDBlocked:
    """内网/元数据地址全部拦截。"""

    @pytest.mark.parametrize("url", [
        "http://localhost/",
        "http://127.0.0.1/",
        "http://127.0.0.1:13000/",
        "http://127.0.0.1:13001/",
        "http://192.168.1.1/",
        "http://10.0.0.5/",
        "http://169.254.169.254/metadata",
        "http://[::1]/",
        "http://172.16.0.1/",
        "http://172.31.255.255/",
    ])
    def test_blocked(self, url):
        from openllm.net import is_blocked_host
        assert is_blocked_host(url) is True, f"应拦截: {url}"

    @pytest.mark.parametrize("url", [
        "https://example.com/",
        "https://www.baidu.com/",
        "https://github.com/openshell",
        "https://arxiv.org/abs/2301.00001",
    ])
    def test_allowed(self, url):
        from openllm.net import is_blocked_host
        assert is_blocked_host(url) is False, f"不应拦截: {url}"


# ═══════════════════════════════════════════════════════
# T2: http_get 重定向拦截
# ═══════════════════════════════════════════════════════

class TestHttpGetRedirect:
    """allow_redirects=False — 3xx 返回标注不跟随。"""

    def test_302_not_followed(self):
        from openllm.net import http_get
        mock_resp = MagicMock()
        mock_resp.status_code = 302
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.text = ""
        mock_resp.__bool__ = lambda self: True

        with patch("openllm.net._requests") as mock_requests:
            mock_requests.get.return_value = mock_resp
            mock_requests.exceptions = MagicMock()
            mock_requests.exceptions.Timeout = TimeoutError
            mock_requests.exceptions.ConnectionError = ConnectionError

            result = http_get("https://httpbin.org/redirect/1")
            assert result["status"] == 302
            # allow_redirects=False → 3xx 返回标注，不跟随
            mock_requests.get.assert_called_once()
            call_kwargs = mock_requests.get.call_args
            assert call_kwargs[1].get("allow_redirects") is False or \
                   call_kwargs.kwargs.get("allow_redirects") is False


# ═══════════════════════════════════════════════════════
# T3: web_fetch SLD 闸拦截
# ═══════════════════════════════════════════════════════

class TestWebFetchSLD:
    """web_fetch 被 SLD 拦截时，http_get 不被调用。"""

    def test_blocked_returns_network_reject(self):
        from openllm import net
        result = net.web_fetch("http://127.0.0.1:13000/secret")
        assert "[网络拒绝]" in result
        assert "本机" in result or "内网" in result

    def test_http_get_not_called(self):
        from openllm import net
        with patch.object(net, "http_get", side_effect=AssertionError("不应调用 http_get")):
            result = net.web_fetch("http://10.0.0.5/admin")
            assert "[网络拒绝]" in result


# ═══════════════════════════════════════════════════════
# T4: 真网络冒烟（skipif 无网则跳过）
# ═══════════════════════════════════════════════════════

def _network_available():
    """探测网络连通性（baidu 200ms 内可通）。"""
    import socket
    try:
        s = socket.create_connection(("www.baidu.com", 443), timeout=2)
        s.close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _network_available(), reason="无网络连接，跳过冒烟测试")
class TestRealNetwork:
    """真实网络冒烟测试（baidu 已实测直连通）。"""

    def test_web_fetch_baidu(self):
        from openllm.net import web_fetch
        result = web_fetch("https://www.baidu.com")
        assert result  # 非空
        assert "[网络拒绝]" not in result
        assert "[网络失败]" not in result
        # 百度首页标题含"百度"（提取后的文本大概率含此字）
        assert "百度" in result or len(result) > 100  # 至少有内容

    def test_http_get_baidu(self):
        from openllm.net import http_get
        result = http_get("https://www.baidu.com", timeout=10)
        assert result["ok"] is True
        assert result["status"] == 200
        assert len(result["text"]) > 100


# ═══════════════════════════════════════════════════════
# T5: web_search 引擎缺失降级
# ═══════════════════════════════════════════════════════

class TestSearchEngineMissing:
    """搜索引擎路径不存在时返回降级消息，不抛异常。"""

    def test_missing_engine_returns_fallback(self):
        from openllm.net import web_search
        with patch("openllm.net._SEARCH_ENGINE", "/nonexistent/path/hermes_search.py"):
            result = web_search("test query")
            assert "[搜索失败]" in result
            # subprocess 退出码 2（文件不存在），信息在 stderr 里
            assert "搜索失败" in result


# ═══════════════════════════════════════════════════════
# T6: ISN 注册面
# ═══════════════════════════════════════════════════════

class TestISNRegistration:
    """ISN 工具注册面：网络工具已注册，ISA 工具清单一致。"""

    def test_isn_tools_contain_network(self):
        from openllm.core.isn_impl import ISN
        isn = ISN()
        assert "web_search" in isn.tools, "ISN.tools 应含 web_search"
        assert "web_fetch" in isn.tools, "ISN.tools 应含 web_fetch"

    def test_full_tools_contain_network(self):
        from openllm.core.isa_impl import FULL_TOOLS
        assert "web_search" in FULL_TOOLS
        assert "web_fetch" in FULL_TOOLS

    def test_tool_descriptions_cover_full_tools(self):
        from openllm.core.isa_impl import FULL_TOOLS, TOOL_DESCRIPTIONS
        missing = [t for t in FULL_TOOLS if t not in TOOL_DESCRIPTIONS]
        assert not missing, f"TOOL_DESCRIPTIONS 缺键: {missing}"

    def test_base_tools_unchanged(self):
        """BASE_TOOLS 不含网络工具（无 provider 降级集）。"""
        from openllm.core.isa_impl import BASE_TOOLS
        assert "web_search" not in BASE_TOOLS
        assert "web_fetch" not in BASE_TOOLS


# ═══════════════════════════════════════════════════════
# T7: octopus 同源验证
# ═══════════════════════════════════════════════════════

class TestOctopusSourceOfTruth:
    """octopus.py 不再有独立的 _tool_desc 字面字典，而是 import TOOL_DESCRIPTIONS。"""

    def test_no_second_tool_desc_dict(self):
        """inspect octopus 源码，确认无独立的 _tool_desc 赋值。"""
        from openllm.iai.octopus import _LeftBrain
        src = inspect.getsource(_LeftBrain.think)
        # 旧代码有 '_tool_desc = {' 这种字面赋值；现在应该是 import
        assert "_tool_desc = {" not in src, \
            "octopus._LeftBrain.think 中仍有独立 _tool_desc 字面字典（假同源未修）"
        # 确认已改为 import
        assert "TOOL_DESCRIPTIONS" in src, \
            "octopus._LeftBrain.think 应 import TOOL_DESCRIPTIONS（同源）"
