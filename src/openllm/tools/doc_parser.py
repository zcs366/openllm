"""
OpenLLM Doc Parser Tool — 通过HTTP调用Hermes端PaddleOCR-VL服务。
文档结构化解析：表格→Markdown、公式→LaTeX、图表→语义描述。

签名约定: func(**kwargs) — registry.execute() 按keyword解包调用（2026-09-10 鲁班修）。
"""
import json
import urllib.request
from typing import Optional


SERVER_URL = "http://127.0.0.1:9873"


def tool_doc_parser(file_path: str = "", pages: Optional[str] = None,
                    elements: Optional[str] = None, output_format: str = "markdown",
                    **kwargs) -> str:
    """文档结构化解析。

    Args:
        file_path: PDF或图片文件路径
        pages: 页码范围（仅PDF），如 "1-5,8"
        elements: 元素类型 table/formula/chart/text
        output_format: markdown(默认) / json
    """
    if not file_path:
        return json.dumps({"error": "file_path is required"}, ensure_ascii=False)

    payload = json.dumps({
        "file_path": file_path,
        "pages": pages,
        "elements": elements,
        "output_format": output_format,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{SERVER_URL}/parse",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        resp = urllib.request.urlopen(req, timeout=600)
        return resp.read().decode()
    except Exception as e:
        return json.dumps({
            "error": f"Doc parser request failed: {str(e)[:300]}"
        }, ensure_ascii=False)


def tool_doc_parser_health(**kwargs) -> str:
    """Doc parser服务健康检查——探测端口9873是否存活。"""
    try:
        resp = urllib.request.urlopen(f"{SERVER_URL}/health", timeout=5)
        return resp.read().decode()
    except Exception as e:
        return json.dumps({
            "status": "offline",
            "server": SERVER_URL,
            "error": str(e)[:200],
        }, ensure_ascii=False)
