"""
OpenLLM Doc Parser Tool — 通过HTTP调用Hermes端PaddleOCR-VL服务。
文档结构化解析：表格→Markdown、公式→LaTeX、图表→语义描述。
"""
import json
import urllib.request


SERVER_URL = "http://127.0.0.1:9873"


def tool_doc_parser(args: dict) -> str:
    """文档结构化解析。

    Args:
        file_path: PDF或图片文件路径
        pages: 页码范围（仅PDF），如 "1-5,8"
        elements: 元素类型 table/formula/chart/text
        output_format: markdown(默认) / json
    """
    file_path = args.get("file_path", "")
    if not file_path:
        return json.dumps({"error": "file_path is required"}, ensure_ascii=False)

    payload = json.dumps({
        "file_path": file_path,
        "pages": args.get("pages"),
        "elements": args.get("elements"),
        "output_format": args.get("output_format", "markdown"),
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
