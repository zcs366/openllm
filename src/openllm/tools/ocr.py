"""
OpenLLM OCR Tool — 通过HTTP调用Hermes端Unlimited-OCR服务。

签名约定: func(**kwargs) — registry.execute() 按keyword解包调用（2026-09-10 鲁班修）。
"""
import json
import urllib.request
import urllib.parse


SERVER_URL = "http://127.0.0.1:9872"


def tool_ocr(file_path: str = "", file_type: str = "auto",
             prompt: str = "<image>document parsing.", **kwargs) -> str:
    """对图片或PDF文件执行OCR识别。

    Args:
        file_path: 文件绝对路径
        file_type: auto/image/pdf (默认auto)
        prompt: OCR提示词
    """
    if not file_path:
        return json.dumps({"error": "file_path is required"}, ensure_ascii=False)

    # 自动检测类型
    if file_type == "auto":
        file_type = "pdf" if file_path.lower().endswith(".pdf") else "image"

    is_pdf = file_type == "pdf"

    data = urllib.parse.urlencode({
        "file_path": file_path,
        "prompt": prompt,
        "is_pdf": str(is_pdf).lower(),
    }).encode()

    req = urllib.request.Request(
        f"{SERVER_URL}/ocr/path",
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    try:
        resp = urllib.request.urlopen(req, timeout=600)
        return resp.read().decode()
    except Exception as e:
        return json.dumps({
            "error": f"OCR request failed: {str(e)[:300]}"
        }, ensure_ascii=False)


def tool_ocr_health(**kwargs) -> str:
    """OCR服务健康检查——探测端口9872是否存活。"""
    try:
        resp = urllib.request.urlopen(f"{SERVER_URL}/health", timeout=5)
        return resp.read().decode()
    except Exception as e:
        return json.dumps({
            "status": "offline",
            "server": SERVER_URL,
            "error": str(e)[:200],
        }, ensure_ascii=False)
