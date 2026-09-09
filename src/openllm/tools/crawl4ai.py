"""
OpenLLM Crawl4AI Tool — subprocess桥接→Hermes端crawl4ai_tool.py。
浏览器级抓取：Playwright JS渲染+反爬+LLM原生Markdown。

桥接选subprocess而非直import：crawl4ai/playwright依赖只装在Hermes venv，
openLLM venv零重依赖、零版本冲突（2026-09-10 鲁班锻造）。
签名约定: func(**kwargs) — registry.execute() 按keyword解包调用。
"""
import json
import subprocess

WRAPPER = "/home/zcs/.hermes/scripts/crawl4ai_wrapper.py"
HERMES_PY = "/home/zcs/.hermes/hermes-agent/venv/bin/python"


def tool_crawl4ai(action: str = "scrape", url: str = "", urls: str = "",
                  query: str = "", depth: int = 2, max_pages: int = 10,
                  css_selector: str = "", **kwargs) -> str:
    """浏览器级网页抓取（JS渲染/SPA/动态加载）。

    Args:
        action: scrape=单页, batch=批量, deep_crawl=多页爬取, extract=结构化抽取
        url: 目标URL (scrape/deep_crawl/extract)
        urls: 逗号分隔URL列表 (batch)
        query: 深爬过滤关键词 (deep_crawl可选)
        depth: 爬取深度 (deep_crawl, 默认2)
        max_pages: 最大页数 (deep_crawl, 默认10)
        css_selector: CSS选择器聚焦提取 (可选)
    """
    if action != "batch" and not url:
        return json.dumps({"error": "url is required"}, ensure_ascii=False)

    payload: dict = {"action": action, "url": url}
    if action == "batch":
        if not urls:
            return json.dumps({"error": "urls is required for batch"}, ensure_ascii=False)
        payload["urls"] = [u.strip() for u in urls.split(",") if u.strip()]
    if action == "deep_crawl":
        payload["depth"] = int(depth)
        payload["max_pages"] = int(max_pages)
    if query:
        payload["query"] = query
    if css_selector:
        payload["css_selector"] = css_selector

    try:
        proc = subprocess.run(
            [HERMES_PY, WRAPPER],
            input=json.dumps(payload),
            capture_output=True, text=True, timeout=180,
        )
        out = proc.stdout.strip()
        if not out:
            return json.dumps({
                "error": f"crawl4ai empty output. stderr: {proc.stderr[:300]}"
            }, ensure_ascii=False)
        return out
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "Timeout 180s"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)[:300]}, ensure_ascii=False)
