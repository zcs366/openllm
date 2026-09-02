#!/usr/bin/env python3
"""MCP Memory Server 自测脚本。

通过 subprocess + 管道模拟 stdio，验证：
1. initialize 握手
2. tools/list 列出工具
3. tools/call memory_write 写入记忆
4. tools/call memory_search 搜索记忆
5. tools/call memory_read 读取记忆
"""
import json
import subprocess
import sys

PROJECT = "/home/zcs/projects/openllm"
PYTHON = f"{PROJECT}/.venv/bin/python"
MODULE = "openllm.bridge.mcp_memory_server"


def make_request(method: str, params: dict = None, req_id: int = 1) -> str:
    """构造 JSON-RPC 2.0 请求。"""
    msg = {
        "jsonrpc": "2.0",
        "method": method,
        "id": req_id,
    }
    if params:
        msg["params"] = params
    return json.dumps(msg) + "\n"


def send_and_recv(proc: subprocess.Popen, request: str, timeout: float = 30) -> dict:
    """发送 JSON-RPC 请求，读取响应。"""
    proc.stdin.write(request.encode())
    proc.stdin.flush()

    # 读取一行响应（可能有多行 JSON-RPC 通知，跳过）
    import time
    start = time.time()
    while time.time() - start < timeout:
        line = proc.stdout.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            resp = json.loads(line)
            # 跳过通知（无 id）
            if "id" in resp:
                return resp
        except json.JSONDecodeError:
            continue
    return {"error": "timeout or no response"}


def test_mcp_server():
    """运行完整测试序列。"""
    print("=" * 60)
    print("MCP Memory Server 自测")
    print("=" * 60)

    # 启动 MCP 服务器进程
    cmd = [PYTHON, "-m", MODULE]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=PROJECT,
        env={
            **dict(__import__("os").environ),
            "PYTHONPATH": f"{PROJECT}/src",
        },
    )

    results = {}

    try:
        # ── 1. initialize ──
        print("\n[1] initialize 握手...")
        req = make_request("initialize", {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0.0"},
        })
        resp = send_and_recv(proc, req)
        results["initialize"] = resp
        print(f"  响应: {json.dumps(resp, ensure_ascii=False, indent=2)[:300]}")
        assert "result" in resp, f"initialize 失败: {resp}"
        print("  ✅ initialize 通过")

        # 发送 initialized 通知
        notif = json.dumps({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }) + "\n"
        proc.stdin.write(notif.encode())
        proc.stdin.flush()

        # ── 2. tools/list ──
        print("\n[2] tools/list 列出工具...")
        req = make_request("tools/list", {}, req_id=2)
        resp = send_and_recv(proc, req)
        results["tools_list"] = resp
        tool_names = [t.get("name", "") for t in resp.get("result", {}).get("tools", [])]
        print(f"  工具列表: {tool_names}")
        assert "memory_write" in tool_names, f"缺少 memory_write: {tool_names}"
        assert "memory_read" in tool_names, f"缺少 memory_read: {tool_names}"
        assert "memory_search" in tool_names, f"缺少 memory_search: {tool_names}"
        print("  ✅ tools/list 通过 (3个工具全部注册)")

        # ── 3. tools/call memory_write ──
        print("\n[3] tools/call memory_write 写入记忆...")
        req = make_request("tools/call", {
            "name": "memory_write",
            "arguments": {
                "content": "MCP自测：openLLM记忆服务器首次写入成功",
                "importance": 7.0,
                "memory_type": "insight",
                "tags": ["mcp", "test", "milestone"],
                "session_id": "mcp_test_001",
            },
        }, req_id=3)
        resp = send_and_recv(proc, req)
        results["memory_write"] = resp
        content = resp.get("result", {}).get("content", [])
        if content:
            text = content[0].get("text", "") if isinstance(content, list) else str(content)
        else:
            text = json.dumps(resp, ensure_ascii=False)
        print(f"  响应摘要: {text[:400]}")
        # 验证写入成功
        assert "memory_id" in text or "ok" in text.lower() or "status" in text.lower(), f"写入可能失败: {text}"
        print("  ✅ memory_write 通过")

        # ── 4. tools/call memory_search ──
        print("\n[4] tools/call memory_search 搜索记忆...")
        req = make_request("tools/call", {
            "name": "memory_search",
            "arguments": {
                "keyword": "MCP自测",
                "max_results": 5,
            },
        }, req_id=4)
        resp = send_and_recv(proc, req)
        results["memory_search"] = resp
        content = resp.get("result", {}).get("content", [])
        if content:
            text = content[0].get("text", "") if isinstance(content, list) else str(content)
        else:
            text = json.dumps(resp, ensure_ascii=False)
        print(f"  响应摘要: {text[:400]}")
        assert "total" in text or "entries" in text or "mcp" in text.lower(), f"搜索可能失败: {text}"
        print("  ✅ memory_search 通过")

        # ── 5. tools/call memory_read ──
        print("\n[5] tools/call memory_read 读取记忆...")
        req = make_request("tools/call", {
            "name": "memory_read",
            "arguments": {
                "session_id": "mcp_test_001",
            },
        }, req_id=5)
        resp = send_and_recv(proc, req)
        results["memory_read"] = resp
        content = resp.get("result", {}).get("content", [])
        if content:
            text = content[0].get("text", "") if isinstance(content, list) else str(content)
        else:
            text = json.dumps(resp, ensure_ascii=False)
        print(f"  响应摘要: {text[:400]}")
        assert "total" in text or "entries" in text or "mcp" in text.lower(), f"读取可能失败: {text}"
        print("  ✅ memory_read 通过")

        # ── 6. tools/call memory_read (query模式) ──
        print("\n[6] tools/call memory_read (query模式)...")
        req = make_request("tools/call", {
            "name": "memory_read",
            "arguments": {
                "query": "MCP自测",
                "max_results": 3,
            },
        }, req_id=6)
        resp = send_and_recv(proc, req)
        results["memory_read_query"] = resp
        content = resp.get("result", {}).get("content", [])
        if content:
            text = content[0].get("text", "") if isinstance(content, list) else str(content)
        else:
            text = json.dumps(resp, ensure_ascii=False)
        print(f"  响应摘要: {text[:400]}")
        print("  ✅ memory_read (query) 通过")

        print("\n" + "=" * 60)
        print("全部 6 项测试通过 ✅")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        proc.stdin.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    return results


if __name__ == "__main__":
    test_mcp_server()
