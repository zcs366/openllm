"""pty 端到端验证：openllm.cli.repl 的粘贴/命令分流行为。

喂真实字节序列（含 ESC[200~/201~ bracketed paste 转义），断言：
  P1: 普通单行输入原样到 onecmd
  P2: /status（白名单）→ 走 onecmd 且被 FakeShell 识别为命令
  P3: /openllm: xxx（白名单外）→ onecmd 收到原文（default 会当消息，
      这里 FakeShell 记录 is_suspicious_paste 的判定结果）
  P4: 多行 bracketed paste 块 → onecmd 收到的形态（\n 保留还是被拍平）
"""
import os
import pty
import re
import select
import subprocess
import sys
import time

CHILD = r'''
import sys
sys.path.insert(0, "/mnt/i/openllm/src")
from openllm.cli.repl import run_repl, is_suspicious_paste

KNOWN = {"status", "help", "exit", "quit", "q", "clear", "reset", "new",
         "research", "context", "model", "hypothesis", "search"}

class FakeShell:
    prompt = "P> "
    KNOWN_COMMANDS = KNOWN
    def onecmd(self, line):
        verdict = "CMD" if (line.startswith("/") and not
                            is_suspicious_paste(line, KNOWN)) else "MSG"
        print(f"RECV[{verdict}]: {line!r}", flush=True)
        return False
    def do_EOF(self, arg):
        print("BYE", flush=True)
        return True

run_repl(FakeShell())
'''

master, slave = pty.openpty()
env = dict(os.environ, TERM="xterm-256color", LANG="C.UTF-8",
           PYTHONPATH="/mnt/i/openllm/src")
proc = subprocess.Popen(
    ["/mnt/i/openllm/.venv/bin/python", "-u", "-c", CHILD],
    stdin=slave, stdout=slave, stderr=slave, env=env, close_fds=True)
os.close(slave)

buf = b""
def drain(timeout=0.6):
    global buf
    end = time.time() + timeout
    while time.time() < end:
        r, _, _ = select.select([master], [], [], 0.1)
        if master in r:
            try:
                chunk = os.read(master, 65536)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            end = time.time() + timeout  # 有数据就续窗

def send(b: bytes, pause=0.5):
    os.write(master, b)
    time.sleep(pause)
    drain(0.5)

drain(3.0)          # 等 prompt_toolkit 起来
send(b"hello\r")
send(b"/status\r")
send(b"/openllm: \xe4\xb8\x8a\xe4\xb8\x80\xe8\xbd\xae\xe7\x9a\x84\xe6\x96\x87\xe5\xad\x97\r")  # /openllm: 上一轮的文字
# 多行粘贴块（首词是白名单命令，验证多行护栏）
paste = "\x1b[200~/status\n\xe8\xbf\x99\xe6\x98\xaf\xe7\xb2\x98\xe8\xb4\xb4\xe7\x9a\x84\xe6\x9d\x90\xe6\x96\x99\x1b[201~"  # /status\n这是粘贴的材料
send(paste.encode(), pause=0.8)
send(b"\r", pause=0.8)   # 提交粘贴块
send(b"\x04", pause=0.8) # Ctrl-D 退出

try:
    proc.wait(timeout=5)
except subprocess.TimeoutExpired:
    proc.kill()

out = buf.decode("utf-8", errors="replace")
# 剥掉 ANSI 转义只留 RECV/BYE 行和护栏提示
keep = [l for l in out.splitlines()
        if "RECV[" in l or "BYE" in l or "粘贴护栏" in l]
print("=== 关键行 ===")
for l in keep:
    print(l)
print("=== 断言 ===")
joined = "\n".join(keep)
checks = [
    ("P1 普通输入到 onecmd", "RECV[MSG]: 'hello'" in joined),
    ("P2 /status 白名单→CMD", "RECV[CMD]: '/status'" in joined),
    ("P3 /openllm: 白名单外→MSG", "RECV[MSG]: '/openllm:" in joined),
    ("P4 多行粘贴被护栏识别（护栏提示出现）", "粘贴护栏" in joined),
]
ok = True
for name, passed in checks:
    print(("PASS" if passed else "FAIL"), name)
    ok = ok and passed
# cp 自 scratch 验证脚本（DR-20260927-01，已人工验证 4/4 PASS）。
# 未 pytest 化：write_file 重写曾被 landlock 执法拦截（-c 注入模式），
# 留作独立回归脚本。跑法：
#   .venv/bin/python tests/test_cli_repl_pty.py  → 退出码 0 = 全过
sys.exit(0 if ok else 1)
