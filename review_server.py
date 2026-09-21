#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
review_server.py —— 复习站的「静态文件 + /ask 问答代理」二合一服务器

用法（在 Pointcept 仓库根目录）：
    python -X utf8 review_server.py

然后浏览器开  http://127.0.0.1:8765/step_all_review.html  （记得 Ctrl+F5 强刷）

它做了两件事：
  1. 像原来的 `python -m http.server 8765` 一样静态伺服当前目录
     （step_all_review.html / step5_practice.html 照常访问）
  2. 额外提供 POST /ask 接口：页面里每章讲解和每道题下面的 💬 问答窗口
     把问题发到这里，本服务调用 claude CLI 的无头模式（claude -p），
     拿到回答再返回给页面显示。

没有这个服务时页面也能用（双击打开 / 纯静态服务器都行），
只是 💬 问答窗口会提示「连不上问答服务」，做题功能完全不受影响。
"""
import json
import shutil
import subprocess
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = 8765
REPO = Path(__file__).resolve().parent
CLAUDE_TIMEOUT = 300          # claude 思考 + 可能读仓库代码验证，给足时间
PROMPT_CAP = 24000            # Windows 命令行参数长度远小于这个数，安全垫

PREAMBLE = """你是「Pointcept 全程复习」网页里内嵌的答疑助手。用户是大前端图形渲染程序员（WebGL/WebGPU 背景，懂 Transformer 原理但缺 PyTorch/numpy 实践），正在学习 Pointcept 3D 点云框架（LiDAR 语义分割方向）。回答要求：
- 用中文，语气像同事间答疑，多用 WebGL/GLSL/前端类比（比如 gather≈index buffer 选顶点、cumsum≈GPU prefix sum、切片≈stride/offset 视图）
- 紧扣下面给出的页面上下文回答；涉及仓库代码时，你现在就在 Pointcept 仓库根目录，可以读代码核实后再答
- 代码/命令用 markdown 代码块；回答长度以讲清为准，不灌水
- 如果上下文标明「用户还没作答这道题」，先给提示和思路引导，不要直接说出最终答案；用户明确要求答案时再给"""


def find_claude():
    p = shutil.which("claude")
    if p:
        return p
    # PATH 里找不到时，试试 npm 全局目录的已知位置
    for cand in (Path.home() / "AppData" / "Roaming" / "npm" / "claude.cmd",
                 Path.home() / ".local" / "bin" / "claude"):
        if cand.exists():
            return str(cand)
    return None


CLAUDE = find_claude()


def ask_claude(payload):
    q = (payload.get("question") or "").strip()
    if not q:
        return {"ok": False, "error": "问题为空"}
    parts = [PREAMBLE]
    ctx = (payload.get("context") or "").strip()
    if ctx:
        parts.append("【页面上下文】\n" + ctx)
    hist = (payload.get("history") or "").strip()
    if hist:
        parts.append("【此前的对话（供衔接，不必重复回答）】\n" + hist)
    parts.append("【用户的新问题】\n" + q)
    prompt = "\n\n".join(parts)
    if len(prompt) > PROMPT_CAP:
        prompt = prompt[:PROMPT_CAP] + "\n…（上下文超长已截断）"
    try:
        # 注意：prompt 走 stdin 而不是命令行参数——Windows 上 .cmd 包装层
        # 传参时换行符会截断命令行（实测只传到第一个 \n 为止）
        r = subprocess.run(
            [CLAUDE, "-p", "--output-format", "text",
             "--allowedTools", "Read", "Grep", "Glob"],
            input=prompt,
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            cwd=str(REPO), timeout=CLAUDE_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "claude 响应超时（%d 秒）——问题可能太大了，拆小一点再问" % CLAUDE_TIMEOUT}
    except OSError as e:
        return {"ok": False, "error": "启动 claude 失败：%s" % e}
    out = (r.stdout or "").strip()
    if not out:
        err = (r.stderr or "").strip()[:400]
        return {"ok": False, "error": "claude 没有返回内容（退出码 %s）%s" % (r.returncode, ("：" + err) if err else "")}
    return {"ok": True, "answer": out}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(REPO), **kw)

    def do_POST(self):
        if self.path.split("?")[0] != "/ask":
            self._json(404, {"ok": False, "error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b""
            payload = json.loads(raw.decode("utf-8", "replace") or "{}")
        except Exception as e:
            self._json(400, {"ok": False, "error": "请求体解析失败：%s" % e})
            return
        if payload.get("probe"):          # 页面启动时的连通性探测，不惊动 claude
            if CLAUDE:
                self._json(200, {"ok": True, "claude": Path(CLAUDE).name})
            else:
                self._json(200, {"ok": False, "error": "服务器找不到 claude CLI（装好并加入 PATH 后重启本服务）"})
            return
        print("[ask] %s" % (payload.get("question") or "")[:80], flush=True)
        if not CLAUDE:
            self._json(500, {"ok": False, "error": "服务器找不到 claude CLI（装好并加入 PATH 后重启本服务）"})
            return
        self._json(200, ask_claude(payload))

    def do_OPTIONS(self):
        # 浏览器跨源 POST 前的预检请求：直接放行（本工具仅本机使用，放开无妨）
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        sys.stderr.write("%s\n" % (fmt % args))


def main():
    if not CLAUDE:
        print("⚠ 未找到 claude CLI —— 静态伺服照常，但 💬 问答窗口不可用", flush=True)
    else:
        print("claude CLI：%s" % CLAUDE, flush=True)
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError as e:
        print("❌ 端口 %d 起不来：%s" % (PORT, e), flush=True)
        print("   多半是旧服务器（python -m http.server 8765）还开着：关掉那个窗口再重跑本脚本", flush=True)
        sys.exit(1)
    srv.daemon_threads = True
    print("", flush=True)
    print("复习站服务器已启动：http://127.0.0.1:%d/step_all_review.html" % PORT, flush=True)
    print("（Ctrl+F5 强刷页面，💬 问答窗口即可用）", flush=True)
    print("静态根目录：%s" % REPO, flush=True)
    print("问答代理：POST /ask → claude CLI 无头模式（超时 %d 秒/次）" % CLAUDE_TIMEOUT, flush=True)
    print("Ctrl+C 停止", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止", flush=True)


if __name__ == "__main__":
    main()
