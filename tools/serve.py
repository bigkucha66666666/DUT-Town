"""本地静态服务：给 web/ 查看器用。

不要直接双击打开 web/index.html —— 浏览器会以 file:// 协议拒绝
fetch 本地 JSON（CORS）。用这个脚本起一个带线程池的静态服务：

    python tools/serve.py                 # 默认 http://127.0.0.1:8123/web/
    python tools/serve.py --port 9000

用 ThreadingHTTPServer 而不是标准库默认的单线程 HTTPServer：
浏览器会并发取 index.html / styles.css / app.js / dut_sem.json，
单线程服务会让它们排队，页面有时要等好几秒才出图。
"""

from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
import sys
from pathlib import Path


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        # 开发期禁缓存，改完地图刷新就能看到
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:  # 精简日志
        if "200" not in (fmt % args):
            super().log_message(fmt, *args)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main(argv=None) -> int:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="DUT 小镇 · 本地地图服务")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--root", default=str(root), help="静态根目录（默认项目根）")
    args = ap.parse_args(argv)

    handler = functools.partial(Handler, directory=args.root)
    url = f"http://{args.host}:{args.port}/web/"

    with Server((args.host, args.port), handler) as httpd:
        print(f"静态根目录：{args.root}")
        print(f"地图查看器：{url}")
        print(f"地图数据　：http://{args.host}:{args.port}/world/maps/dut_sem.json")
        print("Ctrl+C 停止")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n已停止")
    return 0


if __name__ == "__main__":
    sys.exit(main())
