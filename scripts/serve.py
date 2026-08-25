#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.server
import socketserver
from functools import partial
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SITE_ROOT = ROOT / "web" / "dist"


def main() -> None:
    parser = argparse.ArgumentParser(description="启动强化学习教学网页")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if not SITE_ROOT.is_dir():
        raise SystemExit(
            "尚未构建教学网页，请先运行：cd web && npm install && npm run build"
        )

    # Ubuntu 20.04 自带的 Python 3.8 在目录请求中仍会把 directory 当作字符串拼接，
    # 因此这里显式转换，保证首页 `/` 和静态资源都能正常访问。
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(SITE_ROOT))
    with socketserver.TCPServer((args.host, args.port), handler) as server:
        print(f"教学网页已启动：http://{args.host}:{args.port}")
        print("按 Ctrl+C 停止服务。")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n网页服务已停止。")


if __name__ == "__main__":
    main()
