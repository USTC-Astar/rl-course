#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEB_ROOT="$ROOT/web"

if [[ ! -d "$WEB_ROOT/node_modules" ]]; then
  echo "首次启动：正在安装 VitePress 网页依赖……"
  npm --prefix "$WEB_ROOT" install
fi

echo "正在同步训练数据并构建 17 课静态网页……"
npm --prefix "$WEB_ROOT" run build

cd "$ROOT"
exec python3 "$ROOT/scripts/serve.py"
