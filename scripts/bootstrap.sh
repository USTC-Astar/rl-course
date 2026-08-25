#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="$ROOT/.tools"
UV="$TOOLS_DIR/uv"

mkdir -p "$TOOLS_DIR"

if [[ ! -x "$UV" ]]; then
  echo "正在从 Astral 官方安装脚本下载 uv 到项目目录……"
  curl -LsSf https://astral.sh/uv/install.sh | env \
    UV_INSTALL_DIR="$TOOLS_DIR" \
    UV_NO_MODIFY_PATH=1 \
    sh
fi

cd "$ROOT"
echo "正在安装隔离的 Python 3.11 和项目依赖……"
UV_PYTHON_INSTALL_DIR="$ROOT/.python" "$UV" sync --python 3.11

echo "环境准备完成。Python 与依赖都位于项目目录，不会替换系统 Python。"

