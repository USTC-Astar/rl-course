#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV="$ROOT/.tools/uv"

if [[ ! -x "$UV" ]]; then
  echo "尚未安装项目环境，请先运行：./scripts/bootstrap.sh" >&2
  exit 1
fi

cd "$ROOT"
exec "$UV" run "$@"

