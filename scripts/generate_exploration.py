#!/usr/bin/env python3
"""生成第 19 课（探索方法）的网页数据。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_learning_lab.exploration import run_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="生成第 19 课数据")
    parser.add_argument("--output", type=Path, default=ROOT / "web" / "data" / "exploration.json")
    args = parser.parse_args()

    result = run_comparison()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    regret = "，".join(f"{k}={v}" for k, v in result["cumulative_regret"].items())
    print(f"第 19 课数据完成：2000 步累计遗憾 {regret}，输出={args.output}")


if __name__ == "__main__":
    main()
