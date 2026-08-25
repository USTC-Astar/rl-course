#!/usr/bin/env python3
"""生成第 21 课（调试指南）的真实失败曲线数据。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_learning_lab.debugging import run_debug_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="生成第 21 课数据")
    parser.add_argument("--output", type=Path, default=ROOT / "web" / "data" / "debug_lessons.json")
    args = parser.parse_args()

    result = run_debug_suite()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    summary = "，".join(
        f"{name}: 末100均值={data['mean_reward_last_100']}"
        for name, data in result["scenarios"].items()
    )
    print(f"第 21 课数据完成：{summary}，输出={args.output}")


if __name__ == "__main__":
    main()
