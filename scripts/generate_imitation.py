#!/usr/bin/env python3
"""生成第 20 课（模仿学习）的网页数据。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_learning_lab.imitation import run_imitation_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="生成第 20 课数据")
    parser.add_argument("--output", type=Path, default=ROOT / "web" / "data" / "imitation.json")
    args = parser.parse_args()

    result = run_imitation_demo()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(
        f"第 20 课数据完成：覆盖起点 {result['bc_states_covered']} 状态，"
        f"分布内专家/BC 成功率={result['expert_success_covered']}/{result['bc_success_covered']}，"
        f"分布外专家/BC 成功率={result['expert_success_uncovered']}/{result['bc_success_uncovered']}，输出={args.output}"
    )


if __name__ == "__main__":
    main()
