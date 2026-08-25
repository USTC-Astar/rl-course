#!/usr/bin/env python3
"""生成第 0 课（动态规划）与第 18 课（蒙特卡洛、SARSA）的网页数据。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from rl_learning_lab.foundations import (
    GOAL,
    GAMMA,
    ROWS,
    COLS,
    policy_iteration,
    value_iteration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成基础理论课程数据")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "web" / "data" / "foundations.json",
        help="网页数据输出路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vi = value_iteration()
    pi = policy_iteration()

    payload = {
        "environment": {
            "rows": ROWS,
            "cols": COLS,
            "goal": list(GOAL),
            "discount": GAMMA,
        },
        "value_iteration": {
            "sweep_errors": [round(float(e), 8) for e in vi.sweep_errors],
            "sweeps": vi.iterations,
            "final_values": vi.values.round(3).tolist(),
            "route": [list(p) for p in vi.route],
            "route_steps": len(vi.route) - 1,
        },
        "policy_iteration": {
            "improvement_changes": [round(float(e), 8) for e in pi.sweep_errors],
            "iterations": pi.iterations,
            "final_values": pi.values.round(3).tolist(),
            "route": [list(p) for p in pi.route],
            "route_steps": len(pi.route) - 1,
        },
        "agreement": {
            "value_gap": float(np.abs(vi.values - pi.values).max()),
            "same_route": [list(p) for p in vi.route] == [list(p) for p in pi.route],
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(
        f"动态规划数据完成：价值迭代 {vi.iterations} 轮收敛，"
        f"策略迭代 {pi.iterations} 轮收敛，"
        f"两法价值差 ≤ {payload['agreement']['value_gap']:.2e}，输出={args.output}"
    )


if __name__ == "__main__":
    main()
