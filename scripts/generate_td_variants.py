#!/usr/bin/env python3
"""生成第 18 课（蒙特卡洛与 SARSA）的网页数据。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_learning_lab.foundations import value_iteration
from rl_learning_lab.td_variants import run_mc_prediction, run_sarsa_vs_q


def evaluate_uniform_policy(gamma: float = 0.95, tolerance: float = 1e-10) -> dict:
    """用动态规划精确评估均匀随机策略（每动作 1/4），作为 MC 估计的真值。

    MC 估计的是“随机策略的价值”，真值必须来自同一策略的精确评估，
    而不是最优价值——两者是完全不同的量。
    """
    import numpy as np

    from rl_learning_lab.foundations import ROWS, COLS, WALLS, expected_reward

    values = np.zeros((ROWS, COLS))
    converged = False
    for _ in range(5000):
        new_values = values.copy()
        for row in range(ROWS):
            for col in range(COLS):
                state = (row, col)
                if state in WALLS:
                    continue
                total = 0.0
                for action in range(4):
                    successor, reward, done = expected_reward(state, action)
                    future = 0.0 if done else values[successor]
                    total += reward + gamma * future
                new_values[row, col] = total / 4
        change = float(np.abs(new_values - values).max())
        values = new_values
        if change < tolerance:
            converged = True
            break
    return {"values": values, "converged": converged}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成第 18 课数据")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "web" / "data" / "td_variants.json",
        help="网页数据输出路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mc = run_mc_prediction()
    random_policy = evaluate_uniform_policy()
    optimal = value_iteration()
    comparison = run_sarsa_vs_q()

    targets = [tuple(t) for t in mc["targets"]]
    payload = {
        "mc_prediction": {
            **mc,
            "dp_random_policy_values": [round(float(random_policy["values"][t]), 4) for t in targets],
            "optimal_values": [round(float(optimal.values[t]), 4) for t in targets],
        },
        "sarsa_vs_q": comparison,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(
        f"第 18 课数据完成：MC 估计={[', '.join(str(v) for v in mc['final_estimates'])]}，"
        f"随机策略真值={[', '.join(str(v) for v in payload['mc_prediction']['dp_random_policy_values'])]}，"
        f"最优价值={[', '.join(str(v) for v in payload['mc_prediction']['optimal_values'])]}，输出={args.output}"
    )


if __name__ == "__main__":
    main()
