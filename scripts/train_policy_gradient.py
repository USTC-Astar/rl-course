#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_learning_lab.policy_gradient import PolicyGradientConfig, train_policy_gradient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 CartPole REINFORCE 策略梯度")
    parser.add_argument("--episodes", type=int, default=600, help="最大训练回合数")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "web" / "data" / "policy_gradient.json",
        help="网页数据输出路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PolicyGradientConfig(episodes=args.episodes)
    result = train_policy_gradient(
        config,
        ROOT / "artifacts" / "cartpole_policy_gradient.pt",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    evaluation = result["evaluation"]
    print(
        "策略梯度训练完成："
        f"实际回合={result['episodes_completed']}，评估平均回报={evaluation['average_return']:.1f}，"
        f"最佳回报={evaluation['best_return']:.1f}，输出={args.output}"
    )


if __name__ == "__main__":
    main()
