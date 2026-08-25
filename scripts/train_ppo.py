#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_learning_lab.ppo import PPOConfig, train_ppo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 CartPole PPO")
    parser.add_argument("--steps", type=int, default=120_000, help="最大环境交互步数")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "web" / "data" / "ppo.json",
        help="网页数据输出路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PPOConfig(total_steps=args.steps)
    result = train_ppo(config, ROOT / "artifacts" / "cartpole_ppo.pt")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    evaluation = result["evaluation"]
    print(
        "PPO 训练完成："
        f"实际步数={result['steps_completed']}，更新={result['update_count']}，"
        f"评估平均回报={evaluation['average_return']:.1f}，"
        f"最佳回报={evaluation['best_return']:.1f}，输出={args.output}"
    )


if __name__ == "__main__":
    main()

