#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_learning_lab.cartpole_dqn import DQNConfig, train_dqn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 CartPole 深度 Q 网络")
    parser.add_argument("--steps", type=int, default=150_000, help="环境交互总步数")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "web" / "data" / "cartpole.json",
        help="网页数据输出路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DQNConfig(total_steps=args.steps)
    result = train_dqn(config, ROOT / "artifacts" / "cartpole_dqn.pt")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    evaluation = result["evaluation"]
    print(
        "CartPole 训练完成："
        f"步数={args.steps}，评估平均回报={evaluation['average_return']:.1f}，"
        f"最佳回报={evaluation['best_return']:.1f}，输出={args.output}"
    )


if __name__ == "__main__":
    main()
