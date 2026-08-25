#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_learning_lab.actor_critic import ActorCriticConfig, train_actor_critic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 CartPole 行动者-评论家")
    parser.add_argument("--steps", type=int, default=300_000, help="最大环境交互步数")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "web" / "data" / "actor_critic.json",
        help="网页数据输出路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ActorCriticConfig(total_steps=args.steps)
    result = train_actor_critic(
        config,
        ROOT / "artifacts" / "cartpole_actor_critic.pt",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    evaluation = result["evaluation"]
    print(
        "行动者-评论家训练完成："
        f"实际步数={result['steps_completed']}，更新={result['update_count']}，"
        f"评估平均回报={evaluation['average_return']:.1f}，"
        f"最佳回报={evaluation['best_return']:.1f}，输出={args.output}"
    )


if __name__ == "__main__":
    main()
