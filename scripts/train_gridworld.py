#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_learning_lab.gridworld import GridWorld, GridWorldConfig, QLearningConfig, train_q_learning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练网格世界 Q 学习智能体")
    parser.add_argument("--episodes", type=int, default=3000, help="训练回合数")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "web" / "data" / "gridworld.json",
        help="网页数据输出路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = GridWorld(GridWorldConfig())
    train_config = QLearningConfig(episodes=args.episodes)
    result = train_q_learning(env, train_config)

    payload = {
        "environment": {
            **asdict(env.config),
            "start": list(env.config.start),
            "goal": list(env.config.goal),
            "walls": [list(item) for item in env.config.walls],
            "traps": [list(item) for item in env.config.traps],
            "actions": list(env.ACTIONS),
        },
        "training": {
            "config": asdict(result["config"]),
            "episode_rewards": [round(value, 4) for value in result["rewards"]],
            "episode_steps": result["episode_steps"],
            "epsilon_history": [round(value, 5) for value in result["epsilon_history"]],
            "snapshots": result["snapshots"],
        },
        "learned": {
            "q_table": result["q_table"].round(4).tolist(),
            "route": [list(position) for position in result["route"]],
            "route_reward": round(float(result["route_reward"]), 4),
            "reached_goal": result["reached_goal"],
            "success_rate": result["success_rate"],
            "average_recent_reward": round(float(result["average_recent_reward"]), 4),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(
        "网格世界训练完成："
        f"回合={args.episodes}，到达终点={result['reached_goal']}，"
        f"路线步数={len(result['route']) - 1}，输出={args.output}"
    )


if __name__ == "__main__":
    main()

