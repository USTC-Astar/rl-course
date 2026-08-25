"""第 18 课蒙特卡洛与 SARSA 实验的正确性测试。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_learning_lab.td_variants import run_mc_prediction, run_sarsa_vs_q


def test_mc_converges_to_random_policy_truth() -> None:
    # 起点每回合必被访问，4000 回合后估计应接近均匀随机策略的真值（约 -9.5）。
    result = run_mc_prediction(episodes=4000)
    assert result["visit_counts"][0] == 4000
    assert abs(result["final_estimates"][0] - result_truth) < 0.3


result_truth = -9.4538  # 由 generate_td_variants.py 的精确评估给出，随环境定义同步维护


def test_sarsa_and_q_both_learn_to_reach_goal() -> None:
    result = run_sarsa_vs_q(episodes=600)
    # 两种 TD 控制都应在最后 100 回合里绝大多数时候到达终点。
    assert result["sarsa"]["success_rate_last_100"] >= 0.9
    assert result["q_learning"]["success_rate_last_100"] >= 0.9
    # 平均奖励为正：到达 +10 的收益超过步数成本。
    assert result["sarsa"]["mean_reward_last_100"] > 0
    assert result["q_learning"]["mean_reward_last_100"] > 0
