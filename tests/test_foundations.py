"""第 0 课动态规划求解器的正确性测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_learning_lab.foundations import (
    GOAL,
    START,
    TRAPS,
    expected_reward,
    policy_iteration,
    value_iteration,
)


def test_terminal_states_have_zero_future() -> None:
    # 终点与陷阱是吸收态：任何动作都不再改变状态，也不再有奖励。
    for state in (GOAL, *TRAPS):
        successor, reward, done = expected_reward(state, 0)
        assert successor == state
        assert reward == 0.0
        assert done


def test_wall_bump_keeps_position() -> None:
    # 在 (1,0) 向右是墙 (1,1)：位置不动、拿墙惩罚、不终止。
    successor, reward, done = expected_reward((1, 0), 1)
    assert successor == (1, 0)
    assert reward == -1.0
    assert not done


def test_value_iteration_converges_and_reaches_goal() -> None:
    result = value_iteration()
    assert result.iterations >= 2
    assert result.sweep_errors[-1] < 1e-6
    # 路线必须真正到达终点，且不穿过陷阱。
    assert result.route[-1] == GOAL
    for position in result.route:
        assert position not in TRAPS


def test_policy_iteration_agrees_with_value_iteration() -> None:
    vi = value_iteration()
    pi = policy_iteration()
    # 两种解法都解同一个贝尔曼最优方程，最优价值必须一致（容差内）。
    assert np.abs(vi.values - pi.values).max() < 1e-4
    assert vi.route == pi.route


def test_optimal_value_at_start_is_positive() -> None:
    # 起点沿最优路线能拿到 +10 终点奖励，扣掉步数成本后仍应为正。
    result = value_iteration()
    assert result.values[START] > 0
