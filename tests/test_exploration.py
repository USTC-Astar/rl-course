"""第 19 课探索方法实验的正确性测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_learning_lab.exploration import (
    ARM_PROBABILITIES,
    average_runs,
    run_epsilon_greedy,
    run_thompson,
    run_ucb,
)


def test_all_strategies_do_better_than_random() -> None:
    # 随机策略的期望单步奖励 = 各臂概率均值；探索策略应显著高于它。
    random_baseline = float(np.mean(ARM_PROBABILITIES))
    for strategy, kwargs in (
        (run_epsilon_greedy, {"epsilon": 0.1}),
        (run_ucb, {"exploration": 2.0}),
        (run_thompson, {}),
    ):
        average = average_runs(strategy, steps=800, runs=30, base_seed=5, **kwargs)
        assert np.mean(average) > random_baseline


def test_thompson_converges_near_best_arm() -> None:
    # 长期看，汤普森采样应把大多数时间花在最优臂（p=0.7）附近。
    average = average_runs(run_thompson, steps=2000, runs=40, base_seed=9)
    assert np.mean(average[-200:]) > 0.6
