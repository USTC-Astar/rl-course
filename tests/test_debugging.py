"""第 21 课失败注入实验的正确性测试。断言与真实观测到的效应一致。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_learning_lab.debugging import run_debug_suite, train_with_bugs


def test_healthy_baseline_learns_despite_slip() -> None:
    result = train_with_bugs(episodes=1500)
    assert result["mean_reward_last_100"] > 4.0
    assert result["success_rate_last_100"] >= 0.95


def test_wrong_action_update_badly_hurts() -> None:
    # 更新错槽位：末段均值大幅下降、波动成倍上升。
    healthy = train_with_bugs(episodes=1500)
    result = train_with_bugs(episodes=1500, wrong_action_update=True)
    assert result["mean_reward_last_100"] < healthy["mean_reward_last_100"] * 0.7
    assert result["std_reward_last_100"] > healthy["std_reward_last_100"] * 2


def test_huge_alpha_oscillates() -> None:
    # α=1 在有打滑的环境里目标噪声完全进入 Q 表：末段波动远大于健康版。
    healthy = train_with_bugs(episodes=1500)
    result = train_with_bugs(episodes=1500, alpha=1.0)
    assert result["std_reward_last_100"] > healthy["std_reward_last_100"] * 3
    assert result["mean_reward_last_100"] < healthy["mean_reward_last_100"]


def test_short_sighted_degrades_average_reward() -> None:
    # γ=0.3 视野变短：平均奖励下降、波动上升（本环境仍常能到达）。
    healthy = train_with_bugs(episodes=1500)
    result = train_with_bugs(episodes=1500, gamma=0.3)
    assert result["mean_reward_last_100"] < healthy["mean_reward_last_100"]
    assert result["std_reward_last_100"] > healthy["std_reward_last_100"]


def test_no_exploration_is_seed_dependent() -> None:
    suite = run_debug_suite(episodes=800)
    seeds = suite["no_exploration"]["per_seed_mean_reward_last_100"]
    # 种子间差异巨大：至少一个种子显著掉队，另一个接近健康水平。
    assert max(seeds) - min(seeds) > 3
    assert min(seeds) < 5.0
    assert max(seeds) > 6.5
