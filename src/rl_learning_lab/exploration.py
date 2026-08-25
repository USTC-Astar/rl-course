"""第 19 课配套：多臂赌博机上的探索方法对照实验。

k 臂伯努利赌博机：每个动作以固定但未知的概率 p_i 给奖励 1，否则 0。
对照三种探索策略在同一组问题、同一批种子下的平均表现：
ε-贪心、UCB、汤普森采样。全部为真实采样运行，无解析近似。
"""

from __future__ import annotations

import numpy as np

# 10 个动作的真实成功概率：最优动作是 0 号臂（p=0.7），还有几个“诱人但次优”的臂。
ARM_PROBABILITIES = (0.7, 0.5, 0.45, 0.6, 0.2, 0.1, 0.35, 0.55, 0.15, 0.4)


def _pull(rng: np.random.Generator, arm: int) -> float:
    return 1.0 if rng.random() < ARM_PROBABILITIES[arm] else 0.0


def run_epsilon_greedy(epsilon: float, steps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    counts = np.zeros(len(ARM_PROBABILITIES))
    values = np.zeros(len(ARM_PROBABILITIES))
    rewards = np.zeros(steps)
    for step in range(steps):
        if rng.random() < epsilon:
            arm = int(rng.integers(0, len(ARM_PROBABILITIES)))
        else:
            arm = int(np.argmax(values))
        reward = _pull(rng, arm)
        counts[arm] += 1
        # 增量平均：等价于对每个臂分别维护样本均值。
        values[arm] += (reward - values[arm]) / counts[arm]
        rewards[step] = reward
    return rewards


def run_ucb(exploration: float, steps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = len(ARM_PROBABILITIES)
    counts = np.zeros(k)
    values = np.zeros(k)
    rewards = np.zeros(steps)
    for step in range(steps):
        # 前 k 步先把每个臂各拉一次，保证 log/计数 有定义。
        if step < k:
            arm = step
        else:
            bonus = exploration * np.sqrt(np.log(step + 1) / counts)
            arm = int(np.argmax(values + bonus))
        reward = _pull(rng, arm)
        counts[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]
        rewards[step] = reward
    return rewards


def run_thompson(steps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = len(ARM_PROBABILITIES)
    # Beta(1,1) 均匀先验：成功 alpha+=1，失败 beta+=1。
    alpha = np.ones(k)
    beta = np.ones(k)
    rewards = np.zeros(steps)
    for step in range(steps):
        samples = rng.beta(alpha, beta)
        arm = int(np.argmax(samples))
        reward = _pull(rng, arm)
        if reward > 0:
            alpha[arm] += 1
        else:
            beta[arm] += 1
        rewards[step] = reward
    return rewards


def average_runs(strategy, steps: int, runs: int, base_seed: int, **kwargs) -> list[float]:
    """同一策略在 runs 个种子上的逐步平均奖励。"""
    total = np.zeros(steps)
    for run in range(runs):
        total += strategy(steps=steps, seed=base_seed + run, **kwargs)
    return [round(float(v / runs), 4) for v in total]


def run_comparison(steps: int = 2000, runs: int = 100, seed: int = 2024) -> dict:
    best = max(ARM_PROBABILITIES)
    strategies = {
        "epsilon_greedy_01": average_runs(run_epsilon_greedy, steps, runs, seed, epsilon=0.1),
        "epsilon_greedy_005": average_runs(run_epsilon_greedy, steps, runs, seed, epsilon=0.05),
        "ucb_c05": average_runs(run_ucb, steps, runs, seed, exploration=0.5),
        # 保留一组激进系数，展示同一算法因超参数不同而表现迥异。
        "ucb_c20": average_runs(run_ucb, steps, runs, seed, exploration=2.0),
        "thompson": average_runs(run_thompson, steps, runs, seed),
    }
    cumulative: dict[str, list[float]] = {}
    for name, per_step in strategies.items():
        running = 0.0
        curve = []
        for reward in per_step:
            running += reward
            curve.append(round(running, 3))
        cumulative[name] = curve
    # 遗憾 = 最优期望 - 实际累计平均
    regret = {
        name: round(best * steps - curve[-1], 2) for name, curve in cumulative.items()
    }
    return {
        "arm_probabilities": list(ARM_PROBABILITIES),
        "best_probability": best,
        "steps": steps,
        "runs": runs,
        "per_step_average": strategies,
        "cumulative_average": cumulative,
        "cumulative_regret": regret,
    }
