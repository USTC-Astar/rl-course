"""第 0 课配套：模型已知时的动态规划求解器。

与 gridworld.py 共用同一张 6×6 地图，但这里假设转移概率 P 和奖励 R
完全已知（动态规划的前提），用价值迭代和策略迭代直接解贝尔曼最优方程。
真实强化学习（第 1 课起）研究的正是 P 与 R 未知时怎么办。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DPResult:
    """动态规划求解结果。

    values：收敛后的状态价值表（rows × cols）。
    policy：每个状态的最优动作编号，0=上 1=右 2=下 3=左。
    sweep_errors：价值迭代每一轮的最大更新量，用于展示收敛过程。
    iterations：迭代轮数（价值迭代=扫描轮数，策略迭代=改进轮数）。
    route：从起点按最优策略走到终点的坐标序列。
    """

    values: np.ndarray
    policy: np.ndarray
    sweep_errors: list[float]
    iterations: int
    route: list[tuple[int, int]]


ROWS = 6
COLS = 6
START = (5, 0)
GOAL = (0, 5)
WALLS = {(1, 1), (2, 1), (3, 1), (1, 3), (2, 3), (4, 4)}
TRAPS = {(3, 3), (4, 2)}
STEP_REWARD = -0.1
WALL_REWARD = -1.0
TRAP_REWARD = -10.0
GOAL_REWARD = 10.0
DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))
GAMMA = 0.95


def _walkable(position: tuple[int, int]) -> bool:
    row, col = position
    inside = 0 <= row < ROWS and 0 <= col < COLS
    return inside and position not in WALLS


def _terminal(position: tuple[int, int]) -> bool:
    return position == GOAL or position in TRAPS


def expected_reward(state: tuple[int, int], action: int) -> tuple[tuple[int, int], float, bool]:
    """返回（后继位置, 期望奖励, 是否终止）。

    环境是确定性的，所以期望就是现实本身；
    真实问题里这里是“对所有可能后继按 P 求加权平均”。
    """
    if _terminal(state):
        return state, 0.0, True
    row, col = state
    candidate = (row + DELTAS[action][0], col + DELTAS[action][1])
    if not _walkable(candidate):
        return state, WALL_REWARD, False
    if candidate == GOAL:
        return candidate, GOAL_REWARD, True
    if candidate in TRAPS:
        return candidate, TRAP_REWARD, True
    return candidate, STEP_REWARD, False


def value_iteration(tolerance: float = 1e-8, max_sweeps: int = 200) -> DPResult:
    """价值迭代：反复应用贝尔曼最优方程直到价值表不再变化。

    每轮扫描把每个状态的价值替换为
        max_a [ r(s,a) + γ · V(s′) ]，
    终止状态的价值恒为 0（没有未来）。
    """
    values = np.zeros((ROWS, COLS))
    policy = np.zeros((ROWS, COLS), dtype=int)
    sweep_errors: list[float] = []
    iterations = 0

    for sweep in range(max_sweeps):
        new_values = values.copy()
        max_change = 0.0
        for row in range(ROWS):
            for col in range(COLS):
                state = (row, col)
                if _terminal(state) or state in WALLS:
                    continue
                best = -np.inf
                best_action = 0
                for action in range(4):
                    successor, reward, done = expected_reward(state, action)
                    future = 0.0 if done else values[successor]
                    candidate = reward + GAMMA * future
                    if candidate > best:
                        best = candidate
                        best_action = action
                new_values[row, col] = best
                policy[row, col] = best_action
                max_change = max(max_change, abs(best - values[row, col]))
        values = new_values
        sweep_errors.append(max_change)
        iterations = sweep + 1
        if max_change < tolerance:
            break

    return DPResult(
        values=values,
        policy=policy,
        sweep_errors=sweep_errors,
        iterations=iterations,
        route=greedy_route(policy),
    )


def policy_evaluation(policy: np.ndarray, tolerance: float = 1e-8, max_sweeps: int = 200) -> np.ndarray:
    """策略评估：固定策略，反复应用贝尔曼期望方程求 V^π。"""
    values = np.zeros((ROWS, COLS))
    for _ in range(max_sweeps):
        max_change = 0.0
        for row in range(ROWS):
            for col in range(COLS):
                state = (row, col)
                if _terminal(state) or state in WALLS:
                    continue
                successor, reward, done = expected_reward(state, policy[row, col])
                future = 0.0 if done else values[successor]
                new_value = reward + GAMMA * future
                max_change = max(max_change, abs(new_value - values[row, col]))
                values[row, col] = new_value
        if max_change < tolerance:
            break
    return values


def policy_iteration() -> DPResult:
    """策略迭代：评估与改进交替，直到策略不再变化。

    每轮改进都对每个状态取 argmax_a [ r + γ V^π(s′) ]。
    记录每轮“策略价值是否变化”作为收敛证据。
    """
    policy = np.zeros((ROWS, COLS), dtype=int)
    sweep_errors: list[float] = []
    iterations = 0

    for step in range(50):
        values = policy_evaluation(policy)
        stable = True
        for row in range(ROWS):
            for col in range(COLS):
                state = (row, col)
                if _terminal(state) or state in WALLS:
                    continue
                old_action = policy[row, col]
                best = -np.inf
                best_action = old_action
                for action in range(4):
                    successor, reward, done = expected_reward(state, action)
                    future = 0.0 if done else values[successor]
                    candidate = reward + GAMMA * future
                    if candidate > best:
                        best = candidate
                        best_action = action
                if best_action != old_action:
                    policy[row, col] = best_action
                    stable = False
        # 用“整轮改进的期望价值增量”近似收敛误差，便于画图对比。
        new_values = policy_evaluation(policy)
        change = float(np.abs(new_values - values).max())
        sweep_errors.append(change)
        iterations = step + 1
        if stable:
            break

    return DPResult(
        values=policy_evaluation(policy),
        policy=policy,
        sweep_errors=sweep_errors,
        iterations=iterations,
        route=greedy_route(policy),
    )


def greedy_route(policy: np.ndarray) -> list[tuple[int, int]]:
    """按给定（整数）策略从起点走到终点，最多 60 步，防止环路死循环。"""
    assert policy.dtype.kind == "i", "greedy_route 需要整数策略表"
    position = START
    route = [position]
    for _ in range(60):
        if _terminal(position):
            break
        successor, _, _ = expected_reward(position, int(policy[position]))
        if successor == position:
            break
        position = successor
        if position in route:
            break
        route.append(position)
    return route


def q_learning_comparison_table() -> list[dict[str, float | str]]:
    """给出第 1 课 Q 学习收敛后与动态规划最优值的对比行（供网页表格展示）。"""
    optimal = value_iteration()
    rows: list[dict[str, float | str]] = []
    for state in (START, (5, 1), (5, 2), (4, 0), (0, 4)):
        action = optimal.policy[state]
        successor, reward, done = expected_reward(state, action)
        future = 0.0 if done else optimal.values[successor]
        rows.append(
            {
                "state": f"({state[0]},{state[1]})",
                "optimal_value": round(float(optimal.values[state]), 3),
                "one_step_target": round(reward + GAMMA * future, 3),
            }
        )
    return rows
