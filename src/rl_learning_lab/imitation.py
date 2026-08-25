"""第 20 课配套：行为克隆的协变量偏移演示（表格级真实模拟）。

专家 = 动态规划最优策略，从固定起点 (5,0) 出发收集数据（带少量动作噪声）。
行为克隆只在专家访问过的状态上学习，其余状态只能均匀乱走。
演示两件事：
1. 分布内（专家走过的起点）：BC 与专家表现几乎一样；
2. 分布外（专家从未到过的起点）：专家策略照常工作，BC 大幅崩溃——
   因为 BC 学的是"专家去过的地方怎么走"，而不是"任务本身怎么解"。
"""

from __future__ import annotations

import numpy as np

from rl_learning_lab.foundations import value_iteration
from rl_learning_lab.gridworld import GridWorld, GridWorldConfig

NOISE = 0.05
EPISODES = 20
SEED = 33


def collect_expert_dataset(episodes: int = EPISODES, noise: float = NOISE, seed: int = SEED):
    """专家从固定起点收集 (状态, 动作) 数据，动作带少量噪声。"""
    expert = value_iteration().policy
    env = GridWorld(GridWorldConfig())
    rng = np.random.default_rng(seed)
    counts: dict[int, np.ndarray] = {}

    for _ in range(episodes):
        state = env.reset()
        for _ in range(env.config.max_steps):
            position = env.state_to_position(state)
            action = int(expert[position])
            if rng.random() < noise:
                action = int(rng.integers(0, env.action_count))
            if state not in counts:
                counts[state] = np.zeros(env.action_count)
            counts[state][action] += 1
            state, _, done = env.step(action)
            if done:
                break
    return counts


def bc_action(counts: dict[int, np.ndarray], state: int, action_count: int, rng: np.random.Generator) -> int:
    """行为克隆策略：见过按动作频率采样；没见过均匀随机。"""
    if state in counts:
        probabilities = counts[state] / counts[state].sum()
        return int(rng.choice(action_count, p=probabilities))
    return int(rng.integers(0, action_count))


def _rollout(env: GridWorld, start_state: int, pick_action, max_steps: int) -> bool:
    state = env.reset()
    env.state = env.state_to_position(state=start_state)
    state = start_state
    for _ in range(max_steps):
        action = pick_action(state)
        state, _, done = env.step(action)
        if done:
            return env.state == env.config.goal
    return False


def evaluate_from_random_starts(counts: dict[int, np.ndarray], episodes: int = 800, seed: int = SEED + 1):
    """从随机起点分别评估专家策略与 BC 策略，按“起点是否被数据覆盖”分组。"""
    expert = value_iteration().policy
    env = GridWorld(GridWorldConfig())
    rng = np.random.default_rng(seed)

    walls = {env.position_to_state(w) for w in env.config.walls}
    traps = {env.position_to_state(t) for t in env.config.traps}
    goal = env.position_to_state(env.config.goal)
    candidates = [s for s in range(env.state_count) if s not in walls and s not in traps and s != goal]

    def expert_pick(state: int) -> int:
        return int(expert[env.state_to_position(state)])

    def bc_pick(state: int) -> int:
        return bc_action(counts, state, env.action_count, rng)

    groups = {"covered": [0, 0, 0], "uncovered": [0, 0, 0]}  # [专家成功, BC成功, 回合数]
    for _ in range(episodes):
        start = int(candidates[rng.integers(0, len(candidates))])
        key = "covered" if start in counts else "uncovered"
        expert_ok = _rollout(env, start, expert_pick, env.config.max_steps)
        bc_ok = _rollout(env, start, bc_pick, env.config.max_steps)
        groups[key][0] += int(expert_ok)
        groups[key][1] += int(bc_ok)
        groups[key][2] += 1

    def rate(group, index):
        runs = groups[group][2]
        return round(groups[group][index] / max(runs, 1), 4)

    return {
        "covered_starts": len([s for s in candidates if s in counts]),
        "uncovered_starts": len([s for s in candidates if s not in counts]),
        "expert_success_covered": rate("covered", 0),
        "bc_success_covered": rate("covered", 1),
        "expert_success_uncovered": rate("uncovered", 0),
        "bc_success_uncovered": rate("uncovered", 1),
        "episodes": episodes,
    }


def run_imitation_demo() -> dict[str, object]:
    counts = collect_expert_dataset()
    stats = evaluate_from_random_starts(counts)
    return {
        "demo_episodes": EPISODES,
        "expert_action_noise": NOISE,
        "bc_states_covered": len(counts),
        "total_non_terminal_states": GridWorld(GridWorldConfig()).state_count,
        **stats,
    }
