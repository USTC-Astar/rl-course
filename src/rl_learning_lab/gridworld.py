from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


Position = tuple[int, int]


@dataclass(frozen=True)
class GridWorldConfig:
    """网格环境参数。

    奖励采用“小步惩罚 + 明确终点奖励”的设计，是为了让智能体不只学会
    “最终到达”，还会倾向于选择更短的路径。
    """

    rows: int = 6
    cols: int = 6
    start: Position = (5, 0)
    goal: Position = (0, 5)
    walls: tuple[Position, ...] = (
        (1, 1),
        (2, 1),
        (3, 1),
        (1, 3),
        (2, 3),
        (4, 4),
    )
    traps: tuple[Position, ...] = ((3, 3), (4, 2))
    step_reward: float = -0.1
    wall_reward: float = -1.0
    trap_reward: float = -10.0
    goal_reward: float = 10.0
    max_steps: int = 100


class GridWorld:
    ACTIONS = ("up", "right", "down", "left")
    ACTION_DELTAS: tuple[Position, ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))

    def __init__(self, config: GridWorldConfig | None = None) -> None:
        self.config = config or GridWorldConfig()
        self.state = self.config.start
        self.steps = 0

    @property
    def state_count(self) -> int:
        return self.config.rows * self.config.cols

    @property
    def action_count(self) -> int:
        return len(self.ACTIONS)

    def reset(self) -> int:
        self.state = self.config.start
        self.steps = 0
        return self.position_to_state(self.state)

    def position_to_state(self, position: Position) -> int:
        row, col = position
        return row * self.config.cols + col

    def state_to_position(self, state: int) -> Position:
        return divmod(state, self.config.cols)

    def step(self, action: int) -> tuple[int, float, bool]:
        if action < 0 or action >= self.action_count:
            raise ValueError(f"未知动作编号: {action}")

        self.steps += 1
        row_delta, col_delta = self.ACTION_DELTAS[action]
        row, col = self.state
        candidate = (row + row_delta, col + col_delta)

        # 撞边界或墙时保持原地。额外惩罚能帮助智能体更快放弃无效动作。
        if not self._is_walkable(candidate):
            reward = self.config.wall_reward
            done = self.steps >= self.config.max_steps
            return self.position_to_state(self.state), reward, done

        self.state = candidate
        if self.state == self.config.goal:
            return self.position_to_state(self.state), self.config.goal_reward, True
        if self.state in self.config.traps:
            return self.position_to_state(self.state), self.config.trap_reward, True

        done = self.steps >= self.config.max_steps
        return self.position_to_state(self.state), self.config.step_reward, done

    def _is_walkable(self, position: Position) -> bool:
        row, col = position
        inside = 0 <= row < self.config.rows and 0 <= col < self.config.cols
        return inside and position not in self.config.walls


@dataclass(frozen=True)
class QLearningConfig:
    episodes: int = 3000
    learning_rate: float = 0.15
    discount_factor: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.997
    seed: int = 42


def choose_action(
    q_table: np.ndarray,
    state: int,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    """按照 ε-贪心策略选择动作。

    多个动作 Q 值相同时随机打破平局，避免智能体因为数组顺序而长期偏爱
    第一个动作，这在训练初期所有 Q 值都是零时尤其重要。
    """

    if rng.random() < epsilon:
        return int(rng.integers(q_table.shape[1]))

    values = q_table[state]
    best_actions = np.flatnonzero(np.isclose(values, values.max()))
    return int(rng.choice(best_actions))


def greedy_action(q_table: np.ndarray, state: int) -> int:
    return int(np.argmax(q_table[state]))


def policy_snapshot(env: GridWorld, q_table: np.ndarray) -> list[list[str]]:
    arrows = ("↑", "→", "↓", "←")
    policy: list[list[str]] = []
    for row in range(env.config.rows):
        line: list[str] = []
        for col in range(env.config.cols):
            position = (row, col)
            if position == env.config.goal:
                line.append("goal")
            elif position in env.config.walls:
                line.append("wall")
            elif position in env.config.traps:
                line.append("trap")
            else:
                state = env.position_to_state(position)
                line.append(arrows[greedy_action(q_table, state)])
        policy.append(line)
    return policy


def evaluate_route(env: GridWorld, q_table: np.ndarray) -> tuple[list[Position], float, bool]:
    state = env.reset()
    route = [env.state]
    total_reward = 0.0
    reached_goal = False
    visited: dict[Position, int] = {env.state: 1}

    for _ in range(env.config.max_steps):
        action = greedy_action(q_table, state)
        state, reward, done = env.step(action)
        total_reward += reward
        route.append(env.state)
        visited[env.state] = visited.get(env.state, 0) + 1

        if env.state == env.config.goal:
            reached_goal = True
        if done or visited[env.state] > 4:
            break

    return route, total_reward, reached_goal


def train_q_learning(
    env: GridWorld,
    config: QLearningConfig | None = None,
    snapshot_episodes: Iterable[int] = (0, 9, 49, 199, 499, 999, 2999),
) -> dict[str, object]:
    train_config = config or QLearningConfig()
    rng = np.random.default_rng(train_config.seed)
    q_table = np.zeros((env.state_count, env.action_count), dtype=np.float64)
    rewards: list[float] = []
    episode_steps: list[int] = []
    epsilon_history: list[float] = []
    snapshots: list[dict[str, object]] = []
    snapshot_set = set(snapshot_episodes)
    epsilon = train_config.epsilon_start

    for episode in range(train_config.episodes):
        state = env.reset()
        total_reward = 0.0

        for step_index in range(env.config.max_steps):
            action = choose_action(q_table, state, epsilon, rng)
            next_state, reward, done = env.step(action)

            # Q 学习使用“下一状态中最好的动作价值”作为未来估计。
            # 若当前动作已经结束回合，未来价值应当为零，不能继续向终点之后预测。
            best_next_value = 0.0 if done else float(np.max(q_table[next_state]))
            target = reward + train_config.discount_factor * best_next_value
            error = target - q_table[state, action]
            q_table[state, action] += train_config.learning_rate * error

            state = next_state
            total_reward += reward
            if done:
                break

        rewards.append(total_reward)
        episode_steps.append(step_index + 1)
        epsilon_history.append(epsilon)

        if episode in snapshot_set:
            snapshots.append(
                {
                    "episode": episode + 1,
                    "epsilon": round(epsilon, 4),
                    "policy": policy_snapshot(env, q_table),
                }
            )

        epsilon = max(train_config.epsilon_end, epsilon * train_config.epsilon_decay)

    route, route_reward, reached_goal = evaluate_route(env, q_table)
    recent_rewards = rewards[-100:]
    success_count = 0
    for _ in range(100):
        _, _, success = evaluate_route(env, q_table)
        success_count += int(success)

    return {
        "q_table": q_table,
        "rewards": rewards,
        "episode_steps": episode_steps,
        "epsilon_history": epsilon_history,
        "snapshots": snapshots,
        "route": route,
        "route_reward": route_reward,
        "reached_goal": reached_goal,
        "success_rate": success_count / 100,
        "average_recent_reward": float(np.mean(recent_rewards)),
        "config": train_config,
    }

