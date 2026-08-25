from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


def _validate_discount(discount_factor: float) -> None:
    if not 0.0 <= discount_factor <= 1.0:
        raise ValueError("discount_factor 必须位于 0 到 1 之间")


def calculate_n_step_return(
    rewards: Iterable[float],
    next_value: float,
    discount_factor: float,
) -> float:
    """计算一段奖励加末尾 bootstrap 价值组成的 n 步回报。"""

    _validate_discount(discount_factor)
    reward_list = [float(reward) for reward in rewards]
    result = float(next_value)

    # 反向递推能自然得到 r_t + γ(r_t+1 + γ(...))，也避免手动维护幂次。
    for reward in reversed(reward_list):
        result = reward + discount_factor * result
    return result


def calculate_lambda_returns(
    rewards: np.ndarray,
    values: np.ndarray,
    discount_factor: float,
    trace_decay: float,
) -> np.ndarray:
    """计算有限轨迹上的 TD(λ) 回报，values 需比 rewards 多一个末尾价值。"""

    _validate_discount(discount_factor)
    if not 0.0 <= trace_decay <= 1.0:
        raise ValueError("trace_decay 必须位于 0 到 1 之间")
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (rewards.size + 1,):
        raise ValueError("values 必须包含每个时刻价值以及最后一个 bootstrap 价值")

    returns = np.empty_like(rewards)
    running_return = float(values[-1])
    for index in range(rewards.size - 1, -1, -1):
        one_step_bootstrap = (1.0 - trace_decay) * values[index + 1]
        longer_return = trace_decay * running_return
        running_return = rewards[index] + discount_factor * (one_step_bootstrap + longer_return)
        returns[index] = running_return
    return returns


def calculate_dqn_targets(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_target_q_values: torch.Tensor,
    discount_factor: float,
) -> torch.Tensor:
    """普通 DQN：目标网络既选最大动作，也给这个最大动作估值。"""

    _validate_discount(discount_factor)
    next_values = next_target_q_values.max(dim=1).values
    return rewards + discount_factor * (1.0 - dones) * next_values


def calculate_double_dqn_targets(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_online_q_values: torch.Tensor,
    next_target_q_values: torch.Tensor,
    discount_factor: float,
) -> torch.Tensor:
    """Double DQN：在线网络选动作，目标网络只评价被选动作。"""

    _validate_discount(discount_factor)
    if next_online_q_values.shape != next_target_q_values.shape:
        raise ValueError("在线网络与目标网络的 Q 值形状必须一致")
    selected_actions = next_online_q_values.argmax(dim=1, keepdim=True)
    selected_values = next_target_q_values.gather(1, selected_actions).squeeze(1)
    return rewards + discount_factor * (1.0 - dones) * selected_values


def combine_dueling_values(
    state_values: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    """把 Dueling 网络的状态价值和动作优势合成为可辨识的 Q 值。"""

    if state_values.ndim != advantages.ndim or state_values.shape[:-1] != advantages.shape[:-1]:
        raise ValueError("state_values 与 advantages 的批次维度必须一致")
    if state_values.shape[-1] != 1:
        raise ValueError("state_values 的最后一维必须为 1")
    centered_advantages = advantages - advantages.mean(dim=-1, keepdim=True)
    return state_values + centered_advantages


def prioritized_replay_probabilities(
    td_errors: np.ndarray,
    priority_exponent: float,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """根据 TD 误差计算优先经验回放的抽样概率。"""

    if priority_exponent < 0.0:
        raise ValueError("priority_exponent 不能为负数")
    errors = np.asarray(td_errors, dtype=np.float64)
    if errors.ndim != 1 or errors.size == 0:
        raise ValueError("td_errors 必须是一维非空数组")
    priorities = (np.abs(errors) + epsilon) ** priority_exponent
    return priorities / priorities.sum()


def importance_sampling_weights(
    probabilities: np.ndarray,
    correction_exponent: float,
) -> np.ndarray:
    """计算归一化的重要性采样权重，最大权重固定为 1。"""

    if not 0.0 <= correction_exponent <= 1.0:
        raise ValueError("correction_exponent 必须位于 0 到 1 之间")
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 1 or probabilities.size == 0 or np.any(probabilities <= 0.0):
        raise ValueError("probabilities 必须是一维正数数组")
    weights = (probabilities.size * probabilities) ** (-correction_exponent)
    return weights / weights.max()


def project_categorical_distribution(
    next_probabilities: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    support: torch.Tensor,
    discount_factor: float,
) -> torch.Tensor:
    """把 C51 的下一状态回报分布投影回固定原子支撑集。"""

    _validate_discount(discount_factor)
    if support.ndim != 1 or support.numel() < 2:
        raise ValueError("support 必须是一维且至少包含两个原子")
    if next_probabilities.ndim != 2 or next_probabilities.shape[1] != support.numel():
        raise ValueError("next_probabilities 的原子数量必须与 support 一致")
    if rewards.shape != dones.shape or rewards.numel() != next_probabilities.shape[0]:
        raise ValueError("rewards、dones 与批次大小必须一致")

    minimum_value = support[0]
    maximum_value = support[-1]
    atom_delta = (maximum_value - minimum_value) / (support.numel() - 1)
    target_atoms = rewards.unsqueeze(1) + discount_factor * (1.0 - dones.unsqueeze(1)) * support
    target_atoms = target_atoms.clamp(float(minimum_value), float(maximum_value))
    positions = (target_atoms - minimum_value) / atom_delta
    lower = positions.floor().long()
    upper = positions.ceil().long()

    projected = torch.zeros_like(next_probabilities)
    for batch_index in range(next_probabilities.shape[0]):
        for atom_index in range(support.numel()):
            probability = next_probabilities[batch_index, atom_index]
            lower_index = lower[batch_index, atom_index]
            upper_index = upper[batch_index, atom_index]
            if lower_index == upper_index:
                projected[batch_index, lower_index] += probability
            else:
                projected[batch_index, lower_index] += probability * (
                    upper_index.float() - positions[batch_index, atom_index]
                )
                projected[batch_index, upper_index] += probability * (
                    positions[batch_index, atom_index] - lower_index.float()
                )
    return projected


def smooth_td3_target_actions(
    target_actions: torch.Tensor,
    standard_noise: torch.Tensor,
    noise_standard_deviation: float,
    noise_clip: float,
    action_low: float,
    action_high: float,
) -> torch.Tensor:
    """为 TD3 目标动作加入截断噪声，并限制在环境合法动作范围。"""

    if target_actions.shape != standard_noise.shape:
        raise ValueError("target_actions 与 standard_noise 的形状必须一致")
    if noise_standard_deviation < 0.0 or noise_clip < 0.0:
        raise ValueError("噪声参数不能为负数")
    noise = (standard_noise * noise_standard_deviation).clamp(-noise_clip, noise_clip)
    return (target_actions + noise).clamp(action_low, action_high)


def calculate_td3_targets(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    target_q_one: torch.Tensor,
    target_q_two: torch.Tensor,
    discount_factor: float,
) -> torch.Tensor:
    """计算 TD3 的双评论家保守目标。"""

    _validate_discount(discount_factor)
    conservative_next_value = torch.minimum(target_q_one, target_q_two)
    return rewards + discount_factor * (1.0 - dones) * conservative_next_value


@dataclass(frozen=True)
class DynaQConfig:
    rows: int = 6
    cols: int = 6
    episodes: int = 60
    planning_steps: int = 10
    learning_rate: float = 0.2
    discount_factor: float = 0.95
    epsilon: float = 0.15
    seed: int = 7


class DynaQAgent:
    """用于教学的小型表格 Dyna-Q 智能体。"""

    def __init__(self, state_count: int, action_count: int, config: DynaQConfig) -> None:
        self.config = config
        self.q_table = np.zeros((state_count, action_count), dtype=np.float64)
        self.model: dict[tuple[int, int], tuple[int, float, bool]] = {}
        self.random_generator = np.random.default_rng(config.seed)

    def choose_action(self, state: int) -> int:
        if self.random_generator.random() < self.config.epsilon:
            return int(self.random_generator.integers(self.q_table.shape[1]))
        best_actions = np.flatnonzero(self.q_table[state] == self.q_table[state].max())
        return int(self.random_generator.choice(best_actions))

    def _update_q(self, state: int, action: int, next_state: int, reward: float, done: bool) -> None:
        next_value = 0.0 if done else float(self.q_table[next_state].max())
        target = reward + self.config.discount_factor * next_value
        self.q_table[state, action] += self.config.learning_rate * (
            target - self.q_table[state, action]
        )

    def learn_from_real_step(
        self,
        state: int,
        action: int,
        next_state: int,
        reward: float,
        done: bool,
    ) -> None:
        self._update_q(state, action, next_state, reward, done)
        self.model[state, action] = (next_state, reward, done)

        # 规划只从已经真实见过的转移中抽样，避免凭空构造未知规则。
        if not self.model:
            return
        model_keys = tuple(self.model)
        for _ in range(self.config.planning_steps):
            simulated_state, simulated_action = model_keys[
                int(self.random_generator.integers(len(model_keys)))
            ]
            simulated_next_state, simulated_reward, simulated_done = self.model[
                simulated_state, simulated_action
            ]
            self._update_q(
                simulated_state,
                simulated_action,
                simulated_next_state,
                simulated_reward,
                simulated_done,
            )


def run_dyna_q(config: DynaQConfig) -> list[int]:
    """在带障碍网格上运行 Dyna-Q，返回每回合到达目标所用步数。"""

    walls = {(1, 1), (2, 1), (3, 1), (1, 3), (2, 3), (4, 4)}
    start = (config.rows - 1, 0)
    goal = (0, config.cols - 1)
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    agent = DynaQAgent(config.rows * config.cols, len(moves), config)
    episode_steps: list[int] = []

    def to_state(position: tuple[int, int]) -> int:
        return position[0] * config.cols + position[1]

    for _ in range(config.episodes):
        position = start
        for step in range(1, 301):
            state = to_state(position)
            action = agent.choose_action(state)
            row = position[0] + moves[action][0]
            col = position[1] + moves[action][1]
            candidate = (row, col)
            if not (0 <= row < config.rows and 0 <= col < config.cols) or candidate in walls:
                candidate = position
                reward = -1.0
            elif candidate == goal:
                reward = 10.0
            else:
                reward = -0.1
            done = candidate == goal
            agent.learn_from_real_step(state, action, to_state(candidate), reward, done)
            position = candidate
            if done:
                episode_steps.append(step)
                break
        else:
            episode_steps.append(300)
    return episode_steps


def independent_q_update(
    q_values: np.ndarray,
    action: int,
    reward: float,
    learning_rate: float,
) -> None:
    """无状态矩阵博弈中的独立 Q 更新，用于展示多智能体非平稳性。"""

    if not 0 <= action < q_values.size:
        raise ValueError("action 超出 q_values 范围")
    q_values[action] += learning_rate * (reward - q_values[action])


def conservative_q_regularizer(
    q_values: torch.Tensor,
    dataset_actions: torch.Tensor,
) -> torch.Tensor:
    """计算离散动作保守 Q 学习中的 logsumexp 数据外惩罚项。"""

    if q_values.ndim != 2:
        raise ValueError("q_values 必须是 [batch, action] 二维张量")
    if dataset_actions.shape != (q_values.shape[0],):
        raise ValueError("dataset_actions 必须与批次大小一致")
    dataset_q_values = q_values.gather(1, dataset_actions.unsqueeze(1)).squeeze(1)
    return (torch.logsumexp(q_values, dim=1) - dataset_q_values).mean()


def sample_domain_parameters(
    seed: int,
    count: int,
    mass_range: tuple[float, float],
    friction_range: tuple[float, float],
    sensor_noise_standard_deviation: float,
) -> dict[str, np.ndarray]:
    """采样域随机化参数，供仿真到现实课程生成可重复实验。"""

    if count <= 0:
        raise ValueError("count 必须为正数")
    if mass_range[0] > mass_range[1] or friction_range[0] > friction_range[1]:
        raise ValueError("参数范围下界不能大于上界")
    if sensor_noise_standard_deviation < 0.0:
        raise ValueError("传感器噪声标准差不能为负数")

    random_generator = np.random.default_rng(seed)
    return {
        "mass": random_generator.uniform(*mass_range, size=count),
        "friction": random_generator.uniform(*friction_range, size=count),
        "sensor_noise": random_generator.normal(
            0.0,
            sensor_noise_standard_deviation,
            size=count,
        ),
    }
