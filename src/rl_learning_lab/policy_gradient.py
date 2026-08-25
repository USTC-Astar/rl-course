from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from gymnasium.vector import AutoresetMode, SyncVectorEnv


@dataclass(frozen=True)
class PolicyGradientConfig:
    episodes: int = 600
    learning_rate: float = 1e-2
    discount_factor: float = 0.99
    entropy_coefficient: float = 1e-3
    hidden_size: int = 64
    vector_env_count: int = 16
    update_batch_episodes: int = 16
    early_stop_average: float = 475.0
    early_stop_window: int = 50
    seed: int = 42


class PolicyNetwork(nn.Module):
    """直接输出动作分数，再由概率分布决定动作。"""

    def __init__(self, observation_size: int, action_count: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_count),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.layers(observations)


def discounted_returns(rewards: list[float], discount_factor: float) -> torch.Tensor:
    """计算每个时刻往后能获得的折扣回报。

    必须从后往前算，因为当前时刻的回报等于“当前奖励 + 折扣后的下一时刻回报”。
    """

    returns: list[float] = []
    running_return = 0.0
    for reward in reversed(rewards):
        running_return = reward + discount_factor * running_return
        returns.append(running_return)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


def normalized_returns(rewards: list[float], discount_factor: float) -> torch.Tensor:
    returns = discounted_returns(rewards, discount_factor)
    if returns.numel() <= 1:
        return returns
    # 标准化不会改变“哪些动作相对更好”的顺序，但能限制梯度尺度，
    # 减少不同长度回合导致的更新幅度剧烈变化。
    return (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)


def evaluate_policy(
    network: PolicyNetwork,
    config: PolicyGradientConfig,
    device: torch.device,
    episodes: int = 10,
) -> dict[str, object]:
    env = gym.make("CartPole-v1")
    episode_returns: list[float] = []
    trajectories: list[list[list[float]]] = []
    probability_traces: list[list[list[float]]] = []
    action_traces: list[list[int]] = []

    for episode in range(episodes):
        observation, _ = env.reset(seed=config.seed + 20_000 + episode)
        trajectory: list[list[float]] = []
        probabilities: list[list[float]] = []
        actions: list[int] = []
        total_reward = 0.0

        for _ in range(500):
            observation_tensor = torch.as_tensor(
                observation, dtype=torch.float32, device=device
            ).unsqueeze(0)
            with torch.no_grad():
                action_probabilities = torch.softmax(network(observation_tensor), dim=1)
            action = int(action_probabilities.argmax(dim=1).item())

            trajectory.append(observation.astype(float).tolist())
            probabilities.append(action_probabilities.squeeze(0).cpu().tolist())
            actions.append(action)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break

        episode_returns.append(total_reward)
        trajectories.append(trajectory)
        probability_traces.append(probabilities)
        action_traces.append(actions)

    env.close()
    best_index = int(np.argmax(episode_returns))
    return {
        "returns": episode_returns,
        "average_return": float(np.mean(episode_returns)),
        "best_return": float(episode_returns[best_index]),
        "trajectory": trajectories[best_index],
        "action_probabilities": probability_traces[best_index],
        "actions": action_traces[best_index],
    }


def train_policy_gradient(
    config: PolicyGradientConfig,
    model_path: Path,
) -> dict[str, object]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    device = torch.device("cpu")
    probe_env = gym.make("CartPole-v1")
    observation_size = int(np.prod(probe_env.observation_space.shape))
    action_count = int(probe_env.action_space.n)
    probe_env.close()

    env = SyncVectorEnv(
        [lambda: gym.make("CartPole-v1") for _ in range(config.vector_env_count)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    env.action_space.seed(config.seed)
    network = PolicyNetwork(observation_size, action_count, config.hidden_size).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    losses: list[float] = []
    entropy_history: list[float] = []
    observations, _ = env.reset(
        seed=[config.seed + index for index in range(config.vector_env_count)]
    )
    episode_observations: list[list[np.ndarray]] = [
        [] for _ in range(config.vector_env_count)
    ]
    episode_actions: list[list[int]] = [[] for _ in range(config.vector_env_count)]
    episode_rewards: list[list[float]] = [[] for _ in range(config.vector_env_count)]
    batch_observations: list[np.ndarray] = []
    batch_actions: list[int] = []
    batch_returns: list[float] = []
    batch_episode_count = 0
    should_stop = False

    while len(episode_returns) < config.episodes and not should_stop:
        observation_tensor = torch.as_tensor(
            observations, dtype=torch.float32, device=device
        )
        # 一次处理所有并行环境，避免对每辆小车分别调用神经网络。
        with torch.no_grad():
            action_tensor = Categorical(logits=network(observation_tensor)).sample()
        actions = action_tensor.cpu().numpy()
        next_observations, rewards, terminated, truncated, _ = env.step(actions)
        dones = np.logical_or(terminated, truncated)

        for index in range(config.vector_env_count):
            episode_observations[index].append(observations[index].copy())
            episode_actions[index].append(int(actions[index]))
            episode_rewards[index].append(float(rewards[index]))

            if not dones[index]:
                continue

            rewards_for_episode = episode_rewards[index]
            returns_for_episode = normalized_returns(
                rewards_for_episode, config.discount_factor
            ).tolist()
            batch_observations.extend(episode_observations[index])
            batch_actions.extend(episode_actions[index])
            batch_returns.extend(returns_for_episode)
            batch_episode_count += 1
            episode_returns.append(float(sum(rewards_for_episode)))
            episode_lengths.append(len(rewards_for_episode))
            episode_observations[index] = []
            episode_actions[index] = []
            episode_rewards[index] = []

            if len(episode_returns) >= config.episodes:
                break

        observations = next_observations

        if batch_episode_count >= config.update_batch_episodes:
            observation_batch = torch.as_tensor(
                np.asarray(batch_observations), dtype=torch.float32, device=device
            )
            action_batch = torch.as_tensor(
                batch_actions, dtype=torch.int64, device=device
            )
            return_batch = torch.as_tensor(
                batch_returns, dtype=torch.float32, device=device
            )
            distribution = Categorical(logits=network(observation_batch))
            log_probability_tensor = distribution.log_prob(action_batch)
            entropy_tensor = distribution.entropy()

            # 高回报动作的对数概率会被提升，低于本回合平均水平的动作概率会被压低。
            # 对多个回合取平均，避免较长回合仅因样本更多就支配整次更新。
            policy_loss = -(log_probability_tensor * return_batch).mean()
            entropy_bonus = entropy_tensor.mean()
            loss = policy_loss - config.entropy_coefficient * entropy_bonus

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), max_norm=10.0)
            optimizer.step()
            losses.append(float(loss.item()))
            entropy_history.append(float(entropy_bonus.item()))
            batch_observations.clear()
            batch_actions.clear()
            batch_returns.clear()
            batch_episode_count = 0

            if len(episode_returns) >= config.early_stop_window:
                recent_average = float(
                    np.mean(episode_returns[-config.early_stop_window :])
                )
                should_stop = recent_average >= config.early_stop_average

    env.close()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(network.state_dict(), model_path)
    evaluation = evaluate_policy(network, config, device)

    return {
        "config": asdict(config),
        "episode_returns": episode_returns,
        "episodes_completed": len(episode_returns),
        "episode_lengths": episode_lengths,
        "losses": losses,
        "entropy_history": entropy_history,
        "evaluation": evaluation,
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in network.parameters()),
    }
