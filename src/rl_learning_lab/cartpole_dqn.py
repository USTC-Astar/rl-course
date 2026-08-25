from __future__ import annotations

import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class DQNConfig:
    total_steps: int = 150_000
    learning_rate: float = 5e-4
    discount_factor: float = 0.99
    batch_size: int = 64
    replay_capacity: int = 50_000
    warmup_steps: int = 1_000
    train_interval: int = 4
    target_sync_interval: int = 2_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 25_000
    hidden_size: int = 128
    seed: int = 42


class QNetwork(nn.Module):
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


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int) -> None:
        self._items: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=capacity
        )
        self._random = random.Random(seed)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        # 环境会复用数组内存，因此必须复制状态，避免旧经验被后续观测覆盖。
        self._items.append((state.copy(), action, reward, next_state.copy(), done))

    def sample(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = self._random.sample(self._items, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device),
            torch.as_tensor(actions, dtype=torch.int64, device=device).unsqueeze(1),
            torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1),
            torch.as_tensor(np.asarray(next_states), dtype=torch.float32, device=device),
            torch.as_tensor(dones, dtype=torch.float32, device=device).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self._items)


def epsilon_at_step(step: int, config: DQNConfig) -> float:
    if step >= config.epsilon_decay_steps:
        return config.epsilon_end
    if step <= 0:
        return config.epsilon_start
    progress = min(max(step, 0) / config.epsilon_decay_steps, 1.0)
    return config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)


def select_action(
    network: QNetwork,
    state: np.ndarray,
    epsilon: float,
    action_count: int,
    rng: np.random.Generator,
    device: torch.device,
) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(action_count))

    with torch.no_grad():
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        return int(network(state_tensor).argmax(dim=1).item())


def optimize_network(
    online_network: QNetwork,
    target_network: QNetwork,
    replay_buffer: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    config: DQNConfig,
    device: torch.device,
) -> float:
    states, actions, rewards, next_states, dones = replay_buffer.sample(
        config.batch_size, device
    )
    current_values = online_network(states).gather(1, actions)

    with torch.no_grad():
        # 目标网络更新较慢，相当于暂时固定“参考答案”，避免学习目标跟着
        # 在线网络一起剧烈移动，从而改善训练稳定性。
        # 在线网络负责挑选动作，目标网络只负责评价该动作，这种双重估计能
        # 减少同一个网络既选答案又给答案打分造成的 Q 值高估。
        next_actions = online_network(next_states).argmax(dim=1, keepdim=True)
        next_values = target_network(next_states).gather(1, next_actions)
        targets = rewards + config.discount_factor * (1.0 - dones) * next_values

    loss = nn.functional.smooth_l1_loss(current_values, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online_network.parameters(), max_norm=10.0)
    optimizer.step()
    return float(loss.item())


def evaluate_network(
    network: QNetwork,
    config: DQNConfig,
    device: torch.device,
    episodes: int = 10,
) -> dict[str, object]:
    env = gym.make("CartPole-v1")
    returns: list[float] = []
    trajectories: list[list[list[float]]] = []

    for episode in range(episodes):
        state, _ = env.reset(seed=config.seed + 10_000 + episode)
        trajectory = [state.astype(float).tolist()]
        total_reward = 0.0

        for _ in range(500):
            action = select_action(
                network,
                state,
                epsilon=0.0,
                action_count=env.action_space.n,
                rng=np.random.default_rng(config.seed + episode),
                device=device,
            )
            state, reward, terminated, truncated, _ = env.step(action)
            trajectory.append(state.astype(float).tolist())
            total_reward += float(reward)
            if terminated or truncated:
                break

        returns.append(total_reward)
        trajectories.append(trajectory)

    env.close()
    best_index = int(np.argmax(returns))
    return {
        "returns": returns,
        "average_return": float(np.mean(returns)),
        "best_return": float(returns[best_index]),
        "trajectory": trajectories[best_index],
    }


def train_dqn(config: DQNConfig, model_path: Path) -> dict[str, object]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    # 入门项目优先保证不同机器上的结果接近，而不是追求极限吞吐量。
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    device = torch.device("cpu")
    env = gym.make("CartPole-v1")
    env.action_space.seed(config.seed)
    state, _ = env.reset(seed=config.seed)
    observation_size = int(np.prod(env.observation_space.shape))
    action_count = int(env.action_space.n)

    online_network = QNetwork(observation_size, action_count, config.hidden_size).to(device)
    target_network = QNetwork(observation_size, action_count, config.hidden_size).to(device)
    target_network.load_state_dict(online_network.state_dict())
    target_network.eval()

    optimizer = torch.optim.Adam(online_network.parameters(), lr=config.learning_rate)
    replay_buffer = ReplayBuffer(config.replay_capacity, config.seed)
    rng = np.random.default_rng(config.seed)
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    losses: list[float] = []
    epsilon_samples: list[dict[str, float]] = []
    current_return = 0.0
    current_length = 0

    for step in range(config.total_steps):
        epsilon = epsilon_at_step(step, config)
        action = select_action(
            online_network, state, epsilon, action_count, rng, device
        )
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.add(state, action, float(reward), next_state, done)
        state = next_state
        current_return += float(reward)
        current_length += 1

        if (
            len(replay_buffer) >= max(config.warmup_steps, config.batch_size)
            and step % config.train_interval == 0
        ):
            losses.append(
                optimize_network(
                    online_network,
                    target_network,
                    replay_buffer,
                    optimizer,
                    config,
                    device,
                )
            )

        if (step + 1) % config.target_sync_interval == 0:
            target_network.load_state_dict(online_network.state_dict())

        if step % max(1, config.total_steps // 100) == 0:
            epsilon_samples.append({"step": step, "epsilon": epsilon})

        if done:
            episode_returns.append(current_return)
            episode_lengths.append(current_length)
            state, _ = env.reset()
            current_return = 0.0
            current_length = 0

    env.close()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(online_network.state_dict(), model_path)
    evaluation = evaluate_network(online_network, config, device)

    return {
        "config": asdict(config),
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "losses": losses,
        "epsilon_samples": epsilon_samples,
        "evaluation": evaluation,
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in online_network.parameters()),
    }
