from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from torch import nn
from torch.distributions import Categorical


@dataclass(frozen=True)
class ActorCriticConfig:
    total_steps: int = 300_000
    learning_rate: float = 7e-4
    discount_factor: float = 0.99
    rollout_steps: int = 5
    vector_env_count: int = 16
    gae_lambda: float = 0.95
    hidden_size: int = 64
    value_loss_coefficient: float = 0.5
    entropy_coefficient: float = 1e-3
    max_gradient_norm: float = 0.5
    early_stop_average: float = 475.0
    evaluation_interval: int = 100
    evaluation_episodes: int = 5
    seed: int = 42


class ActorCriticNetwork(nn.Module):
    """使用独立行动者和评论家，避免两个目标争夺同一组隐藏特征。"""

    def __init__(self, observation_size: int, action_count: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_count),
        )
        self.critic = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        action_logits = self.actor(observations)
        state_values = self.critic(observations).squeeze(-1)
        return action_logits, state_values


def calculate_n_step_returns(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    bootstrap_values: torch.Tensor,
    discount_factor: float,
) -> torch.Tensor:
    """从一段短轨迹的末尾向前计算每一步训练目标。"""

    returns = torch.zeros_like(rewards)
    running_return = bootstrap_values
    for step in reversed(range(rewards.shape[0])):
        # 回合结束时不能把新回合的价值接到旧回合后面，因此 done 会切断未来回报。
        running_return = (
            rewards[step]
            + discount_factor * running_return * (1.0 - dones[step])
        )
        returns[step] = running_return
    return returns


def calculate_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    state_values: torch.Tensor,
    bootstrap_values: torch.Tensor,
    discount_factor: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算广义优势估计和评论家的回报目标。"""

    advantages = torch.zeros_like(rewards)
    running_advantage = torch.zeros_like(bootstrap_values)
    next_values = bootstrap_values

    for step in reversed(range(rewards.shape[0])):
        continuation = 1.0 - dones[step]
        temporal_difference = (
            rewards[step]
            + discount_factor * next_values * continuation
            - state_values[step]
        )
        running_advantage = (
            temporal_difference
            + discount_factor * gae_lambda * continuation * running_advantage
        )
        advantages[step] = running_advantage
        next_values = state_values[step]

    return_targets = advantages + state_values
    return advantages, return_targets


def evaluate_actor_critic(
    network: ActorCriticNetwork,
    config: ActorCriticConfig,
    device: torch.device,
    episodes: int = 10,
) -> dict[str, object]:
    env = gym.make("CartPole-v1")
    episode_returns: list[float] = []
    trajectories: list[list[list[float]]] = []
    probability_traces: list[list[list[float]]] = []
    value_traces: list[list[float]] = []
    realized_return_traces: list[list[float]] = []
    action_traces: list[list[int]] = []

    for episode in range(episodes):
        observation, _ = env.reset(seed=config.seed + 30_000 + episode)
        trajectory: list[list[float]] = []
        probabilities: list[list[float]] = []
        values: list[float] = []
        actions: list[int] = []
        total_reward = 0.0

        for _ in range(500):
            observation_tensor = torch.as_tensor(
                observation, dtype=torch.float32, device=device
            ).unsqueeze(0)
            with torch.no_grad():
                logits, state_value = network(observation_tensor)
                action_probabilities = torch.softmax(logits, dim=1)
            action = int(action_probabilities.argmax(dim=1).item())

            trajectory.append(observation.astype(float).tolist())
            probabilities.append(action_probabilities.squeeze(0).cpu().tolist())
            values.append(float(state_value.item()))
            actions.append(action)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break

        episode_returns.append(total_reward)
        trajectories.append(trajectory)
        probability_traces.append(probabilities)
        value_traces.append(values)
        running_return = 0.0
        realized_returns: list[float] = []
        for _ in reversed(trajectory):
            running_return = 1.0 + config.discount_factor * running_return
            realized_returns.append(running_return)
        realized_returns.reverse()
        realized_return_traces.append(realized_returns)
        action_traces.append(actions)

    env.close()
    best_index = int(np.argmax(episode_returns))
    return {
        "returns": episode_returns,
        "average_return": float(np.mean(episode_returns)),
        "best_return": float(episode_returns[best_index]),
        "trajectory": trajectories[best_index],
        "action_probabilities": probability_traces[best_index],
        "state_values": value_traces[best_index],
        "realized_returns": realized_return_traces[best_index],
        "actions": action_traces[best_index],
    }


def train_actor_critic(
    config: ActorCriticConfig,
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
    observations, _ = env.reset(
        seed=[config.seed + index for index in range(config.vector_env_count)]
    )

    network = ActorCriticNetwork(
        observation_size, action_count, config.hidden_size
    ).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    running_episode_returns = np.zeros(config.vector_env_count, dtype=np.float64)
    running_episode_lengths = np.zeros(config.vector_env_count, dtype=np.int64)
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    total_losses: list[float] = []
    actor_losses: list[float] = []
    critic_losses: list[float] = []
    entropy_history: list[float] = []
    evaluation_history: list[dict[str, float]] = []
    best_evaluation_return = float("-inf")
    best_state_dict: dict[str, torch.Tensor] | None = None
    steps_completed = 0
    update_count = 0
    should_stop = False

    while steps_completed < config.total_steps and not should_stop:
        rollout_observations: list[np.ndarray] = []
        rollout_actions: list[np.ndarray] = []
        rollout_rewards: list[np.ndarray] = []
        rollout_dones: list[np.ndarray] = []

        for _ in range(config.rollout_steps):
            observation_tensor = torch.as_tensor(
                observations, dtype=torch.float32, device=device
            )
            with torch.no_grad():
                action_logits, _ = network(observation_tensor)
                action_tensor = Categorical(logits=action_logits).sample()
            actions = action_tensor.cpu().numpy()
            next_observations, rewards, terminated, truncated, _ = env.step(actions)
            dones = np.logical_or(terminated, truncated)

            rollout_observations.append(observations.copy())
            rollout_actions.append(actions.copy())
            rollout_rewards.append(rewards.astype(np.float32))
            rollout_dones.append(dones.astype(np.float32))
            running_episode_returns += rewards
            running_episode_lengths += 1

            for index in np.flatnonzero(dones):
                episode_returns.append(float(running_episode_returns[index]))
                episode_lengths.append(int(running_episode_lengths[index]))
                running_episode_returns[index] = 0.0
                running_episode_lengths[index] = 0

            observations = next_observations
            steps_completed += config.vector_env_count
            if steps_completed >= config.total_steps:
                break

        observation_batch = torch.as_tensor(
            np.asarray(rollout_observations), dtype=torch.float32, device=device
        )
        action_batch = torch.as_tensor(
            np.asarray(rollout_actions), dtype=torch.int64, device=device
        )
        reward_batch = torch.as_tensor(
            np.asarray(rollout_rewards), dtype=torch.float32, device=device
        )
        done_batch = torch.as_tensor(
            np.asarray(rollout_dones), dtype=torch.float32, device=device
        )

        flat_observations = observation_batch.reshape(-1, observation_size)
        action_logits, flat_state_values = network(flat_observations)
        state_values = flat_state_values.reshape(reward_batch.shape)
        distribution = Categorical(logits=action_logits)
        log_probabilities = distribution.log_prob(action_batch.reshape(-1)).reshape(
            reward_batch.shape
        )
        entropy = distribution.entropy().mean()

        with torch.no_grad():
            next_observation_tensor = torch.as_tensor(
                observations, dtype=torch.float32, device=device
            )
            _, bootstrap_values = network(next_observation_tensor)
            advantages, return_targets = calculate_gae(
                reward_batch,
                done_batch,
                state_values.detach(),
                bootstrap_values,
                config.discount_factor,
                config.gae_lambda,
            )

        # 行动者只关心相对好坏，标准化可以避免评论家数值尺度直接控制更新幅度。
        normalized_advantages = (
            advantages - advantages.mean()
        ) / (advantages.std(unbiased=False) + 1e-8)
        actor_loss = -(log_probabilities * normalized_advantages).mean()
        critic_loss = nn.functional.mse_loss(state_values, return_targets)
        total_loss = (
            actor_loss
            + config.value_loss_coefficient * critic_loss
            - config.entropy_coefficient * entropy
        )

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(network.parameters(), config.max_gradient_norm)
        optimizer.step()

        update_count += 1
        total_losses.append(float(total_loss.item()))
        actor_losses.append(float(actor_loss.item()))
        critic_losses.append(float(critic_loss.item()))
        entropy_history.append(float(entropy.item()))

        if update_count % config.evaluation_interval == 0:
            checkpoint_evaluation = evaluate_actor_critic(
                network,
                config,
                device,
                episodes=config.evaluation_episodes,
            )
            average_return = float(checkpoint_evaluation["average_return"])
            evaluation_history.append(
                {
                    "steps": float(steps_completed),
                    "average_return": average_return,
                }
            )
            if average_return > best_evaluation_return:
                best_evaluation_return = average_return
                # 保存张量副本，后续训练即使退化，也能恢复到独立评估最好的版本。
                best_state_dict = {
                    name: parameter.detach().cpu().clone()
                    for name, parameter in network.state_dict().items()
                }
            should_stop = average_return >= config.early_stop_average

    env.close()
    if best_state_dict is not None:
        network.load_state_dict(best_state_dict)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(network.state_dict(), model_path)
    evaluation = evaluate_actor_critic(network, config, device)

    return {
        "config": asdict(config),
        "steps_completed": steps_completed,
        "update_count": update_count,
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "total_losses": total_losses,
        "actor_losses": actor_losses,
        "critic_losses": critic_losses,
        "entropy_history": entropy_history,
        "evaluation_history": evaluation_history,
        "evaluation": evaluation,
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in network.parameters()),
    }
