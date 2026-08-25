from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from torch import nn
from torch.distributions import Normal

from rl_learning_lab.actor_critic import calculate_gae
from rl_learning_lab.ppo import calculate_clipped_policy_loss


@dataclass(frozen=True)
class ContinuousPPOConfig:
    total_steps: int = 200_000
    learning_rate: float = 3e-4
    discount_factor: float = 0.99
    gae_lambda: float = 0.95
    rollout_steps: int = 256
    vector_env_count: int = 8
    update_epochs: int = 4
    minibatch_size: int = 256
    clip_coefficient: float = 0.2
    value_loss_coefficient: float = 0.5
    entropy_coefficient: float = 1e-3
    max_gradient_norm: float = 0.5
    hidden_size: int = 64
    evaluation_interval: int = 5
    evaluation_episodes: int = 5
    early_stop_average: float = 900.0
    seed: int = 42


class GaussianActorCriticNetwork(nn.Module):
    """行动者输出高斯均值和标准差，评论家输出状态价值。"""

    def __init__(self, observation_size: int, action_size: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_size),
        )
        self.log_standard_deviation = nn.Parameter(torch.zeros(action_size))
        self.critic = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_means = self.actor(observations)
        log_standard_deviations = self.log_standard_deviation.clamp(-5.0, 2.0)
        standard_deviations = log_standard_deviations.exp().expand_as(action_means)
        state_values = self.critic(observations).squeeze(-1)
        return action_means, standard_deviations, state_values


def action_scale_and_bias(
    action_low: torch.Tensor, action_high: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = (action_high - action_low) / 2.0
    bias = (action_high + action_low) / 2.0
    return scale, bias


def sample_squashed_gaussian(
    action_means: torch.Tensor,
    action_standard_deviations: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """从高斯分布采样，并把无限范围动作压缩到环境边界。"""

    distribution = Normal(action_means, action_standard_deviations)
    raw_actions = distribution.rsample()
    squashed_actions = torch.tanh(raw_actions)
    scale, bias = action_scale_and_bias(action_low, action_high)
    bounded_actions = bias + scale * squashed_actions
    correction = torch.log(scale * (1.0 - squashed_actions.pow(2)) + 1e-6)
    log_probabilities = (distribution.log_prob(raw_actions) - correction).sum(dim=-1)
    entropy = distribution.entropy().sum(dim=-1)
    return bounded_actions, log_probabilities, entropy


def squashed_gaussian_log_probability(
    action_means: torch.Tensor,
    action_standard_deviations: torch.Tensor,
    bounded_actions: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算一个已执行连续动作在新策略下的对数概率。"""

    scale, bias = action_scale_and_bias(action_low, action_high)
    normalized_actions = ((bounded_actions - bias) / scale).clamp(-0.999999, 0.999999)
    raw_actions = torch.atanh(normalized_actions)
    distribution = Normal(action_means, action_standard_deviations)
    correction = torch.log(scale * (1.0 - normalized_actions.pow(2)) + 1e-6)
    log_probabilities = (distribution.log_prob(raw_actions) - correction).sum(dim=-1)
    entropy = distribution.entropy().sum(dim=-1)
    return log_probabilities, entropy


def deterministic_continuous_action(
    action_means: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
) -> torch.Tensor:
    scale, bias = action_scale_and_bias(action_low, action_high)
    return bias + scale * torch.tanh(action_means)


def evaluate_continuous_policy(
    network: GaussianActorCriticNetwork,
    config: ContinuousPPOConfig,
    device: torch.device,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    episodes: int = 10,
) -> dict[str, object]:
    env = gym.make("InvertedPendulum-v5")
    episode_returns: list[float] = []
    trajectories: list[list[list[float]]] = []
    action_traces: list[list[list[float]]] = []
    mean_traces: list[list[list[float]]] = []
    standard_deviation_traces: list[list[list[float]]] = []
    value_traces: list[list[float]] = []
    realized_return_traces: list[list[float]] = []

    for episode in range(episodes):
        observation, _ = env.reset(seed=config.seed + 40_000 + episode)
        trajectory: list[list[float]] = []
        actions: list[list[float]] = []
        means: list[list[float]] = []
        standard_deviations: list[list[float]] = []
        values: list[float] = []
        rewards: list[float] = []
        total_reward = 0.0

        for _ in range(1000):
            observation_tensor = torch.as_tensor(
                observation, dtype=torch.float32, device=device
            ).unsqueeze(0)
            with torch.no_grad():
                raw_means, action_std, state_value = network(observation_tensor)
                # 评估阶段关闭随机采样，只执行分布中心，测量模型已经学会的稳定控制能力。
                action = deterministic_continuous_action(
                    raw_means, action_low, action_high
                )

            trajectory.append(observation.astype(float).tolist())
            actions.append(action.squeeze(0).cpu().tolist())
            means.append(action.squeeze(0).cpu().tolist())
            scale, _ = action_scale_and_bias(action_low, action_high)
            standard_deviations.append((action_std * scale).squeeze(0).cpu().tolist())
            values.append(float(state_value.item()))
            observation, reward, terminated, truncated, _ = env.step(
                action.squeeze(0).cpu().numpy()
            )
            rewards.append(float(reward))
            total_reward += float(reward)
            if terminated or truncated:
                break

        running_return = 0.0
        realized_returns: list[float] = []
        # 使用环境真实返回的奖励倒推折扣回报，避免把“每步奖励恒为 1”写死在评估逻辑里。
        for reward in reversed(rewards):
            running_return = reward + config.discount_factor * running_return
            realized_returns.append(running_return)
        realized_returns.reverse()
        episode_returns.append(total_reward)
        trajectories.append(trajectory)
        action_traces.append(actions)
        mean_traces.append(means)
        standard_deviation_traces.append(standard_deviations)
        value_traces.append(values)
        realized_return_traces.append(realized_returns)

    env.close()
    best_index = int(np.argmax(episode_returns))
    return {
        "returns": episode_returns,
        "average_return": float(np.mean(episode_returns)),
        "best_return": float(episode_returns[best_index]),
        "trajectory": trajectories[best_index],
        "actions": action_traces[best_index],
        "action_means": mean_traces[best_index],
        "action_standard_deviations": standard_deviation_traces[best_index],
        "state_values": value_traces[best_index],
        "realized_returns": realized_return_traces[best_index],
    }


def train_continuous_ppo(
    config: ContinuousPPOConfig,
    model_path: Path,
) -> dict[str, object]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(1)

    device = torch.device("cpu")
    probe_env = gym.make("InvertedPendulum-v5")
    observation_size = int(np.prod(probe_env.observation_space.shape))
    action_size = int(np.prod(probe_env.action_space.shape))
    action_low = torch.as_tensor(
        probe_env.action_space.low, dtype=torch.float32, device=device
    )
    action_high = torch.as_tensor(
        probe_env.action_space.high, dtype=torch.float32, device=device
    )
    probe_env.close()

    # 多个同步环境一次产生一批经验，CPU 上也能减少纯 Python 控制循环的开销。
    env = SyncVectorEnv(
        [lambda: gym.make("InvertedPendulum-v5") for _ in range(config.vector_env_count)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    env.action_space.seed(config.seed)
    observations, _ = env.reset(
        seed=[config.seed + index for index in range(config.vector_env_count)]
    )

    network = GaussianActorCriticNetwork(
        observation_size, action_size, config.hidden_size
    ).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate, eps=1e-5)
    running_episode_returns = np.zeros(config.vector_env_count, dtype=np.float64)
    running_episode_lengths = np.zeros(config.vector_env_count, dtype=np.int64)
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    actor_losses: list[float] = []
    critic_losses: list[float] = []
    entropy_history: list[float] = []
    clip_fraction_history: list[float] = []
    approximate_kl_history: list[float] = []
    action_std_history: list[float] = []
    evaluation_history: list[dict[str, float]] = []
    best_evaluation_return = float("-inf")
    best_state_dict: dict[str, torch.Tensor] | None = None
    steps_completed = 0
    update_count = 0
    should_stop = False

    while steps_completed < config.total_steps and not should_stop:
        rollout_observations: list[np.ndarray] = []
        rollout_actions: list[np.ndarray] = []
        rollout_log_probabilities: list[np.ndarray] = []
        rollout_rewards: list[np.ndarray] = []
        rollout_dones: list[np.ndarray] = []
        rollout_values: list[np.ndarray] = []

        for _ in range(config.rollout_steps):
            observation_tensor = torch.as_tensor(
                observations, dtype=torch.float32, device=device
            )
            with torch.no_grad():
                action_means, action_stds, state_values = network(observation_tensor)
                actions, log_probabilities, _ = sample_squashed_gaussian(
                    action_means,
                    action_stds,
                    action_low,
                    action_high,
                )
            action_array = actions.cpu().numpy()
            next_observations, rewards, terminated, truncated, _ = env.step(action_array)
            dones = np.logical_or(terminated, truncated)
            rollout_observations.append(observations.copy())
            rollout_actions.append(action_array.copy())
            rollout_log_probabilities.append(log_probabilities.cpu().numpy())
            rollout_rewards.append(rewards.astype(np.float32))
            rollout_dones.append(dones.astype(np.float32))
            rollout_values.append(state_values.cpu().numpy())

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
            np.asarray(rollout_actions), dtype=torch.float32, device=device
        )
        old_log_probability_batch = torch.as_tensor(
            np.asarray(rollout_log_probabilities), dtype=torch.float32, device=device
        )
        reward_batch = torch.as_tensor(
            np.asarray(rollout_rewards), dtype=torch.float32, device=device
        )
        done_batch = torch.as_tensor(
            np.asarray(rollout_dones), dtype=torch.float32, device=device
        )
        old_value_batch = torch.as_tensor(
            np.asarray(rollout_values), dtype=torch.float32, device=device
        )

        with torch.no_grad():
            next_observation_tensor = torch.as_tensor(
                observations, dtype=torch.float32, device=device
            )
            _, _, bootstrap_values = network(next_observation_tensor)
            # GAE 同时利用即时奖励和评论家的下一状态估值，降低纯蒙特卡洛回报的波动。
            advantages, return_targets = calculate_gae(
                reward_batch,
                done_batch,
                old_value_batch,
                bootstrap_values,
                config.discount_factor,
                config.gae_lambda,
            )

        flat_observations = observation_batch.reshape(-1, observation_size)
        flat_actions = action_batch.reshape(-1, action_size)
        flat_old_log_probabilities = old_log_probability_batch.reshape(-1)
        flat_old_values = old_value_batch.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_returns = return_targets.reshape(-1)
        flat_advantages = (
            flat_advantages - flat_advantages.mean()
        ) / (flat_advantages.std(unbiased=False) + 1e-8)
        sample_count = flat_observations.shape[0]
        update_actor_losses: list[float] = []
        update_critic_losses: list[float] = []
        update_entropies: list[float] = []
        update_clip_fractions: list[float] = []
        update_approximate_kls: list[float] = []

        for _ in range(config.update_epochs):
            shuffled_indices = torch.randperm(sample_count, device=device)
            for start in range(0, sample_count, config.minibatch_size):
                indices = shuffled_indices[start : start + config.minibatch_size]
                action_means, action_stds, new_values = network(
                    flat_observations[indices]
                )
                new_log_probabilities, entropies = squashed_gaussian_log_probability(
                    action_means,
                    action_stds,
                    flat_actions[indices],
                    action_low,
                    action_high,
                )
                entropy = entropies.mean()
                actor_loss, probability_ratio = calculate_clipped_policy_loss(
                    new_log_probabilities,
                    flat_old_log_probabilities[indices],
                    flat_advantages[indices],
                    config.clip_coefficient,
                )
                value_change = new_values - flat_old_values[indices]
                clipped_values = flat_old_values[indices] + torch.clamp(
                    value_change,
                    -config.clip_coefficient,
                    config.clip_coefficient,
                )
                value_loss_unclipped = (new_values - flat_returns[indices]).pow(2)
                value_loss_clipped = (clipped_values - flat_returns[indices]).pow(2)
                critic_loss = 0.5 * torch.maximum(
                    value_loss_unclipped, value_loss_clipped
                ).mean()
                total_loss = (
                    actor_loss
                    + config.value_loss_coefficient * critic_loss
                    - config.entropy_coefficient * entropy
                )

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), config.max_gradient_norm)
                optimizer.step()

                with torch.no_grad():
                    log_ratio = (
                        new_log_probabilities - flat_old_log_probabilities[indices]
                    )
                    approximate_kl = ((probability_ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        (probability_ratio - 1.0).abs() > config.clip_coefficient
                    ).float().mean()
                update_actor_losses.append(float(actor_loss.item()))
                update_critic_losses.append(float(critic_loss.item()))
                update_entropies.append(float(entropy.item()))
                update_clip_fractions.append(float(clip_fraction.item()))
                update_approximate_kls.append(float(approximate_kl.item()))

        update_count += 1
        actor_losses.append(float(np.mean(update_actor_losses)))
        critic_losses.append(float(np.mean(update_critic_losses)))
        entropy_history.append(float(np.mean(update_entropies)))
        clip_fraction_history.append(float(np.mean(update_clip_fractions)))
        approximate_kl_history.append(float(np.mean(update_approximate_kls)))
        scale, _ = action_scale_and_bias(action_low, action_high)
        with torch.no_grad():
            current_std = network.log_standard_deviation.clamp(-5.0, 2.0).exp()
        action_std_history.append(float((current_std * scale).mean().item()))

        if update_count % config.evaluation_interval == 0:
            checkpoint_evaluation = evaluate_continuous_policy(
                network,
                config,
                device,
                action_low,
                action_high,
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
                # 保存独立评估最好的参数，防止后续随机探索把已经学好的策略覆盖掉。
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
    evaluation = evaluate_continuous_policy(
        network,
        config,
        device,
        action_low,
        action_high,
    )

    return {
        "environment": "InvertedPendulum-v5",
        "action_low": action_low.cpu().tolist(),
        "action_high": action_high.cpu().tolist(),
        "config": asdict(config),
        "steps_completed": steps_completed,
        "update_count": update_count,
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "actor_losses": actor_losses,
        "critic_losses": critic_losses,
        "entropy_history": entropy_history,
        "clip_fraction_history": clip_fraction_history,
        "approximate_kl_history": approximate_kl_history,
        "action_std_history": action_std_history,
        "evaluation_history": evaluation_history,
        "evaluation": evaluation,
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in network.parameters()),
    }
