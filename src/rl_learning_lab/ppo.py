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

from rl_learning_lab.actor_critic import (
    ActorCriticNetwork,
    calculate_gae,
    evaluate_actor_critic,
)


@dataclass(frozen=True)
class PPOConfig:
    total_steps: int = 120_000
    learning_rate: float = 3e-4
    discount_factor: float = 0.99
    gae_lambda: float = 0.95
    rollout_steps: int = 128
    vector_env_count: int = 16
    update_epochs: int = 4
    minibatch_size: int = 256
    clip_coefficient: float = 0.2
    value_loss_coefficient: float = 0.5
    entropy_coefficient: float = 1e-2
    max_gradient_norm: float = 0.5
    hidden_size: int = 64
    evaluation_interval: int = 5
    evaluation_episodes: int = 5
    early_stop_average: float = 475.0
    seed: int = 42


def calculate_clipped_policy_loss(
    new_log_probabilities: torch.Tensor,
    old_log_probabilities: torch.Tensor,
    advantages: torch.Tensor,
    clip_coefficient: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算 PPO 裁剪策略损失，并返回新旧策略概率比。"""

    probability_ratio = torch.exp(new_log_probabilities - old_log_probabilities)
    unclipped_loss = -advantages * probability_ratio
    clipped_ratio = torch.clamp(
        probability_ratio,
        1.0 - clip_coefficient,
        1.0 + clip_coefficient,
    )
    clipped_loss = -advantages * clipped_ratio
    # 对行动者来说取更保守的损失，防止“表现不错的一批数据”让策略改变过猛。
    return torch.maximum(unclipped_loss, clipped_loss).mean(), probability_ratio


def train_ppo(config: PPOConfig, model_path: Path) -> dict[str, object]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    # PPO 会对许多小批次反复反向传播。小网络使用多线程时，线程调度成本
    # 反而高于矩阵计算本身，因此这里固定单线程以缩短实际课程等待时间。
    torch.set_num_threads(1)

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
                action_logits, state_values = network(observation_tensor)
                distribution = Categorical(logits=action_logits)
                action_tensor = distribution.sample()
                log_probability_tensor = distribution.log_prob(action_tensor)

            actions = action_tensor.cpu().numpy()
            next_observations, rewards, terminated, truncated, _ = env.step(actions)
            dones = np.logical_or(terminated, truncated)
            rollout_observations.append(observations.copy())
            rollout_actions.append(actions.copy())
            rollout_log_probabilities.append(log_probability_tensor.cpu().numpy())
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
            np.asarray(rollout_actions), dtype=torch.int64, device=device
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
            _, bootstrap_values = network(next_observation_tensor)
            advantages, return_targets = calculate_gae(
                reward_batch,
                done_batch,
                old_value_batch,
                bootstrap_values,
                config.discount_factor,
                config.gae_lambda,
            )

        flat_observations = observation_batch.reshape(-1, observation_size)
        flat_actions = action_batch.reshape(-1)
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
                action_logits, new_values = network(flat_observations[indices])
                distribution = Categorical(logits=action_logits)
                new_log_probabilities = distribution.log_prob(flat_actions[indices])
                entropy = distribution.entropy().mean()
                actor_loss, probability_ratio = calculate_clipped_policy_loss(
                    new_log_probabilities,
                    flat_old_log_probabilities[indices],
                    flat_advantages[indices],
                    config.clip_coefficient,
                )

                # 评论家也限制单次变化，避免价值预测在重复训练同一批数据时跳动过大。
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
        "actor_losses": actor_losses,
        "critic_losses": critic_losses,
        "entropy_history": entropy_history,
        "clip_fraction_history": clip_fraction_history,
        "approximate_kl_history": approximate_kl_history,
        "evaluation_history": evaluation_history,
        "evaluation": evaluation,
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in network.parameters()),
    }
