from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


@dataclass(frozen=True)
class SACConfig:
    total_steps: int = 100_000
    replay_capacity: int = 100_000
    learning_starts: int = 1_000
    batch_size: int = 128
    discount_factor: float = 0.99
    soft_update_rate: float = 0.005
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    temperature_learning_rate: float = 3e-4
    initial_temperature: float = 0.2
    hidden_size: int = 128
    evaluation_interval: int = 5_000
    evaluation_episodes: int = 5
    early_stop_average: float = -190.0
    metric_interval: int = 100
    seed: int = 42


class ReplayBuffer:
    """保存跨越多个旧策略的经验，并进行均匀随机抽样。"""

    def __init__(
        self,
        capacity: int,
        observation_size: int,
        action_size: int,
        seed: int,
    ) -> None:
        self.capacity = capacity
        self.observations = np.zeros(
            (capacity, observation_size), dtype=np.float32
        )
        self.actions = np.zeros((capacity, action_size), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_observations = np.zeros(
            (capacity, observation_size), dtype=np.float32
        )
        self.terminated = np.zeros((capacity, 1), dtype=np.float32)
        self.position = 0
        self.size = 0
        self.random = np.random.default_rng(seed)

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
    ) -> None:
        self.observations[self.position] = observation
        self.actions[self.position] = action
        self.rewards[self.position, 0] = reward
        self.next_observations[self.position] = next_observation
        self.terminated[self.position, 0] = float(terminated)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.size < batch_size:
            raise ValueError("经验回放池中的样本数量不足")
        indices = self.random.integers(0, self.size, size=batch_size)
        return tuple(
            torch.as_tensor(array[indices], dtype=torch.float32, device=device)
            for array in (
                self.observations,
                self.actions,
                self.rewards,
                self.next_observations,
                self.terminated,
            )
        )

    def __len__(self) -> int:
        return self.size


class SACActor(nn.Module):
    """输出高斯分布参数，并通过 tanh 把动作限制在环境范围内。"""

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_size, action_size)
        self.log_standard_deviation_head = nn.Linear(hidden_size, action_size)
        self.register_buffer(
            "action_scale",
            torch.as_tensor((action_high - action_low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.as_tensor((action_high + action_low) / 2.0, dtype=torch.float32),
        )

    def distribution_parameters(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(observations)
        means = self.mean_head(features)
        # 限制标准差范围，避免接近零导致梯度爆炸，也避免探索噪声无限增大。
        log_standard_deviations = self.log_standard_deviation_head(features).clamp(
            -5.0, 2.0
        )
        return means, log_standard_deviations

    def sample(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        means, log_standard_deviations = self.distribution_parameters(observations)
        standard_deviations = log_standard_deviations.exp()
        distribution = Normal(means, standard_deviations)
        raw_actions = distribution.rsample()
        squashed_actions = torch.tanh(raw_actions)
        actions = self.action_bias + self.action_scale * squashed_actions
        # tanh 会挤压坐标间距；若不修正，SAC 会把变换后的动作概率记错。
        correction = torch.log(
            self.action_scale * (1.0 - squashed_actions.pow(2)) + 1e-6
        )
        log_probabilities = (
            distribution.log_prob(raw_actions) - correction
        ).sum(dim=-1, keepdim=True)
        deterministic_actions = self.action_bias + self.action_scale * torch.tanh(
            means
        )
        return actions, log_probabilities, deterministic_actions, standard_deviations


class SoftQNetwork(nn.Module):
    """估计某个状态下执行某个连续动作的长期价值。"""

    def __init__(
        self, observation_size: int, action_size: int, hidden_size: int
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        return self.network(torch.cat((observations, actions), dim=-1))


def soft_update(
    source_network: nn.Module, target_network: nn.Module, update_rate: float
) -> None:
    """让目标网络缓慢跟随在线网络，避免价值目标每次突然跳变。"""

    with torch.no_grad():
        for source_parameter, target_parameter in zip(
            source_network.parameters(), target_network.parameters(), strict=True
        ):
            target_parameter.mul_(1.0 - update_rate)
            target_parameter.add_(update_rate * source_parameter)


def evaluate_sac_policy(
    actor: SACActor,
    critic_one: SoftQNetwork,
    critic_two: SoftQNetwork,
    config: SACConfig,
    device: torch.device,
    episodes: int = 10,
) -> dict[str, object]:
    env = gym.make("Pendulum-v1")
    episode_returns: list[float] = []
    trajectories: list[list[list[float]]] = []
    action_traces: list[list[list[float]]] = []
    action_center_traces: list[list[list[float]]] = []
    action_std_traces: list[list[list[float]]] = []
    reward_traces: list[list[float]] = []
    q_one_traces: list[list[float]] = []
    q_two_traces: list[list[float]] = []
    realized_return_traces: list[list[float]] = []

    for episode in range(episodes):
        observation, _ = env.reset(seed=config.seed + 60_000 + episode)
        trajectory: list[list[float]] = []
        actions: list[list[float]] = []
        action_centers: list[list[float]] = []
        action_stds: list[list[float]] = []
        rewards: list[float] = []
        q_values_one: list[float] = []
        q_values_two: list[float] = []
        total_reward = 0.0

        for _ in range(200):
            observation_tensor = torch.as_tensor(
                observation, dtype=torch.float32, device=device
            ).unsqueeze(0)
            with torch.no_grad():
                _, _, deterministic_action, raw_standard_deviation = actor.sample(
                    observation_tensor
                )
                q_value_one = critic_one(observation_tensor, deterministic_action)
                q_value_two = critic_two(observation_tensor, deterministic_action)

            action_array = deterministic_action.squeeze(0).cpu().numpy()
            trajectory.append(observation.astype(float).tolist())
            actions.append(action_array.astype(float).tolist())
            action_centers.append(action_array.astype(float).tolist())
            action_stds.append(
                (raw_standard_deviation * actor.action_scale)
                .squeeze(0)
                .cpu()
                .tolist()
            )
            q_values_one.append(float(q_value_one.item()))
            q_values_two.append(float(q_value_two.item()))
            observation, reward, terminated, truncated, _ = env.step(action_array)
            rewards.append(float(reward))
            total_reward += float(reward)
            if terminated or truncated:
                break

        running_return = 0.0
        realized_returns: list[float] = []
        for reward in reversed(rewards):
            running_return = reward + config.discount_factor * running_return
            realized_returns.append(running_return)
        realized_returns.reverse()
        episode_returns.append(total_reward)
        trajectories.append(trajectory)
        action_traces.append(actions)
        action_center_traces.append(action_centers)
        action_std_traces.append(action_stds)
        reward_traces.append(rewards)
        q_one_traces.append(q_values_one)
        q_two_traces.append(q_values_two)
        realized_return_traces.append(realized_returns)

    env.close()
    best_index = int(np.argmax(episode_returns))
    average_return = float(np.mean(episode_returns))
    # 最好一局可能只是初始角度碰巧接近竖直；网页应展示更接近整体水平的代表性轨迹。
    representative_index = int(
        np.argmin(np.abs(np.asarray(episode_returns) - average_return))
    )
    return {
        "returns": episode_returns,
        "average_return": average_return,
        "best_return": float(episode_returns[best_index]),
        "trajectory_return": float(episode_returns[representative_index]),
        "trajectory": trajectories[representative_index],
        "actions": action_traces[representative_index],
        "action_centers": action_center_traces[representative_index],
        "action_standard_deviations": action_std_traces[representative_index],
        "rewards": reward_traces[representative_index],
        "q_values_one": q_one_traces[representative_index],
        "q_values_two": q_two_traces[representative_index],
        "realized_returns": realized_return_traces[representative_index],
    }


def train_sac(config: SACConfig, model_path: Path) -> dict[str, object]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(1)
    device = torch.device("cpu")

    env = gym.make("Pendulum-v1")
    env.action_space.seed(config.seed)
    observation, _ = env.reset(seed=config.seed)
    observation_size = int(np.prod(env.observation_space.shape))
    action_size = int(np.prod(env.action_space.shape))
    action_low = env.action_space.low.astype(np.float32)
    action_high = env.action_space.high.astype(np.float32)

    actor = SACActor(
        observation_size,
        action_size,
        action_low,
        action_high,
        config.hidden_size,
    ).to(device)
    critic_one = SoftQNetwork(
        observation_size, action_size, config.hidden_size
    ).to(device)
    critic_two = SoftQNetwork(
        observation_size, action_size, config.hidden_size
    ).to(device)
    target_critic_one = copy.deepcopy(critic_one).to(device)
    target_critic_two = copy.deepcopy(critic_two).to(device)
    for target_network in (target_critic_one, target_critic_two):
        for parameter in target_network.parameters():
            parameter.requires_grad_(False)

    actor_optimizer = torch.optim.Adam(
        actor.parameters(), lr=config.actor_learning_rate
    )
    critic_optimizer = torch.optim.Adam(
        list(critic_one.parameters()) + list(critic_two.parameters()),
        lr=config.critic_learning_rate,
    )
    log_temperature = torch.tensor(
        math.log(config.initial_temperature),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    temperature_optimizer = torch.optim.Adam(
        [log_temperature], lr=config.temperature_learning_rate
    )
    target_entropy = -float(action_size)
    replay_buffer = ReplayBuffer(
        config.replay_capacity,
        observation_size,
        action_size,
        config.seed,
    )

    episode_return = 0.0
    episode_length = 0
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    actor_losses: list[float] = []
    critic_losses: list[float] = []
    temperature_losses: list[float] = []
    temperature_history: list[float] = []
    entropy_history: list[float] = []
    q_disagreement_history: list[float] = []
    replay_size_history: list[int] = []
    metric_steps: list[int] = []
    evaluation_history: list[dict[str, float]] = []
    update_count = 0
    best_evaluation_return = float("-inf")
    best_checkpoint: dict[str, object] | None = None
    steps_completed = 0

    for step in range(1, config.total_steps + 1):
        if step <= config.learning_starts:
            action = env.action_space.sample()
        else:
            observation_tensor = torch.as_tensor(
                observation, dtype=torch.float32, device=device
            ).unsqueeze(0)
            with torch.no_grad():
                sampled_action, _, _, _ = actor.sample(observation_tensor)
            action = sampled_action.squeeze(0).cpu().numpy()

        next_observation, reward, terminated, truncated, _ = env.step(action)
        # 时间上限截断不代表摆控制失败，因此 Bellman 目标仍允许从下一状态继续估值。
        replay_buffer.add(
            observation,
            action,
            float(reward),
            next_observation,
            terminated,
        )
        episode_return += float(reward)
        episode_length += 1
        observation = next_observation
        steps_completed = step

        if terminated or truncated:
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            observation, _ = env.reset()
            episode_return = 0.0
            episode_length = 0

        if step >= config.learning_starts and len(replay_buffer) >= config.batch_size:
            (
                observations,
                actions,
                rewards,
                next_observations,
                terminated_batch,
            ) = replay_buffer.sample(config.batch_size, device)

            with torch.no_grad():
                next_actions, next_log_probabilities, _, _ = actor.sample(
                    next_observations
                )
                next_q_one = target_critic_one(next_observations, next_actions)
                next_q_two = target_critic_two(next_observations, next_actions)
                temperature = log_temperature.exp()
                next_soft_value = torch.minimum(next_q_one, next_q_two) - (
                    temperature * next_log_probabilities
                )
                q_targets = rewards + config.discount_factor * (
                    1.0 - terminated_batch
                ) * next_soft_value

            q_one = critic_one(observations, actions)
            q_two = critic_two(observations, actions)
            critic_loss = nn.functional.mse_loss(q_one, q_targets) + nn.functional.mse_loss(
                q_two, q_targets
            )
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # 更新行动者时冻结评论家参数，但保留动作到 Q 值的梯度路径。
            for parameter in list(critic_one.parameters()) + list(
                critic_two.parameters()
            ):
                parameter.requires_grad_(False)
            sampled_actions, log_probabilities, _, _ = actor.sample(observations)
            sampled_q_one = critic_one(observations, sampled_actions)
            sampled_q_two = critic_two(observations, sampled_actions)
            minimum_sampled_q = torch.minimum(sampled_q_one, sampled_q_two)
            temperature = log_temperature.exp().detach()
            actor_loss = (
                temperature * log_probabilities - minimum_sampled_q
            ).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            for parameter in list(critic_one.parameters()) + list(
                critic_two.parameters()
            ):
                parameter.requires_grad_(True)

            temperature_loss = -(
                log_temperature
                * (log_probabilities + target_entropy).detach()
            ).mean()
            temperature_optimizer.zero_grad()
            temperature_loss.backward()
            temperature_optimizer.step()

            soft_update(critic_one, target_critic_one, config.soft_update_rate)
            soft_update(critic_two, target_critic_two, config.soft_update_rate)
            update_count += 1

            if update_count % config.metric_interval == 0:
                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(critic_loss.item()))
                temperature_losses.append(float(temperature_loss.item()))
                temperature_history.append(float(log_temperature.exp().item()))
                entropy_history.append(float((-log_probabilities).mean().item()))
                q_disagreement_history.append(
                    float((sampled_q_one - sampled_q_two).abs().mean().item())
                )
                replay_size_history.append(len(replay_buffer))
                metric_steps.append(step)

        if step % config.evaluation_interval == 0 and step > config.learning_starts:
            evaluation = evaluate_sac_policy(
                actor,
                critic_one,
                critic_two,
                config,
                device,
                episodes=config.evaluation_episodes,
            )
            average_return = float(evaluation["average_return"])
            evaluation_history.append(
                {"steps": float(step), "average_return": average_return}
            )
            if average_return > best_evaluation_return:
                best_evaluation_return = average_return
                best_checkpoint = {
                    "actor": copy.deepcopy(actor.state_dict()),
                    "critic_one": copy.deepcopy(critic_one.state_dict()),
                    "critic_two": copy.deepcopy(critic_two.state_dict()),
                    "temperature": float(log_temperature.exp().item()),
                }
            if average_return >= config.early_stop_average:
                break

    env.close()
    if best_checkpoint is not None:
        actor.load_state_dict(best_checkpoint["actor"])
        critic_one.load_state_dict(best_checkpoint["critic_one"])
        critic_two.load_state_dict(best_checkpoint["critic_two"])

    final_evaluation = evaluate_sac_policy(
        actor, critic_one, critic_two, config, device
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic_one": critic_one.state_dict(),
            "critic_two": critic_two.state_dict(),
            "temperature": (
                best_checkpoint["temperature"]
                if best_checkpoint is not None
                else float(log_temperature.exp().item())
            ),
        },
        model_path,
    )

    return {
        "environment": "Pendulum-v1",
        "action_low": action_low.astype(float).tolist(),
        "action_high": action_high.astype(float).tolist(),
        "config": asdict(config),
        "steps_completed": steps_completed,
        "update_count": update_count,
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "actor_losses": actor_losses,
        "critic_losses": critic_losses,
        "temperature_losses": temperature_losses,
        "temperature_history": temperature_history,
        "entropy_history": entropy_history,
        "q_disagreement_history": q_disagreement_history,
        "replay_size_history": replay_size_history,
        "metric_steps": metric_steps,
        "evaluation_history": evaluation_history,
        "evaluation": final_evaluation,
        "final_temperature": (
            best_checkpoint["temperature"]
            if best_checkpoint is not None
            else float(log_temperature.exp().item())
        ),
        "target_entropy": target_entropy,
        "device": str(device),
        "parameter_count": sum(
            parameter.numel()
            for network in (actor, critic_one, critic_two)
            for parameter in network.parameters()
        ),
    }
