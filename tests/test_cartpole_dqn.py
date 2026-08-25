import numpy as np
import torch

from rl_learning_lab.cartpole_dqn import DQNConfig, QNetwork, ReplayBuffer, epsilon_at_step


def test_epsilon_schedule_reaches_configured_floor() -> None:
    config = DQNConfig(epsilon_decay_steps=100)

    assert epsilon_at_step(0, config) == config.epsilon_start
    assert epsilon_at_step(100, config) == config.epsilon_end
    assert epsilon_at_step(1000, config) == config.epsilon_end


def test_q_network_output_shape() -> None:
    network = QNetwork(observation_size=4, action_count=2, hidden_size=32)
    output = network(torch.zeros((3, 4), dtype=torch.float32))

    assert output.shape == (3, 2)


def test_replay_buffer_returns_requested_batch() -> None:
    buffer = ReplayBuffer(capacity=10, seed=1)
    for index in range(5):
        state = np.full(4, index, dtype=np.float32)
        buffer.add(state, index % 2, 1.0, state + 1, False)

    batch = buffer.sample(batch_size=4, device=torch.device("cpu"))

    assert batch[0].shape == (4, 4)
    assert batch[1].shape == (4, 1)
