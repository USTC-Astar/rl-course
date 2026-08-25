import numpy as np
import torch
from torch import nn

from rl_learning_lab.sac import ReplayBuffer, SACActor, SoftQNetwork, soft_update


def test_replay_buffer_wraps_and_samples_expected_shapes() -> None:
    buffer = ReplayBuffer(capacity=3, observation_size=2, action_size=1, seed=7)
    for index in range(5):
        observation = np.array([index, index + 1], dtype=np.float32)
        buffer.add(
            observation,
            np.array([index / 10], dtype=np.float32),
            float(index),
            observation + 1,
            terminated=False,
        )

    batch = buffer.sample(batch_size=2, device=torch.device("cpu"))

    assert len(buffer) == 3
    assert batch[0].shape == (2, 2)
    assert batch[1].shape == (2, 1)
    assert batch[2].shape == (2, 1)


def test_sac_actor_actions_stay_inside_bounds() -> None:
    actor = SACActor(
        observation_size=3,
        action_size=1,
        action_low=np.array([-2.0], dtype=np.float32),
        action_high=np.array([2.0], dtype=np.float32),
        hidden_size=32,
    )
    actions, log_probabilities, deterministic_actions, _ = actor.sample(
        torch.zeros((64, 3))
    )

    assert torch.all(actions >= -2.0)
    assert torch.all(actions <= 2.0)
    assert torch.all(deterministic_actions >= -2.0)
    assert torch.all(deterministic_actions <= 2.0)
    assert torch.isfinite(log_probabilities).all()


def test_twin_q_networks_return_one_value_per_sample() -> None:
    critic = SoftQNetwork(observation_size=3, action_size=1, hidden_size=32)
    values = critic(torch.zeros((5, 3)), torch.zeros((5, 1)))

    assert values.shape == (5, 1)


def test_soft_update_moves_target_toward_source() -> None:
    source = nn.Linear(2, 1, bias=False)
    target = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        source.weight.fill_(2.0)
        target.weight.fill_(0.0)

    soft_update(source, target, update_rate=0.25)

    assert torch.allclose(target.weight, torch.full_like(target.weight, 0.5))
