import numpy as np
import torch

from rl_learning_lab.policy_gradient import PolicyNetwork, discounted_returns, normalized_returns


def test_discounted_returns_are_computed_backwards() -> None:
    result = discounted_returns([1.0, 1.0, 1.0], discount_factor=0.5)

    assert torch.allclose(result, torch.tensor([1.75, 1.5, 1.0]))


def test_normalized_returns_have_near_zero_mean() -> None:
    result = normalized_returns([1.0, 1.0, 1.0, 1.0], discount_factor=0.99)

    assert np.isclose(float(result.mean()), 0.0, atol=1e-6)


def test_policy_network_outputs_one_logit_per_action() -> None:
    network = PolicyNetwork(observation_size=4, action_count=2, hidden_size=32)
    output = network(torch.zeros((5, 4), dtype=torch.float32))

    assert output.shape == (5, 2)

