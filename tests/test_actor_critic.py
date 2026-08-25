import torch

from rl_learning_lab.actor_critic import (
    ActorCriticNetwork,
    calculate_gae,
    calculate_n_step_returns,
)


def test_actor_critic_network_has_two_outputs() -> None:
    network = ActorCriticNetwork(observation_size=4, action_count=2, hidden_size=32)
    logits, values = network(torch.zeros((5, 4), dtype=torch.float32))

    assert logits.shape == (5, 2)
    assert values.shape == (5,)


def test_terminal_state_cuts_future_return() -> None:
    rewards = torch.tensor([[1.0], [1.0], [1.0]])
    dones = torch.tensor([[0.0], [1.0], [0.0]])
    bootstrap = torch.tensor([10.0])
    result = calculate_n_step_returns(rewards, dones, bootstrap, discount_factor=0.5)

    expected = torch.tensor([[1.5], [1.0], [6.0]])
    assert torch.allclose(result, expected)


def test_gae_returns_advantage_plus_value() -> None:
    rewards = torch.tensor([[1.0], [1.0]])
    dones = torch.tensor([[0.0], [1.0]])
    values = torch.tensor([[0.4], [0.6]])
    advantages, returns = calculate_gae(
        rewards,
        dones,
        values,
        bootstrap_values=torch.tensor([3.0]),
        discount_factor=0.9,
        gae_lambda=0.95,
    )

    assert torch.allclose(returns, advantages + values)
    assert torch.allclose(returns[-1], torch.tensor([1.0]))
