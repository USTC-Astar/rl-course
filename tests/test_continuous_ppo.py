import torch

from rl_learning_lab.continuous_ppo import (
    GaussianActorCriticNetwork,
    sample_squashed_gaussian,
    squashed_gaussian_log_probability,
)


def test_sampled_actions_stay_inside_environment_bounds() -> None:
    means = torch.zeros((64, 1))
    standard_deviations = torch.ones((64, 1))
    low = torch.tensor([-3.0])
    high = torch.tensor([3.0])
    actions, log_probabilities, _ = sample_squashed_gaussian(
        means, standard_deviations, low, high
    )

    assert torch.all(actions >= low)
    assert torch.all(actions <= high)
    assert torch.isfinite(log_probabilities).all()


def test_sampled_log_probability_can_be_recomputed() -> None:
    torch.manual_seed(7)
    means = torch.tensor([[0.2], [-0.4], [1.1]])
    standard_deviations = torch.tensor([[0.7], [1.2], [0.5]])
    low = torch.tensor([-3.0])
    high = torch.tensor([3.0])
    actions, sampled_log_probabilities, _ = sample_squashed_gaussian(
        means, standard_deviations, low, high
    )
    recomputed_log_probabilities, _ = squashed_gaussian_log_probability(
        means, standard_deviations, actions, low, high
    )

    assert torch.allclose(
        sampled_log_probabilities, recomputed_log_probabilities, atol=1e-5
    )


def test_recomputed_log_probability_has_expected_shape() -> None:
    network = GaussianActorCriticNetwork(4, 1, hidden_size=32)
    means, standard_deviations, values = network(torch.zeros((5, 4)))
    actions = torch.zeros((5, 1))
    log_probabilities, entropies = squashed_gaussian_log_probability(
        means,
        standard_deviations,
        actions,
        torch.tensor([-3.0]),
        torch.tensor([3.0]),
    )

    assert log_probabilities.shape == (5,)
    assert entropies.shape == (5,)
    assert values.shape == (5,)
