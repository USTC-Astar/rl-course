import torch

from rl_learning_lab.ppo import calculate_clipped_policy_loss


def test_positive_advantage_limits_excessive_probability_increase() -> None:
    old_log_probability = torch.log(torch.tensor([0.4]))
    new_log_probability = torch.log(torch.tensor([0.6]))
    loss, ratio = calculate_clipped_policy_loss(
        new_log_probability,
        old_log_probability,
        advantages=torch.tensor([1.0]),
        clip_coefficient=0.2,
    )

    assert torch.allclose(ratio, torch.tensor([1.5]))
    assert torch.allclose(loss, torch.tensor(-1.2))


def test_negative_advantage_limits_excessive_probability_decrease() -> None:
    old_log_probability = torch.log(torch.tensor([0.4]))
    new_log_probability = torch.log(torch.tensor([0.2]))
    loss, ratio = calculate_clipped_policy_loss(
        new_log_probability,
        old_log_probability,
        advantages=torch.tensor([-1.0]),
        clip_coefficient=0.2,
    )

    assert torch.allclose(ratio, torch.tensor([0.5]))
    assert torch.allclose(loss, torch.tensor(0.8))

