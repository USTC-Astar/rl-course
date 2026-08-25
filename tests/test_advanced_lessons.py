from __future__ import annotations

import numpy as np
import pytest
import torch

from rl_learning_lab.advanced_lessons import (
    DynaQConfig,
    calculate_double_dqn_targets,
    calculate_dqn_targets,
    calculate_lambda_returns,
    calculate_n_step_return,
    calculate_td3_targets,
    combine_dueling_values,
    conservative_q_regularizer,
    importance_sampling_weights,
    prioritized_replay_probabilities,
    project_categorical_distribution,
    run_dyna_q,
    sample_domain_parameters,
    smooth_td3_target_actions,
)


def test_n_step_return_and_lambda_endpoints() -> None:
    assert calculate_n_step_return([1.0, 1.0, 1.0], 5.0, 0.9) == pytest.approx(6.355)

    rewards = np.array([1.0, 2.0])
    values = np.array([0.0, 4.0, 5.0])
    one_step = calculate_lambda_returns(rewards, values, 0.9, trace_decay=0.0)
    full_return = calculate_lambda_returns(rewards, values, 0.9, trace_decay=1.0)
    assert one_step.tolist() == pytest.approx([4.6, 6.5])
    assert full_return.tolist() == pytest.approx([6.85, 6.5])


def test_double_dqn_separates_selection_and_evaluation() -> None:
    rewards = torch.tensor([0.0])
    dones = torch.tensor([0.0])
    online = torch.tensor([[10.0, 8.0]])
    target = torch.tensor([[4.0, 7.0]])

    standard = calculate_dqn_targets(rewards, dones, target, discount_factor=1.0)
    double = calculate_double_dqn_targets(rewards, dones, online, target, discount_factor=1.0)
    assert standard.item() == pytest.approx(7.0)
    assert double.item() == pytest.approx(4.0)


def test_dueling_combination_centers_advantages() -> None:
    values = torch.tensor([[5.0]])
    advantages = torch.tensor([[2.0, -1.0]])
    q_values = combine_dueling_values(values, advantages)
    torch.testing.assert_close(q_values, torch.tensor([[6.5, 3.5]]))
    assert q_values.mean().item() == pytest.approx(5.0)


def test_prioritized_replay_probabilities_and_weights() -> None:
    probabilities = prioritized_replay_probabilities(np.array([0.1, 1.0, 5.0]), 0.6)
    weights = importance_sampling_weights(probabilities, 0.4)
    assert probabilities.sum() == pytest.approx(1.0)
    assert probabilities[2] > probabilities[1] > probabilities[0]
    assert weights[0] == pytest.approx(1.0)
    assert weights[2] < weights[1] < weights[0]


def test_categorical_projection_preserves_probability_mass() -> None:
    next_probabilities = torch.tensor([[0.2, 0.5, 0.3]])
    support = torch.tensor([-1.0, 0.0, 1.0])
    projected = project_categorical_distribution(
        next_probabilities,
        rewards=torch.tensor([0.5]),
        dones=torch.tensor([0.0]),
        support=support,
        discount_factor=0.9,
    )
    assert projected.sum().item() == pytest.approx(1.0)
    assert torch.all(projected >= 0.0)


def test_td3_smoothing_and_target_are_bounded_and_conservative() -> None:
    actions = torch.tensor([-0.95, 0.0, 0.95])
    noise = torch.tensor([-10.0, 1.0, 10.0])
    smoothed = smooth_td3_target_actions(actions, noise, 0.2, 0.5, -1.0, 1.0)
    assert smoothed.tolist() == pytest.approx([-1.0, 0.2, 1.0])

    targets = calculate_td3_targets(
        rewards=torch.tensor([1.0]),
        dones=torch.tensor([0.0]),
        target_q_one=torch.tensor([8.0]),
        target_q_two=torch.tensor([5.0]),
        discount_factor=0.9,
    )
    assert targets.item() == pytest.approx(5.5)


def test_dyna_q_planning_improves_recent_route_length() -> None:
    no_planning = run_dyna_q(DynaQConfig(episodes=50, planning_steps=0, seed=9))
    with_planning = run_dyna_q(DynaQConfig(episodes=50, planning_steps=20, seed=9))
    assert np.mean(with_planning[-10:]) <= np.mean(no_planning[-10:])


def test_conservative_regularizer_penalizes_unseen_high_q_action() -> None:
    q_values = torch.tensor([[2.0, 8.0, 1.0], [3.0, 7.0, 2.0]])
    dataset_actions = torch.tensor([0, 0])
    regularizer = conservative_q_regularizer(q_values, dataset_actions)
    assert regularizer.item() > 4.0


def test_domain_randomization_is_reproducible_and_bounded() -> None:
    first = sample_domain_parameters(3, 100, (0.8, 1.2), (0.5, 1.5), 0.03)
    second = sample_domain_parameters(3, 100, (0.8, 1.2), (0.5, 1.5), 0.03)
    assert np.array_equal(first["mass"], second["mass"])
    assert np.all((first["mass"] >= 0.8) & (first["mass"] <= 1.2))
    assert np.all((first["friction"] >= 0.5) & (first["friction"] <= 1.5))
