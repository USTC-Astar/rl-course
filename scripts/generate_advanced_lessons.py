#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_learning_lab.advanced_lessons import (
    DynaQConfig,
    calculate_double_dqn_targets,
    calculate_dqn_targets,
    calculate_lambda_returns,
    calculate_n_step_return,
    conservative_q_regularizer,
    importance_sampling_weights,
    independent_q_update,
    prioritized_replay_probabilities,
    run_dyna_q,
    sample_domain_parameters,
    smooth_td3_target_actions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成第 8—17 课的轻量可重复实验数据")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "web" / "data" / "advanced_lessons.json",
    )
    return parser.parse_args()


def build_payload() -> dict[str, object]:
    random_generator = np.random.default_rng(2026)

    n_values = list(range(1, 9))
    n_step_targets = [
        calculate_n_step_return([1.0] * n, next_value=6.0, discount_factor=0.9)
        for n in n_values
    ]
    lambda_values = np.linspace(0.0, 1.0, 21)
    lambda_targets = [
        float(
            calculate_lambda_returns(
                rewards=np.ones(5),
                values=np.array([4.0, 4.5, 5.0, 5.5, 6.0, 6.5]),
                discount_factor=0.9,
                trace_decay=float(trace_decay),
            )[0]
        )
        for trace_decay in lambda_values
    ]

    standard_estimates: list[float] = []
    double_estimates: list[float] = []
    true_action_values = torch.tensor([[5.0, 5.0]])
    for _ in range(200):
        online_noise = torch.tensor(random_generator.normal(0.0, 1.5, size=(1, 2)), dtype=torch.float32)
        target_noise = torch.tensor(random_generator.normal(0.0, 1.5, size=(1, 2)), dtype=torch.float32)
        rewards = torch.tensor([0.0])
        dones = torch.tensor([0.0])
        standard = calculate_dqn_targets(
            rewards,
            dones,
            true_action_values + target_noise,
            discount_factor=1.0,
        )
        double = calculate_double_dqn_targets(
            rewards,
            dones,
            true_action_values + online_noise,
            true_action_values + target_noise,
            discount_factor=1.0,
        )
        standard_estimates.append(float(standard.item()))
        double_estimates.append(float(double.item()))

    td_errors = np.array([0.1, 0.4, 1.0, 2.0, 6.0, 10.0])
    replay_probabilities = prioritized_replay_probabilities(td_errors, priority_exponent=0.6)
    replay_weights = importance_sampling_weights(replay_probabilities, correction_exponent=0.4)
    sampled_indices = random_generator.choice(
        td_errors.size,
        size=5000,
        p=replay_probabilities,
    )
    replay_counts = np.bincount(sampled_indices, minlength=td_errors.size)

    rainbow_components = ["Double", "Dueling", "PER", "N-step", "C51", "NoisyNet"]
    # 这里是教学用复杂度刻度，不冒充基准成绩；网页会明确标记为结构比较。
    rainbow_complexity = [1.2, 1.1, 1.5, 1.1, 1.8, 1.4]

    base_actions = torch.linspace(-1.0, 1.0, 101)
    noise = torch.tensor(random_generator.normal(0.0, 1.0, size=101), dtype=torch.float32)
    smoothed_actions = smooth_td3_target_actions(
        base_actions,
        noise,
        noise_standard_deviation=0.2,
        noise_clip=0.5,
        action_low=-1.0,
        action_high=1.0,
    )

    dyna_zero = run_dyna_q(DynaQConfig(planning_steps=0, seed=17))
    dyna_five = run_dyna_q(DynaQConfig(planning_steps=5, seed=17))
    dyna_twenty = run_dyna_q(DynaQConfig(planning_steps=20, seed=17))

    first_q = np.zeros(2, dtype=np.float64)
    second_q = np.zeros(2, dtype=np.float64)
    cooperation_rates: list[float] = []
    cooperation_window: list[float] = []
    epsilon = 0.2
    for _ in range(400):
        first_action = int(random_generator.integers(2)) if random_generator.random() < epsilon else int(np.argmax(first_q))
        second_action = int(random_generator.integers(2)) if random_generator.random() < epsilon else int(np.argmax(second_q))
        if first_action == 0 and second_action == 0:
            first_reward, second_reward = 4.0, 4.0
        elif first_action == 1 and second_action == 1:
            first_reward, second_reward = 1.0, 1.0
        elif first_action == 0:
            first_reward, second_reward = 0.0, 2.0
        else:
            first_reward, second_reward = 2.0, 0.0
        independent_q_update(first_q, first_action, first_reward, learning_rate=0.1)
        independent_q_update(second_q, second_action, second_reward, learning_rate=0.1)
        cooperation_window.append(float(first_action == 0 and second_action == 0))
        cooperation_rates.append(float(np.mean(cooperation_window[-40:])))

    conservative_penalties: list[float] = []
    conservative_alphas = np.linspace(0.0, 2.0, 21)
    offline_q_values = torch.tensor(
        [[2.0, 7.5, 1.0], [3.0, 8.0, 2.0], [2.5, 6.5, 1.5]],
        dtype=torch.float32,
    )
    dataset_actions = torch.tensor([0, 0, 2])
    regularizer = conservative_q_regularizer(offline_q_values, dataset_actions)
    for alpha in conservative_alphas:
        conservative_penalties.append(float(alpha * regularizer.item()))

    randomized_parameters = sample_domain_parameters(
        seed=33,
        count=200,
        mass_range=(0.7, 1.3),
        friction_range=(0.5, 1.5),
        sensor_noise_standard_deviation=0.05,
    )
    widths = np.linspace(0.0, 1.0, 21)
    deployment_gap = 0.6
    robustness = [
        float(np.clip(1.0 - max(0.0, deployment_gap - width) * 1.1 - 0.12 * width, 0.0, 1.0))
        for width in widths
    ]

    return {
        "lesson08": {
            "n_values": n_values,
            "n_step_targets": n_step_targets,
            "lambda_values": lambda_values.round(2).tolist(),
            "lambda_targets": lambda_targets,
        },
        "lesson09": {
            "standard_estimates": standard_estimates,
            "double_estimates": double_estimates,
            "standard_average": float(np.mean(standard_estimates)),
            "double_average": float(np.mean(double_estimates)),
            "true_value": 5.0,
        },
        "lesson10": {
            "state_values": [2.0, 4.0, 6.0, 8.0],
            "left_advantages": [1.0, 2.0, -1.0, 0.5],
            "right_advantages": [-1.0, -2.0, 1.0, -0.5],
        },
        "lesson11": {
            "td_errors": td_errors.tolist(),
            "probabilities": replay_probabilities.tolist(),
            "weights": replay_weights.tolist(),
            "sample_counts": replay_counts.tolist(),
        },
        "lesson12": {
            "components": rainbow_components,
            "relative_complexity": rainbow_complexity,
        },
        "lesson13": {
            "base_actions": base_actions.tolist(),
            "smoothed_actions": smoothed_actions.tolist(),
        },
        "lesson14": {
            "planning_0_steps": dyna_zero,
            "planning_5_steps": dyna_five,
            "planning_20_steps": dyna_twenty,
        },
        "lesson15": {
            "cooperation_rates": cooperation_rates,
            "final_first_q": first_q.tolist(),
            "final_second_q": second_q.tolist(),
        },
        "lesson16": {
            "alphas": conservative_alphas.round(2).tolist(),
            "penalties": conservative_penalties,
            "raw_regularizer": float(regularizer.item()),
        },
        "lesson17": {
            "randomization_widths": widths.round(2).tolist(),
            "robustness": robustness,
            "mass_samples": randomized_parameters["mass"].tolist(),
            "friction_samples": randomized_parameters["friction"].tolist(),
            "sensor_noise_samples": randomized_parameters["sensor_noise"].tolist(),
        },
    }


def main() -> None:
    args = parse_args()
    payload = build_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"第 8—17 课实验数据已生成：{args.output}")


if __name__ == "__main__":
    main()
