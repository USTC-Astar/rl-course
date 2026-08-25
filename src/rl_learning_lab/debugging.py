"""第 21 课配套：常见训练 bug 的真实失败曲线。

在 6×6 网格 Q 学习上注入四种经典错误，与健康基线对照。
环境加入 20% 动作打滑（每步有概率被替换成随机动作），
因为确定性表格环境太宽容，多数 bug 在其中并不发作——
这个事实本身就是第 21 课的教学内容之一。

场景（均为真实训练，种子固定）：
1. healthy         健康基线（ε 衰减、α=0.15、γ=0.95、正确更新所做动作）；
2. no_exploration  ε 恒为 0，跑 10 个种子展示"成败看运气"；
3. wrong_action    经典代码笔误：更新 argmax 动作的 Q 而不是实际做的动作；
4. huge_alpha      α=1.0：目标一变就完全覆盖旧估计，在有打滑的环境里振荡；
5. short_sighted   γ=0.3：远处 +10 折到近乎为零，学会也提不起兴趣。
"""

from __future__ import annotations

import numpy as np

from rl_learning_lab.gridworld import GridWorld, GridWorldConfig

SLIP = 0.2


def train_with_bugs(
    episodes: int = 2000,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.997,
    alpha: float = 0.15,
    gamma: float = 0.95,
    wrong_action_update: bool = False,
    slip: float = SLIP,
    seed: int = 5,
) -> dict[str, object]:
    """带可注入 bug 的 Q 学习训练循环；slip 模拟执行打滑。"""
    env = GridWorld(GridWorldConfig())
    rng = np.random.default_rng(seed)
    q_table = np.zeros((env.state_count, env.action_count))
    rewards: list[float] = []
    epsilon = epsilon_start

    for _ in range(episodes):
        state = env.reset()
        total = 0.0
        for _ in range(env.config.max_steps):
            if rng.random() < epsilon:
                action = int(rng.integers(0, env.action_count))
            else:
                best = q_table[state].max()
                ties = np.flatnonzero(np.abs(q_table[state] - best) < 1e-9)
                action = int(ties[rng.integers(0, len(ties))])

            # 模拟打滑：期望动作有一定概率被环境替换成随机动作。
            executed = action
            if rng.random() < slip:
                executed = int(rng.integers(0, env.action_count))
            next_state, reward, done = env.step(executed)
            total += reward

            future = 0.0 if done else float(q_table[next_state].max())
            target = reward + gamma * future

            if wrong_action_update:
                # 笔误：想更新"实际做的动作"，写成了"当前表上最优的动作"。
                update_slot = int(np.argmax(q_table[state]))
            else:
                update_slot = action
            q_table[state, update_slot] += alpha * (target - q_table[state, update_slot])

            state = next_state
            if done:
                break
        rewards.append(round(total, 4))
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    last_100 = rewards[-100:]
    return {
        "episode_rewards": rewards,
        "mean_reward_last_100": round(float(np.mean(last_100)), 4),
        "std_reward_last_100": round(float(np.std(last_100)), 4),
        "success_rate_last_100": round(
            len([r for r in last_100 if r > 0]) / max(len(last_100), 1), 4
        ),
        "mean_abs_q": round(float(np.abs(q_table).mean()), 4),
    }


def _mean_curve(runs: list[dict[str, object]]) -> list[float]:
    curves = np.array([r["episode_rewards"] for r in runs], dtype=float)
    return [round(float(v), 4) for v in curves.mean(axis=0)]


def run_debug_suite(episodes: int = 2000) -> dict[str, object]:
    """健康基线 + 四种注入 bug 的对照实验（含多种子探索失效分析）。"""
    healthy = train_with_bugs(episodes=episodes)
    wrong_action = train_with_bugs(episodes=episodes, wrong_action_update=True)
    huge_alpha = train_with_bugs(episodes=episodes, alpha=1.0)
    short_sighted = train_with_bugs(episodes=episodes, gamma=0.3)

    # ε=0 跑 10 个种子：展示同一算法、同一环境，只因初始运气不同而命运不同。
    greedy_runs = [
        train_with_bugs(
            episodes=episodes,
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay=1.0,
            seed=seed,
        )
        for seed in range(10)
    ]
    greedy_finals = [run["mean_reward_last_100"] for run in greedy_runs]

    return {
        "episodes": episodes,
        "slip": SLIP,
        "scenarios": {
            "healthy": healthy,
            "wrong_action": wrong_action,
            "huge_alpha": huge_alpha,
            "short_sighted": short_sighted,
        },
        "no_exploration": {
            "per_seed_mean_reward_last_100": greedy_finals,
            "best_seed": max(greedy_finals),
            "worst_seed": min(greedy_finals),
            "mean_curve": _mean_curve(greedy_runs),
            "seeds": 10,
        },
    }
