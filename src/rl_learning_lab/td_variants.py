"""第 18 课配套：蒙特卡洛评估与 SARSA 对照实验。

全部在 6×6 网格世界上真实运行：
1. 蒙特卡洛首访评估——随机策略下估计若干状态的价值，对照动态规划真值；
2. SARSA 与 Q 学习——同一随机种子、同参数训练，对比回合奖励曲线。
"""

from __future__ import annotations

from rl_learning_lab.gridworld import GridWorld, GridWorldConfig, choose_action


def run_mc_prediction(
    episodes: int = 4000,
    targets: tuple[tuple[int, int], ...] = ((5, 0), (0, 0), (5, 5)),
    gamma: float = 0.95,
    seed: int = 7,
) -> dict[str, object]:
    """随机策略下的首访蒙特卡洛价值估计。

    每回合从起点走到终点/陷阱/超时，倒推折扣回报，
    记录目标状态首次被访问时的回报的累计平均。
    """
    import numpy as np

    env = GridWorld(GridWorldConfig())
    rng = np.random.default_rng(seed)
    sums = {position: 0.0 for position in targets}
    counts = {position: 0 for position in targets}
    history = {position: [] for position in targets}

    for _ in range(episodes):
        env.reset()
        visited_states: list[int] = []
        rewards: list[float] = []
        done = False
        while not done:
            state = env.position_to_state(env.state)
            action = int(rng.integers(0, env.action_count))
            _, reward, done = env.step(action)
            visited_states.append(state)
            rewards.append(reward)

        # 倒推折扣回报：G_t = r_t + γ·r_{t+1} + …
        returns = [0.0] * len(rewards)
        running = 0.0
        for index in range(len(rewards) - 1, -1, -1):
            running = rewards[index] + gamma * running
            returns[index] = running

        cols = env.config.cols
        seen: set[tuple[int, int]] = set()
        for index, state in enumerate(visited_states):
            position = divmod(state, cols)
            if position in targets and position not in seen:
                seen.add(position)
                sums[position] += returns[index]
                counts[position] += 1

        for position in targets:
            if counts[position] > 0:
                history[position].append(round(sums[position] / counts[position], 4))

    return {
        "targets": [list(p) for p in targets],
        "estimate_history": [history[p] for p in targets],
        "final_estimates": [round(sums[p] / max(counts[p], 1), 4) for p in targets],
        "visit_counts": [counts[p] for p in targets],
        "episodes": episodes,
        "gamma": gamma,
    }


def run_sarsa_vs_q(
    episodes: int = 800,
    alpha: float = 0.15,
    gamma: float = 0.95,
    epsilon: float = 0.1,
    seed: int = 11,
) -> dict[str, object]:
    """同一种子、同一环境分别训练 SARSA 与 Q 学习，返回对照数据。

    两者唯一的差别是 TD 目标：
    SARSA     用行为策略实际选出的下一个动作 a′ 的 Q 值（同策略）；
    Q 学习    用下一状态的最大 Q 值（异策略）。
    """
    import numpy as np

    def train(use_sarsa: bool) -> dict[str, object]:
        env = GridWorld(GridWorldConfig())
        rng = np.random.default_rng(seed)
        q_table = np.zeros((env.state_count, env.action_count))
        rewards: list[float] = []
        successes: list[bool] = []

        for _ in range(episodes):
            state = env.reset()
            total = 0.0
            reached_goal = False
            for _ in range(env.config.max_steps):
                action = choose_action(q_table, state, epsilon, rng)
                next_state, reward, done = env.step(action)
                total += reward

                if use_sarsa:
                    # SARSA：下一个动作同样由 ε-贪心行为策略选出。
                    next_action = choose_action(q_table, next_state, epsilon, rng)
                    target = reward + gamma * q_table[next_state, next_action]
                else:
                    # Q 学习：目标直接取下一状态的最大价值。
                    target = reward + gamma * float(q_table[next_state].max())
                q_table[state, action] += alpha * (target - q_table[state, action])

                state = next_state
                if done:
                    if env.state == env.config.goal:
                        reached_goal = True
                    break
            rewards.append(round(total, 4))
            successes.append(reached_goal)

        last_100_rewards = rewards[-100:]
        last_100_success = successes[-100:]
        return {
            "episode_rewards": rewards,
            "success_rate_last_100": round(
                sum(1 for s in last_100_success if s) / max(len(last_100_success), 1), 4
            ),
            "mean_reward_last_100": round(sum(last_100_rewards) / max(len(last_100_rewards), 1), 4),
        }

    return {
        "sarsa": train(use_sarsa=True),
        "q_learning": train(use_sarsa=False),
        "episodes": episodes,
        "epsilon": epsilon,
        "alpha": alpha,
        "gamma": gamma,
        "seed": seed,
    }
