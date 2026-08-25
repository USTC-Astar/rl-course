import numpy as np

from rl_learning_lab.gridworld import GridWorld, QLearningConfig, train_q_learning


def test_goal_is_reachable_after_training() -> None:
    env = GridWorld()
    result = train_q_learning(
        env,
        QLearningConfig(episodes=1500, seed=7),
        snapshot_episodes=(),
    )

    assert result["reached_goal"] is True
    assert result["route"][-1] == env.config.goal
    assert len(result["route"]) <= 20


def test_wall_collision_keeps_agent_in_place() -> None:
    env = GridWorld()
    state = env.reset()
    next_state, reward, done = env.step(2)

    assert next_state == state
    assert np.isclose(reward, env.config.wall_reward)
    assert done is False

