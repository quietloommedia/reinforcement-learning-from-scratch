import numpy as np


def greedy_action(q, state, rng):
    row = q[state]
    winners = np.flatnonzero(row == row.max())
    return int(rng.choice(winners))


def choose_action(q, state, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(q.shape[1]))
    return greedy_action(q, state, rng)


def update(
    q, state, action, reward, next_state,
    terminated, alpha, gamma
):
    if terminated:
        target = reward
    else:
        target = reward + gamma * q[next_state].max()
    q[state, action] += alpha * (target - q[state, action])


def exploration_rate(episode):
    return max(0.05, 1 - episode / 2400)
