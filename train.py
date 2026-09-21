import argparse
from pathlib import Path
import gymnasium as gym
import numpy as np
from agent import choose_action, update, exploration_rate
from records import save_agent, save_rows


def train(seed=7, slippery=False, episodes=3000, output=None):
    alpha, gamma = 0.10, 0.99
    rng = np.random.default_rng(seed)
    env = gym.make(
        "FrozenLake-v1", map_name="4x4",
        is_slippery=slippery
    )
    q = np.zeros((env.observation_space.n, env.action_space.n))
    rows = []
    for episode in range(episodes):
        state, _ = env.reset(seed=seed * 10000 + episode)
        epsilon = exploration_rate(episode)
        steps, total = 0, 0.0
        while True:
            action = choose_action(q, state, epsilon, rng)
            next_state, reward, terminated, truncated, _ = env.step(action)
            update(
                q, state, action, reward, next_state,
                terminated, alpha, gamma
            )
            state = next_state
            steps += 1
            total += reward
            if terminated or truncated:
                break
        rows.append(dict(
            episode=episode, epsilon=epsilon,
            reward=total, steps=steps
        ))
        if (episode + 1) % 500 == 0:
            goals = sum(row["reward"] for row in rows[-500:])
            print(
                f"Episode {episode + 1:4d} | epsilon {epsilon:.3f}"
                f" | last 500: {goals:.0f} goals"
            )
    env.close()
    config = dict(
        environment="FrozenLake-v1", map_name="4x4",
        is_slippery=slippery, seed=seed, episodes=episodes,
        alpha=alpha, gamma=gamma,
        epsilon="max(0.05, 1 - episode / 2400)",
        gymnasium_version=gym.__version__,
        numpy_version=np.__version__, time_limit=100,
        rewards="goal=1; frozen=0; hole=0"
    )
    if output is not None:
        save_agent(output, q, config)
        save_rows(Path(output) / "training.csv", rows)
        print("Saved:", output)
    return q, rows, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--slippery", action="store_true")
    args = parser.parse_args()
    condition = "slippery" if args.slippery else "dry"
    folder = Path("runs") / f"{condition}-seed-{args.seed}"
    train(args.seed, args.slippery, output=folder)


if __name__ == "__main__":
    main()
