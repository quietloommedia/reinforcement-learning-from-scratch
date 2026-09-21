import argparse
from pathlib import Path
import gymnasium as gym
import numpy as np
from records import save_rows


def run_random(slippery=False, seed=7, episodes=500):
    env = gym.make(
        "FrozenLake-v1", map_name="4x4",
        is_slippery=slippery
    )
    rng = np.random.default_rng(seed + 900000)
    rows = []
    for episode in range(episodes):
        state, _ = env.reset(seed=100000 + episode)
        steps, total = 0, 0.0
        while True:
            action = int(rng.integers(4))
            state, reward, terminated, truncated, _ = env.step(action)
            steps += 1
            total += reward
            if terminated or truncated:
                if reward == 1:
                    outcome = "goal"
                elif terminated:
                    outcome = "hole"
                else:
                    outcome = "timeout"
                rows.append(dict(
                    episode=episode, seed=100000 + episode,
                    steps=steps, reward=total, outcome=outcome
                ))
                break
    env.close()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slippery", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rows = run_random(args.slippery, args.seed)

    goals = sum(row["outcome"] == "goal" for row in rows)
    holes = sum(row["outcome"] == "hole" for row in rows)
    timeouts = sum(row["outcome"] == "timeout" for row in rows)
    print(f"Random policy: {goals}/{len(rows)} goals")
    print(f"  holes = {holes}, timeouts = {timeouts}")

    condition = "slippery" if args.slippery else "dry"
    folder = Path("runs") / f"{condition}-seed-{args.seed}"
    save_rows(folder / "random.csv", rows)


if __name__ == "__main__":
    main()
