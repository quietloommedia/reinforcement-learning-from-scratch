import argparse
from pathlib import Path
import gymnasium as gym
import numpy as np
from agent import greedy_action
from records import load_agent, save_rows


def evaluate(q, slippery=False, seed=7, episodes=500):
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
            if q is None:
                action = int(rng.integers(4))
            else:
                action = greedy_action(q, state, rng)
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


def summarize(rows, label):
    counts = {
        kind: sum(row["outcome"] == kind for row in rows)
        for kind in ("goal", "hole", "timeout")
    }
    goals = counts["goal"]
    percent = 100 * goals / len(rows)
    print(f"{label}: {goals}/{len(rows)} goals ({percent:.1f}%)")
    print(f"  holes = {counts['hole']}, timeouts = {counts['timeout']}")
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=Path,
        default=Path("runs/dry-seed-7")
    )
    args = parser.parse_args()
    q, config = load_agent(args.model)
    original = q.copy()
    rows = evaluate(q, config["is_slippery"], config["seed"])
    assert np.array_equal(q, original), "Evaluation changed Q!"
    summarize(rows, "Trained greedy policy")
    save_rows(args.model / "evaluation.csv", rows)


if __name__ == "__main__":
    main()
