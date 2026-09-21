import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=Path, default=Path("runs/dry-seed-7")
    )
    args = parser.parse_args()
    with (args.model / "training.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    rewards = np.array([float(row["reward"]) for row in rows])
    window = 100
    success = np.convolve(
        rewards, np.ones(window) / window, mode="valid"
    )
    episodes = np.arange(window, len(rewards) + 1)
    fig, ax = plt.subplots(
        figsize=(10, 5), layout="constrained"
    )
    ax.plot(episodes, success * 100, color="#087f73", linewidth=2)
    ax.set(
        xlabel="Completed training episodes",
        ylabel="Goals in trailing 100 episodes (%)",
        title="Exploratory training performance",
        ylim=(0, 100)
    )
    ax.grid(alpha=0.2)
    path = args.model / "learning-curve.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("Saved:", path)


if __name__ == "__main__":
    main()
