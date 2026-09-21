import argparse
from pathlib import Path
import gymnasium as gym
import numpy as np
from agent import greedy_action
from records import load_agent
from lake_view import LakeWindow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=Path,
        default=Path("runs/dry-seed-7")
    )
    parser.add_argument("--episode-seed", type=int, default=100000)
    args = parser.parse_args()
    q, config = load_agent(args.model)
    env = gym.make(
        config["environment"], map_name=config["map_name"],
        is_slippery=config["is_slippery"]
    )
    window = LakeWindow(
        env.unwrapped.desc, config["is_slippery"]
    )
    rng = np.random.default_rng(config["seed"] + 900000)
    state, _ = env.reset(seed=args.episode_seed)
    try:
        window.show(state, "Ready", seconds=1.2)
        while window.open:
            action = greedy_action(q, state, rng)
            next_state, reward, terminated, truncated, _ = env.step(action)
            print(
                f"state {state} -> {next_state} | "
                f"action={action} | reward={reward}"
            )
            state = next_state
            action_name = ("LEFT", "DOWN", "RIGHT", "UP")[action]
            window.show(
                state, f"requested {action_name} / reward {reward:g}"
            )
            if terminated or truncated:
                if reward == 1:
                    outcome = "Goal!"
                elif terminated:
                    outcome = "Hole."
                else:
                    outcome = "Time limit."
                print(outcome)
                window.show(state, outcome, seconds=3)
                break
    finally:
        env.close()
        window.close()


if __name__ == "__main__":
    main()
