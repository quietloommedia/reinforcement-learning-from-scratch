import gymnasium as gym
import numpy as np
from agent import greedy_action

env = gym.make("FrozenLake-v1", is_slippery=False)
q = np.zeros((16, 4))
state, _ = env.reset(seed=7)
for step in range(5):
    action = int(np.argmax(q[state]))
    state, reward, terminated, truncated, _ = env.step(action)
    print(
        f"step {step + 1}: action={action}, "
        f"state={state}, reward={reward}"
    )
env.close()

print("All ties:", np.flatnonzero(q[0] == q[0].max()))
rng = np.random.default_rng(7)
choices = [greedy_action(q, 0, rng) for _ in range(12)]
print("Random tie-breaking:", choices)
print("Tie-breaking fixes this tie bias.")
print("Epsilon exploration still matters after ties disappear.")
