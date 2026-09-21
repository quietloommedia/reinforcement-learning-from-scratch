import gymnasium as gym

env = gym.make(
    "FrozenLake-v1", map_name="4x4",
    is_slippery=False, render_mode="ansi"
)

state, info = env.reset(seed=7)
print(env.render())
print("Actions: 0 LEFT, 1 DOWN, 2 RIGHT, 3 UP")
print("Start state:", state)

next_state, reward, terminated, truncated, info = env.step(1)
print(env.render())
print("Next state:", next_state)
print("Reward:", reward)
print("Terminated:", terminated, "Truncated:", truncated)
env.close()
