import sys
import importlib.metadata as metadata
import gymnasium as gym

print("Python:", sys.version.split()[0])
print("Interpreter:", sys.executable)
for package in ("gymnasium", "numpy", "pygame-ce", "matplotlib"):
    print(f"{package}: {metadata.version(package)}")

env = gym.make("FrozenLake-v1", is_slippery=False)
state, info = env.reset(seed=7)
print(f"Ready: {env.observation_space.n} states, "
      f"{env.action_space.n} actions; start = {state}")
env.close()
