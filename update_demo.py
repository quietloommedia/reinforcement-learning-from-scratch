import numpy as np
from agent import update

q = np.zeros((16, 4))
q[14, 2] = 0.20
update(
    q, state=14, action=2, reward=1, next_state=15,
    terminated=True, alpha=0.10, gamma=0.99
)
print(f"Terminal goal: {q[14, 2]:.4f}")

q[8, 2] = 0.20
q[9, 1] = 0.80
update(
    q, state=8, action=2, reward=0, next_state=9,
    terminated=False, alpha=0.10, gamma=0.99
)
print(f"Continuing move: {q[8, 2]:.4f}")
