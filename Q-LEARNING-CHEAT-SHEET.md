# Q-learning: a compact working reference

QuietLoom Media • [Free code-along](https://www.youtube.com/watch?v=q9UxQozYXwA) • [Project files](https://github.com/quietloommedia/reinforcement-learning-from-scratch)

**State** is the current situation, such as a square on the lake. **Action** is a requested move. **Reward** is immediate feedback. **Q(s,a)** estimates discounted future return after action a in state s; it is not just the last reward.

```python
target = reward if terminated else reward + gamma * q[next_state].max()
q[state, action] += alpha * (target - q[state, action])
```

`alpha` controls how far this update moves toward the target. `gamma` discounts future rewards. For a true terminal state, there is no future-value term. A time-limit truncation is not automatically a true terminal state; stop/reset the episode, and handle bootstrapping according to the task's formulation. The workshop bootstraps on truncation and suppresses it on termination.

**Choose moves:** with probability epsilon choose any valid action; otherwise choose a best estimated action, breaking ties randomly. Exploration may choose the same action as the greedy branch. `argmax` alone selects the first maximum, which creates a preference when values tie.

**Toy arithmetic:** current Q=.4, reward=0, best next Q=.8, gamma=.9, alpha=.1. Target=.72; error=.32; updated Q=.432. If instead this is terminal with reward1: target=1 and updated Q=.46. These are illustration values, not a training benchmark.

**Evaluate separately:** stop updates, use a documented evaluation policy, freeze map/settings/seeds, count goals/holes/timeouts with an episode denominator. Success on one fixed dry map does not imply success on unfamiliar maps or slippery dynamics.

**Remember:** seed and versions → baseline → training → fresh evaluation → inspect failures → change one thing → compare. Do not select parameters on a final test set and then call it untouched evidence.
