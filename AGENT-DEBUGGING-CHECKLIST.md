# Agent stuck? Inspect these in order

[QuietLoom's free workshop](https://www.youtube.com/watch?v=q9UxQozYXwA) • [Runnable project](https://github.com/quietloommedia/reinforcement-learning-from-scratch)

1. Print current state, requested action, next state, reward, terminated and truncated. Does the environment actually move? A wall collision can legitimately leave state unchanged.
2. Confirm action mapping. Standard FrozenLake: 0 left, 1 down, 2 right, 3 up.
3. Inspect the active Q row. Equal entries plus plain `argmax` choose the first tied index. Random tie-breaking removes this bias, not every training problem.
4. Inspect actual epsilon values. Exploration needs to run during training; tie-breaking alone doesn't explore inferior estimates. Evaluation should use the documented evaluation policy.
5. Confirm the update function runs and writes to `q[state, action]`, not the next-state row. Print one before/target/after example.
6. Check true termination versus a time limit. Suppress bootstrap for true terminal transitions. Use the task's intended truncation treatment.
7. Check reward frequency. In a sparse-reward task, no discovered goal can leave many zero estimates. More episodes alone may not repair a broken selector or update.
8. Confirm reset between episodes and retain the observation returned by each step. Stop on termination or truncation.
9. Verify the saved table's shape, file and environment settings. Load in a new process; don't accidentally evaluate a new zero table.
10. Evaluate with episode counts and failure categories. Log seed, map, slippery setting and policy. A pretty route is not a success-rate measurement.

When requesting help, include the smallest reproducible code, versions, settings and a short trace. Remove personal paths and credentials. Describe what you expected and what happened; avoid pasting an entire private project.
