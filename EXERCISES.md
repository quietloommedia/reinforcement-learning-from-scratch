# Follow-along exercises

Use the matching chapter checkpoint in a new folder if you need to resume. Recreate its virtual environment; do not copy a .venv. Keep the original run folders so comparisons remain possible.

## Chapter 8: verify evaluation before watching the agent

Create `test_evaluation.py` as demonstrated. Its 25 episodes check code behavior, not trained performance. It checks that Q remains unchanged, exactly 25 records return, each outcome is goal/hole/timeout, and each episode uses 1–100 moves. Select Terminal → Run Task → 7 Test algorithm. Expect six passing tests after adding this file; Chapter 7 has five.

## Chapter 9: a worked separate experiment

Create `experiment.py` from the recorded example or compare it with the reference. It varies the training seed to 19, keeps the dry 4×4 map and 3,000-episode budget, and uses the same evaluation protocol. It saves to `runs/dry-seed-19-experiment`. Select the file and run the default build task. Read the training blocks separately from the final 500-episode greedy evaluation. The recorded result is 500 goals, zero holes and zero timeouts. A repeated count is not evidence of an independent new task; evaluation shares episode seeds 100000–100499.

Before changing anything else, write your prediction and the one factor you will vary. Keep the resulting configuration, table, training.csv and evaluation.csv. Report every outcome, including failures.

## Extension exercise: change the exploration schedule

This is a learner extension, not a claim that the video demonstrates a completed schedule comparison. Work in a separate copy of the Chapter 9 checkpoint. In `agent.py`, change the divisor in `exploration_rate` from 2400 to a chosen value such as 1200. In `train.py`, change the recorded epsilon formula in `config` to match the new code. In `experiment.py`, set the training/evaluation seed back to 7 and use a new output name such as `runs/dry-decay-1200-seed-7`. Keep map, budget, alpha, gamma and evaluation protocol unchanged. Run the script and compare both training history and the final greedy outcomes against dry-seed-7. Do not infer an improvement from a smoother training curve alone.

## Extension exercise: a different map

This requires coordinated changes, not only a new drawing. Update the environment creation in training, baseline, evaluation, setup and replay; record the map identity in config; review the loader's fixed 4×4 and (16,4) validation; verify observation/action sizes and termination/time-limit assumptions. Retrain a fresh table for that state space. Add tests for the new dimensions and saved configuration before making performance claims. Never load the old table as though it learned a universal navigation skill.
