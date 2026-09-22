# Reinforcement Learning from Scratch

**Code an Agent That Learns a Maze — QuietLoom / Alex Mercer**

An original code-with-me workshop for learners who know basic Python. Gymnasium provides FrozenLake; we implement tabular Q-learning ourselves. No GPU, paid API or neural network is required.

## Start here

Use Python 3.12 and Visual Studio Code. The recording uses Windows. The macOS/Linux commands below are provided for portability; the complete release is tested on Windows. Obtain Python from https://www.python.org/downloads/ and VS Code from https://code.visualstudio.com/. In VS Code, install **Python** by **Microsoft** (`ms-python.python`).

Unzip the course download somewhere writable. Open **starter** as a folder in VS Code. Use **reference** to compare with completed files. **checkpoints** contains cumulative chapter endpoints: copy a checkpoint into a separate working folder if you need to resume. Do not copy a virtual environment between folders; recreate it in the working folder.

The starter includes `records.py` (file helpers), `test_agent.py` (tests used after the algorithm exists), dependency versions and prepared VS Code tasks. It intentionally does not include the completed algorithm. The tests cannot run until the files they import have been written.

### Windows PowerShell

Run these commands from the project folder. No activation or execution-policy change is required.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe check_setup.py
```

### macOS / Linux

```sh
python3.12 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python check_setup.py
```

On Linux, your distribution may package the `venv` module separately. Install that package for your Python version if the standard environment-creation command reports it missing. A graphical desktop is required for `watch.py`; training and evaluation do not need one.

### VS Code tasks and interpreter

The supplied tasks are project configuration, not built-in FrozenLake features. Use **Terminal → Run Task** for environment creation and dependency installation. Use **Python: Select Interpreter** in the Command Palette to select this folder's `.venv` executable. The exact path differs by operating system.

**Terminal → Run Build Task** runs the active Python file using the supplied default task. Keep the intended `.py` file active, not a CSV or Markdown file. Other tasks train/evaluate the slippery condition or run the tests. The tasks explicitly name the local interpreter; selecting a different interpreter in the Python extension does not rewrite them.

If VS Code opens in Restricted Mode, inspect the source before choosing whether to trust your own workshop folder. Do not disable security protections globally.

## Course sequence

| Chapter | Build / inspect | Successful checkpoint |
|---|---|---|
| 1 | Real random and trained episodes | Understand the goal and limits |
| 2 | `check_setup.py` | Project interpreter; 16 states, 4 actions, start 0 |
| 3 | `environment.py` | Down from 0 reaches 4 on the dry map |
| 4 | `baseline.py` | 500 outcome rows; counts sum to 500 |
| 5 | `agent.py`, `update_demo.py` | Terminal 0.2800; continuing 0.2592 |
| 6 | `train.py`, `plot_results.py` | 3000 training rows, saved table, measured plot |
| 7 | `debug_ties.py`, tests | Explain stuck-left bias and terminal handling |
| 8 | `evaluate.py`, `watch.py` | Frozen saved policy evaluated and replayed |
| 9 | Slippery condition | Separate run, failures included, controlled next experiment |

## Run the completed reference

Replace `python` below with your environment's executable, or activate it yourself. Run from the folder containing the Python files.

```text
python check_setup.py
python environment.py
python baseline.py
python update_demo.py
python train.py
python plot_results.py
python debug_ties.py
python -m unittest -v
python evaluate.py
python watch.py
python baseline.py --slippery
python train.py --slippery
python evaluate.py --model runs/slippery-seed-7
python watch.py --model runs/slippery-seed-7
python plot_results.py --model runs/slippery-seed-7
```

Training starts a new zero table each time and replaces files in that condition/seed's run folder. Preserve a copy before modifying an experiment. Use `--seed 19` or `--seed 41` to repeat with the other recorded training seeds. Evaluation accepts the matching `--model` folder.

`watch.py` opens the workshop's custom fantasy window, driven by real Gymnasium observations, and closes when the episode ends. The printed trace remains in the terminal. Video replays may hold each actual state longer for readability; they do not fabricate transitions.

## What the result means

- Default 4×4 map, rewards 1 for goal and 0 otherwise, external 100-step cap.
- Action order: 0 left, 1 down, 2 right, 3 up. State address: row × 4 + column.
- Q-learning uses alpha 0.1, gamma 0.99, 3000 episodes, and epsilon `max(0.05, 1 - episode / 2400)`.
- Both ending flags stop a rollout. Only true termination removes the future-value term; this workshop treats the wrapper's time limit as external to the underlying task.
- Evaluation has no updates and uses episode seeds 100000–100499, separate from training. It is still the same map, not a new-maze generalization test.
- Three training seeds share the evaluation seeds. Their results are not 1500 independent samples. No finite-run optimality guarantee is claimed.
- `learning-curve.png` shows training success in a trailing 100-episode window. Adjacent points overlap. It is not held-out performance or a confidence interval.
- `update_demo.py` assigns illustrative numbers to test arithmetic; those numbers are not presented as learned values.

## Troubleshooting

**Module not found:** print `sys.executable` and use that executable with `-m pip install -r requirements.txt`. Check the task's interpreter as well as the extension's selection.

**File not found:** run from the project folder. Preserve `q.npy` beside its `config.json`. Check `--model`.

**Task runs the wrong file:** our default task uses the active editor. Select the intended Python file first.

**No graphical window:** use a graphical desktop; test training/evaluation separately. The ANSI inspection works without a game window. Keep `lake_view.py` beside `watch.py` and preserve the supplied `assets/lake-fantasy-v1` folder.

**Agent never succeeds:** inspect a short state/action/reward trace; check exploration, reset placement, update ordering, terminal handling, and whether you accidentally recreate Q inside the episode loop. Do not change every hyperparameter at once.

**Different counts:** compare seeds, package versions, map/dynamics and code. Report your measured counts honestly. Repeated runs under different versions need not match.

## Credits and technical references

This is an original course and implementation, not a transcript rewrite of another instructor's course.

- Gymnasium FrozenLake specification: https://gymnasium.farama.org/environments/toy_text/frozen_lake/
- Termination and external time limits: https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/
- Python virtual environments: https://docs.python.org/3.12/library/venv.html
- VS Code Python environments: https://code.visualstudio.com/docs/python/environments
- Sutton and Barto, *Reinforcement Learning: An Introduction*: http://incompleteideas.net/book/the-book-2nd.html

The custom fantasy renderer in `lake_view.py` is supplied presentation code. It draws the actual Gymnasium map and current observation; it never chooses actions, changes rewards or advances the game. Its high-resolution adventurer, treasure chest, ice and hazard artwork was generated with built-in Codex ImageGen for this course. The renderer is independent of Q-learning, so you can learn and test the algorithm without a graphical desktop. State changes remain discrete; a brief visual hop connects the two actual observed cells. The short viewing delay makes each actual observation readable; it does not change the environment's dynamics.

The stock Gymnasium renderer used in earlier previews credits Franuka's Snowy Town pack and Mel Tillery; those stock sprites are not used by the new course renderer. Exact diagrams, state numbers and code remain programmatic. Narration is synthetic, using the channel's established Alex Mercer voice profile; its reference attribution is retained in the production records.


## Live demonstration revision

This local revision adds the explicitly printed random success rate, the recorded separate experiment script, and EXERCISES.md. The starter stays intentionally incomplete. Chapter 8 includes the evaluation regression test before episode playback. Existing algorithm and environment behavior are preserved. The video shows Windows; other-platform setup commands remain references, not filmed demonstrations.
