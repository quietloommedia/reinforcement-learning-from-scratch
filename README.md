# Reinforcement Learning from Scratch

**Code an Agent That Learns a Maze — QuietLoom Media / Alex Mercer**

An original FrozenLake Q-learning workshop for learners with basic Python. No GPU or paid API is required.

## Download the complete code-with-me project

**[Download the course project](https://github.com/quietloommedia/reinforcement-learning-from-scratch/releases/latest)** — choose `rl-code-with-me.zip` under Assets.

The release contains the **starter**, cumulative **chapter checkpoints**, completed **reference**, VS Code tasks, custom fantasy assets, saved policies and recorded experiments, plus checksums. Start with its `START-HERE.md`, then open `starter` as a folder in VS Code. Files absent from the starter are created with the lesson. A checkpoint is the completed endpoint of its chapter.

The Python files in this repository's root are the finished reference implementation, provided for browsing. For the exact ready-to-run teaching layout and supplied artwork, use the release ZIP. Its code is identical to these files. [Read the complete setup and lesson guide](START-HERE.md).

## Run the completed reference (Windows)

Extract the release, open `rl-code-with-me/reference` in VS Code, and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe check_setup.py
.\.venv\Scripts\python.exe -m unittest -v
.\.venv\Scripts\python.exe watch.py
```

Use Python 3.12. The package was tested on Windows; the guide includes macOS/Linux commands. A graphical desktop is required for the fantasy replay; learning and evaluation use the CPU.

## What you build

1. Verify Python, VS Code, the interpreter and the environment.
2. Inspect actual states, actions and rewards; measure a random baseline.
3. Write Q-learning and epsilon-greedy action selection.
4. Train and inspect real logs and learning curves.
5. Debug tie-breaking, terminal handling and update arithmetic.
6. Save, evaluate and replay the policy.
7. Repeat with slippery movement and design the next experiment.

## Results and reproducibility

The dry-map result is on one fixed map, not arbitrary-maze generalization. Slippery evaluations include holes and timeouts. Three training seeds share evaluation seeds; they are not 1,500 independent tests. Exact results can vary with package versions and seeds. Training replaces the selected run folder's files, so preserve recorded results before experimenting.

## Course and credits

[Course video](https://www.youtube.com/watch?v=qOCVpt7q2XM) — currently private for review. The project download is available independently.

[QuietLoom Media](https://www.youtube.com/@quietloom-media)

Original teaching code and custom generated fantasy artwork. Gymnasium and other dependencies retain their own licenses. See the guide for technical references and asset notes. Channel credentials, voice recordings, production logs and video masters are not included.
