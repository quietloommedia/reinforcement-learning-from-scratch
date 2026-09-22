from pathlib import Path
from train import train
from evaluate import evaluate, summarize
from records import save_rows

# Predict: a new training seed may learn different values.
# Fix: dry map, 3000 episodes, and the evaluation protocol.
folder = Path("runs/dry-seed-19-experiment")
q, rows, config = train(
    seed=19, slippery=False, episodes=3000, output=folder
)
evaluation = evaluate(q, slippery=False, seed=19, episodes=500)
summarize(evaluation, "Seed 19 experiment")
save_rows(folder / "evaluation.csv", evaluation)
