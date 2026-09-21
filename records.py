"""Small file helpers, kept outside the algorithm we are learning."""
import csv
import json
from pathlib import Path
import numpy as np


def save_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_agent(directory, q, config):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "q.npy", q, allow_pickle=False)
    (directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def load_agent(directory):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    q = np.load(directory / "q.npy", allow_pickle=False)
    if config.get("environment") != "FrozenLake-v1" or config.get("map_name") != "4x4":
        raise ValueError("This workshop supports the default FrozenLake-v1 4x4 map.")
    if type(config.get("is_slippery")) is not bool:
        raise ValueError("Missing or invalid is_slippery setting.")
    if q.shape != (16, 4) or not np.issubdtype(q.dtype, np.floating) or not np.isfinite(q).all():
        raise ValueError("Expected a finite floating-point Q table with shape (16, 4).")
    return q, config
