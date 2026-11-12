import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "thunder-results" / "long-a100x8-20261112" / "runs"
RESULTS_DIR = ROOT / "eval" / "results"
OUT_DIR = ROOT / "images"

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
BLUE_LIGHT = "#a9c8ef"
GRAY = "#8f8e89"

plt.rcParams.update({
    "svg.fonttype": "none",
    "svg.hashsalt": "micrograd",
    "font.size": 11,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_SECONDARY,
    "axes.titlecolor": TEXT,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "legend.frameon": False,
    "legend.labelcolor": TEXT,
})


def read_log(path):
    run, train, val = None, [], []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if row["kind"] == "run" and run is None:
                run = row
            elif row["kind"] == "train":
                train.append(row)
            elif row["kind"] == "val":
                val.append(row)
    return run, train, val


def binned(steps, values, width):
    xs, ys = [], []
    for start in range(0, len(values), width):
        chunk = values[start:start + width]
        xs.append(steps[start + len(chunk) // 2])
        ys.append(sum(chunk) / len(chunk))
    return xs, ys


def save(fig, out_dir, name):
    path = out_dir / name
    fig.savefig(path, metadata={"Date": None}, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


def plot_loss(run, train, val, out_dir):
    per_step = run["tokens_per_step"] / 1e9
    steps = [r["step"] for r in train]
    xs, ys = binned(steps, [r["loss"] for r in train], 200)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot([x * per_step for x in xs], ys, color=BLUE_LIGHT, linewidth=1.5, label="Train")
    ax.plot([r["step"] * per_step for r in val], [r["val_loss"] for r in val],
            color=BLUE, linewidth=2, label="Validation")
    final = val[-1]
    ax.annotate(f"{final['val_loss']:.3f}", (final["step"] * per_step, final["val_loss"]),
                xytext=(0, -16), textcoords="offset points", ha="right", color=TEXT, fontsize=10)
    ax.set_ylim(2.8, 4.6)
    ax.set_xlim(0, final["step"] * per_step * 1.01)
    ax.set_xlabel("Training tokens (billions)")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Loss", pad=12)
    ax.legend(loc="upper right")
    save(fig, out_dir, "gpt2-loss.svg")


def hellaswag(path):
    data = json.loads(path.read_text())
    scores = data.get("harness", data)["results"]["hellaswag"]
    return scores["acc_norm,none"] * 100


def plot_hellaswag(results_dir, out_dir):
    rows = [
        ("GPT-2 124M", hellaswag(results_dir / "gpt2-hellaswag-reference.json"), GRAY),
        ("taby-mm", hellaswag(results_dir / "taby-mm-hellaswag.json"), BLUE),
    ]
    fig, ax = plt.subplots(figsize=(8, 2.2))
    for i, (name, score, color) in enumerate(rows):
        ax.barh(i, score, height=0.55, color=color)
        ax.text(score + 0.6, i, f"{score:.2f}%", va="center", color=TEXT, fontsize=11)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.tick_params(axis="y", colors=TEXT)
    ax.set_xlim(0, 45)
    ax.set_xticks([])
    ax.grid(visible=False)
    ax.spines["bottom"].set_visible(False)
    ax.set_title("HellaSwag score", pad=12)
    save(fig, out_dir, "gpt2-hellaswag.svg")


def plot_learning_rate(run, train, out_dir):
    per_step = run["tokens_per_step"] / 1e9
    steps = [r["step"] for r in train]
    xs, ys = binned(steps, [r["lr"] * 1e4 for r in train], 100)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot([x * per_step for x in xs], ys, color=BLUE, linewidth=2)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}e-4" if v else "0")
    ax.set_ylim(0, 6.6)
    ax.set_yticks([0, 2, 4, 6])
    ax.set_xlim(0, steps[-1] * per_step * 1.01)
    ax.set_xlabel("Training tokens (billions)")
    ax.set_ylabel("Learning rate")
    ax.set_title("Learning rate", pad=12)
    save(fig, out_dir, "gpt2-learning-rate.svg")


def main():
    parser = argparse.ArgumentParser(description="Draw the README charts from the training log and HellaSwag results.")
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run, train, val = read_log(args.run_dir / "long.jsonl")
    plot_loss(run, train, val, args.out_dir)
    plot_hellaswag(args.results_dir, args.out_dir)
    plot_learning_rate(run, train, args.out_dir)


if __name__ == "__main__":
    main()
