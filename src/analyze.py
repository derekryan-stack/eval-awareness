"""Summarize evaluate.py outputs (and training logs) into a table and a figure.

Reads results/eval/<label>.jsonl for each label and, for trained models,
results/<run>/metrics.json for the per-step reasoning length during training.

Usage:
    python src/analyze.py --labels base cot_length --runs cot_length=results/prob_cot_length
Writes results/eval/summary.md, summary.csv and cot_length.png.
"""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rewards import is_correct

REPO_ROOT = Path(__file__).resolve().parent.parent

# Categorical slots 1-3 of the reference palette (validated all-pairs for 3 series), in fixed order.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def summarize(label, records, base_mean):
    toks = [r["reasoning_tokens"] for r in records]
    correct = sum(r["correct"] for r in records)
    lo, hi = wilson(correct, len(records))
    mean = statistics.mean(toks)
    se = statistics.stdev(toks) / math.sqrt(len(toks)) if len(toks) > 1 else float("nan")
    correct_toks = [r["reasoning_tokens"] for r in records if r["correct"]]
    return {
        "model": label,
        "n": len(records),
        "accuracy": correct / len(records),
        "accuracy_ci95": f"[{lo:.2f}, {hi:.2f}]",
        "finished_thinking": sum(r["finished_thinking"] for r in records) / len(records),
        "reasoning_tokens_mean": mean,
        "reasoning_tokens_se": se,
        "reasoning_tokens_median": statistics.median(toks),
        "reasoning_tokens_mean_if_correct": statistics.mean(correct_toks) if correct_toks else float("nan"),
        "change_vs_base": (mean - base_mean) / base_mean if base_mean else float("nan"),
    }


def training_curve(run_dir):
    path = Path(run_dir) / "metrics.json"
    if not path.exists():
        return [], []
    steps, values = [], []
    for entry in json.loads(path.read_text())["log_history"]:
        key = next((k for k in entry if "cot_tokens" in k), None)
        if key is not None and "step" in entry:
            steps.append(entry["step"])
            values.append(entry[key])
    return steps, values


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot(labels, data, rows, curves, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.2, 0.8, 1.4]})
    color = {label: COLORS[i % len(COLORS)] for i, label in enumerate(labels)}

    ax = axes[0]
    style(ax)
    for i, label in enumerate(labels):
        toks = [r["reasoning_tokens"] for r in data[label]]
        jitter = [i + ((j * 0.618) % 1 - 0.5) * 0.35 for j in range(len(toks))]
        ax.scatter(jitter, toks, s=16, color=color[label], alpha=0.55, edgecolors=SURFACE, linewidths=0.6, zorder=2)
        mean = statistics.mean(toks)
        ax.hlines(mean, i - 0.3, i + 0.3, color=INK, linewidth=2, zorder=3)
        ax.annotate(f"{mean:,.0f}", (i + 0.32, mean), va="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(labels)), labels, color=INK)
    ax.set_ylabel("Reasoning tokens (before </think>)", color=INK_2, fontsize=9)
    ax.set_title("Reasoning length per completion\n(bar = mean)", loc="left", fontsize=11, color=INK)
    ax.set_ylim(bottom=0)

    ax = axes[1]
    style(ax)
    for i, row in enumerate(rows):
        lo, hi = (float(x) for x in row["accuracy_ci95"].strip("[]").split(", "))
        ax.bar(i, row["accuracy"], width=0.55, color=color[row["model"]], zorder=2)
        ax.errorbar(i, row["accuracy"], yerr=[[row["accuracy"] - lo], [hi - row["accuracy"]]], color=INK_2, capsize=4, linewidth=1.2, zorder=3)
        ax.annotate(f"{row['accuracy']:.0%}", (i, hi + 0.02), ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(rows)), [row["model"] for row in rows], color=INK)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Accuracy on held-out problems\n(95% Wilson CI)", loc="left", fontsize=11, color=INK)

    ax = axes[2]
    style(ax)
    for label, (steps, values) in curves.items():
        if steps:
            # Each step sees only a couple of prompts, so show the raw series faintly under a rolling mean.
            window = 10
            smooth = [statistics.mean(values[max(0, i - window + 1):i + 1]) for i in range(len(values))]
            ax.plot(steps, values, color=color[label], linewidth=1, alpha=0.3)
            ax.plot(steps, smooth, color=color[label], linewidth=2, label=f"{label} ({window}-step mean)")
            ax.annotate(f"{smooth[-1]:,.0f}", (steps[-1], smooth[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    if sum(bool(steps) for steps, _ in curves.values()) > 1:
        ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    ax.set_xlabel("Training step", color=INK_2, fontsize=9)
    ax.set_ylabel("Mean reasoning tokens per completion", color=INK_2, fontsize=9)
    ax.set_title("Reasoning length during training\n(faint = per step, bold = 10-step mean)", loc="left", fontsize=11, color=INK)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", nargs="+", required=True, help="Eval labels; the first is the baseline")
    parser.add_argument("--runs", nargs="*", default=[], help="label=run_dir pairs for training curves")
    parser.add_argument("--eval-dir", default=str(REPO_ROOT / "results" / "eval"))
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    data = {label: [json.loads(line) for line in (eval_dir / f"{label}.jsonl").open()] for label in args.labels}
    # Re-grade with the current checker so every label is scored by the same code.
    for records in data.values():
        for r in records:
            r["correct"] = r["finished_thinking"] and is_correct(r["response"], r["answer"])
    base_mean = statistics.mean(r["reasoning_tokens"] for r in data[args.labels[0]])
    rows = [summarize(label, data[label], base_mean) for label in args.labels]

    with (eval_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    families = sorted({r["family"] or "?" for records in data.values() for r in records})
    lines = [
        "| Model | n | Accuracy (95% CI) | Finished thinking | Reasoning tokens, mean ± SE | Median | Mean if correct | Δ vs base |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['n']} | {row['accuracy']:.0%} {row['accuracy_ci95']} | {row['finished_thinking']:.0%} "
            f"| {row['reasoning_tokens_mean']:,.0f} ± {row['reasoning_tokens_se']:,.0f} | {row['reasoning_tokens_median']:,.0f} "
            f"| {row['reasoning_tokens_mean_if_correct']:,.0f} | {row['change_vs_base']:+.0%} |"
        )
    lines += ["", "Mean reasoning tokens by problem family:", "", "| Family | " + " | ".join(args.labels) + " |", "|---|" + "---|" * len(args.labels)]
    for fam in families:
        cells = []
        for label in args.labels:
            toks = [r["reasoning_tokens"] for r in data[label] if (r["family"] or "?") == fam]
            cells.append(f"{statistics.mean(toks):,.0f}" if toks else "-")
        lines.append(f"| {fam} | " + " | ".join(cells) + " |")
    (eval_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    curves = dict(pair.split("=", 1) for pair in args.runs)
    plot(args.labels, data, rows, {label: training_curve(run) for label, run in curves.items()}, eval_dir / "cot_length.png")
    print(f"Wrote {eval_dir / 'summary.md'}, summary.csv, cot_length.png")


if __name__ == "__main__":
    main()
