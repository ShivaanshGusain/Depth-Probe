"""Self-contained plots and paired document-bootstrap changes for segment runs."""

import csv
import json
from pathlib import Path

import numpy as np

from depthprobe.analysis import DocBootstrap
from depthprobe.provenance import write_json


def report(run, *, out=None, show=False):
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = Path(run)
    out = Path(out) if out else run / "report"
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((run / "summary.json").read_text())
    metrics = summary["evaluation"]
    regularizer = summary["args"].get("regularizer", "l2")
    intervals = {}
    if (run / "before.npz").exists() and (run / "after.npz").exists():
        with np.load(run / "before.npz") as before, np.load(run / "after.npz") as after:
            for key in ("target_index", "target_id", "doc"):
                if not np.array_equal(before[key], after[key]):
                    raise ValueError(f"Before/after {key} do not align")
            boot = DocBootstrap(before["doc"], 1000, summary["args"].get("seed", 0))
            intervals = {"documents": boot.n_docs,
                         "delta_cross_entropy": boot.mean(after["nll"].astype(float) - before["nll"]),
                         "delta_accuracy": boot.mean(after["correct"].astype(float) - before["correct"].astype(float))}
    write_json(out / "paired_changes.json", intervals)
    rows = []
    if (run / "training.csv").exists():
        with (run / "training.csv").open() as stream:
            rows = list(csv.DictReader(stream))

    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    for axis, key, label in zip(axes[0], ("cross_entropy", "perplexity", "accuracy"),
                                ("Full-model cross-entropy ↓", "Full-model perplexity ↓", "Full-model accuracy ↑")):
        values = [metrics[phase][key] for phase in ("before", "after")]
        axis.bar(["Before", "After"], values, color=["#718096", "#3182ce"])
        axis.set_title(label)
        axis.set_ylim(0, max(values) * 1.2 or 1)
        for index, value in enumerate(values):
            axis.text(index, value, f"{value:.2%}" if key == "accuracy" else f"{value:.4f}", ha="center", va="bottom")
    if rows:
        tokens = np.array([float(row["tokens"]) for row in rows])
        for key, label in (("cross_entropy", "CE"), ("objective", "CE + λR")):
            axes[1, 0].plot(tokens, [float(row[key]) for row in rows], label=label, alpha=0.8)
        axes[1, 0].legend()
        axes[1, 0].set_title("Truncated training path (not full-model eval)")
        if regularizer == "both":
            for key, label in (("l2_penalty", "L2"), ("kl_penalty", "KL")):
                axes[1, 1].plot(tokens, [float(row[key]) for row in rows], label=label)
            axes[1, 1].legend()
        else:
            axes[1, 1].plot(tokens, [float(row["regularizer"]) for row in rows])
        axes[1, 1].set_title("Unweighted regularizers" if regularizer == "both" else "Unweighted regularizer")
        axes[1, 2].plot(tokens, [float(row["accuracy"]) for row in rows])
        axes[1, 2].set_title("Truncated-path training accuracy")
        for axis in axes[1]:
            axis.set_xlabel("Cumulative scored training tokens")
            axis.grid(alpha=0.2)
    else:
        for axis in axes[1]:
            axis.set_visible(False)
    first, last = summary["args"]["first_block"], summary["args"]["last_block"]
    figure.suptitle(f"Blocks {first}–{last} · {summary['args']['split']} · {metrics['before']['tokens']:,} held-out tokens")
    figure.tight_layout()
    figure.savefig(out / "results.png", dpi=160)

    lines = [f"# Segment fine-tuning: blocks {first}–{last}", "",
             "| Full-model metric | Before | After | Change (after − before) |",
             "|---|---:|---:|---:|"]
    for key in ("cross_entropy", "perplexity", "accuracy"):
        values = [metrics["before"][key], metrics["after"][key], metrics["delta_after_minus_before"][key]]
        formatted = [f"{value:.6f}" for value in values]
        lines.append(f"| {key} | {' | '.join(formatted)} |")
    lines += ["", "Accuracy is a fraction; changes are absolute, not relative.",
              "Lower CE/perplexity and higher accuracy indicate improvement.",
              "Training curves use the truncated path; the table uses the restored full model.", ""]
    if "partial_evaluation" in summary:
        partial = summary["partial_evaluation"]
        exit_block = summary["args"].get("partial_exit_block") or last
        head_label = "final norm/head" if summary["args"].get("train_head") else "original final norm/head"
        lines += [f"## Partial model: blocks 1–{exit_block} → {head_label}", "",
                  "Same held-out validation/evaluation tokens as the full model (not training averages).", "",
                  "| Partial-model metric | Before | After | Change (after − before) |",
                  "|---|---:|---:|---:|"]
        for key in ("cross_entropy", "perplexity", "accuracy"):
            values = [partial[phase][key] for phase in ("before", "after", "delta_after_minus_before")]
            lines.append(f"| {key} | {' | '.join(f'{value:.6f}' for value in values)} |")
        lines.append("")
    if summary["args"].get("train_head"):
        lines += ["Prediction head is trainable; final normalization remains frozen.", ""]
    if regularizer in ("l2", "both"):
        space = summary["args"].get("l2_space", "residual")
        lines += [f"L2 space: {space}; strength: {summary['args'].get('reg_strength', 1.0)}."]
        if space == "rms":
            lines += ["Parameter-free per-token RMS normalization (no learned gain/bias or mean subtraction).",
                      "Scales are the initial per-feature sample STDs in that same RMS-normalized space."]
        lines += ["L2 is a soft mean penalty, not a hard per-activation bound.", ""]
    if regularizer in ("kl", "both"):
        strength = summary["args"].get("kl_strength")
        strength = summary["args"].get("reg_strength", 1.0) if strength is None else strength
        lines += [f"KL strength: {strength}.", "KL is against the original untuned exit-lens distribution.", ""]
    if intervals:
        lines.append(f"Paired 95% document-bootstrap intervals ({intervals['documents']} documents):")
        for key in ("delta_cross_entropy", "delta_accuracy"):
            point, low, high = intervals[key]
            lines.append(f"- {key}: {point:+.6f} [{low:+.6f}, {high:+.6f}]")
    lines += ["", "Boundary protocol: continuous token stream; cross-document attention allowed within a window.",
              "No cross-batch-row or future-token attention. No EOS separators inserted."]
    text = "\n".join(lines) + "\n"
    (out / "results.md").write_text(text)
    print(text)
    print(f"Plot: {out / 'results.png'}")
    if show:
        plt.show()
    plt.close(figure)
    return intervals
