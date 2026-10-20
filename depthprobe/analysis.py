"""Curves with document-level bootstrap CIs (E4), per-token monotonicity (E3), gain profile (E9).

Confidence intervals resample documents, not tokens: tokens within a document are correlated.
Every mean is a ratio of sums over resampled documents (token-weighted).
"""

import csv
import json
from pathlib import Path

import numpy as np

LENS_METRICS = ("nll", "kl", "top1", "agree", "entropy")
HIDDEN_METRICS = ("cos", "rel_dist", "norm")


def load_run(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    with np.load(path / "metrics.npz") as data:
        arrays = {key: data[key] for key in data.files}
    return manifest, arrays


class DocBootstrap:
    def __init__(self, doc, n_boot, seed):
        _, self.inverse = np.unique(doc, return_inverse=True)
        self.n_docs = int(self.inverse.max()) + 1
        rng = np.random.default_rng(seed)
        self.weights = rng.multinomial(self.n_docs, np.full(self.n_docs, 1 / self.n_docs), size=n_boot).astype(np.float64)

    def _doc_sums(self, values):
        values = np.atleast_2d(values).astype(np.float64)
        return np.stack([np.bincount(self.inverse, weights=row, minlength=self.n_docs) for row in values], axis=1)

    def ratio(self, numerator, denominator):
        """sum(numerator) / sum(denominator) with a 95% percentile interval; [S, N] -> [S], [N] -> scalar."""
        top, bottom = self._doc_sums(numerator), self._doc_sums(denominator)
        point = top.sum(0) / bottom.sum(0)
        boot = (self.weights @ top) / (self.weights @ bottom)
        low, high = np.nanpercentile(boot, [2.5, 97.5], axis=0)
        if np.ndim(numerator) == 1:
            return float(point[0]), float(low[0]), float(high[0])
        return point, low, high

    def mean(self, values):
        values = np.asarray(values, dtype=np.float64)
        return self.ratio(values, np.ones_like(values))


def per_token_e3(arrays, lens, from_layer, eps):
    """Per-token monotonicity over states from_layer..L for one lens ('raw' or 'tuned')."""
    nll = arrays[f"{lens}_nll"][from_layer:].astype(np.float64)
    kl = arrays[f"{lens}_kl"][from_layer:].astype(np.float64)
    top1 = arrays[f"{lens}_top1"][from_layer:]

    def reversals(values):  # quantities that should fall with depth
        increase = np.diff(values, axis=0)
        bad = increase > eps
        return bad.sum(0), np.where(bad, increase, 0.0).sum(0)

    nll_count, nll_size = reversals(nll)
    kl_count, kl_size = reversals(kl)
    # settled_from[l]: the prediction is correct at l and at every later state.
    settled_from = np.flip(np.logical_and.accumulate(np.flip(top1, 0), axis=0), 0)
    settled = settled_from.any(0)
    return {
        "monotone_nll": nll_count == 0,
        "nll_reversals": nll_count,
        "nll_reversal_size": nll_size,
        "monotone_kl": kl_count == 0,
        "kl_reversals": kl_count,
        "kl_reversal_size": kl_size,
        "correct_then_wrong": top1[:-1].any(0) & ~top1[-1],
        "final_correct": top1[-1],
    }, np.where(settled, settled_from.argmax(0) + from_layer, 0), settled


def summarize(run_dir, *, from_layer, min_context, eps, n_boot, seed):
    manifest, arrays = load_run(run_dir)
    keep = arrays["context"] >= min_context
    arrays = {key: value[..., keep] for key, value in arrays.items()}
    boot = DocBootstrap(arrays["doc"], n_boot, seed)
    n_states = arrays["raw_nll"].shape[0]
    lenses = [lens for lens in ("raw", "tuned") if f"{lens}_nll" in arrays]
    curves, e3 = {}, {}
    for lens in lenses:
        for metric in LENS_METRICS:
            curves[f"{lens}_{metric}"] = boot.mean(arrays[f"{lens}_{metric}"])
        point, low, high = curves[f"{lens}_nll"]
        curves[f"{lens}_ppl"] = (np.exp(point), np.exp(low), np.exp(high))
        for metric in ("nll", "kl"):  # E9: improvement contributed by each block
            values = arrays[f"{lens}_{metric}"].astype(np.float64)
            curves[f"{lens}_gain_{metric}"] = boot.mean(values[:-1] - values[1:])
        flags, depth, settled = per_token_e3(arrays, lens, from_layer, eps)
        e3[lens] = {key: boot.mean(value) for key, value in flags.items()}
        e3[lens]["prediction_depth"] = boot.ratio(depth * settled, settled)
        e3[lens]["settled_fraction"] = boot.mean(settled)
    for metric in HIDDEN_METRICS:
        curves[metric] = boot.mean(arrays[metric])
    baselines = {"native_nll": boot.mean(arrays["native_nll"])}
    for key in ("unigram_nll", "unigram_kl"):
        if key in arrays:
            baselines[key] = boot.mean(arrays[key])
    return {
        "run": str(Path(run_dir).resolve()),
        "label": manifest["args"].get("label") or Path(run_dir).name,
        "variant": manifest.get("variant"),
        "n_states": n_states,
        "tokens": int(keep.sum()),
        "documents": boot.n_docs,
        "settings": {"from_layer": from_layer, "min_context": min_context, "eps": eps, "bootstrap": n_boot, "seed": seed},
        "curves": {key: [np.asarray(x).tolist() for x in value] for key, value in curves.items()},
        "e3": e3,
        "baselines": baselines,
    }


def write_csv(path, summaries):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "curve", "state", "depth", "value", "ci_low", "ci_high"])
        for summary in summaries:
            for curve, (point, low, high) in summary["curves"].items():
                steps = len(point)
                for state in range(steps):
                    # Gain curves have one entry per block (state l-1 -> l), indexed by the block.
                    index = state + 1 if "gain" in curve else state
                    writer.writerow([summary["label"], curve, index, index / (summary["n_states"] - 1),
                                     point[state], low[state], high[state]])


def plot(out_dir, summaries):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("nll", "NLL of true next token (nats)", "linear"),
        ("ppl", "Perplexity", "log"),
        ("kl", "KL(p_final || p_layer) (nats)", "symlog"),
        ("top1", "Top-1 accuracy (true token)", "linear"),
        ("agree", "Top-1 agreement with final layer", "linear"),
        ("gain_nll", "NLL improvement from this block", "linear"),
        ("gain_kl", "KL improvement from this block", "symlog"),
        ("cos", "cos(h_l, h_L)", "linear"),
        ("rel_dist", "||h_L - h_l|| / ||h_L||", "linear"),
        ("norm", "||h_l||", "log"),
    ]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for name, title, scale in panels:
        figure, axis = plt.subplots(figsize=(7, 4.2))
        for index, summary in enumerate(summaries):
            color = colors[index % len(colors)]
            last = summary["n_states"] - 1
            styles = [("tuned", "-"), ("raw", "--")] if name not in HIDDEN_METRICS else [(None, "-")]
            for lens, style in styles:
                key = name if lens is None else f"{lens}_{name}"
                if key not in summary["curves"]:
                    continue
                point, low, high = (np.asarray(x) for x in summary["curves"][key])
                first = 1 if "gain" in name else 0
                depth = np.arange(first, first + len(point)) / last
                label = summary["label"] if lens is None else f"{summary['label']} ({lens})"
                axis.plot(depth, point, style, color=color, label=label, linewidth=1.4)
                axis.fill_between(depth, low, high, color=color, alpha=0.15, linewidth=0)
        if scale == "symlog":
            axis.set_yscale("symlog", linthresh=1e-2)
        elif scale == "log":
            axis.set_yscale("log")
        axis.set_xlabel("Depth (state index / number of blocks)")
        axis.set_ylabel(title)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(Path(out_dir) / f"{name}.png", dpi=150)
        plt.close(figure)

    keys = ["monotone_nll", "monotone_kl", "correct_then_wrong"]
    figure, axis = plt.subplots(figsize=(7, 4.2))
    bars = [(summary["label"], lens, summary["e3"][lens]) for summary in summaries for lens in summary["e3"]]
    width = 0.8 / max(1, len(bars))
    for index, (label, lens, stats) in enumerate(bars):
        points = np.array([stats[key][0] for key in keys])
        errors = np.array([[stats[key][0] - stats[key][1], stats[key][2] - stats[key][0]] for key in keys]).T
        axis.bar(np.arange(len(keys)) + index * width, points, width, yerr=errors, capsize=2, label=f"{label} ({lens})")
    axis.set_xticks(np.arange(len(keys)) + width * (len(bars) - 1) / 2, keys)
    axis.set_ylabel("Fraction of tokens")
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(Path(out_dir) / "e3_monotonicity.png", dpi=150)
    plt.close(figure)
