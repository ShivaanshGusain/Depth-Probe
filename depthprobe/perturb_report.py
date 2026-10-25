"""E6 analysis: per-document sums -> means with document-bootstrap intervals, selectivity, depth trend,
control contrasts, the pre-registered gate and plots. Every statistic is computed per bootstrap replicate,
so derived quantities (differences, ratios, slopes) get honest intervals. Runs compared with each other
must share data, windows, seed, conditions and measured tokens; the same document weights are then used
for all of them (checked in the CLI).
"""

import csv
import json
from pathlib import Path

import numpy as np

STATE_KEYS = ("log_survival", "log_A", "log_A0", "retained", "log_off", "log_off0", "log_on", "log_on0")
SELECTIVITY_PAIRS = (("pca", "on", "off"), ("readout", "read", "unread"))
# Subspace metrics are summed only over tokens where they are defined; averaged over those tokens' counts.
SUBSPACE_KEYS = {"log_off": "off", "log_off0": "off", "log_on": "on", "log_on0": "on"}


def load_perturb_run(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    with np.load(path / "perturb.npz") as data:
        arrays = {key: data[key] for key in data.files}
    return manifest, arrays


class DocSums:
    """Bootstrap over documents from per-document sums (the token counts per document are shared)."""

    def __init__(self, counts, n_boot, seed, keep=None):
        # Runs that are compared must pass the same `keep` (document set) and seed: the weights then coincide.
        self.keep = counts > 0 if keep is None else keep
        self.counts = counts[self.keep].astype(np.float64)
        n_docs = int(self.keep.sum())
        rng = np.random.default_rng(seed)
        self.weights = rng.multinomial(n_docs, np.full(n_docs, 1 / n_docs), size=n_boot).astype(np.float64)

    def mean(self, sums):
        """sums [..., D] -> (point [...], replicates [..., B]) of the token-weighted mean."""
        selected = sums[..., self.keep]
        point = selected.sum(-1) / self.counts.sum()
        replicates = (selected @ self.weights.T) / (self.weights @ self.counts)
        return point, replicates

    def ratio(self, top, bottom):
        """sum(top) / sum(bottom) over documents, both [..., D]."""
        top, bottom = top[..., self.keep], bottom[..., self.keep]
        with np.errstate(divide="ignore", invalid="ignore"):
            return top.sum(-1) / bottom.sum(-1), (top @ self.weights.T) / (bottom @ self.weights.T)


def interval(point, replicates):
    point = float(point)
    if not np.isfinite(point):
        return {"value": point, "low": float("nan"), "high": float("nan")}
    with np.errstate(invalid="ignore"):
        low, high = np.nanpercentile(replicates, [2.5, 97.5], axis=-1)
    return {"value": point, "low": float(low), "high": float(high)}


def _condition_index(manifest):
    return {(c["site"], c["family"], c["delta"], c["sign"]): i for i, c in enumerate(manifest["perturb"]["conditions"])}


def _pair(index, site, family, delta):
    return index[(site, family, delta, 1)], index[(site, family, delta, -1)]


def summarize_run(manifest, arrays, boot):
    """Statistics per (family, delta, site) at the last state, plus curves over later states."""
    meta = manifest["perturb"]
    index = _condition_index(manifest)
    last = meta["n_states"] - 1
    sites, families, deltas = meta["sites"], meta["families"], meta["deltas"]
    cells, store = [], {}

    def keep(family, delta, site, name, point, replicates, cell):
        store[(family, delta, site, name)] = (point, replicates)
        cell[name] = interval(point, replicates)

    for family in families:
        for delta in deltas:
            for site_index, site in enumerate(sites):
                plus, minus = _pair(index, site, family, delta)
                cell = {"family": family, "delta": delta, "site": site}
                for key in STATE_KEYS:  # mean over the antithetic pair
                    summed = arrays[f"state_{key}"][plus, last] + arrays[f"state_{key}"][minus, last]
                    defined = f"defined_{SUBSPACE_KEYS.get(key)}"
                    if key in SUBSPACE_KEYS and defined in arrays:
                        point, replicates = boot.ratio(summed, arrays[defined][plus, last]
                                                       + arrays[defined][minus, last])
                    else:
                        point, replicates = boot.mean(summed / 2)
                    keep(family, delta, site, key, point, replicates, cell)
                for name in ("off", "on"):
                    if f"defined_{name}" in arrays:
                        point, replicates = boot.mean((arrays[f"defined_{name}"][plus, last]
                                                       + arrays[f"defined_{name}"][minus, last]) / 2)
                        keep(family, delta, site, f"defined_fraction_{name}", point, replicates, cell)
                for net, raw, null in (("log_Lambda", "log_A", "log_A0"), ("log_off_net", "log_off", "log_off0"),
                                       ("log_on_net", "log_on", "log_on0")):
                    a, b = store[(family, delta, site, raw)], store[(family, delta, site, null)]
                    keep(family, delta, site, net, a[0] - b[0], a[1] - b[1], cell)
                summed = {key: arrays[f"output_{key}"][plus] + arrays[f"output_{key}"][minus]
                          for key in ("kl", "kl_carry", "kl_last", "dnll", "flip", "nl", "delta_eff")
                          if f"output_{key}" in arrays}  # delta_eff is absent from runs made before it existed
                point, replicates = boot.ratio(summed["kl"], summed["kl_carry"])
                keep(family, delta, site, "repaired_damage_ratio", point, replicates, cell)
                with np.errstate(divide="ignore", invalid="ignore"):
                    store[(family, delta, site, "log_repaired")] = (np.log(point), np.log(replicates))
                point, replicates = boot.ratio(summed["kl"], summed["kl_last"])
                keep(family, delta, site, "damage_vs_last_state", point, replicates, cell)
                for key in ("kl", "dnll", "flip", "nl", "delta_eff"):
                    if key not in summed:
                        continue
                    point, replicates = boot.mean(summed[key] / 2)
                    keep(family, delta, site, key, point, replicates, cell)
                for name, key in (("signal_growth", "clean_signal"), ("content_growth", "clean_content")):
                    point, replicates = boot.mean(arrays[key][site_index, last])
                    keep(family, delta, site, name, point, replicates, cell)
                cells.append(cell)
    curves = {}
    for family in families:
        for delta in deltas:
            for site in sites:
                plus, minus = _pair(index, site, family, delta)
                points = []
                for state in range(site, last + 1):
                    point = {"state": state}
                    for key in ("log_survival", "log_A", "log_A0"):
                        value, _ = boot.mean((arrays[f"state_{key}"][plus, state]
                                              + arrays[f"state_{key}"][minus, state]) / 2)
                        point[key] = float(value)
                    points.append(point)
                curves[f"{family}|{delta}|{site}"] = points
    return {"cells": cells, "curves": curves, "derived": _derived(store, families, deltas, sites)}, store


def _derived(store, families, deltas, sites):
    """Selectivity (content kept minus noise kept) and depth trends, per delta and site."""
    out = {"selectivity": [], "depth_trend": []}
    for delta in deltas:
        for name, content, noise in SELECTIVITY_PAIRS:
            if content not in families or noise not in families:
                continue
            for site in sites:
                for key in ("log_A", "log_Lambda", "log_survival", "log_repaired"):
                    c, x = store[(content, delta, site, key)], store[(noise, delta, site, key)]
                    out["selectivity"].append({"definition": name, "delta": delta, "site": site, "metric": key,
                                               **interval(c[0] - x[0], c[1] - x[1])})
        if len(sites) < 3:
            continue
        x = np.array(sites, dtype=np.float64)
        centred = x - x.mean()
        for family in families:
            for key in ("log_A", "log_survival", "log_Lambda", "log_off_net"):
                y_point = np.array([store[(family, delta, s, key)][0] for s in sites])
                if not np.all(np.isfinite(y_point)):
                    continue  # e.g. off-manifold survival of an on-manifold injection is undefined
                y_boot = np.stack([store[(family, delta, s, key)][1] for s in sites])  # [sites, B]
                slope = (centred @ (y_point - y_point.mean())) / (centred @ centred)
                slopes = centred @ (y_boot - y_boot.mean(0)) / (centred @ centred)
                # Positive slope: later injection leaves more of the perturbation, i.e. earlier is absorbed more.
                out["depth_trend"].append({"family": family, "delta": delta, "metric": key,
                                           **interval(slope, slopes)})
    return out


def contrasts(runs, reference_label, metric="log_A"):
    """Trained-model minus control differences of selectivity and of iso absorption, shared document weights."""
    by_label = {run["label"]: run for run in runs}
    reference = by_label[reference_label]
    meta = reference["manifest"]["perturb"]
    out = []
    for run in runs:
        if run["label"] == reference_label:
            continue
        for delta in meta["deltas"]:
            for site in meta["sites"]:
                for name, content, noise in SELECTIVITY_PAIRS:
                    if content not in meta["families"] or noise not in meta["families"]:
                        continue
                    ref_c, ref_x = reference["store"][(content, delta, site, metric)], reference["store"][(noise, delta, site, metric)]
                    ctl_c, ctl_x = run["store"][(content, delta, site, metric)], run["store"][(noise, delta, site, metric)]
                    point = (ref_c[0] - ref_x[0]) - (ctl_c[0] - ctl_x[0])
                    replicates = (ref_c[1] - ref_x[1]) - (ctl_c[1] - ctl_x[1])
                    out.append({"control": run["label"], "statistic": f"selectivity_{name}", "delta": delta,
                                "site": site, **interval(point, replicates)})
                if "iso" in meta["families"]:
                    ref, ctl = reference["store"][("iso", delta, site, metric)], run["store"][("iso", delta, site, metric)]
                    out.append({"control": run["label"], "statistic": f"{metric}_iso", "delta": delta, "site": site,
                                **interval(ref[0] - ctl[0], ref[1] - ctl[1])})
    return out


def gate(contrast_rows, deciding, delta, margin, min_sites, rule):
    """The v1 decision in its pre-registered form. A control passes a definition when trained-minus-control
    selectivity exceeds `margin` with the interval's lower end above zero at >= min_sites sites. The overall
    gate passes only if every deciding control is present and passes under `rule` ('both' definitions or
    'either')."""
    if rule not in ("both", "either"):
        raise ValueError(f"Unknown gate rule {rule!r}; use 'both' or 'either'")
    per_control = {}
    for row in contrast_rows:
        if row["delta"] != delta or not row["statistic"].startswith("selectivity_"):
            continue
        definition = row["statistic"].removeprefix("selectivity_")
        entry = per_control.setdefault(row["control"], {}).setdefault(definition, {"sites_passing": 0, "sites": 0})
        entry["sites"] += 1
        entry["sites_passing"] += int(row["value"] > margin and row["low"] > 0)
    reasons = []
    for control, definitions in per_control.items():
        for entry in definitions.values():
            entry["passes"] = entry["sites_passing"] >= min_sites
        verdicts = [entry["passes"] for entry in definitions.values()]
        needed = len(SELECTIVITY_PAIRS) if rule == "both" else 1
        definitions["passes"] = (len(verdicts) >= needed) and (all(verdicts) if rule == "both" else any(verdicts))
        definitions["deciding"] = control in deciding
    missing = [control for control in deciding if control not in per_control]
    if missing:
        reasons.append(f"deciding controls without contrasts: {missing}")
    if not deciding:
        reasons.append("no deciding controls")
    failed = [control for control in deciding if control in per_control and not per_control[control]["passes"]]
    if failed:
        reasons.append(f"deciding controls not beaten: {failed}")
    return {"passes": not reasons, "reasons": reasons, "rule": rule, "delta": delta, "margin": margin,
            "min_sites": min_sites, "deciding": list(deciding), "per_control": per_control}


def write_tables(out, runs, contrast_rows):
    with open(out / "cells.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "family", "delta", "site", "statistic", "value", "low", "high"])
        for run in runs:
            for cell in run["summary"]["cells"]:
                for key, value in cell.items():
                    if isinstance(value, dict):
                        writer.writerow([run["label"], cell["family"], cell["delta"], cell["site"], key,
                                         value["value"], value["low"], value["high"]])
    with open(out / "contrasts.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["control", "statistic", "delta", "site", "value", "low", "high"])
        for row in contrast_rows:
            writer.writerow([row["control"], row["statistic"], row["delta"], row["site"], row["value"], row["low"],
                             row["high"]])


def _error_bars(rows):
    values = np.array([r["value"] for r in rows])
    errors = np.array([[r["value"] - r["low"], r["high"] - r["value"]] for r in rows]).T
    return values, np.clip(np.nan_to_num(errors), 0, None)  # a point can fall outside its percentile interval


def plot(out, runs, delta):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for run in runs:  # absorption curves: raw survival and normalized absorption from every site, iso noise
        meta = run["manifest"]["perturb"]
        if "iso" not in meta["families"]:
            continue
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        last = meta["n_states"] - 1
        for i, site in enumerate(meta["sites"]):
            points = run["summary"]["curves"][f"iso|{delta}|{site}"]
            depth = [p["state"] / last for p in points]
            color = colors[i % len(colors)]
            axes[0].plot(depth, [p["log_survival"] for p in points], color=color, linewidth=1.2)
            axes[1].plot(depth, [p["log_A"] for p in points], color=color, linewidth=1.2)
            axes[1].plot(depth, [p["log_A0"] for p in points], color=color, linewidth=0.8, linestyle=":")
        axes[0].set_title("Raw survival log ||D_k|| / ||D_l|| (noise removed)", fontsize=9)
        axes[1].set_title("Normalized log A (solid) vs dilution null log A0 (dotted)", fontsize=9)
        for axis in axes:
            axis.axhline(0, color="grey", linewidth=0.6)
            axis.set_xlabel("Depth of measured state (state / blocks)")
            axis.grid(alpha=0.3)
        figure.suptitle(f"{run['label']}: iso noise, delta = {delta}, one line per injection site", fontsize=10)
        figure.tight_layout()
        figure.savefig(out / f"absorption_{run['label']}.png", dpi=150)
        plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, run in enumerate(runs):
        rows = [r for r in run["summary"]["derived"]["selectivity"] if r["delta"] == delta and r["metric"] == "log_A"]
        for axis, definition in zip(axes, ("pca", "readout")):
            selected = [r for r in rows if r["definition"] == definition]
            if not selected:
                continue
            values, errors = _error_bars(selected)
            axis.errorbar([r["site"] for r in selected], values, yerr=errors, marker="o", capsize=2,
                          color=colors[i % len(colors)], label=run["label"])
    for axis, title in zip(axes, ("Selectivity, PCA definition: log A(on) - log A(off)",
                                  "Selectivity, readout definition: log A(read) - log A(unread)")):
        axis.axhline(0, color="grey", linewidth=0.6)
        axis.set_title(title, fontsize=9)
        axis.set_xlabel("Injection site (state)")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(out / "selectivity.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4.2))
    for i, run in enumerate(runs):
        cells = [c for c in run["summary"]["cells"] if c["family"] == "iso" and c["delta"] == delta]
        if not cells:
            continue
        sites = [c["site"] for c in cells]
        axis.plot(sites, [c["log_survival"]["value"] for c in cells], marker="o", color=colors[i % len(colors)],
                  label=f"{run['label']}: noise removed (raw survival)")
        axis.plot(sites, [-c["content_growth"]["value"] for c in cells], marker="x", linestyle="--",
                  color=colors[i % len(colors)], label=f"{run['label']}: minus content growth")
    axis.axhline(0, color="grey", linewidth=0.6)
    axis.set_xlabel("Injection site (state)")
    axis.set_ylabel("log ratio, last state vs injection state")
    axis.set_title(f"Noise removed vs signal added, iso, delta = {delta}", fontsize=9)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(out / "noise_vs_signal.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4.2))
    for i, run in enumerate(runs):
        meta = run["manifest"]["perturb"]
        if "iso" not in meta["families"]:
            continue
        chosen = sorted({meta["sites"][0], meta["sites"][len(meta["sites"]) // 2], meta["sites"][-1]})
        for site in chosen:
            cells = [c for c in run["summary"]["cells"] if c["family"] == "iso" and c["site"] == site]
            axis.plot([c["delta"] for c in cells], [c["repaired_damage_ratio"]["value"] for c in cells], marker="o",
                      color=colors[i % len(colors)], alpha=0.4 + 0.6 * (site == chosen[-1]),
                      label=f"{run['label']}, site {site}")
    axis.set_xscale("log")
    axis.axhline(1, color="grey", linewidth=0.6)
    axis.set_xlabel("Noise size delta (relative to the token's bulk norm)")
    axis.set_ylabel("Output KL / KL if the perturbation were carried unchanged")
    axis.set_title("Dose response, iso noise (below 1: later layers repaired the output)", fontsize=9)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(out / "dose_response.png", dpi=150)
    plt.close(figure)
