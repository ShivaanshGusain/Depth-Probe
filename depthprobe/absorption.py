"""Reproduction of the perturbation-absorption probe of Jing et al., arXiv 2606.15161 (2026).

Their recipe (Sec. 3): replace the output of layer l by h + delta with delta = eta * ||h||_F * xi, where xi
is a unit-Frobenius Gaussian over the whole layer-output tensor; propagate through the later layers; record
the relative error R_k = ||h~_k - h_k||_F / ||h_k||_F at every later layer and the absorption coefficient
r_l = R_L / R_l (r < 1 absorbs, r > 1 amplifies). They report eta = 1e-2 ("pruning scale": early layers
amplify, middle and late layers absorb) and eta = 1e-4 (every layer amplifies), in FP16.

Two choices in that recipe can produce the result without any absorption, so this reproduction can vary them:
  --dtype float16 vs float32: FP16 resolves about 5e-4 relative, so eta = 1e-4 is largely rounded away and
      rounding noise from the later layers can then dominate R_k.
  --scale frobenius vs token: whole-tensor scaling lets a few tokens with massive activations (attention
      sinks) set the noise size, so ordinary tokens receive far more than eta; per-token scaling gives every
      token eta times its own norm.
Not stated in the paper and therefore chosen here: the probe data (pass C4 or WikiText), the positions
(all positions of each window, as the Frobenius norm over the tensor suggests), the injection site (the
residual stream after block l), and the unit of the Frobenius norm (one window, i.e. one sequence).
Every ratio is taken against the in-batch clean replica, and R at the injection site is the realised value
after the cast to the model's dtype.
Layer indexing: state l is the output of block l (state 0 = embeddings), as HF's hidden_states[l] for l < L.
The paper does not say whether its "layers 1, 12, 24" use that numbering or 0-based block indices (our states
2, 13, 25); measure both if the exact sites matter.
"""

import hashlib
import time
import warnings

import numpy as np
import torch

from depthprobe.data import iterate_batches
from depthprobe.finetune import make_segment
from depthprobe.model import ResidualCapture, find_parts, n_blocks
from depthprobe.perturb import check_layer_types, sink_mask


def _xi(seed, start, length, width):
    digest = hashlib.sha256(f"{seed}:xi:{start}".encode()).digest()
    generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little") % (2 ** 63))
    return torch.randn(length, width, generator=generator)


def _starts(batch):
    starts = []
    for row in range(batch.inputs.shape[0]):
        selected = batch.rows == row
        first = np.argmin(batch.positions[selected])
        starts.append(int(batch.targets[selected][first] - batch.context[selected][first]))
    return starts


def perturbation(base, xi, eta, scale, keep):
    """base [B, T, d]; xi [B, T, d] Gaussian; keep [B, T] positions that may be perturbed."""
    xi = xi * keep[..., None]
    base = base.float() * keep[..., None]
    if scale == "frobenius":  # the paper: one unit-norm direction over the whole tensor of each sequence
        unit = xi / xi.flatten(1).norm(dim=-1).clamp_min(1e-30)[:, None, None]
        return eta * base.flatten(1).norm(dim=-1)[:, None, None] * unit
    if scale == "token":  # each token gets eta times its own norm
        unit = xi / xi.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        return eta * base.norm(dim=-1, keepdim=True) * unit
    raise ValueError(f"Unknown scale {scale}")


def relative_error(perturbed, reference, keep):
    """Per sequence: ||h~ - h||_F / ||h||_F and the cosine distance 1 - cos(vec h~, vec h), over kept positions."""
    perturbed, reference = perturbed.float() * keep[..., None], reference.float() * keep[..., None]
    difference = (perturbed - reference).flatten(1)
    reference, perturbed = reference.flatten(1), perturbed.flatten(1)
    error = difference.norm(dim=-1) / reference.norm(dim=-1).clamp_min(1e-30)
    cosine = (perturbed * reference).sum(-1) / (perturbed.norm(dim=-1) * reference.norm(dim=-1)).clamp_min(1e-30)
    return error, 1 - cosine


@torch.no_grad()
def run_absorption(model, corpus, windows, *, sites, etas, seeds, batch_size, bos_id, device, scale="frobenius",
                   exclude_sinks=False, sink_factor=10.0, replicas=4, log=print):
    """R_k and cosine distance per (site, eta, seed, state k, window), plus R after the final norm.

    HF's last hidden state (output_hidden_states[-1]) is taken AFTER the final norm, so if the paper used it,
    its R_L is `relative_error_final`; `relative_error[..., L, :]` is the last block's raw output. eta = 0
    adds nothing and measures the numerical floor of the comparison itself.
    """
    check_layer_types(model.config)
    blocks_count = n_blocks(model)
    states_count = blocks_count + 1
    parts = find_parts(model)
    segments = {site: make_segment(model, site + 1, blocks_count, trainable=False) for site in sites}
    conditions = [(eta, seed) for eta in etas for seed in range(seeds)]
    n_windows = len(windows)
    shape = (len(sites), len(etas), seeds, states_count, n_windows)
    error, cosine = np.full(shape, np.nan), np.full(shape, np.nan)
    final = np.full(shape[:3] + (n_windows,), np.nan)
    floor = np.zeros((len(sites), states_count))
    bos_offset = 0 if bos_id is None else 1
    window_index = 0
    began = time.time()
    for step, batch in enumerate(iterate_batches(corpus.ids, windows, batch_size, bos_id)):
        inputs = torch.from_numpy(batch.inputs).to(device)
        with ResidualCapture(model) as capture:
            parts.container(input_ids=inputs, use_cache=False)
        states = capture.states
        rows, length = inputs.shape
        keep = torch.ones(rows, length, device=device)
        if exclude_sinks:
            keep = (~sink_mask(states, bos_offset, sink_factor)).float()
        starts = _starts(batch)
        # One draw per (seed, window), reused for every site and eta (common random numbers, drawn once).
        xis = {seed: torch.stack([_xi(seed, start, length, states[0].shape[-1]) for start in starts]).to(device)
               for seed in range(seeds)}
        columns = slice(window_index, window_index + rows)
        for site_index, site in enumerate(sites):
            base = states[site]
            segment = segments[site]
            blocks = getattr(segment, parts.blocks_attr)
            for begin in range(0, len(conditions), replicas):
                group = conditions[begin:begin + replicas]
                copies = [base] + [(base.float() + perturbation(base, xis[seed], eta, scale, keep)).to(base.dtype)
                                   for eta, seed in group]
                with ResidualCapture(blocks=blocks) as seg_capture:
                    segment(inputs_embeds=torch.cat(copies), use_cache=False)
                outputs = [[state[g * rows:(g + 1) * rows] for state in seg_capture.states] for g in range(len(copies))]
                reference = outputs[0]
                reference_final = parts.norm(reference[-1])
                for offset, (clean_state, replica_state) in enumerate(zip(states[site:], reference)):
                    deviation, _ = relative_error(replica_state, clean_state, keep)
                    floor[site_index, site + offset] = np.fmax(floor[site_index, site + offset],
                                                               deviation.max().item())
                for g, (eta, seed) in enumerate(group, start=1):
                    e = etas.index(eta)
                    for offset, (noisy_state, reference_state) in enumerate(zip(outputs[g], reference)):
                        values, distance = relative_error(noisy_state, reference_state, keep)
                        error[site_index, e, seed, site + offset, columns] = values.cpu().numpy()
                        cosine[site_index, e, seed, site + offset, columns] = distance.cpu().numpy()
                    values, _ = relative_error(parts.norm(outputs[g][-1]), reference_final, keep)
                    final[site_index, e, seed, columns] = values.cpu().numpy()
        window_index += rows
        if step % 5 == 0:
            elapsed = time.time() - began
            log(f"absorption probe: {window_index}/{n_windows} windows, {elapsed / 60:.1f} min, "
                f"about {elapsed / window_index * (n_windows - window_index) / 60:.1f} min left")
    meta = {"sites": list(sites), "etas": list(etas), "seeds": seeds, "scale": scale, "exclude_sinks": exclude_sinks,
            "n_states": states_count, "windows": n_windows, "floor_max": float(floor.max())}
    return {"relative_error": error, "cosine_distance": cosine, "relative_error_final": final, "floor": floor}, meta


def _ratio(top, bottom, valid, picks, name):
    """Mean over windows of top / bottom where valid, its window-bootstrap interval, median and ratio of means."""
    if not valid.any():
        return {name: None, f"{name}_low": None, f"{name}_high": None, f"{name}_median": None,
                f"{name}_ratio_of_means": None}
    ratio = np.where(valid, top / np.where(valid, bottom, 1.0), np.nan)
    with warnings.catch_warnings():  # a resample can draw only invalid windows
        warnings.simplefilter("ignore", category=RuntimeWarning)
        boot = np.nanmean(ratio[picks], axis=-1)
    return {name: float(np.nanmean(ratio)), f"{name}_low": float(np.nanpercentile(boot, 2.5)),
            f"{name}_high": float(np.nanpercentile(boot, 97.5)), f"{name}_median": float(np.nanmedian(ratio)),
            f"{name}_ratio_of_means": float(top[valid].mean() / bottom[valid].mean())}


def _mean(values):
    finite = np.isfinite(values)
    return float(values[finite].mean()) if finite.any() else None


def summarize(arrays, meta, n_boot=1000, seed=0):
    """Absorption coefficient r_l = R_L / R_l (realised R at the site) and R_L / eta, with window-bootstrap
    intervals; per-window values are first averaged over noise seeds. Windows where R overflowed (inf/NaN, e.g.
    FP16) are counted, not averaged; windows where the perturbation rounded away entirely (R_l = 0) likewise."""
    rng = np.random.default_rng(seed)
    with warnings.catch_warnings():  # states before the site are all NaN by design
        warnings.simplefilter("ignore", category=RuntimeWarning)
        error = np.nanmean(arrays["relative_error"], axis=2)  # [sites, etas, states, windows]
        final = np.nanmean(arrays["relative_error_final"], axis=2) if "relative_error_final" in arrays else None
    last = meta["n_states"] - 1
    windows = error.shape[-1]
    picks = rng.integers(0, windows, size=(n_boot, windows))
    rows = []
    for i, site in enumerate(meta["sites"]):
        for e, eta in enumerate(meta["etas"]):
            at_site, at_last = error[i, e, site], error[i, e, last]
            finite = np.isfinite(at_site) & np.isfinite(at_last)
            valid = finite & (at_site > 0)
            realised = _mean(np.where(finite, at_site, np.nan))
            row = {"site": site, "eta": eta, "realised_R_at_site": realised,
                   # The main FP16 diagnostic: 1 when the injected perturbation is what was asked for.
                   "realised_over_eta": None if eta == 0 or realised is None else realised / eta,
                   "rounded_away_fraction": None if eta == 0 else float((finite & (at_site == 0)).mean()),
                   "nonfinite_fraction": float(1 - finite.mean()),
                   "R_last": _mean(np.where(finite, at_last, np.nan)),
                   **_ratio(at_last, at_site, valid, picks, "absorption")}
            row["R_last_over_eta"] = None if eta == 0 or row["R_last"] is None else row["R_last"] / eta
            if final is not None:
                at_final = final[i, e]
                row["R_final"] = _mean(np.where(finite, at_final, np.nan))
                row.update(_ratio(at_final, at_site, valid & np.isfinite(at_final), picks, "absorption_final"))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                row["curve"] = [None if not np.isfinite(v) else float(v)
                                for v in np.nanmean(np.where(np.isfinite(error[i, e, site:]), error[i, e, site:],
                                                             np.nan), axis=-1)]
            rows.append(row)
    return rows


def plot(out, rows, meta, label):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    last = meta["n_states"] - 1
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for e, eta in enumerate(meta["etas"]):
        selected = [r for r in rows if r["eta"] == eta]
        sites = [r["site"] for r in selected]
        measured = [r for r in selected if r["absorption"] is not None]
        if measured:
            values = np.array([r["absorption"] for r in measured])
            errors = np.clip(np.nan_to_num(np.array([[r["absorption"] - r["absorption_low"],
                                                      r["absorption_high"] - r["absorption"]] for r in measured]).T),
                             0, None)
            axes[0].errorbar([r["site"] for r in measured], values, yerr=errors, marker="o", capsize=2,
                             color=colors[e % len(colors)], label=f"eta = {eta:g}")
        if eta == 0:  # nothing injected: the floor, not a curve on a log axis
            continue
        chosen = sorted({sites[0], sites[len(sites) // 2], sites[-1]})
        for r in selected:
            if r["site"] in chosen:
                depth = np.arange(r["site"], last + 1)
                curve = [np.nan if v is None else v for v in r["curve"]]
                axes[1].plot(depth, curve, color=colors[e % len(colors)],
                             alpha=0.35 + 0.65 * (r["site"] == chosen[-1]))
    axes[0].axhline(1, color="grey", linewidth=0.6)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Injection layer l (state)")
    axes[0].set_ylabel("Absorption r_l = R_L / R_l (below 1: absorbed)")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Layer k")
    axes[1].set_ylabel("Relative error R_k (early, middle, late injection)")
    for axis in axes:
        axis.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    figure.suptitle(f"{label}: scale={meta['scale']}, sinks {'excluded' if meta['exclude_sinks'] else 'included'}",
                    fontsize=10)
    figure.tight_layout()
    figure.savefig(out / "absorption.png", dpi=150)
    plt.close(figure)
