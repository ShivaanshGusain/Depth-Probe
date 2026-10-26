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

Beyond the reproduction (see `summarize_tests`): absolute sizes of the noise and of the clean state, real
perturbations from token swaps, and output KL, which separate dilution, convergence and denoising.
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


KINDS = ("noise", "real")
# Token sets every perturbation is measured on. all: every kept token. swapped: kept tokens after the
# counterfactual swap that it moves; changed / preserved: those whose top-1 prediction the swap does / does not
# change. Noise is measured on the same sets as the real perturbation, so each comparison uses one population.
MASKS = ("all", "swapped", "changed", "preserved")
# A swap defines a real direction at a token only if it moves that token's state by at least this fraction.
MIN_REAL = 1e-5


def _generator(seed, key):
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest[:8], "little") % (2 ** 63))


def _xi(seed, start, length, width):
    return torch.randn(length, width, generator=_generator(seed, f"xi:{start}"))


def swap(seed, start, inputs_row, corpus_ids, bos_offset, reach=64):
    """The counterfactual for one window: a position among its first `reach` tokens (never the first token or
    BOS, which act as sinks) and a replacement drawn from the corpus, keyed like the noise so that every model
    variant sees the same swap."""
    generator = _generator(seed, f"swap:{start}")
    span = max(1, min(reach, len(inputs_row) - bos_offset - 2))
    position = bos_offset + 1 + int(torch.randint(0, span, (1,), generator=generator))
    for _ in range(100):
        token = int(corpus_ids[int(torch.randint(0, len(corpus_ids), (1,), generator=generator))])
        if token != int(inputs_row[position]):
            return position, token
    raise ValueError("Could not draw a replacement token that differs from the original")


def _token_sq(hidden):
    return hidden.double().square().sum(-1)  # [B, T]


def _masked_norm(token_sq, mask):
    return (token_sq * mask).sum(-1).sqrt()


def _masked_spread(hidden, token_sq, mask):
    """Frobenius norm of the masked tokens around their own mean: size measured against the spread between
    tokens, so a common direction shared by all tokens does not count."""
    total = (hidden.double() * mask[..., None]).sum(1)
    count = mask.sum(-1).clamp_min(1)
    return ((token_sq * mask).sum(-1) - total.square().sum(-1) / count).clamp_min(0).sqrt()


def _readout_chunks(readout, hidden, chunk):
    flat = hidden.reshape(-1, hidden.shape[-1])
    return [readout(flat[i:i + chunk]) for i in range(0, len(flat), chunk)]


def _top1(readout, hidden, chunk):
    return torch.cat([logits.argmax(-1) for logits in _readout_chunks(readout, hidden, chunk)]).view(hidden.shape[:2])


def _kl(logp_reference, readout, hidden, chunk):
    """Per token KL(reference || perturbed) of the next-token distribution."""
    flat = hidden.reshape(-1, hidden.shape[-1])
    values = []
    for j, i in enumerate(range(0, len(flat), chunk)):
        logp = torch.log_softmax(readout(flat[i:i + chunk]), dim=-1)
        values.append((logp_reference[j].exp() * (logp_reference[j] - logp)).sum(-1))
    return torch.cat(values).view(hidden.shape[:2])


def _sinks(states, starts, bos_offset, sink_factor, fixed_sinks):
    if fixed_sinks is None:
        return sink_mask(states, bos_offset, sink_factor)
    sinks = torch.zeros(states[0].shape[:2], dtype=torch.bool, device=states[0].device)
    for row, start in enumerate(starts):
        if start not in fixed_sinks:
            raise ValueError(f"Window starting at {start} is not in the reference sink masks; runs must use the "
                             "same data, context, stride and --every")
        sinks[row, sorted(fixed_sinks[start])] = True
    sinks[:, : 1 + bos_offset] = True
    return sinks


@torch.no_grad()
def run_absorption(model, corpus, windows, *, sites, etas, seeds, batch_size, bos_id, device, scale="frobenius",
                   exclude_sinks=False, sink_factor=10.0, replicas=4, readout=None, swaps=False, fixed_sinks=None,
                   chunk=256, log=print):
    """R_k and cosine distance per (site, eta, seed, state k, window), plus R after the final norm.

    HF's last hidden state (output_hidden_states[-1]) is taken AFTER the final norm, so if the paper used it,
    its R_L is `relative_error_final`; `relative_error[..., L, :]` is the last block's raw output. eta = 0
    adds nothing and measures the numerical floor of the comparison itself.

    Also recorded, to separate dilution, convergence and denoising (see `summarize_tests`):
      delta_norm [kind, mask, site, eta, seed, state, window]  ||h~_k - h_k||_F over the tokens of the mask
      clean_norm, clean_spread [site, mask, seed, state, window]  ||h_k||_F and the spread around the mean
      counts [site, mask, seed, window]; kl [kind, mask, site, eta, seed, window] (mean over the mask; readout)
    swaps=True adds the real perturbation: for noise seed s, one earlier token of each window is swapped (seed s),
    and the difference it makes at state l, scaled exactly like the noise, is injected at l with the original
    context. Comparing its survival with the noise's on the same tokens asks whether the later layers remove
    random perturbations more than real ones. fixed_sinks reuses a reference run's sink positions.
    """
    check_layer_types(model.config)
    if swaps and readout is None:
        raise ValueError("swaps need the readout, to split tokens by whether the swap changes the prediction")
    if swaps and 0 in sites:
        raise ValueError("State 0 holds token embeddings only: a swap moves no other token there, so real "
                         "perturbations need injection sites >= 1")
    blocks_count = n_blocks(model)
    states_count = blocks_count + 1
    last = blocks_count
    parts = find_parts(model)
    segments = {site: make_segment(model, site + 1, blocks_count, trainable=False) for site in sites}
    conditions = [("noise", eta, seed) for eta in etas for seed in range(seeds)]
    if swaps:
        conditions += [("real", eta, seed) for eta in etas if eta > 0 for seed in range(seeds)]
    n_windows = len(windows)
    shape = (len(sites), len(etas), seeds, states_count, n_windows)
    error, cosine = np.full(shape, np.nan), np.full(shape, np.nan)
    final = np.full(shape[:3] + (n_windows,), np.nan)
    delta_norm = np.full((len(KINDS), len(MASKS)) + shape, np.nan)
    clean_norm = np.full((len(sites), len(MASKS), seeds, states_count, n_windows), np.nan)
    clean_spread = np.full_like(clean_norm, np.nan)
    counts = np.zeros((len(sites), len(MASKS), seeds, n_windows))
    kl = np.full((len(KINDS), len(MASKS)) + shape[:3] + (n_windows,), np.nan)
    kl_carry = np.full_like(kl, np.nan)
    swap_positions = np.full((seeds, n_windows), -1, dtype=np.int64)
    swap_tokens = np.full((seeds, n_windows), -1, dtype=np.int64)
    floor = np.zeros((len(sites), states_count))
    bos_offset = 0 if bos_id is None else 1
    window_record, sink_record_start, sink_record_position = [], [], []
    window_index = 0
    began = time.time()
    for step, batch in enumerate(iterate_batches(corpus.ids, windows, batch_size, bos_id)):
        inputs = torch.from_numpy(batch.inputs).to(device)
        with ResidualCapture(model) as capture:
            parts.container(input_ids=inputs, use_cache=False)
        states = capture.states
        rows, length = inputs.shape
        starts = _starts(batch)
        sinks = _sinks(states, starts, bos_offset, sink_factor, fixed_sinks)
        for row, start in enumerate(starts):
            window_record.append(start)
            for position in torch.nonzero(sinks[row], as_tuple=False)[:, 0].tolist():
                sink_record_start.append(start)
                sink_record_position.append(position)
        keep = (~sinks).float() if exclude_sinks else torch.ones(rows, length, device=device)
        columns = slice(window_index, window_index + rows)
        token_sq = [_token_sq(state) for state in states]
        # One draw per (seed, window), reused for every site and eta (common random numbers, drawn once).
        xis = {seed: torch.stack([_xi(seed, start, length, states[0].shape[-1]) for start in starts]).to(device)
               for seed in range(seeds)}
        # Per seed: the swap's real directions at each site and the token sets it defines.
        directions, masks = {}, {}
        for seed in range(seeds):
            masks[seed] = {site: {"all": keep} for site in sites}
            if not swaps:
                continue
            swapped_inputs = inputs.clone()
            after = torch.zeros(rows, length, device=device)
            for row, start in enumerate(starts):
                position, token = swap(seed, start, batch.inputs[row], corpus.ids, bos_offset)
                swapped_inputs[row, position] = token
                after[row, position + 1:] = 1
                swap_positions[seed, window_index + row], swap_tokens[seed, window_index + row] = position, token
            with ResidualCapture(model) as swapped_capture:
                parts.container(input_ids=swapped_inputs, use_cache=False)
            swapped = swapped_capture.states
            changed = (_top1(readout, states[last], chunk) != _top1(readout, swapped[last], chunk)).float()
            directions[seed] = {}
            for site in sites:
                difference = swapped[site].float() - states[site].float()
                moved = (difference.norm(dim=-1) > MIN_REAL * states[site].float().norm(dim=-1)).float()
                selected = keep * after * moved
                directions[seed][site] = difference
                masks[seed][site].update(swapped=selected, changed=selected * changed,
                                         preserved=selected * (1 - changed))
            del swapped, swapped_capture
        for site_index, site in enumerate(sites):
            for seed in range(seeds):
                for m, name in enumerate(MASKS):
                    mask = masks[seed][site].get(name)
                    if mask is None:
                        continue
                    counts[site_index, m, seed, columns] = mask.sum(-1).cpu().numpy()
                    for k in range(site, states_count):
                        clean_norm[site_index, m, seed, k, columns] = _masked_norm(token_sq[k], mask).cpu().numpy()
                        clean_spread[site_index, m, seed, k, columns] = \
                            _masked_spread(states[k], token_sq[k], mask).cpu().numpy()
            base = states[site]
            segment = segments[site]
            blocks = getattr(segment, parts.blocks_attr)
            for begin in range(0, len(conditions), replicas):
                group = conditions[begin:begin + replicas]
                copies = [base]
                for kind, eta, seed in group:
                    if kind == "noise":
                        injected = perturbation(base, xis[seed], eta, scale, keep)
                    else:
                        injected = perturbation(base, directions[seed][site], eta, scale,
                                                masks[seed][site]["swapped"])
                    copies.append((base.float() + injected).to(base.dtype))
                with ResidualCapture(blocks=blocks) as seg_capture:
                    segment(inputs_embeds=torch.cat(copies), use_cache=False)
                outputs = [[state[g * rows:(g + 1) * rows] for state in seg_capture.states] for g in range(len(copies))]
                reference = outputs[0]
                reference_final = parts.norm(reference[-1])
                for offset, (clean_state, replica_state) in enumerate(zip(states[site:], reference)):
                    deviation, _ = relative_error(replica_state, clean_state, keep)
                    floor[site_index, site + offset] = np.fmax(floor[site_index, site + offset],
                                                               deviation.max().item())
                logp_reference = None
                if readout is not None:
                    logp_reference = [torch.log_softmax(logits, dim=-1)
                                      for logits in _readout_chunks(readout, reference[-1], chunk)]
                for g, (kind, eta, seed) in enumerate(group, start=1):
                    e, c = etas.index(eta), KINDS.index(kind)
                    if kind == "noise":
                        for offset, (noisy_state, reference_state) in enumerate(zip(outputs[g], reference)):
                            values, distance = relative_error(noisy_state, reference_state, keep)
                            error[site_index, e, seed, site + offset, columns] = values.cpu().numpy()
                            cosine[site_index, e, seed, site + offset, columns] = distance.cpu().numpy()
                        values, _ = relative_error(parts.norm(outputs[g][-1]), reference_final, keep)
                        final[site_index, e, seed, columns] = values.cpu().numpy()
                    measured = [(m, mask) for m, name in enumerate(MASKS)
                                if (mask := masks[seed][site].get(name)) is not None
                                and not (kind == "real" and name == "all")]
                    for offset, (noisy_state, reference_state) in enumerate(zip(outputs[g], reference)):
                        difference_sq = _token_sq(noisy_state.float() - reference_state.float())
                        for m, mask in measured:
                            delta_norm[c, m, site_index, e, seed, site + offset, columns] = \
                                _masked_norm(difference_sq, mask).cpu().numpy()
                    if logp_reference is not None:
                        # KL(clean prediction || perturbed prediction), and the same for the injected perturbation
                        # carried unchanged to the last state: the output damage with no repair by the layers.
                        divergence = _kl(logp_reference, readout, outputs[g][-1], chunk).double()
                        injected = outputs[g][0].float() - reference[0].float()
                        carried = _kl(logp_reference, readout, reference[-1].float() + injected, chunk).double()
                        for m, mask in measured:
                            count = mask.sum(-1).clamp_min(1)
                            kl[c, m, site_index, e, seed, columns] = ((divergence * mask).sum(-1) / count).cpu().numpy()
                            kl_carry[c, m, site_index, e, seed, columns] = \
                                ((carried * mask).sum(-1) / count).cpu().numpy()
        window_index += rows
        if step % 5 == 0:
            elapsed = time.time() - began
            log(f"absorption probe: {window_index}/{n_windows} windows, {elapsed / 60:.1f} min, "
                f"about {elapsed / window_index * (n_windows - window_index) / 60:.1f} min left")
    meta = {"sites": list(sites), "etas": list(etas), "seeds": seeds, "scale": scale, "exclude_sinks": exclude_sinks,
            "n_states": states_count, "windows": n_windows, "floor_max": float(floor.max()), "swaps": swaps,
            "kinds": list(KINDS), "masks": list(MASKS), "sinks_fixed": fixed_sinks is not None,
            "output_kl": readout is not None}
    arrays = {"relative_error": error, "cosine_distance": cosine, "relative_error_final": final, "floor": floor,
              "delta_norm": delta_norm, "clean_norm": clean_norm, "clean_spread": clean_spread, "counts": counts,
              "window_starts": np.asarray(window_record, dtype=np.int64),
              "sink_starts": np.asarray(sink_record_start, dtype=np.int64),
              "sink_positions": np.asarray(sink_record_position, dtype=np.int64)}
    if readout is not None:
        arrays.update(kl=kl, kl_carry=kl_carry)
    if swaps:
        arrays.update(swap_positions=swap_positions, swap_tokens=swap_tokens)
    return arrays, meta


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


# ---------------------------------------------------------------------------------------------- the three tests
# Q1  dilution or real shrinkage: does the noise itself get smaller (absolute survival ||D_L|| / ||D_l||)?
# Q2  learned: does the trained model shrink noise clearly more than random-init and permuted-weight controls?
# Q3  denoising, convergence or compression: survival of a real perturbation (a token swap's difference) over
#     that of noise of the same size on the same tokens, split by whether the swap changes the prediction.
# "Clearly" means a log ratio beyond `margin` (log 1.25 by default) with its 95% interval on the same side of 0;
# "about equal" means the whole interval lies within +/- margin. Anything else is "unclear".
MARGIN = float(np.log(1.25))


def _seed_sums(top, bottom):
    """top, bottom [seeds, windows] -> per-window sums over the seeds where both are defined."""
    valid = np.isfinite(top) & np.isfinite(bottom) & (bottom > 0)
    return np.where(valid, top, 0.0).sum(0), np.where(valid, bottom, 0.0).sum(0)


def _log_ratio(top, bottom, picks):
    """log(sum top / sum bottom) over windows with window-bootstrap replicates; NaN when undefined."""
    with np.errstate(divide="ignore", invalid="ignore"):
        point = np.log(top.sum() / bottom.sum())
        replicates = np.log(top[picks].sum(-1) / bottom[picks].sum(-1))
    return point, replicates


def _interval(point, replicates):
    finite = replicates[np.isfinite(replicates)]
    # A statistic undefined in more than 5% of resamples (e.g. a rare token set) gets no interval: "unclear".
    if not np.isfinite(point) or finite.size < 0.95 * replicates.size:
        return {"value": float(point), "low": float("nan"), "high": float("nan")}
    low, high = np.percentile(finite, [2.5, 97.5])
    return {"value": float(point), "low": float(low), "high": float(high)}


def _above(x, margin):
    return x["value"] > margin and x["low"] > 0


def _below(x, margin):
    return x["value"] < -margin and x["high"] < 0


def _near(x, margin):
    return x["low"] > -margin and x["high"] < margin


def q1_verdict(log_survival, margin=MARGIN):
    if _below(log_survival, margin):
        return "shrinks"
    if _near(log_survival, margin):
        return "carried unchanged (dilution)"
    if _above(log_survival, margin):
        return "grows"
    return "unclear"


def q3_verdict(selectivity_changed, selectivity_preserved, margin=MARGIN):
    if _above(selectivity_preserved, margin) and _above(selectivity_changed, margin):
        return "denoising"
    if _near(selectivity_preserved, margin) and _above(selectivity_changed, margin):
        return "convergence"
    if _near(selectivity_preserved, margin) and _near(selectivity_changed, margin):
        return "compression"
    if _below(selectivity_preserved, margin) or _below(selectivity_changed, margin):
        return "none of the three (real perturbations shrink more than noise)"
    return "unclear"


def _picks(windows, n_boot, seed):
    return np.random.default_rng(seed).integers(0, windows, size=(n_boot, windows))


def _survival(arrays, kind, mask, site_index, eta_index, site, last):
    delta = arrays["delta_norm"][KINDS.index(kind), MASKS.index(mask), site_index, eta_index]  # [seeds, states, W]
    return _seed_sums(delta[:, last], delta[:, site])


def summarize_tests(arrays, meta, n_boot=1000, seed=0, margin=MARGIN):
    """Per (site, eta > 0): log noise survival, log clean growth (norm and spread), relative absorption, and with
    swaps the log selectivity (real over noise survival) on changed and preserved tokens, plus the verdicts."""
    last = meta["n_states"] - 1
    picks = _picks(meta["windows"], n_boot, seed)
    rows = []
    for i, site in enumerate(meta["sites"]):
        for e, eta in enumerate(meta["etas"]):
            if eta == 0:
                continue
            row = {"site": site, "eta": eta}
            noise = _log_ratio(*_survival(arrays, "noise", "all", i, e, site, last), picks)
            growth = _log_ratio(*_seed_sums(arrays["clean_norm"][i, 0, :, last], arrays["clean_norm"][i, 0, :, site]),
                                picks)
            spread = _log_ratio(*_seed_sums(arrays["clean_spread"][i, 0, :, last],
                                            arrays["clean_spread"][i, 0, :, site]), picks)
            row["log_noise_survival"] = _interval(*noise)
            row["log_signal_growth"] = _interval(*growth)
            row["log_spread_growth"] = _interval(*spread)
            row["log_relative_absorption"] = _interval(noise[0] - growth[0], noise[1] - growth[1])
            row["log_absorption_vs_spread"] = _interval(noise[0] - spread[0], noise[1] - spread[1])
            row["q1"] = q1_verdict(row["log_noise_survival"], margin)
            if meta.get("swaps"):
                for name in ("changed", "preserved"):
                    real = _log_ratio(*_survival(arrays, "real", name, i, e, site, last), picks)
                    noisy = _log_ratio(*_survival(arrays, "noise", name, i, e, site, last), picks)
                    row[f"log_real_survival_{name}"] = _interval(*real)
                    row[f"log_noise_survival_{name}"] = _interval(*noisy)
                    row[f"log_selectivity_{name}"] = _interval(real[0] - noisy[0], real[1] - noisy[1])
                counts = arrays["counts"][i]
                changed, swapped = counts[MASKS.index("changed")].sum(0), counts[MASKS.index("swapped")].sum(0)
                row["changed_fraction"] = float(changed.sum() / max(swapped.sum(), 1))
                row["q3"] = q3_verdict(row["log_selectivity_changed"], row["log_selectivity_preserved"], margin)
            if "kl" in arrays:
                for kind in KINDS if meta.get("swaps") else ("noise",):
                    mask = MASKS.index("swapped" if kind == "real" else "all")
                    weights = arrays["counts"][i, mask]
                    sums = {}
                    for name in ("kl", "kl_carry"):
                        values = arrays[name][KINDS.index(kind), mask, i, e] if name in arrays else None
                        if values is None:
                            continue
                        valid = np.isfinite(values)
                        sums[name] = (np.where(valid, values, 0) * weights).sum(0)  # token-weighted, per window
                        row[f"output_{name}_{kind}"] = float(sums[name].sum() / max((weights * valid).sum(), 1))
                    if "kl_carry" in sums:
                        # Below 1: the layers repaired the output beyond carrying the perturbation unchanged.
                        row[f"log_repaired_{kind}"] = _interval(*_log_ratio(sums["kl"], sums["kl_carry"], picks))
            rows.append(row)
    return rows


def compare_runs(reference, controls, n_boot=1000, seed=0, margin=MARGIN):
    """Q2: log(control noise survival / trained noise survival) per (site, eta), paired over the same windows.
    reference, controls: (label, arrays, meta). Positive = the trained model shrinks noise more."""
    _, ref_arrays, meta = reference
    last = meta["n_states"] - 1
    picks = _picks(meta["windows"], n_boot, seed)
    out = []
    for label, arrays, control_meta in controls:
        for key in ("sites", "etas", "seeds", "windows", "n_states", "scale", "exclude_sinks"):
            if control_meta[key] != meta[key]:
                raise ValueError(f"Control {label!r} differs from the reference in {key}")
        if not np.array_equal(arrays["window_starts"], ref_arrays["window_starts"]):
            raise ValueError(f"Control {label!r} measured different windows")
        if meta["exclude_sinks"] and not control_meta.get("sinks_fixed"):
            raise ValueError(f"Control {label!r} must reuse the reference sinks (--sinks-from)")
        for i, site in enumerate(meta["sites"]):
            for e, eta in enumerate(meta["etas"]):
                if eta == 0:
                    continue
                trained = _log_ratio(*_survival(ref_arrays, "noise", "all", i, e, site, last), picks)
                control = _log_ratio(*_survival(arrays, "noise", "all", i, e, site, last), picks)
                difference = _interval(control[0] - trained[0], control[1] - trained[1])
                if _above(difference, margin):
                    answer = "yes"
                elif difference["high"] < margin:
                    answer = "no"
                else:
                    answer = "unclear"
                out.append({"control": label, "site": site, "eta": eta, "log_control_over_trained": difference,
                            "trained_shrinks_more": answer})
    return out


def q2_verdict(comparisons, site, eta):
    """yes if every control is clearly beaten at this site, no if any control is not, otherwise unclear."""
    answers = [c["trained_shrinks_more"] for c in comparisons if c["site"] == site and c["eta"] == eta]
    if not answers:
        return "no controls"
    if all(answer == "yes" for answer in answers):
        return "yes"
    if any(answer == "no" for answer in answers):
        return "no"
    return "unclear"


def plot_tests(out, rows, comparisons, eta, label):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [row for row in rows if row["eta"] == eta]
    sites = [row["site"] for row in selected]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    series = [("log_noise_survival", "noise, all tokens"), ("log_signal_growth", "clean signal growth")]
    if selected and "log_selectivity_changed" in selected[0]:
        series += [("log_noise_survival_changed", "noise, prediction-changed tokens"),
                   ("log_real_survival_changed", "real, prediction-changed tokens"),
                   ("log_noise_survival_preserved", "noise, prediction-preserved tokens"),
                   ("log_real_survival_preserved", "real, prediction-preserved tokens")]
    for j, (key, name) in enumerate(series):
        centre = np.exp([row[key]["value"] for row in selected])
        lows, highs = np.exp([row[key]["low"] for row in selected]), np.exp([row[key]["high"] for row in selected])
        errors = np.clip(np.nan_to_num(np.stack([centre - lows, highs - centre])), 0, None)
        axes[0].errorbar(np.array(sites) + 0.25 * (j - len(series) / 2), centre, yerr=errors, marker="o", capsize=2,
                         linestyle="none", color=colors[j % len(colors)], label=name)
    axes[0].axhline(1, color="grey", linewidth=0.6)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Injection layer l (state)")
    axes[0].set_ylabel("Size at the last layer / size at l")
    axes[0].set_title(f"{label}: absolute survival, eta = {eta:g}", fontsize=9)
    axes[0].legend(fontsize=6)
    for j, control in enumerate(sorted({c["control"] for c in comparisons})):
        chosen = [c for c in comparisons if c["control"] == control and c["eta"] == eta]
        values = np.array([c["log_control_over_trained"]["value"] for c in chosen])
        errors = np.clip(np.nan_to_num(np.array([[c["log_control_over_trained"]["value"]
                                                   - c["log_control_over_trained"]["low"],
                                                   c["log_control_over_trained"]["high"]
                                                   - c["log_control_over_trained"]["value"]] for c in chosen]).T),
                         0, None)
        axes[1].errorbar([c["site"] + 0.3 * j for c in chosen], values, yerr=errors, marker="o", capsize=2,
                         linestyle="none", color=colors[j % len(colors)], label=control)
    for level in (-MARGIN, 0, MARGIN):
        axes[1].axhline(level, color="grey", linewidth=0.6, linestyle="-" if level == 0 else ":")
    axes[1].set_xlabel("Injection layer l (state)")
    axes[1].set_ylabel("log(control survival / trained survival)")
    axes[1].set_title("Q2: above the dotted line = trained model shrinks noise clearly more", fontsize=9)
    if comparisons:
        axes[1].legend(fontsize=7)
    for axis in axes:
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(out / "tests.png", dpi=150)
    plt.close(figure)
