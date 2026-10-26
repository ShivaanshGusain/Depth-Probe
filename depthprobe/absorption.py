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


# noise: independent Gaussian per token. real: the difference a swapped earlier token makes, shared across the
# later tokens. rotated: that difference with its feature axis turned by a fixed random rotation, which keeps
# every token's size and the token-to-token correlations but makes the feature direction random. real vs
# rotated isolates the feature direction; rotated vs noise is the effect of the correlation across tokens alone.
KINDS = ("noise", "real", "rotated")
# Token sets every perturbation is measured on. all: every kept token. swapped: kept tokens after the
# counterfactual swap that it moves; changed / preserved: those whose top-1 prediction the swap does / does not
# change. Noise is measured on the same sets as the real perturbation, so each comparison uses one population.
MASKS = ("all", "swapped", "changed", "preserved")
# A swap defines a real direction at a token only if it moves that token's state by at least this fraction.
MIN_REAL = 1e-5
# Output-relevance bookkeeping: KL after the suffix; KL of the injected perturbation carried unchanged; KL of a
# random direction with exactly the propagated perturbation's per-token size at the last state (Q4 reference).
VARIANTS = ("after", "carried", "matched")
# Coordinates whose mean |h| over the batch's kept tokens exceeds this many times the median coordinate are
# "massive" (as in the E6 calibration); selectivity is also measured without them.
MASSIVE_FACTOR = 20.0


def _generator(seed, key):
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest[:8], "little") % (2 ** 63))


def _xi(seed, start, length, width):
    return torch.randn(length, width, generator=_generator(seed, f"xi:{start}"))


def rotation(seed, width):
    """A fixed random orthogonal [width, width] matrix per seed (QR of a Gaussian, signs fixed)."""
    gaussian = torch.randn(width, width, generator=_generator(seed, "rotation"), dtype=torch.float64)
    q, r = torch.linalg.qr(gaussian)
    return (q * torch.sign(torch.diagonal(r))).float()


def _unit_directions(seed, key, starts, length, width, device):
    """Per-token unit Gaussian directions [B, T, d], keyed by (seed, key, window start)."""
    draws = torch.stack([torch.randn(length, width, generator=_generator(seed, f"{key}:{start}"))
                         for start in starts]).to(device)
    return draws / draws.norm(dim=-1, keepdim=True).clamp_min(1e-30)


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


def _logp_chunks(readout, hidden, chunk):
    """Log-probabilities in float64: the KL of a small perturbation (~1e-7) is below float32 log-softmax rounding."""
    return [torch.log_softmax(logits.double(), dim=-1) for logits in _readout_chunks(readout, hidden, chunk)]


def _kl(logp_reference, readout, hidden, chunk):
    """Per token KL(reference || perturbed) of the next-token distribution, in float64."""
    flat = hidden.reshape(-1, hidden.shape[-1])
    values = []
    for j, i in enumerate(range(0, len(flat), chunk)):
        logp = torch.log_softmax(readout(flat[i:i + chunk]).double(), dim=-1)
        values.append((logp_reference[j].exp() * (logp_reference[j] - logp)).sum(-1))
    return torch.cat(values).view(hidden.shape[:2])


def _bulk_coordinates(state, keep):
    """[d] 1 at ordinary coordinates, 0 at massive ones, from this batch's kept tokens."""
    weights = keep[..., None]
    mean_abs = (state.float().abs() * weights).sum((0, 1)) / weights.sum().clamp_min(1)
    return (mean_abs <= MASSIVE_FACTOR * mean_abs.median()).double()


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
                   fixed_changed=None, chunk=256, log=print):
    """R_k and cosine distance per (site, eta, seed, state k, window), plus R after the final norm.

    HF's last hidden state (output_hidden_states[-1]) is taken AFTER the final norm, so if the paper used it,
    its R_L is `relative_error_final`; `relative_error[..., L, :]` is the last block's raw output. eta = 0
    adds nothing and measures the numerical floor of the comparison itself.

    Also recorded, to separate dilution, convergence and denoising (see `summarize_tests`):
      delta_norm [kind, mask, site, eta, seed, state, window]  ||h~_k - h_k||_F over the tokens of the mask
      clean_norm, clean_spread [site, mask, seed, state, window]  ||h_k||_F and the spread around the mean
      counts [site, mask, seed, window]; kl [kind, mask, site, eta, seed, window] (mean over the mask; readout)
      delta_norm_bulk: as delta_norm with massive coordinates left out (massive_counts [state, window])
      kl_total, size_total [kind, variant, mask, site, eta, seed, window]: summed KL and summed squared relative
          size at the last state for the variants after / carried / matched; size_post_total the same size
          measured after the final norm. "matched" adds, at the last state, a random direction with exactly the
          propagated perturbation's per-token size, so KL(after) / KL(matched) compares direction only: the Q4
          reference for "is noise steered away from the directions the prediction reads?".
          kl_random_total, size_random_total [site, mask, seed, window]: a random direction of fixed relative
          size max(etas) at the last state (secondary reference).
      changed_tokens [seed, window, position]: the prediction-changed flags (with swaps); fixed_changed reuses a
          reference run's flags, so controls compare the trained model's token populations.
    swaps=True adds the real perturbation: for noise seed s, one earlier token of each window is swapped (seed s),
    and the difference it makes at state l, scaled exactly like the noise, is injected at l with the original
    context, together with the same difference rotated (KINDS). Comparing survivals on the same tokens asks
    whether the later layers keep real feature directions more than random ones. fixed_sinks reuses a reference
    run's sink positions.
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
        conditions += [(kind, eta, seed) for kind in ("real", "rotated") for eta in etas if eta > 0
                       for seed in range(seeds)]
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
    delta_norm_bulk = np.full_like(delta_norm, np.nan)
    kl_total = np.full((len(KINDS), len(VARIANTS), len(MASKS)) + shape[:3] + (n_windows,), np.nan)
    size_total = np.full_like(kl_total, np.nan)
    size_post_total = np.full_like(kl_total, np.nan)
    kl_random_total = np.full((len(sites), len(MASKS), seeds, n_windows), np.nan)
    size_random_total = np.full_like(kl_random_total, np.nan)
    massive_counts = np.zeros((states_count, n_windows))
    eta_random = max(etas)
    swap_positions = np.full((seeds, n_windows), -1, dtype=np.int64)
    max_length = max(window.end - window.start for window in windows) + (0 if bos_id is None else 1)
    changed_tokens = np.zeros((seeds, n_windows, max_length), dtype=bool)
    rotations = {}
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
        bulk = [_bulk_coordinates(state, keep) for state in states]
        massive_counts[:, columns] = np.array([float((b == 0).sum()) for b in bulk])[:, None]
        # One draw per (seed, window), reused for every site and eta (common random numbers, drawn once).
        xis = {seed: torch.stack([_xi(seed, start, length, states[0].shape[-1]) for start in starts]).to(device)
               for seed in range(seeds)}
        width = states[0].shape[-1]
        # Per seed: the swap's real (and rotated) directions at each site and the token sets it defines.
        directions, rotated, masks = {}, {}, {}
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
            if fixed_changed is None:
                changed = (_top1(readout, states[last], chunk) != _top1(readout, swapped[last], chunk)).float()
            else:  # the reference (trained) run's token populations
                changed = torch.zeros(rows, length, device=device)
                for row, start in enumerate(starts):
                    if start not in fixed_changed:
                        raise ValueError(f"Window starting at {start} has no reference token sets; runs must use "
                                         "the same data, context, stride and --every")
                    changed[row] = torch.from_numpy(fixed_changed[start][seed, :length]).float().to(device)
            changed_tokens[seed, columns, :length] = changed.bool().cpu().numpy()
            if seed not in rotations:
                rotations[seed] = rotation(seed, width).to(device)
            directions[seed], rotated[seed] = {}, {}
            for site in sites:
                difference = swapped[site].float() - states[site].float()
                moved = (difference.norm(dim=-1) > MIN_REAL * states[site].float().norm(dim=-1)).float()
                selected = keep * after * moved
                directions[seed][site] = difference
                rotated[seed][site] = difference @ rotations[seed]
                masks[seed][site].update(swapped=selected, changed=selected * changed,
                                         preserved=selected * (1 - changed))
            del swapped, swapped_capture
        units = {}
        if readout is not None:
            units = {seed: _unit_directions(seed, "matched", starts, length, width, device) for seed in range(seeds)}
        random_kl = {}
        if readout is not None and eta_random > 0:
            logp_clean = _logp_chunks(readout, states[last], chunk)
            everywhere = torch.ones(rows, length, device=device)
            for seed in range(seeds):
                direction = torch.stack([torch.randn(length, states[last].shape[-1],
                                                     generator=_generator(seed, f"final:{start}"))
                                         for start in starts]).to(device)
                bump = perturbation(states[last], direction, eta_random, "token", everywhere)
                random_kl[seed] = _kl(logp_clean, readout, states[last].float() + bump, chunk).double()
            del logp_clean
        for site_index, site in enumerate(sites):
            for seed in range(seeds):
                for m, name in enumerate(MASKS):
                    mask = masks[seed][site].get(name)
                    if mask is None:
                        continue
                    counts[site_index, m, seed, columns] = mask.sum(-1).cpu().numpy()
                    if seed in random_kl:
                        kl_random_total[site_index, m, seed, columns] = (random_kl[seed] * mask).sum(-1).cpu().numpy()
                        size_random_total[site_index, m, seed, columns] = \
                            (mask.sum(-1).double() * eta_random ** 2).cpu().numpy()
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
                        source = directions if kind == "real" else rotated
                        injected = perturbation(base, source[seed][site], eta, scale, masks[seed][site]["swapped"])
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
                    logp_reference = _logp_chunks(readout, reference[-1], chunk)
                    reference_sq = _token_sq(reference[-1]).clamp_min(1e-30)
                    normed_reference = readout.norm(reference[-1].to(readout.dtype)).double()
                    normed_sq = normed_reference.square().sum(-1).clamp_min(1e-30)

                    def post_norm_size(hidden):  # squared size after the final norm, relative to the token
                        return (readout.norm(hidden.to(readout.dtype)).double() - normed_reference).square().sum(-1) \
                            / normed_sq
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
                                and not (kind != "noise" and name == "all")]
                    for offset, (noisy_state, reference_state) in enumerate(zip(outputs[g], reference)):
                        squared = (noisy_state.float() - reference_state.float()).double().square()
                        difference_sq, bulk_sq = squared.sum(-1), (squared * bulk[site + offset]).sum(-1)
                        for m, mask in measured:
                            delta_norm[c, m, site_index, e, seed, site + offset, columns] = \
                                _masked_norm(difference_sq, mask).cpu().numpy()
                            delta_norm_bulk[c, m, site_index, e, seed, site + offset, columns] = \
                                _masked_norm(bulk_sq, mask).cpu().numpy()
                    if logp_reference is not None:
                        # KL(clean prediction || perturbed prediction), and the same for the injected perturbation
                        # carried unchanged to the last state: the output damage with no repair by the layers.
                        divergence = _kl(logp_reference, readout, outputs[g][-1], chunk)
                        injected = outputs[g][0].float() - reference[0].float()
                        carried_state = reference[-1].float() + injected
                        carried = _kl(logp_reference, readout, carried_state, chunk)
                        # A random direction with exactly the propagated perturbation's size at every token.
                        propagated = outputs[g][-1].float() - reference[-1].float()
                        matched_state = reference[-1].float() + units[seed] * propagated.norm(dim=-1, keepdim=True)
                        matched = _kl(logp_reference, readout, matched_state, chunk)
                        # Squared size relative to the token's last state, per token: KL is quadratic in it here.
                        size_after = _token_sq(propagated) / reference_sq
                        size_carried = _token_sq(injected) / reference_sq
                        variants = ((divergence, size_after, post_norm_size(outputs[g][-1])),
                                    (carried, size_carried, post_norm_size(carried_state)),
                                    (matched, size_after, post_norm_size(matched_state)))
                        for m, mask in measured:
                            count = mask.sum(-1).clamp_min(1)
                            kl[c, m, site_index, e, seed, columns] = ((divergence * mask).sum(-1) / count).cpu().numpy()
                            kl_carry[c, m, site_index, e, seed, columns] = \
                                ((carried * mask).sum(-1) / count).cpu().numpy()
                            for v, (values, sizes, post_sizes) in enumerate(variants):
                                kl_total[c, v, m, site_index, e, seed, columns] = (values * mask).sum(-1).cpu().numpy()
                                size_total[c, v, m, site_index, e, seed, columns] = (sizes * mask).sum(-1).cpu().numpy()
                                size_post_total[c, v, m, site_index, e, seed, columns] = \
                                    (post_sizes * mask).sum(-1).cpu().numpy()
        window_index += rows
        if step % 5 == 0:
            elapsed = time.time() - began
            log(f"absorption probe: {window_index}/{n_windows} windows, {elapsed / 60:.1f} min, "
                f"about {elapsed / window_index * (n_windows - window_index) / 60:.1f} min left")
    meta = {"sites": list(sites), "etas": list(etas), "seeds": seeds, "scale": scale, "exclude_sinks": exclude_sinks,
            "n_states": states_count, "windows": n_windows, "floor_max": float(floor.max()), "swaps": swaps,
            "kinds": list(KINDS), "masks": list(MASKS), "variants": list(VARIANTS),
            "sinks_fixed": fixed_sinks is not None, "token_sets_fixed": fixed_changed is not None,
            "output_kl": readout is not None, "eta_random": eta_random, "massive_factor": MASSIVE_FACTOR}
    arrays = {"relative_error": error, "cosine_distance": cosine, "relative_error_final": final, "floor": floor,
              "delta_norm": delta_norm, "clean_norm": clean_norm, "clean_spread": clean_spread, "counts": counts,
              "delta_norm_bulk": delta_norm_bulk, "massive_counts": massive_counts,
              "window_starts": np.asarray(window_record, dtype=np.int64),
              "sink_starts": np.asarray(sink_record_start, dtype=np.int64),
              "sink_positions": np.asarray(sink_record_position, dtype=np.int64)}
    if readout is not None:
        arrays.update(kl=kl, kl_carry=kl_carry, kl_total=kl_total, size_total=size_total,
                      size_post_total=size_post_total, kl_random_total=kl_random_total,
                      size_random_total=size_random_total)
    if swaps:
        arrays.update(swap_positions=swap_positions, swap_tokens=swap_tokens, changed_tokens=changed_tokens)
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


def _index(meta, field, name, default):
    """Position of `name` in a run's recorded axis (older runs recorded fewer kinds or variants), or None."""
    names = meta.get(field, default)
    return names.index(name) if name in names else None


def _survival(arrays, meta, kind, mask, site_index, eta_index, site, last, key="delta_norm"):
    k = _index(meta, "kinds", kind, ["noise", "real"])
    if k is None:
        return None
    delta = arrays[key][k, MASKS.index(mask), site_index, eta_index]  # [seeds, states, W]
    return _seed_sums(delta[:, last], delta[:, site])


def _growth(arrays, site_index, site, last, key="clean_norm"):
    return _seed_sums(arrays[key][site_index, 0, :, last], arrays[key][site_index, 0, :, site])


def _contrast(arrays, meta, numerator, denominator, name, site_index, eta_index, site, last, picks,
              key="delta_norm"):
    """log(survival of `numerator` / survival of `denominator`) on one token set, as (point, replicates)."""
    top = _survival(arrays, meta, numerator, name, site_index, eta_index, site, last, key)
    bottom = _survival(arrays, meta, denominator, name, site_index, eta_index, site, last, key)
    if top is None or bottom is None:
        return None
    top, bottom = _log_ratio(*top, picks), _log_ratio(*bottom, picks)
    return top[0] - bottom[0], top[1] - bottom[1]


def _selectivity(arrays, meta, name, site_index, eta_index, site, last, picks, key="delta_norm"):
    """The Q3 statistic: real over rotated-real survival (v2); real over noise for runs without rotated."""
    reference = "rotated" if _index(meta, "kinds", "rotated", ["noise", "real"]) is not None else "noise"
    return _contrast(arrays, meta, "real", reference, name, site_index, eta_index, site, last, picks, key)


def _kl_sum(arrays, meta, kind, variant, mask, site_index, eta_index, key="kl_total"):
    k, v = _index(meta, "kinds", kind, ["noise", "real"]), _index(meta, "variants", variant, ["after", "carried"])
    if k is None or v is None or key not in arrays:
        return None
    return arrays[key][k, v, MASKS.index(mask), site_index, eta_index]  # [seeds, W]


def _steering(arrays, meta, kind, mask, site_index, eta_index, picks, post_norm=False):
    """log(KL after the layers / KL of a random direction of the same per-token size at the last state).
    post_norm: per unit of size measured after the final norm instead (removes what the norm itself discards)."""
    after = _kl_sum(arrays, meta, kind, "after", mask, site_index, eta_index)
    matched = _kl_sum(arrays, meta, kind, "matched", mask, site_index, eta_index)
    if after is None or matched is None:
        return None
    if not post_norm:
        return _log_ratio(*_seed_sums(after, matched), picks)
    after_size = _kl_sum(arrays, meta, kind, "after", mask, site_index, eta_index, "size_post_total")
    matched_size = _kl_sum(arrays, meta, kind, "matched", mask, site_index, eta_index, "size_post_total")
    if after_size is None:
        return None
    top = _log_ratio(*_seed_sums(after, after_size), picks)
    bottom = _log_ratio(*_seed_sums(matched, matched_size), picks)
    return top[0] - bottom[0], top[1] - bottom[1]


def _massive_share(arrays, meta, kind, mask, site_index, eta_index, state):
    """Fraction of the perturbation's energy at massive coordinates, at one state, over all seeds and windows."""
    k = _index(meta, "kinds", kind, ["noise", "real"])
    if k is None:
        return None
    index = (k, MASKS.index(mask), site_index, eta_index, slice(None), state)
    full, bulk = arrays["delta_norm"][index], arrays["delta_norm_bulk"][index]
    valid = np.isfinite(full) & np.isfinite(bulk)
    total = np.square(full[valid]).sum()
    return float(1 - np.square(bulk[valid]).sum() / total) if total > 0 else None


def q4_verdict(log_steering, margin=MARGIN):
    """Output effect of the propagated noise over that of a random direction of the same per-token size at the
    last state: does the model move noise out of (or into) the directions the prediction reads?"""
    if _below(log_steering, margin):
        return "steered away from output-relevant directions"
    if _near(log_steering, margin):
        return "not steered (as output-relevant as a random direction)"
    if _above(log_steering, margin):
        return "steered toward output-relevant directions"
    return "unclear"


def _versus(difference, margin):
    """Control comparisons: yes if clearly positive, no if the interval rules out a margin-sized effect."""
    if _above(difference, margin):
        return "yes"
    if difference["high"] < margin:
        return "no"
    return "unclear"


def summarize_tests(arrays, meta, n_boot=1000, seed=0, margin=MARGIN):
    """Per (site, eta > 0): log noise survival, log clean growth (norm and spread), relative absorption; with
    swaps, Q3 on real vs rotated-real survival (and real vs noise, the v1 statistic), with and without massive
    coordinates; with the readout, Q4 steering against a size-matched random direction. Plus the verdicts."""
    last = meta["n_states"] - 1
    picks = _picks(meta["windows"], n_boot, seed)
    has_rotated = _index(meta, "kinds", "rotated", ["noise", "real"]) is not None
    rows = []
    for i, site in enumerate(meta["sites"]):
        for e, eta in enumerate(meta["etas"]):
            if eta == 0:
                continue
            row = {"site": site, "eta": eta}
            noise = _log_ratio(*_survival(arrays, meta, "noise", "all", i, e, site, last), picks)
            growth = _log_ratio(*_growth(arrays, i, site, last), picks)
            spread = _log_ratio(*_growth(arrays, i, site, last, "clean_spread"), picks)
            row["log_noise_survival"] = _interval(*noise)
            row["log_signal_growth"] = _interval(*growth)
            row["log_spread_growth"] = _interval(*spread)
            row["log_relative_absorption"] = _interval(noise[0] - growth[0], noise[1] - growth[1])
            row["log_absorption_vs_spread"] = _interval(noise[0] - spread[0], noise[1] - spread[1])
            row["q1"] = q1_verdict(row["log_noise_survival"], margin)
            if meta.get("swaps"):
                for name in ("changed", "preserved"):
                    for kind in ("real", "rotated", "noise"):
                        survival = _survival(arrays, meta, kind, name, i, e, site, last)
                        if survival is not None:
                            row[f"log_{kind}_survival_{name}"] = _interval(*_log_ratio(*survival, picks))
                    row[f"log_selectivity_{name}"] = _interval(*_selectivity(arrays, meta, name, i, e, site, last,
                                                                             picks))
                    row[f"log_real_over_noise_{name}"] = _interval(
                        *_contrast(arrays, meta, "real", "noise", name, i, e, site, last, picks))
                    if has_rotated:  # the effect of correlation across tokens alone
                        row[f"log_rotated_over_noise_{name}"] = _interval(
                            *_contrast(arrays, meta, "rotated", "noise", name, i, e, site, last, picks))
                    if "delta_norm_bulk" in arrays:
                        bulk = _selectivity(arrays, meta, name, i, e, site, last, picks, "delta_norm_bulk")
                        full = _selectivity(arrays, meta, name, i, e, site, last, picks)
                        row[f"log_selectivity_{name}_bulk"] = _interval(*bulk)
                        row[f"log_bulk_minus_full_{name}"] = _interval(bulk[0] - full[0], bulk[1] - full[1])
                counts = arrays["counts"][i]
                changed, swapped = counts[MASKS.index("changed")].sum(0), counts[MASKS.index("swapped")].sum(0)
                row["changed_fraction"] = float(changed.sum() / max(swapped.sum(), 1))
                row["q3_statistic"] = "real vs rotated real" if has_rotated else "real vs noise"
                row["q3"] = q3_verdict(row["log_selectivity_changed"], row["log_selectivity_preserved"], margin)
                row["q3_vs_noise"] = q3_verdict(row["log_real_over_noise_changed"],
                                                row["log_real_over_noise_preserved"], margin)
                if "delta_norm_bulk" in arrays:
                    if arrays["massive_counts"][[site, last]].max() == 0:
                        row["q3_bulk"] = row["q3_massive_check"] = \
                            "not applicable (no massive coordinates at the site or the last state)"
                    else:
                        row["q3_bulk"] = q3_verdict(row["log_selectivity_changed_bulk"],
                                                    row["log_selectivity_preserved_bulk"], margin)
                        row["q3_massive_check"] = ("robust (same verdict on bulk coordinates)"
                                                   if row["q3_bulk"] == row["q3"]
                                                   else f"depends on massive coordinates ({row['q3_bulk']} on bulk "
                                                        "coordinates)")
                    kinds = ("noise", "real", "rotated") if has_rotated else ("noise", "real")
                    for kind in kinds:
                        mask = "all" if kind == "noise" else "swapped"
                        for where, state in (("site", site), ("last", last)):
                            row[f"massive_share_{kind}_{where}"] = _massive_share(arrays, meta, kind, mask, i, e,
                                                                                  state)
            if "kl" in arrays:
                for kind in KINDS if meta.get("swaps") else ("noise",):
                    k = _index(meta, "kinds", kind, ["noise", "real"])
                    if k is None:
                        continue
                    mask = MASKS.index("all" if kind == "noise" else "swapped")
                    weights = arrays["counts"][i, mask]
                    sums = {}
                    for name in ("kl", "kl_carry"):
                        values = arrays[name][k, mask, i, e] if name in arrays else None
                        if values is None:
                            continue
                        valid = np.isfinite(values)
                        sums[name] = (np.where(valid, values, 0) * weights).sum(0)  # token-weighted, per window
                        row[f"output_{name}_{kind}"] = float(sums[name].sum() / max((weights * valid).sum(), 1))
                    if "kl_carry" in sums:
                        # Below 1: the layers repaired the output beyond carrying the perturbation unchanged.
                        row[f"log_repaired_{kind}"] = _interval(*_log_ratio(sums["kl"], sums["kl_carry"], picks))
            steering = _steering(arrays, meta, "noise", "all", i, e, picks)
            if steering is not None:
                row["log_steering_noise"] = _interval(*steering)
                row["q4"] = q4_verdict(row["log_steering_noise"], margin)
                post = _steering(arrays, meta, "noise", "all", i, e, picks, post_norm=True)
                if post is not None:  # per unit of size after the final norm: excludes what the norm discards
                    row["log_steering_noise_post_norm"] = _interval(*post)
                for kind in ("real", "rotated") if meta.get("swaps") else ():
                    other = _steering(arrays, meta, kind, "swapped", i, e, picks)
                    if other is not None:
                        row[f"log_steering_{kind}"] = _interval(*other)
            if "kl_random_total" in arrays:  # secondary: a random direction of fixed size max(etas)
                after = _kl_sum(arrays, meta, "noise", "after", "all", i, e)
                after_size = _kl_sum(arrays, meta, "noise", "after", "all", i, e, "size_total")
                random = _log_ratio(*_seed_sums(arrays["kl_random_total"][i, 0], arrays["size_random_total"][i, 0]),
                                    picks)
                noise_unit = _log_ratio(*_seed_sums(after, after_size), picks)
                row["log_steering_noise_fixed_size"] = _interval(noise_unit[0] - random[0], noise_unit[1] - random[1])
                if steering is None:  # runs without the matched direction: the fixed-size reference is all there is
                    row["log_steering_noise"] = row["log_steering_noise_fixed_size"]
                    row["q4"] = q4_verdict(row["log_steering_noise"], margin)
            rows.append(row)
    return rows


# A control whose noise reaches this relative size at the last state (window mean, R_L at the primary eta) is
# saturated: any perturbation decorrelates its state, so comparisons against it are not applicable.
SATURATION = 0.1


def compare_runs(reference, controls, n_boot=1000, seed=0, margin=MARGIN):
    """Trained model vs controls, per (site, eta), paired over the same windows. reference, controls:
    (label, arrays, meta).
      trained_shrinks_more (Q2, v1): log(control / trained absolute noise survival) clearly positive
      trained_relative_lower (Q2, v2): log(control / trained relative noise, i.e. survival over signal growth)
      selectivity_learned (Q3, v2): trained Q3 selectivity clearly above the control's on both token sets
      trained_steers_away_more (Q4, secondary): trained log steering clearly below the control's
    Every answer is "not applicable (control saturated)" where the control's R_L exceeds SATURATION."""
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
        both_swapped = meta.get("swaps") and control_meta.get("swaps")
        if both_swapped:
            if not np.array_equal(arrays["swap_positions"], ref_arrays["swap_positions"]):
                raise ValueError(f"Control {label!r} used different token swaps")
            if "changed_tokens" in ref_arrays and (not control_meta.get("token_sets_fixed") or not np.array_equal(
                    arrays.get("changed_tokens"), ref_arrays["changed_tokens"])):
                raise ValueError(f"Control {label!r} must reuse the reference prediction-changed tokens "
                                 "(--real-swaps with --sinks-from the trained run)")
        for i, site in enumerate(meta["sites"]):
            for e, eta in enumerate(meta["etas"]):
                if eta == 0:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    saturation = float(np.nanmean(arrays["relative_error"][i, e, :, last]))
                entry = {"control": label, "site": site, "eta": eta, "control_last_relative_error": saturation,
                         "control_saturated": bool(not saturation <= SATURATION)}
                trained = _log_ratio(*_survival(ref_arrays, meta, "noise", "all", i, e, site, last), picks)
                control = _log_ratio(*_survival(arrays, control_meta, "noise", "all", i, e, site, last), picks)
                difference = _interval(control[0] - trained[0], control[1] - trained[1])
                trained_growth = _log_ratio(*_growth(ref_arrays, i, site, last), picks)
                control_growth = _log_ratio(*_growth(arrays, i, site, last), picks)
                relative = _interval((control[0] - control_growth[0]) - (trained[0] - trained_growth[0]),
                                     (control[1] - control_growth[1]) - (trained[1] - trained_growth[1]))
                entry.update(log_control_over_trained=difference, trained_shrinks_more=_versus(difference, margin),
                             log_relative_control_over_trained=relative,
                             trained_relative_lower=_versus(relative, margin))
                if both_swapped:
                    answers = []
                    for name in ("changed", "preserved"):
                        mine = _selectivity(ref_arrays, meta, name, i, e, site, last, picks)
                        theirs = _selectivity(arrays, control_meta, name, i, e, site, last, picks)
                        gap = _interval(mine[0] - theirs[0], mine[1] - theirs[1])
                        entry[f"log_selectivity_trained_over_control_{name}"] = gap
                        answers.append(_versus(gap, margin))
                    entry["selectivity_learned"] = ("yes" if all(a == "yes" for a in answers)
                                                    else "no" if "no" in answers else "unclear")
                mine = _steering(ref_arrays, meta, "noise", "all", i, e, picks)
                theirs = _steering(arrays, control_meta, "noise", "all", i, e, picks)
                if mine is not None and theirs is not None:
                    gap = _interval(theirs[0] - mine[0], theirs[1] - mine[1])  # positive: trained steers away more
                    entry["log_steering_control_minus_trained"] = gap
                    entry["trained_steers_away_more"] = _versus(gap, margin)
                if entry["control_saturated"]:
                    for key in ("trained_shrinks_more", "trained_relative_lower", "selectivity_learned",
                                "trained_steers_away_more"):
                        if key in entry:
                            entry[key] = "not applicable (control saturated)"
                out.append(entry)
    return out


def control_verdict(comparisons, site, eta, key="trained_shrinks_more", deciding=None):
    """yes if every applicable deciding control is clearly beaten at this site, no if any is not, otherwise
    unclear. Saturated controls are not applicable; controls outside `deciding` are reported only."""
    rows = [c for c in comparisons if c["site"] == site and c["eta"] == eta and key in c
            and (deciding is None or c["control"] in deciding)]
    if not rows:
        return "not measured"
    answers = [c[key] for c in rows if not c[key].startswith("not applicable")]
    if not answers:
        return "not applicable (every deciding control saturated)"
    if all(answer == "yes" for answer in answers):
        return "yes"
    if any(answer == "no" for answer in answers):
        return "no"
    return "unclear"


def control_detail(comparisons, site, eta, key, deciding=None):
    """'control: answer' for every control, marking those that do not decide."""
    return ", ".join(f"{c['control']}: {c[key]}" + ("" if deciding is None or c["control"] in deciding
                                                    else " (reported only)")
                     for c in comparisons if c["site"] == site and c["eta"] == eta and key in c)


def q2_verdict(comparisons, site, eta):
    return control_verdict(comparisons, site, eta, "trained_shrinks_more")


def plot_tests(out, rows, comparisons, eta, label, margin=MARGIN):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [row for row in rows if row["eta"] == eta]
    sites = [row["site"] for row in selected]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    series = [("log_noise_survival", "noise, all tokens"), ("log_signal_growth", "clean signal growth")]
    for name in ("changed", "preserved"):
        for kind in ("noise", "rotated", "real"):
            key = f"log_{kind}_survival_{name}"
            if selected and key in selected[0]:
                series.append((key, f"{kind}, prediction-{name} tokens"))
    for j, (key, name) in enumerate(series):
        centre = np.exp([row[key]["value"] for row in selected])
        lows, highs = np.exp([row[key]["low"] for row in selected]), np.exp([row[key]["high"] for row in selected])
        errors = np.clip(np.nan_to_num(np.stack([centre - lows, highs - centre])), 0, None)
        axes[0].errorbar(np.array(sites) + 0.2 * (j - len(series) / 2), centre, yerr=errors, marker="o", capsize=2,
                         linestyle="none", color=colors[j % len(colors)], label=name)
    axes[0].axhline(1, color="grey", linewidth=0.6)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Injection layer l (state)")
    axes[0].set_ylabel("Size at the last layer / size at l")
    axes[0].set_title(f"{label}: absolute survival, eta = {eta:g}", fontsize=9)
    axes[0].legend(fontsize=6)
    for j, control in enumerate(sorted({c["control"] for c in comparisons})):
        chosen = [c for c in comparisons if c["control"] == control and c["eta"] == eta]
        statistic = [c["log_relative_control_over_trained"] for c in chosen]
        values = np.array([s["value"] for s in statistic])
        errors = np.clip(np.nan_to_num(np.array([[s["value"] - s["low"], s["high"] - s["value"]]
                                                 for s in statistic]).T), 0, None)
        axes[1].errorbar([c["site"] + 0.3 * j for c in chosen], values, yerr=errors, marker="o", capsize=2,
                         linestyle="none", color=colors[j % len(colors)],
                         label=control + (" (saturated)" if any(c["control_saturated"] for c in chosen) else ""))
    for level in (-margin, 0, margin):
        axes[1].axhline(level, color="grey", linewidth=0.6, linestyle="-" if level == 0 else ":")
    axes[1].set_xlabel("Injection layer l (state)")
    axes[1].set_ylabel("log(control relative noise / trained relative noise)")
    axes[1].set_title("Q2 relative: above the dotted line = trained model keeps noise clearly smaller", fontsize=9)
    if comparisons:
        axes[1].legend(fontsize=7)
    for axis in axes:
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(out / "tests.png", dpi=150)
    plt.close(figure)
