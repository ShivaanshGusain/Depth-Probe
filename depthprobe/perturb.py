"""E6 noise-injection (denoise) test, v1: do later layers remove a perturbation added at layer l?

Clean pass: every state of the model at the measured positions (no LM head is built). Noisy pass for
injection state l: the suffix blocks l+1..L run as a segment on [clean, noisy x G] copies of state l. Row
group 0 is the in-batch clean reference: every metric compares the noisy rows with it, so kernel effects of
the larger batch (bf16 tiling, MoE expert grouping) cancel. Its deviation from the separate clean pass is
reported as the numerical floor.

State indexing as everywhere: state 0 = embeddings, state l = output of block l, state L = last block.
Noise is sized in normalized coordinates (relative to each token's bulk norm, massive coordinates left
out) and drawn once per window, so the same draws are reused across sites, sizes and model variants.
The metric functions are pure functions of reference states, noisy states and the injected noise; the
analytic toy stacks in depthprobe/toys.py reuse them.
"""

import hashlib

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from depthprobe.data import iterate_batches
from depthprobe.finetune import make_segment
from depthprobe.model import ResidualCapture, find_parts, hidden_width, n_blocks

FAMILIES = ("iso", "off", "on", "read", "unread", "shared", "signed")
SUBSPACE_FAMILIES = ("off", "on", "read", "unread")
STATE_METRICS = ("log_survival", "log_A", "log_A0", "retained", "log_off", "log_off0", "log_on", "log_on0")
OUTPUT_METRICS = ("kl", "dnll", "flip", "kl_carry", "dnll_carry", "kl_last", "dnll_last", "nl", "delta_eff")
TINY = 1e-30
# Subspace survival is undefined when the injected perturbation has (almost) no component in that subspace.
MIN_SHARE = 1e-3


# ---------------------------------------------------------------------------------------------- coordinates

def norm_kind(model):
    return "layer" if isinstance(find_parts(model).norm, nn.LayerNorm) else "rms"


def normalize(hidden, kind, eps=1e-6):
    """Parameter-free version of the final norm: the direction the next norm reads, without gain or bias."""
    hidden = hidden.float()
    if kind == "layer":
        hidden = hidden - hidden.mean(-1, keepdim=True)
    return hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + eps)


def _log_norm(values):
    return torch.log(values.norm(dim=-1).clamp_min(TINY))


def check_layer_types(config):
    """Segments index per-layer attention types by list position, so mixed types would get wrong masks."""
    types = getattr(config, "layer_types", None)
    if types and len(set(types)) > 1:
        raise ValueError(f"Mixed attention layer types {sorted(set(types))} are not supported by suffix segments")


# ---------------------------------------------------------------------------------------------- metrics

def propagation_metrics(reference, noisy, kind, *, off=None, on=None):
    """How a perturbation injected at reference[0] -> noisy[0] evolves through the later states, per token.

    reference, noisy: lists of [n, d] tensors for states l, l+1, ..., L (index 0 is the injection state).
    off, on: optional lists of [d, r] orthonormal bases per state, for subspace-projected survival.
    Returns [m + 1, n] float32 tensors (row 0 is the injection state itself):
      log_survival   log ||D_k|| / ||D_l||, raw coordinates (noise removed in absolute terms)
      log_A          the same in normalized coordinates (what the next norm reads)
      log_A0         dilution null: D_l carried unchanged along the identity path, then normalized
      retained       <D_k, D_l> / ||D_l||^2, the part of the original perturbation still present
      log_off/on     survival of the perturbation's off- / on-manifold component (each state's basis)
      log_off0/on0   their null: D_l carried unchanged and projected on state k's basis, which captures
                     subspace drift and rank changes with no processing. NaN where the injected
                     perturbation has less than MIN_SHARE of its energy in that subspace.
    """
    d0 = noisy[0].float() - reference[0].float()
    u0 = normalize(noisy[0], kind) - normalize(reference[0], kind)
    base_raw, base_norm = _log_norm(d0), _log_norm(u0)
    d0_energy = d0.square().sum(-1).clamp_min(TINY)
    projected = off is not None
    keys = STATE_METRICS if projected else ("log_survival", "log_A", "log_A0", "retained")
    out = {key: [] for key in keys}
    if projected:
        base_off, base_on = d0 @ off[0], d0 @ on[0]
        off_defined = base_off.square().sum(-1) / d0_energy >= MIN_SHARE
        on_defined = base_on.square().sum(-1) / d0_energy >= MIN_SHARE
        log_base_off, log_base_on = _log_norm(base_off), _log_norm(base_on)
    nan = float("nan")
    for index, (state, perturbed) in enumerate(zip(reference, noisy)):
        state, perturbed = state.float(), perturbed.float()
        difference = perturbed - state
        normalized_state = normalize(state, kind)
        out["log_survival"].append(_log_norm(difference) - base_raw)
        out["log_A"].append(_log_norm(normalize(perturbed, kind) - normalized_state) - base_norm)
        out["log_A0"].append(_log_norm(normalize(state + d0, kind) - normalized_state) - base_norm)
        out["retained"].append((difference * d0).sum(-1) / d0_energy)
        if projected:
            out["log_off"].append(torch.where(off_defined, _log_norm(difference @ off[index]) - log_base_off, nan))
            out["log_off0"].append(torch.where(off_defined, _log_norm(d0 @ off[index]) - log_base_off, nan))
            out["log_on"].append(torch.where(on_defined, _log_norm(difference @ on[index]) - log_base_on, nan))
            out["log_on0"].append(torch.where(on_defined, _log_norm(d0 @ on[index]) - log_base_on, nan))
    return {key: torch.stack(values) for key, values in out.items()}


def output_damage(logp_reference, logits_noisy, targets):
    """KL(reference || noisy) over the vocabulary, change in NLL of the true token, and top-1 flip, per token."""
    logp_noisy = torch.log_softmax(logits_noisy.float(), dim=-1)
    kl = (logp_reference.exp() * (logp_reference - logp_noisy)).sum(-1)
    dnll = (logp_reference.gather(-1, targets[:, None]) - logp_noisy.gather(-1, targets[:, None]))[:, 0]
    flip = (logp_reference.argmax(-1) != logp_noisy.argmax(-1)).float()
    return kl, dnll, flip


def nonlinearity(plus, minus, reference):
    """Antithetic pair at one state: 0 when the response is odd (linear), 1 when the two responses coincide.
    In bfloat16 the rounding of h +/- eps is not sign-symmetric, so small sizes have a floor above 0."""
    up, down = plus.float() - reference.float(), minus.float() - reference.float()
    return (up + down).norm(dim=-1) / (up.norm(dim=-1) + down.norm(dim=-1)).clamp_min(TINY)


# ---------------------------------------------------------------------------------------------- noise

def _generator(seed, key):
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest[:8], "little") % (2 ** 63))


def window_noise(seed, starts, length, width):
    """Base Gaussian draws [B, T, d] and per-token random signs [B, T], keyed only by (seed, window start)."""
    z = torch.stack([torch.randn(length, width, generator=_generator(seed, f"z:{start}")) for start in starts])
    signs = torch.stack([torch.randint(0, 2, (length,), generator=_generator(seed, f"sign:{start}")) * 2.0 - 1.0
                         for start in starts])
    return z, signs


def shared_direction(seed, width):
    return torch.randn(width, generator=_generator(seed, "shared"))


def unit_directions(family, z, shared, signs, basis, bulk):
    """Unit perturbation directions per token, restricted to bulk coordinates (and to `basis` if given)."""
    if family in ("shared", "signed"):
        directions = shared.to(z.device).expand_as(z).clone()
        if family == "signed":
            directions = directions * signs[..., None]
    else:
        directions = z.clone()
    directions = directions * bulk
    if basis is not None:
        directions = (directions @ basis) @ basis.T
    return directions / directions.norm(dim=-1, keepdim=True).clamp_min(TINY)


def bulk_norm(hidden, bulk):
    return (hidden.float() * bulk).norm(dim=-1)


def family_basis(calibration, family, state):
    return calibration[family][state] if family in SUBSPACE_FAMILIES else None


def conditions_for(sites, families, deltas):
    """Ordered so that each +/- antithetic pair is adjacent (sign innermost)."""
    return [{"site": site, "family": family, "delta": float(delta), "sign": sign}
            for site in sites for family in families for delta in deltas for sign in (1, -1)]


# ---------------------------------------------------------------------------------------------- calibration

def _subspace_bases(matrix, bulk, keep_fraction):
    """Top eigenvectors carrying keep_fraction of the energy (at most half the bulk width) and the low-energy
    half, embedded in full width with zero rows at massive coordinates. Also returns the realised rank and
    energy share of the top part, since the half-width cap can bind in near-isotropic (control) models."""
    index = torch.nonzero(bulk, as_tuple=False)[:, 0]
    sub = matrix[index][:, index].double()
    values, vectors = torch.linalg.eigh((sub + sub.T) / 2)
    values, vectors = values.flip(0).clamp_min(0), vectors.flip(1)
    share = values.cumsum(0) / values.sum().clamp_min(TINY)
    top = int((share < keep_fraction).sum().item()) + 1  # first index reaching keep_fraction (device-agnostic)
    top = min(max(top, 1), len(index) // 2)
    width = matrix.shape[0]

    def embed(columns):
        basis = torch.zeros(width, columns.shape[1], dtype=torch.float32, device=matrix.device)
        basis[index] = columns.float()
        return basis

    return embed(vectors[:, :top]), embed(vectors[:, len(index) // 2:]), values.float(), top, float(share[top - 1])


@torch.no_grad()
def calibrate(model, readout, corpus, windows, *, batch_size, bos_id, device, massive_factor=20.0,
              keep_fraction=0.9, log=print):
    """Massive-coordinate masks and on/off-manifold and readable/unreadable bases per state.

    on/off: principal directions of the normalized clean states (top directions carrying keep_fraction of
    the variance, at most half the bulk width / the low-variance half), with massive coordinates left out.
    read/unread: the same split of the readout's sensitivity (final-norm gain times unembedding), so
    'content' can also be defined by what the head reads rather than by variance.
    """
    kind, states_count, width = norm_kind(model), n_blocks(model) + 1, hidden_width(model)
    container = find_parts(model).container
    first = torch.zeros(states_count, width, dtype=torch.float64, device=device)
    second = torch.zeros(states_count, width, width, dtype=torch.float64, device=device)
    magnitude = torch.zeros(states_count, width, dtype=torch.float64, device=device)
    count = 0
    for batch in iterate_batches(corpus.ids, windows, batch_size, bos_id):
        keep = batch.positions > 0  # skip the first position of each window (attention sink)
        rows = torch.from_numpy(batch.rows[keep]).to(device)
        positions = torch.from_numpy(batch.positions[keep]).to(device)
        with ResidualCapture(model, select=(rows, positions)) as capture:
            container(input_ids=torch.from_numpy(batch.inputs).to(device), use_cache=False)
        for state, hidden in enumerate(capture.states):
            normalized = normalize(hidden, kind).double()
            first[state] += normalized.sum(0)
            second[state] += normalized.T @ normalized
            magnitude[state] += hidden.double().abs().sum(0)
        count += len(rows)
    if count < 2:
        raise ValueError("Calibration needs at least two tokens")
    norm_weight = getattr(find_parts(model).norm, "weight", None)
    gain = norm_weight.detach().float().to(device) if norm_weight is not None else torch.ones(width, device=device)
    sensitivity = torch.zeros(width, width, dtype=torch.float64, device=device)
    weight = readout.head.weight.detach()
    for start in range(0, weight.shape[0], 16384):  # vocabulary chunks: a float64 copy of a 152k head is ~5 GB
        rows_block = weight[start:start + 16384].float().to(device) * gain
        sensitivity += (rows_block.T @ rows_block).double()
    calibration = {"bulk": [], "on": [], "off": [], "read": [], "unread": [], "eigenvalues": [], "massive": [],
                   "on_rank": [], "on_share": [], "read_rank": [], "read_share": []}
    for state in range(states_count):
        mean_abs = magnitude[state] / count
        bulk = mean_abs <= massive_factor * mean_abs.median()
        mean = first[state] / count
        covariance = second[state] / count - torch.outer(mean, mean)
        on, off, values, on_rank, on_share = _subspace_bases(covariance, bulk, keep_fraction)
        read, unread, _, read_rank, read_share = _subspace_bases(sensitivity, bulk, keep_fraction)
        calibration["bulk"].append(bulk.float())
        calibration["massive"].append(torch.nonzero(~bulk, as_tuple=False)[:, 0].tolist())
        for key, value in (("on", on), ("off", off), ("read", read), ("unread", unread), ("eigenvalues", values),
                           ("on_rank", on_rank), ("on_share", on_share), ("read_rank", read_rank),
                           ("read_share", read_share)):
            calibration[key].append(value)
    calibration["tokens"] = count
    log(f"calibrated on {count} tokens; massive coordinates per state: {[len(m) for m in calibration['massive']]}; "
        f"on-manifold variance share per state: {[round(s, 3) for s in calibration['on_share']]}")
    return calibration


def calibration_summary(calibration):
    return {key: calibration[key] for key in ("massive", "on_rank", "on_share", "read_rank", "read_share", "tokens")}


def save_calibration(path, calibration, meta):
    torch.save({"calibration": dict(calibration), "meta": meta}, path)


def load_calibration(path, device):
    blob = torch.load(path, map_location=device, weights_only=True)
    return blob["calibration"], blob["meta"]


# ---------------------------------------------------------------------------------------------- running

def sink_mask(states, bos_offset, factor):
    """[B, T] True at attention-sink positions: position 0 (and BOS) plus tokens whose largest activation in
    any state after the embeddings exceeds `factor` times that window's median token."""
    peak = torch.stack([state.float().abs().amax(-1) for state in states[1:]]).amax(0)  # [B, T]
    sinks = peak > factor * peak.median(dim=-1, keepdim=True).values
    sinks[:, : 1 + bos_offset] = True
    return sinks


def _scatter(sums, condition, first_state, values, docs, n_docs):
    """Add per-token values [m, n] into per-document sums for states first_state..first_state+m-1."""
    per_doc = torch.zeros(values.shape[0], n_docs, dtype=torch.float64, device=values.device)
    per_doc.index_add_(1, docs, values.double())
    sums[condition, first_state:first_state + values.shape[0]] += per_doc.cpu().numpy()


def _scatter_tokens(sums, condition, values, docs, n_docs):
    per_doc = torch.zeros(n_docs, dtype=torch.float64, device=values.device)
    per_doc.index_add_(0, docs, values.double())
    sums[condition] += per_doc.cpu().numpy()


def _window_starts(batch):
    """Corpus index of each row's first token: the noise key, independent of batching and of the model."""
    starts = []
    for row in range(batch.inputs.shape[0]):
        selected = batch.rows == row
        first = np.argmin(batch.positions[selected])
        starts.append(int(batch.targets[selected][first] - batch.context[selected][first]))
    return starts


def _chunked_logp(readout, final, chunk):
    return [torch.log_softmax(readout(final[i:i + chunk]), dim=-1) for i in range(0, len(final), chunk)]


def _damage(readout, logp_parts, final, targets, chunk):
    parts = [output_damage(logp_parts[j], readout(final[i:i + chunk]), targets[i:i + chunk])
             for j, i in enumerate(range(0, len(final), chunk))]
    return [torch.cat(values) for values in zip(*parts)]


@torch.no_grad()
def run_perturbation(model, readout, corpus, windows, calibration, *, sites, families, deltas, seed, batch_size,
                     bos_id, replicas, min_context, sink_factor, chunk, device, fixed_sinks=None, log=print):
    """Inject every condition at every site and accumulate per-document sums of the metrics.

    fixed_sinks: {window start: sink positions} from a reference run, so every model variant measures and
    perturbs exactly the same tokens (trained-minus-control contrasts then compare one population).
    """
    check_layer_types(model.config)
    if replicas < 2 or replicas % 2:
        raise ValueError("replicas must be an even number >= 2 so antithetic pairs share a pass")
    unknown = sorted(set(families) - set(FAMILIES))
    if unknown:
        raise ValueError(f"Unknown families {unknown}; choose from {FAMILIES}")
    blocks_count, kind, width = n_blocks(model), norm_kind(model), hidden_width(model)
    states_count = blocks_count + 1
    if any(not 0 <= site < blocks_count for site in sites):
        raise ValueError(f"Injection sites must lie in 0..{blocks_count - 1}")
    parts_info = find_parts(model)
    container = parts_info.container
    segments = {site: make_segment(model, site + 1, blocks_count, trainable=False) for site in sites}
    conditions = conditions_for(sites, families, deltas)
    index_of = {(c["site"], c["family"], c["delta"], c["sign"]): i for i, c in enumerate(conditions)}
    n_docs = int(corpus.doc.max()) + 1
    state_sums = {key: np.zeros((len(conditions), states_count, n_docs)) for key in STATE_METRICS}
    output_sums = {key: np.zeros((len(conditions), n_docs)) for key in OUTPUT_METRICS}
    clean_sums = {key: np.zeros((len(sites), states_count, n_docs)) for key in ("signal", "content")}
    # Tokens at which log_off / log_on are defined; the report averages those metrics over these counts only.
    defined_sums = {key: np.zeros((len(conditions), states_count, n_docs)) for key in ("off", "on")}
    counts = np.zeros(n_docs)
    floor_max = np.zeros((len(sites), states_count))
    floor_sum = np.zeros((len(sites), states_count))
    floor_count = np.zeros(len(sites))
    bulk = [b.to(device) for b in calibration["bulk"]]
    shared = shared_direction(seed, width).to(device)
    bos_offset = 0 if bos_id is None else 1
    measured_total = 0
    sink_record_start, sink_record_position, window_record = [], [], []

    for step, batch in enumerate(iterate_batches(corpus.ids, windows, batch_size, bos_id)):
        inputs = torch.from_numpy(batch.inputs).to(device)
        with ResidualCapture(model) as capture:
            container(input_ids=inputs, use_cache=False)
        states = capture.states  # S x [B, T, d]
        starts = _window_starts(batch)
        if fixed_sinks is None:
            sinks = sink_mask(states, bos_offset, sink_factor)
        else:
            sinks = torch.zeros(inputs.shape, dtype=torch.bool, device=device)
            for row, start in enumerate(starts):
                if start not in fixed_sinks:
                    raise ValueError(f"Window starting at {start} is not in the reference sink masks; "
                                     "runs must use the same data, context, stride and --every")
                sinks[row, list(fixed_sinks[start])] = True
            sinks[:, : 1 + bos_offset] = True
        for row, start in enumerate(starts):
            window_record.append(start)
            for position in torch.nonzero(sinks[row], as_tuple=False)[:, 0].tolist():
                sink_record_start.append(start)
                sink_record_position.append(position)
        rows_all = torch.from_numpy(batch.rows).to(device)
        positions_all = torch.from_numpy(batch.positions).to(device)
        keep = (torch.from_numpy(batch.context).to(device) >= min_context) & ~sinks[rows_all, positions_all]
        rows, positions = rows_all[keep], positions_all[keep]
        n = len(rows)
        if n == 0:
            continue
        docs = torch.from_numpy(corpus.doc[batch.targets]).to(device)[keep].long()
        targets = torch.from_numpy(corpus.ids[batch.targets]).to(device)[keep]
        clean = [state[rows, positions] for state in states]
        counts += np.bincount(docs.cpu().numpy(), minlength=n_docs)
        measured_total += n
        noise_mask = (~sinks).float()[..., None]  # [B, T, 1]
        z, signs = window_noise(seed, starts, batch.inputs.shape[1], width)
        z, signs = z.to(device), signs.to(device)

        # Same-relative-size perturbation added at the last state, where no layers remain (context, not repair).
        last = blocks_count
        logp_last = _chunked_logp(readout, clean[last], chunk)
        last_scale = bulk_norm(clean[last], bulk[last])
        last_damage = {}
        for family in families:
            directions = unit_directions(family, z[rows, positions], shared, signs[rows, positions],
                                         family_basis(calibration, family, last), bulk[last])
            for delta in deltas:
                for sign in (1, -1):
                    eps = sign * delta * last_scale[:, None] * directions
                    kl, dnll, _ = _damage(readout, logp_last, clean[last].float() + eps, targets, chunk)
                    last_damage[(family, float(delta), sign)] = (kl, dnll)

        for site_index, site in enumerate(sites):
            for later in range(site, states_count):  # clean-only signal growth from the site onward
                growth = torch.log(bulk_norm(clean[later], bulk[later]).clamp_min(TINY)) \
                    - torch.log(bulk_norm(clean[site], bulk[site]).clamp_min(TINY))
                content = _log_norm(clean[later].float() @ calibration["on"][later]) \
                    - _log_norm(clean[site].float() @ calibration["on"][site])
                _scatter(clean_sums["signal"], site_index, later, growth[None], docs, n_docs)
                _scatter(clean_sums["content"], site_index, later, content[None], docs, n_docs)
            base = states[site]
            scale = bulk_norm(base, bulk[site])[..., None]  # [B, T, 1]
            site_conditions = [c for c in conditions if c["site"] == site]
            directions = {family: unit_directions(family, z, shared, signs, family_basis(calibration, family, site),
                                                  bulk[site]) for family in families}
            offs = [calibration["off"][k] for k in range(site, states_count)]
            ons = [calibration["on"][k] for k in range(site, states_count)]
            segment = segments[site]
            segment_blocks = getattr(segment, parts_info.blocks_attr)
            batch_rows = base.shape[0]
            for begin in range(0, len(site_conditions), replicas):
                group = site_conditions[begin:begin + replicas]
                copies = [base] + [(base.float() + c["sign"] * c["delta"] * scale * directions[c["family"]]
                                    * noise_mask).to(base.dtype) for c in group]
                select_rows = torch.cat([rows + g * batch_rows for g in range(len(copies))])
                select_positions = positions.repeat(len(copies))
                with ResidualCapture(blocks=segment_blocks, select=(select_rows, select_positions)) as seg_capture:
                    segment(inputs_embeds=torch.cat(copies), use_cache=False)
                outputs = [[state[g * n:(g + 1) * n] for state in seg_capture.states] for g in range(len(copies))]
                reference = outputs[0]  # in-batch clean replica: same kernels as the noisy rows
                for offset, (clean_state, replica_state) in enumerate(zip(clean[site:], reference)):
                    deviation = (replica_state.float() - clean_state.float()).norm(dim=-1) \
                        / bulk_norm(clean_state, bulk[site + offset]).clamp_min(TINY)
                    floor_max[site_index, site + offset] = max(floor_max[site_index, site + offset],
                                                               deviation.max().item())
                    floor_sum[site_index, site + offset] += deviation.sum().item()
                floor_count[site_index] += n
                logp_reference = _chunked_logp(readout, reference[-1], chunk)
                for g, condition in enumerate(group, start=1):
                    c = index_of[(condition["site"], condition["family"], condition["delta"], condition["sign"])]
                    metrics = propagation_metrics(reference, outputs[g], kind, off=offs, on=ons)
                    # Cross pairs have no intended component (the on and off bases are orthogonal at the site);
                    # whatever share they show is rounding of h + eps, so they are undefined by construction.
                    for name, family in (("off", "on"), ("on", "off")):
                        if condition["family"] == family:
                            metrics[f"log_{name}"].fill_(float("nan"))
                            metrics[f"log_{name}0"].fill_(float("nan"))
                    for key in STATE_METRICS:
                        values = metrics[key]
                        _scatter(state_sums[key], c, site, torch.where(torch.isfinite(values), values, 0.0), docs,
                                 n_docs)
                    for name in ("off", "on"):
                        _scatter(defined_sums[name], c, site, torch.isfinite(metrics[f"log_{name}"]).float(), docs,
                                 n_docs)
                    kl, dnll, flip = _damage(readout, logp_reference, outputs[g][-1], targets, chunk)
                    # Output dilution null: the realised injected perturbation carried unchanged to the last state.
                    injected = outputs[g][0].float() - reference[0].float()
                    kl_carry, dnll_carry, _ = _damage(readout, logp_reference, reference[-1].float() + injected,
                                                      targets, chunk)
                    kl_last, dnll_last = last_damage[(condition["family"], condition["delta"], condition["sign"])]
                    # Effective size as the next RMSNorm sees it: realised noise over the FULL norm, massive
                    # coordinates included (delta itself is relative to the bulk norm, so this can be smaller).
                    delta_eff = injected.norm(dim=-1) / reference[0].float().norm(dim=-1).clamp_min(TINY)
                    for key, values in (("kl", kl), ("dnll", dnll), ("flip", flip), ("kl_carry", kl_carry),
                                        ("dnll_carry", dnll_carry), ("kl_last", kl_last), ("dnll_last", dnll_last),
                                        ("delta_eff", delta_eff)):
                        _scatter_tokens(output_sums[key], c, values, docs, n_docs)
                    if condition["sign"] == -1:  # its + partner is the previous replica in this pass
                        nl = nonlinearity(outputs[g - 1][-1], outputs[g][-1], reference[-1])
                        for partner in (c - 1, c):
                            _scatter_tokens(output_sums["nl"], partner, nl, docs, n_docs)
        if step % 5 == 0:
            log(f"perturbed batch {step + 1}: {measured_total} measured tokens")

    if measured_total == 0:
        raise ValueError("No measured tokens; lower --min-context or use more data")
    arrays = {f"state_{key}": value for key, value in state_sums.items()}
    arrays.update({f"output_{key}": value for key, value in output_sums.items()})
    arrays.update({f"clean_{key}": value for key, value in clean_sums.items()})
    arrays.update({f"defined_{key}": value for key, value in defined_sums.items()})
    # Floor: in-batch clean replica versus the separate clean pass, relative to the bulk norm. Zero for
    # deterministic kernels; nonzero when batch composition changes kernels (bf16 tiling, MoE grouping).
    floor_mean = floor_sum / np.maximum(floor_count, 1)[:, None]
    arrays.update({"counts": counts, "floor_max": floor_max, "floor_mean": floor_mean,
                   "window_starts": np.asarray(window_record, dtype=np.int64),
                   "sink_starts": np.asarray(sink_record_start, dtype=np.int64),
                   "sink_positions": np.asarray(sink_record_position, dtype=np.int64)})
    meta = {"conditions": conditions, "sites": list(sites), "families": list(families),
            "deltas": [float(d) for d in deltas], "n_states": states_count, "norm_kind": kind, "seed": seed,
            "measured_tokens": int(measured_total), "documents": int((counts > 0).sum()),
            "floor_max": float(floor_max.max()), "sinks_fixed": fixed_sinks is not None,
            "calibration": calibration_summary(calibration)}
    return arrays, meta


def load_sinks(path):
    """{window start: set of sink positions} from an earlier run's perturb.npz."""
    with np.load(path) as data:
        sinks = {int(start): set() for start in data["window_starts"]}
        for start, position in zip(data["sink_starts"], data["sink_positions"]):
            sinks[int(start)].add(int(position))
    return sinks


def parse_sites(text, blocks_count):
    if text == "all":
        return list(range(blocks_count))
    sites = sorted({int(part) for part in text.split(",") if part.strip()})
    if not sites or sites[0] < 0 or sites[-1] >= blocks_count:
        raise ValueError(f"--sites must be 'all' or indices in 0..{blocks_count - 1}")
    return sites
