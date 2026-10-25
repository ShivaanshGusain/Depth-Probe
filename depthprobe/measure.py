"""E0 audit and per-token, per-state metrics for E1 (lens curves), E2 (convergence to self)
and E3 (per-token monotonicity). Per-token values are kept so E3/E4 can be computed later."""

import math

import numpy as np
import torch
import torch.nn.functional as F

from depthprobe.data import iterate_batches
from depthprobe.lens import kl_to
from depthprobe.model import ResidualCapture, forward_states, n_blocks


def layer_metrics(logits, logp_final, final_argmax, targets):
    logits = logits.float()
    logp = torch.log_softmax(logits, dim=-1)
    predicted = logits.argmax(-1)
    target_logit = logits.gather(-1, targets[:, None])
    return {
        "nll": -logp.gather(-1, targets[:, None])[:, 0],  # reference B: the data
        "kl": (logp_final.exp() * (logp_final - logp)).sum(-1),  # reference A: KL(p_L || p_l)
        "top1": predicted == targets,
        "agree": predicted == final_argmax,
        "rank": (logits > target_logit).sum(-1),
        "entropy": -(logp.exp() * logp).sum(-1),
    }


def hidden_metrics(hidden, final):
    hidden, final = hidden.float(), final.float()
    return {
        "cos": F.cosine_similarity(hidden, final, dim=-1),
        "rel_dist": (final - hidden).norm(dim=-1) / final.norm(dim=-1),
        "norm": hidden.norm(dim=-1),
    }


class Accumulator:
    def __init__(self):
        self.per_state = {}
        self.per_token = {}

    def state(self, key, layer, values):
        self.per_state.setdefault(key, {}).setdefault(layer, []).append(values.cpu().numpy())

    def token(self, key, values):
        if isinstance(values, torch.Tensor):
            values = values.cpu().numpy()
        self.per_token.setdefault(key, []).append(values)

    def arrays(self):
        arrays = {key: np.concatenate(parts) for key, parts in self.per_token.items()}
        for key, layers in self.per_state.items():
            arrays[key] = np.stack([np.concatenate(layers[layer]) for layer in sorted(layers)])
        for key, value in arrays.items():
            if value.dtype == np.float64:
                arrays[key] = value.astype(np.float32)
            elif value.dtype == np.int64 and key.endswith("rank"):
                arrays[key] = value.astype(np.int32)
        return arrays


def _to_device(batch, device):
    rows = torch.from_numpy(batch.rows).to(device)
    positions = torch.from_numpy(batch.positions).to(device)
    return torch.from_numpy(batch.inputs).to(device), rows, positions


@torch.no_grad()
def check(model, readout, corpus, windows, *, batch_size, bos_id, chunk, device, log=print):
    """E0: label alignment against the library's own loss, and readout(final state) against native logits."""
    native_sum, readout_sum, error_sum, count = 0.0, 0.0, 0.0, 0
    worst_nll_error, worst_loss_gap, worst_logit_error = 0.0, 0.0, 0.0
    for batch in iterate_batches(corpus.ids, windows, batch_size, bos_id):
        inputs, rows, positions = _to_device(batch, device)
        targets = torch.from_numpy(corpus.ids[batch.targets]).to(device)
        if not torch.equal(inputs[rows, positions + 1], targets):
            raise RuntimeError("Scoring positions do not line up with target tokens")
        labels = torch.full_like(inputs, -100)
        labels[rows, positions + 1] = targets
        with ResidualCapture(model) as capture:
            output = model(input_ids=inputs, labels=labels, use_cache=False)
        if len(capture.states) != n_blocks(model) + 1:
            raise RuntimeError("Did not capture every block output")
        final = capture.states[-1][rows, positions]
        native = output.logits[rows, positions]
        batch_native = 0.0
        for start in range(0, len(targets), chunk):
            part = slice(start, start + chunk)
            native_logits = native[part].float()
            readout_logits = readout(final[part])
            native_nll = F.cross_entropy(native_logits, targets[part], reduction="none")
            readout_nll = F.cross_entropy(readout_logits, targets[part], reduction="none")
            nll_error = (native_nll - readout_nll).abs()
            worst_nll_error = max(worst_nll_error, nll_error.max().item())
            error_sum += nll_error.sum().item()
            worst_logit_error = max(worst_logit_error, (native_logits - readout_logits).abs().max().item())
            batch_native += native_nll.sum().item()
            readout_sum += readout_nll.sum().item()
        worst_loss_gap = max(worst_loss_gap, abs(output.loss.item() - batch_native / len(targets)))
        native_sum += batch_native
        count += len(targets)
        log(f"checked {count} tokens: running native PPL {math.exp(native_sum / count):.4f}")
    return {
        "scored_tokens": count,
        "native_ppl": math.exp(native_sum / count),
        "readout_ppl": math.exp(readout_sum / count),
        "max_abs_final_nll_error": worst_nll_error,
        # In bfloat16 a single rounding step near a logit of 30 is 0.125 nats, so the mean shows whether
        # a large maximum is isolated rounding or a systematic mismatch.
        "mean_abs_final_nll_error": error_sum / count,
        "max_abs_final_logit_error": worst_logit_error,
        "max_abs_library_loss_gap": worst_loss_gap,
    }


@torch.no_grad()
def measure(model, readout, corpus, windows, *, batch_size, bos_id, chunk, device, lens=None,
            unigram_logp=None, max_final_nll_error=0.1, log=print, log_every=10):
    n_states = n_blocks(model) + 1
    accumulator = Accumulator()
    unigram = None if unigram_logp is None else torch.from_numpy(unigram_logp).float().to(device)
    worst_error, error_sum, scored = 0.0, 0.0, 0
    for step, batch in enumerate(iterate_batches(corpus.ids, windows, batch_size, bos_id)):
        inputs, rows, positions = _to_device(batch, device)
        states, output = forward_states(model, inputs)
        hidden = [state[rows, positions] for state in states]
        native = output.logits[rows, positions]
        del states, output
        targets = torch.from_numpy(corpus.ids[batch.targets]).to(device)
        for start in range(0, len(targets), chunk):
            part = slice(start, start + chunk)
            target = targets[part]
            final = hidden[-1][part]
            native_nll = F.cross_entropy(native[part].float(), target, reduction="none")
            final_logits = readout(final)
            logp_final = torch.log_softmax(final_logits, dim=-1)
            final_argmax = final_logits.argmax(-1)
            error = (F.nll_loss(logp_final, target, reduction="none") - native_nll).abs()
            worst_error = max(worst_error, error.max().item())
            error_sum += error.sum().item()
            if worst_error > max_final_nll_error:
                raise RuntimeError(f"Readout of the final state differs from native logits by {worst_error:.4f} nats")
            accumulator.token("native_nll", native_nll)
            for layer in range(n_states):
                state = hidden[layer][part]
                for key, value in layer_metrics(readout(state), logp_final, final_argmax, target).items():
                    accumulator.state(f"raw_{key}", layer, value)
                if lens is not None:
                    tuned = layer_metrics(readout(lens(state, layer)), logp_final, final_argmax, target)
                    for key, value in tuned.items():
                        accumulator.state(f"tuned_{key}", layer, value)
                for key, value in hidden_metrics(state, final).items():
                    accumulator.state(key, layer, value)
            if unigram is not None:
                accumulator.token("unigram_nll", -unigram[target])
                accumulator.token("unigram_kl", kl_to(logp_final, unigram.expand_as(logp_final)))
        accumulator.token("target_id", corpus.ids[batch.targets])
        accumulator.token("target_index", batch.targets)
        accumulator.token("doc", corpus.doc[batch.targets])
        accumulator.token("context", batch.context)
        scored += len(targets)
        if step % log_every == 0:
            log(f"measured batch {step + 1}: {scored} tokens")
    arrays = accumulator.arrays()
    stats = {
        "scored_tokens": int(arrays["doc"].size),
        "n_states": n_states,
        "native_ppl": float(np.exp(arrays["native_nll"].astype(np.float64).mean())),
        "max_abs_final_nll_error": worst_error,
        "mean_abs_final_nll_error": error_sum / max(scored, 1),
    }
    return arrays, stats


def unigram_logp(ids, vocab_size, alpha=1.0):
    """Add-alpha unigram log-probabilities: the token-frequency baseline (E5)."""
    counts = np.bincount(ids, minlength=vocab_size).astype(np.float64) + alpha
    return np.log(counts / counts.sum()).astype(np.float32)
