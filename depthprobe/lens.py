"""Tuned lens (Belrose et al., 2023, arXiv 2303.08112), reimplemented.

One affine translator per non-final state, h -> h + A h + b, initialized to the identity and
trained to minimize KL(p_final || p_layer) through the frozen readout. Reimplemented rather
than using the `tuned-lens` package, whose compatibility with current transformers is unverified.
"""

import math

import numpy as np
import torch
from torch import nn

from depthprobe.data import iterate_batches
from depthprobe.model import forward_states, hidden_width, n_blocks


class TunedLens(nn.Module):
    def __init__(self, n_states, width):
        super().__init__()
        self.translators = nn.ModuleList(nn.Linear(width, width) for _ in range(n_states - 1))
        for translator in self.translators:
            nn.init.zeros_(translator.weight)
            nn.init.zeros_(translator.bias)

    @property
    def n_states(self):
        return len(self.translators) + 1

    def forward(self, hidden, layer):
        hidden = hidden.float()
        if layer == len(self.translators):
            return hidden
        return hidden + self.translators[layer](hidden)


def save_lens(path, lens, meta):
    torch.save({"state_dict": lens.state_dict(), "meta": meta}, path)


def load_lens(path, device):
    blob = torch.load(path, map_location=device, weights_only=True)
    meta = blob["meta"]
    lens = TunedLens(meta["n_states"], meta["width"]).to(device)
    lens.load_state_dict(blob["state_dict"])
    lens.requires_grad_(False).eval()
    return lens, meta


def kl_to(logp_reference, logits):
    """KL(reference || softmax(logits)) per token, in nats."""
    logp = torch.log_softmax(logits, dim=-1)
    return (logp_reference.exp() * (logp_reference - logp)).sum(-1)


def _selected(states, batch, device):
    rows = torch.from_numpy(batch.rows).to(device)
    positions = torch.from_numpy(batch.positions).to(device)
    return [state[rows, positions] for state in states]


def train_lens(model, readout, ids, windows, *, batch_size, bos_id, chunk, lr, warmup_fraction, device,
               log=print, log_every=20):
    n_states = n_blocks(model) + 1
    lens = TunedLens(n_states, hidden_width(model)).to(device)
    optimizer = torch.optim.AdamW(lens.parameters(), lr=lr, weight_decay=0.0)
    batches = list(iterate_batches(ids, windows, batch_size, bos_id))
    total = len(batches)
    warmup = max(1, math.ceil(warmup_fraction * total))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup) * max(0.0, 1.0 - step / total))
    history = []
    for step, batch in enumerate(batches):
        states, _ = forward_states(model, torch.from_numpy(batch.inputs).to(device))
        hidden = _selected(states, batch, device)
        del states
        count = hidden[0].shape[0]
        layer_kl = torch.zeros(n_states - 1)
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, count, chunk):
            part = slice(start, start + chunk)
            with torch.no_grad():
                logp_final = torch.log_softmax(readout(hidden[-1][part]), dim=-1)
            for layer in range(n_states - 1):
                kl = kl_to(logp_final, readout(lens(hidden[layer][part], layer)))
                # Mean over tokens and layers; backward per layer keeps only one vocab-sized graph alive.
                (kl.sum() / (count * (n_states - 1))).backward()
                layer_kl[layer] += kl.detach().sum().cpu()
        torch.nn.utils.clip_grad_norm_(lens.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        mean_kl = (layer_kl / count).mean().item()
        history.append(mean_kl)
        if step % log_every == 0 or step == total - 1:
            log(f"lens step {step + 1}/{total}: mean KL {mean_kl:.4f} nats")
    return lens, history


@torch.no_grad()
def evaluate_lens(model, readout, lens, ids, windows, *, batch_size, bos_id, chunk, device):
    """Mean KL(p_final || p_layer) per state for the raw logit lens and the tuned lens."""
    n_states = n_blocks(model) + 1
    raw, tuned, count = np.zeros(n_states), np.zeros(n_states), 0
    for batch in iterate_batches(ids, windows, batch_size, bos_id):
        states, _ = forward_states(model, torch.from_numpy(batch.inputs).to(device))
        hidden = _selected(states, batch, device)
        del states
        for start in range(0, hidden[0].shape[0], chunk):
            part = slice(start, start + chunk)
            logp_final = torch.log_softmax(readout(hidden[-1][part]), dim=-1)
            for layer in range(n_states):
                raw[layer] += kl_to(logp_final, readout(hidden[layer][part])).sum().item()
                tuned[layer] += kl_to(logp_final, readout(lens(hidden[layer][part], layer))).sum().item()
        count += hidden[0].shape[0]
    return {"tokens": count, "raw_kl": (raw / count).tolist(), "tuned_kl": (tuned / count).tolist()}
