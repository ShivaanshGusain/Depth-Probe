"""Local segment CE fine-tuning; the untouched suffix is restored for evaluation.

Blocks are numbered 1..L, inclusive. Cached reference hidden states reconstruct
exact untuned-lens probabilities through the frozen float32 readout, avoiding
vocabulary-sized caches. Only the segment and readout live on the training device.
"""

import copy
import csv
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from depthprobe.data import iterate_batches
from depthprobe.lens import kl_to
from depthprobe.model import find_parts, hidden_width, n_blocks
from depthprobe.provenance import write_json


def autocast(device, amp):
    if amp == "none":
        return nullcontext()
    if amp != "bfloat16" or torch.device(device).type != "cuda":
        raise ValueError("bfloat16 AMP requires a CUDA device; otherwise use --amp none")
    return torch.autocast("cuda", dtype=torch.bfloat16)


def memory_gb(device):
    return torch.cuda.max_memory_allocated(device) / 2**30 if torch.device(device).type == "cuda" else 0.0


def token_count(windows):
    return sum(window.end - window.first_target for window in windows)


class CachedWindows(Dataset):
    """Memory-map disk shards, selecting individual windows for independent rebatching."""

    def __init__(self, path, meta):
        self.path = Path(path)
        self.shards = meta["shards"]
        sizes = meta.get("shard_sizes")
        if sizes is None:  # Compatibility with the first cache format.
            sizes = []
            for filename in self.shards:
                batch = torch.load(self.path / filename, weights_only=True, mmap=True)
                sizes.append(list(batch["prefix"].shape[:2]))
        self.index = [(shard, row) for shard, (rows, _) in enumerate(sizes) for row in range(rows)]
        self.lengths = [sizes[shard][1] for shard, _ in self.index]
        self._loaded_index, self._loaded = None, None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        shard, row = self.index[index]
        if shard != self._loaded_index:
            self._loaded = torch.load(self.path / self.shards[shard], weights_only=True, mmap=True)
            self._loaded_index = shard
        batch = self._loaded
        keep = batch["rows"] == row
        return {"prefix": batch["prefix"][row], "positions": batch["positions"][keep],
                "targets": batch["targets"][keep], "reference_exit": batch["reference_exit"][keep]}


def collate_windows(windows):
    """Equal-length groups need no padding; only recorded target positions enter the loss."""
    return {"prefix": torch.stack([window["prefix"] for window in windows]),
            "rows": torch.cat([torch.full_like(window["positions"], row) for row, window in enumerate(windows)]),
            **{key: torch.cat([window[key] for window in windows])
               for key in ("positions", "targets", "reference_exit")}}


def training_batches(dataset, batch_size, rng):
    groups = {}
    for index, length in enumerate(dataset.lengths):
        groups.setdefault(length, []).append(index)
    batches = []
    for indices in groups.values():
        indices = rng.permutation(indices).tolist()
        batches.extend(indices[start:start + batch_size] for start in range(0, len(indices), batch_size))
    rng.shuffle(batches)
    return batches


def validate_range(model, first, last):
    if not 1 <= first <= last <= n_blocks(model):
        raise ValueError(f"Need 1 <= first <= last <= {n_blocks(model)} (inclusive block numbers)")


def training_postfix(row, regularizer):
    """Do not present an unmeasured L2 diagnostic as zero drift in KL/CE-only runs."""
    postfix = {"CE": f"{row['cross_entropy']:.3f}", "acc": f"{row['accuracy']:.1%}",
               "lr": f"{row['lr']:.2g}", "peak_GB": f"{row['peak_gpu_gb']:.1f}"}
    if regularizer in ("l2", "both"):
        postfix["L2"] = f"{float(row.get('l2_penalty', row['regularizer'])):.3g}"
    if regularizer in ("kl", "both"):
        postfix["KL"] = f"{float(row.get('kl_penalty', row['regularizer'])):.3g}"
    if row["max_l2_deviation_std"] is not None:
        postfix["max_dev_std"] = f"{row['max_l2_deviation_std']:.3g}"
    return postfix


class RunningMoments:
    """Stable per-feature sample moments, computed on original scored activations."""

    def __init__(self, width):
        self.mean = torch.zeros(width, dtype=torch.float64)
        self.squares = torch.zeros_like(self.mean)
        self.count = 0

    def update(self, values):
        values = values.detach().cpu().double()
        count = len(values)
        mean = values.mean(0)
        delta = mean - self.mean
        self.squares += ((values - mean) ** 2).sum(0) + delta.square() * self.count * count / (self.count + count)
        self.mean += delta * count / (self.count + count)
        self.count += count

    def stats(self, floor):
        if self.count < 2:
            raise ValueError("Need at least two scored tokens for activation sample STD")
        std = (self.squares / (self.count - 1)).clamp_min(0).sqrt().float()
        return {"mean": self.mean.float(), "sample_std": std, "scale": std.clamp_min(floor)}


def parameter_free_rms(hidden, eps=1e-6):
    """RMS normalization without learned gain/bias, centering, or model norm parameters."""
    if eps <= 0:
        raise ValueError("RMS epsilon must be positive")
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        hidden = hidden.float()
        return hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + eps)


def l2_deviation(hidden, reference, scale, space, rms_eps=1e-6):
    """Per-activation changes in original-STD units, optionally after parameter-free RMS."""
    if space not in ("residual", "rms"):
        raise ValueError("Unknown L2 space")
    # Match the FP32 normalization used to collect the reference moments,
    # independently of mixed-precision head/segment compute.
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        hidden, reference = hidden.float(), reference.float()
        if space == "rms":
            hidden = parameter_free_rms(hidden, rms_eps)
            with torch.no_grad():
                reference = parameter_free_rms(reference, rms_eps)
        return (hidden - reference) / scale


def segment_step(segment, readout, batch, optimizer, parameters, scale, *, device, amp, chunk, regularizer, strength,
                 l2_space="residual", rms_eps=1e-6, kl_strength=None):
    """One optimizer update, shared by training and GPU throughput benchmarking.

    ``both`` adds L2 and KL; ``strength`` weights L2 and ``kl_strength`` weights KL.
    """
    if regularizer not in ("l2", "kl", "both", "none"):
        raise ValueError("Unknown regularizer")
    kl_strength = strength if kl_strength is None else kl_strength
    optimizer.zero_grad(set_to_none=True)
    with autocast(device, amp):
        output = segment(inputs_embeds=batch["prefix"], use_cache=False, return_dict=True)
    selected = output.last_hidden_state[batch["rows"], batch["positions"]]
    # Accumulate vocabulary-chunk gradients at a leaf, then traverse the segment once.
    leaf = selected.detach().float().requires_grad_(True)
    n = len(leaf)
    metrics = torch.zeros(4, device=device)  # CE, L2, KL, correct
    max_deviation = torch.zeros((), device=device)
    for start in range(0, n, chunk):
        part = slice(start, start + chunk)
        hidden = leaf[part]
        with autocast(device, amp):
            logits = readout(hidden)
        logits = logits.float()
        targets = batch["targets"][part]
        ce = F.cross_entropy(logits, targets, reduction="sum")
        reference = batch["reference_exit"][part].float()
        l2 = kl = hidden.new_zeros(())
        if regularizer in ("l2", "both"):
            deviation = l2_deviation(hidden, reference, scale, l2_space, rms_eps)
            l2 = deviation.square().mean(-1).sum()
            max_deviation = torch.maximum(max_deviation, deviation.detach().abs().amax())
        if regularizer in ("kl", "both"):
            with torch.no_grad(), autocast(device, amp):
                logp_original = readout(reference).float().log_softmax(-1)
            kl = kl_to(logp_original, logits).sum()
        ((ce + strength * l2 + kl_strength * kl) / n).backward()
        metrics += torch.stack((ce.detach(), l2.detach(), kl.detach(),
                                (logits.argmax(-1) == targets).sum()))
    selected.backward(leaf.grad.to(selected.dtype))
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    values = torch.cat((metrics, grad_norm.detach().reshape(1), max_deviation.reshape(1))).cpu().tolist()
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("Nonfinite loss/gradient; refusing to update weights")
    optimizer.step()
    return {"cross_entropy": values[0], "l2": values[1], "kl": values[2], "correct": values[3],
            "grad_norm": values[4],
            "max_l2_deviation_std": values[5] if regularizer in ("l2", "both") else None}


class ZeroPositions(nn.Module):
    """GPT-2 prefix states already include positional embeddings."""

    def __init__(self, width, dtype):
        super().__init__()
        self.width = width
        self.register_buffer("anchor", torch.empty(0, dtype=dtype), persistent=False)

    def forward(self, positions):
        return torch.zeros((*positions.shape, self.width), device=positions.device, dtype=self.anchor.dtype)


def make_segment(model, first, last, trainable=True):
    """Reuse native backbone forward/masks/rotary embeddings, sharing only selected blocks.

    The shallow backbone shell owns a separate module dictionary; changing its
    embeddings/norm/block list does not change the full model's structure.
    trainable=False leaves the shared blocks' requires_grad flags as they are (inference use).
    """
    validate_range(model, first, last)
    parts = find_parts(model)
    segment = copy.copy(parts.container)
    segment._modules = parts.container._modules.copy()
    segment._buffers = parts.container._buffers.copy()
    segment._parameters = parts.container._parameters.copy()
    segment.config = copy.deepcopy(segment.config)
    setattr(segment, parts.blocks_attr, nn.ModuleList(list(parts.blocks)[first - 1:last]))
    if parts.family == "gpt2":
        segment.wte = nn.Identity()
        segment.wpe = ZeroPositions(hidden_width(model), parts.container.wpe.weight.dtype)
        segment.ln_f = nn.Identity()
    elif parts.family == "gpt_neox":
        segment.set_input_embeddings(nn.Identity())  # embed_in, under whatever name this version uses
        segment.final_layer_norm = nn.Identity()
    else:
        segment.embed_tokens = nn.Identity()
        segment.norm = nn.Identity()
    # Copy non-block modules (e.g. rotary buffers/dropout) rather than moving
    # shared prefix/suffix infrastructure when the segment changes device.
    for name, module in list(segment._modules.items()):
        if name != parts.blocks_attr:
            segment._modules[name] = copy.deepcopy(module)
    if trainable:
        segment.requires_grad_(False)
        getattr(segment, parts.blocks_attr).requires_grad_(True)
    else:
        for name, module in segment._modules.items():
            if name != parts.blocks_attr:
                module.requires_grad_(False)
    # Disable dropout for deterministic reference matching, without disabling gradients.
    return segment.eval()


@torch.no_grad()
def cache_training(model, corpus, windows, path, *, first, last, batch_size, bos_id,
                   device, chunk, std_floor=1e-4, log=print, progress=True, amp="none", rms_eps=1e-6):
    """Disk shards; per-feature sample STD over all scored segment-exit activations."""
    validate_range(model, first, last)
    if std_floor <= 0 or chunk <= 0 or rms_eps <= 0:
        raise ValueError("std_floor, chunk, and RMS epsilon must be positive")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    parts = find_parts(model)
    blocks = parts.blocks
    residual_moments = RunningMoments(hidden_width(model))
    rms_moments = RunningMoments(hidden_width(model))
    count, shards, shard_sizes = 0, [], []
    captured = {}

    def entering(module, args, kwargs):
        hidden = args[0] if args else kwargs["hidden_states"]
        captured["prefix"] = hidden.detach().cpu().clone()

    def capture(name):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[name] = hidden.detach().cpu().clone()
        return hook

    handles = [blocks[first - 1].register_forward_pre_hook(entering, with_kwargs=True),
               blocks[last - 1].register_forward_hook(capture("exit")),
               blocks[-1].register_forward_hook(capture("final"))]
    bar = tqdm(total=token_count(windows), desc="Cache original activations", unit="tok", unit_scale=True,
               disable=not progress, dynamic_ncols=True)
    try:
        for step, batch in enumerate(iterate_batches(corpus.ids, windows, batch_size, bos_id)):
            # Native backbone includes the native final norm; chunk the native head
            # instead of materializing [batch, context, vocabulary] inference logits.
            with autocast(device, amp):
                output = parts.container(input_ids=torch.from_numpy(batch.inputs).to(device),
                                         use_cache=False, return_dict=True)
            rows, positions = torch.from_numpy(batch.rows), torch.from_numpy(batch.positions)
            targets = torch.from_numpy(corpus.ids[batch.targets])
            reference = captured["exit"][rows, positions]
            residual_moments.update(reference)
            for start in range(0, len(reference), chunk):
                rms_moments.update(parameter_free_rms(reference[start:start + chunk], rms_eps))
            count += len(reference)
            native_nll, native_top1 = [], []
            for start in range(0, len(targets), chunk):
                part = slice(start, start + chunk)
                with autocast(device, amp):
                    logits = parts.head(output.last_hidden_state[rows[part].to(device), positions[part].to(device)]).float()
                native_nll.append(F.cross_entropy(logits, targets[part].to(device), reduction="none").cpu())
                native_top1.append(logits.argmax(-1).cpu())
            shard = {"prefix": captured["prefix"], "reference_exit": reference,
                     "reference_final": captured["final"][rows, positions],
                     "rows": rows, "positions": positions, "targets": targets,
                     "target_index": torch.from_numpy(batch.targets),
                     "doc": torch.from_numpy(corpus.doc[batch.targets]),
                     "context": torch.from_numpy(batch.context),
                     "native_nll": torch.cat(native_nll), "native_prediction": torch.cat(native_top1)}
            filename = f"batch-{step:06d}.pt"
            torch.save(shard, path / filename)
            shards.append(filename)
            shard_sizes.append(list(batch.inputs.shape))
            captured.clear()
            del output
            bar.update(len(targets))
            bar.set_postfix(shards=len(shards), peak_GB=f"{memory_gb(device):.1f}", refresh=False)
    finally:
        bar.close()
        for handle in handles:
            handle.remove()
    torch.save({**residual_moments.stats(std_floor), "rms": rms_moments.stats(std_floor), "rms_eps": rms_eps},
               path / "activation_stats.pt")
    meta = {"first_block": first, "last_block": last, "tokens": count, "shards": shards, "shard_sizes": shard_sizes,
             "std_floor": std_floor, "std_definition": "per-feature sample STD over scored segment-exit tokens",
             "rms_std_definition": "per-feature sample STD after parameter-free per-token RMS normalization, float32",
             "rms_eps": rms_eps, "rms_normalization": "h / sqrt(mean_feature(h^2) + eps); no learned parameters",
            "amp": amp, "boundary_policy": "continuous stream; cross-document attention allowed; no EOS inserted",
            "references": "original exit/final hidden states; decode with original frozen norm/head"}
    write_json(path / "cache.json", meta)
    log(f"Cached {count} scored tokens in {len(shards)} shards")
    return meta


def train_segment(segment, readout, cache_path, meta, *, device, epochs, lr, strength,
                  regularizer, chunk, seed, log=print, batch_size=None, workers=0, progress=True,
                  amp="none", lr_schedule="none", warmup_fraction=0.05, log_path=None, log_every=1,
                  l2_space="residual", rms_eps=1e-6, kl_strength=None, train_head=False):
    kl_strength = strength if kl_strength is None else kl_strength
    if epochs < 1 or lr <= 0 or strength < 0 or kl_strength < 0 or chunk < 1:
        raise ValueError("Need epochs >= 1, lr > 0, nonnegative strengths, chunk >= 1")
    if regularizer not in ("l2", "kl", "both", "none"):
        raise ValueError("Unknown regularizer")
    if train_head and regularizer != "none":
        raise ValueError("Trainable head requires regularizer=none; reference readout must stay frozen otherwise")
    if l2_space not in ("residual", "rms") or rms_eps <= 0:
        raise ValueError("Unknown L2 space")
    if workers < 0 or log_every < 1 or not 0 <= warmup_fraction < 1:
        raise ValueError("Need workers >= 0, log_every >= 1, 0 <= warmup_fraction < 1")
    if lr_schedule not in ("none", "cosine"):
        raise ValueError("Unknown LR schedule")
    cache_path = Path(cache_path)
    dataset = CachedWindows(cache_path, meta)
    batch_size = batch_size or meta.get("shard_sizes", [[1]])[0][0]
    if batch_size < 1 or not len(dataset):
        raise ValueError("Need a positive batch size and nonempty training cache")
    segment.to(device).eval()
    readout.to(device).eval().requires_grad_(False)
    if train_head:
        readout.head.requires_grad_(True)
    stats = torch.load(cache_path / "activation_stats.pt", weights_only=True)
    if regularizer in ("l2", "both") and l2_space == "rms":
        if "rms" not in stats or stats.get("rms_eps") != rms_eps:
            raise ValueError("Cache lacks matching parameter-free RMS statistics; regenerate it with the requested RMS epsilon")
        stats = stats["rms"]
    scale = stats["scale"].to(device)
    parameters = [p for p in segment.parameters() if p.requires_grad]
    parameters += [p for p in readout.parameters() if p.requires_grad]
    cuda = torch.device(device).type == "cuda"
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.0, fused=cuda)
    rng = np.random.default_rng(seed)
    history = []
    # Fail before any update if an architecture/library version cannot faithfully
    # execute the extracted segment (position handling is especially important).
    probe = torch.load(cache_path / meta["shards"][0], weights_only=True)
    with torch.no_grad(), autocast(device, amp):
        actual = segment(inputs_embeds=probe["prefix"].to(device), use_cache=False).last_hidden_state
        actual = actual[probe["rows"].to(device), probe["positions"].to(device)].float()
        tolerance = 1e-4 if probe["prefix"].dtype == torch.float32 and amp == "none" else 5e-2
        torch.testing.assert_close(actual, probe["reference_exit"].to(device).float(), atol=tolerance, rtol=tolerance,
                                   msg="Extracted segment differs from cached original behavior; refusing to train")
    del probe, actual
    steps_per_epoch = len(training_batches(dataset, batch_size, np.random.default_rng(seed)))
    total_steps, global_step, total_tokens = epochs * steps_per_epoch, 0, 0
    warmup_steps = math.ceil(total_steps * warmup_fraction) if lr_schedule != "none" else 0
    started = time.perf_counter()
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    fields = ["step", "epoch", "tokens", "batch_tokens", "cross_entropy", "accuracy", "regularizer",
              "l2_penalty", "kl_penalty", "objective", "lr", "grad_norm", "tokens_per_second", "peak_gpu_gb",
              "max_l2_deviation_std"]
    logfile = open(log_path, "w", newline="") if log_path else nullcontext()
    with logfile as stream:
        writer = csv.DictWriter(stream, fieldnames=fields) if stream else None
        if writer:
            writer.writeheader()
        for epoch in range(epochs):
            ce_sum, l2_sum, kl_sum, correct, count = 0.0, 0.0, 0.0, 0, 0
            max_deviation = 0.0 if regularizer in ("l2", "both") else None
            loader = DataLoader(dataset, batch_sampler=training_batches(dataset, batch_size, rng),
                                collate_fn=collate_windows, num_workers=workers, pin_memory=cuda,
                                **({"prefetch_factor": 2} if workers else {}))
            bar = tqdm(total=meta["tokens"], desc=f"Train {epoch + 1}/{epochs}", unit="tok", unit_scale=True,
                       disable=not progress, dynamic_ncols=True)
            try:
                for batch in loader:
                    if lr_schedule == "cosine":
                        factor = min(1.0, (global_step + 1) / max(1, warmup_steps))
                        if global_step >= warmup_steps:
                            fraction = (global_step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
                            factor = 0.5 * (1 + math.cos(math.pi * fraction))
                        for group in optimizer.param_groups:
                            group["lr"] = lr * factor
                    batch = {key: value.to(device, non_blocking=cuda) for key, value in batch.items()}
                    n = len(batch["targets"])
                    values = segment_step(segment, readout, batch, optimizer, parameters, scale,
                                          device=device, amp=amp, chunk=chunk, regularizer=regularizer, strength=strength,
                                          l2_space=l2_space, rms_eps=rms_eps, kl_strength=kl_strength)
                    global_step += 1
                    total_tokens += n
                    ce_sum += values["cross_entropy"]
                    l2_sum += values["l2"]
                    kl_sum += values["kl"]
                    correct += values["correct"]
                    count += n
                    if values["max_l2_deviation_std"] is not None:
                        max_deviation = max(max_deviation, values["max_l2_deviation_std"])
                    row = {
                        "step": global_step, "epoch": epoch + 1, "tokens": total_tokens, "batch_tokens": n,
                        "cross_entropy": values["cross_entropy"] / n, "accuracy": values["correct"] / n,
                        "regularizer": (values["l2"] + values["kl"]) / n,
                        "l2_penalty": values["l2"] / n if regularizer in ("l2", "both") else None,
                        "kl_penalty": values["kl"] / n if regularizer in ("kl", "both") else None,
                        "objective": (values["cross_entropy"] + strength * values["l2"]
                                      + kl_strength * values["kl"]) / n,
                        "lr": optimizer.param_groups[0]["lr"], "grad_norm": values["grad_norm"],
                        "tokens_per_second": total_tokens / (time.perf_counter() - started),
                        "peak_gpu_gb": memory_gb(device),
                        "max_l2_deviation_std": values["max_l2_deviation_std"],
                    }
                    if writer:
                        writer.writerow({key: "" if value is None else value for key, value in row.items()})
                        stream.flush()
                    bar.update(n)
                    if global_step % log_every == 0:
                        bar.set_postfix(training_postfix(row, regularizer), refresh=False)
            finally:
                bar.close()
            record = {"epoch": epoch + 1, "tokens": count, "cross_entropy": ce_sum / count,
                      "accuracy": correct / count,
                      "regularizer": (l2_sum + kl_sum) / count,
                      "l2_penalty": l2_sum / count if regularizer in ("l2", "both") else None,
                      "kl_penalty": kl_sum / count if regularizer in ("kl", "both") else None,
                      "reg_strength": strength, "kl_strength": kl_strength if regularizer in ("kl", "both") else None,
                      "l2_space": l2_space if regularizer in ("l2", "both") else None,
                      "max_l2_deviation_std": max_deviation,
                      "optimizer_steps": global_step, "tokens_per_second": total_tokens / (time.perf_counter() - started),
                      "peak_gpu_gb": memory_gb(device)}
            history.append(record)
            log(f"segment training: {record}")
    return history


@torch.no_grad()
def evaluate_full(model, corpus, windows, *, batch_size, bos_id, device, chunk,
                  progress=True, amp="none", description="Evaluate full model", last_block=None):
    """Held-out metrics; optionally omit blocks after a 1-based inclusive exit.

    The partial path retains the current final norm/head and prefix positions.
    A shallow backbone shell avoids changing the original model's block list.
    """
    model.eval()
    parts = find_parts(model)
    backbone = parts.container
    if last_block is not None:
        validate_range(model, 1, last_block)
        backbone = copy.copy(parts.container)
        backbone._modules = parts.container._modules.copy()
        setattr(backbone, parts.blocks_attr, nn.ModuleList(list(parts.blocks)[:last_block]))
    arrays = {key: [] for key in ("nll", "correct", "prediction", "target_id", "target_index", "doc")}
    count, nll_sum, correct = 0, 0.0, 0
    with tqdm(total=token_count(windows), desc=description, unit="tok", unit_scale=True,
              disable=not progress, dynamic_ncols=True) as bar:
        for batch in iterate_batches(corpus.ids, windows, batch_size, bos_id):
            with autocast(device, amp):
                output = backbone(input_ids=torch.from_numpy(batch.inputs).to(device),
                                         use_cache=False, return_dict=True)
            for start in range(0, len(batch.targets), chunk):
                part = slice(start, start + chunk)
                indices = batch.targets[part]
                with autocast(device, amp):
                    logits = parts.head(output.last_hidden_state[torch.from_numpy(batch.rows[part]).to(device),
                                                                torch.from_numpy(batch.positions[part]).to(device)]).float()
                targets = torch.from_numpy(corpus.ids[indices]).to(device)
                prediction = logits.argmax(-1)
                values = {"nll": F.cross_entropy(logits, targets, reduction="none").cpu().numpy(),
                          "correct": (prediction == targets).cpu().numpy(), "prediction": prediction.cpu().numpy(),
                          "target_id": corpus.ids[indices], "target_index": indices, "doc": corpus.doc[indices]}
                for key, value in values.items():
                    arrays[key].append(value)
                nll_sum += values["nll"].sum(dtype=np.float64)
                correct += values["correct"].sum()
                count += len(indices)
            bar.update(len(batch.targets))
            bar.set_postfix(CE=f"{nll_sum / count:.3f}", acc=f"{correct / count:.1%}", refresh=False)
    if not arrays["nll"]:
        raise ValueError("Evaluation has no scored tokens")
    arrays = {key: np.concatenate(value) for key, value in arrays.items()}
    ce = float(arrays["nll"].astype(np.float64).mean())
    return {"tokens": len(arrays["nll"]), "cross_entropy": ce,
            "perplexity": math.exp(ce), "accuracy": float(arrays["correct"].mean())}, arrays
