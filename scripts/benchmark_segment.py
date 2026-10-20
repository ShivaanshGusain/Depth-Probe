"""Short, synthetic GPU benchmark of the actual segment CE+L2 optimizer step.

No downloads or saved model changes. Excludes tokenization/cache I/O; final
training.csv is the authoritative end-to-end throughput measurement.
Run from the project root: python scripts/benchmark_segment.py --help
"""

import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from depthprobe.finetune import autocast, make_segment, parameter_free_rms, segment_step
from depthprobe.model import Readout, hidden_width, load_model, max_context
from depthprobe.provenance import environment, write_json


def candidate(model, readout, args, batch_size, chunk):
    segment = make_segment(model, args.first_block, args.last_block).to(args.device)
    width = hidden_width(model)
    parameters = [p for p in segment.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=0.0, fused=True)
    prefix = torch.randn(batch_size, args.ctx, width, device=args.device) * 0.1
    rows = torch.arange(batch_size, device=args.device).repeat_interleave(args.ctx - 1)
    positions = torch.arange(args.ctx - 1, device=args.device).repeat(batch_size)
    with torch.no_grad(), autocast(args.device, args.amp):
        reference = segment(inputs_embeds=prefix, use_cache=False).last_hidden_state[rows, positions].float()
    batch = {"prefix": prefix, "rows": rows, "positions": positions, "reference_exit": reference,
             "targets": torch.randint(model.config.vocab_size, (len(rows),), device=args.device)}
    scale = (parameter_free_rms(reference) if args.l2_space == "rms" else reference).std(0).clamp_min(1e-4)
    kwargs = dict(device=args.device, amp=args.amp, chunk=chunk, regularizer="l2", strength=1.0, l2_space=args.l2_space)
    for _ in range(args.warmup):
        segment_step(segment, readout, batch, optimizer, parameters, scale, **kwargs)
    torch.cuda.synchronize(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    started = time.perf_counter()
    for _ in range(args.steps):
        segment_step(segment, readout, batch, optimizer, parameters, scale, **kwargs)
    torch.cuda.synchronize(args.device)
    elapsed = time.perf_counter() - started
    return {"batch_size": batch_size, "chunk": chunk, "tokens_per_second": len(rows) * args.steps / elapsed,
            "seconds_per_step": elapsed / args.steps,
            "peak_gpu_gb": torch.cuda.max_memory_allocated(args.device) / 2**30}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--first-block", type=int, default=4)
    parser.add_argument("--last-block", type=int, default=8)
    parser.add_argument("--ctx", type=int, default=1024)
    parser.add_argument("--batch-sizes", default="8,16,32,64,96,128")
    parser.add_argument("--chunks", default="1024,2048")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--amp", choices=["none", "bfloat16"], default="bfloat16")
    parser.add_argument("--l2-space", choices=["residual", "rms"], default="rms")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.85, help="Headroom limit for recommendations")
    parser.add_argument("--out", default="runs/4090-benchmark.json")
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 1 or not 0 < args.memory_fraction < 1:
        parser.error("Need positive steps/warmup and 0 < memory-fraction < 1")
    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        parser.error("CUDA GPU required")
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = load_model(args.model, device="cpu", attn="sdpa")
    if not 2 <= args.ctx <= max_context(model):
        parser.error("ctx must be between 2 and the model context limit")
    readout = Readout(model).to(args.device)
    free, _ = torch.cuda.mem_get_info(args.device)
    budget = (free + torch.cuda.memory_allocated(args.device)) / 2**30 * args.memory_fraction
    results = []
    for chunk in [int(value) for value in args.chunks.split(",")]:
        for batch_size in [int(value) for value in args.batch_sizes.split(",")]:
            if min(chunk, batch_size) < 1:
                parser.error("Batch sizes and chunks must be positive")
            try:
                result = candidate(model, readout, args, batch_size, chunk)
                result["within_budget"] = result["peak_gpu_gb"] <= budget
                print(f"batch={batch_size:3d} chunk={chunk:5d}: {result['tokens_per_second']:,.0f} tok/s; "
                      f"{result['peak_gpu_gb']:.2f} GB peak", flush=True)
            except torch.cuda.OutOfMemoryError:
                result = {"batch_size": batch_size, "chunk": chunk, "oom": True}
                print(f"batch={batch_size:3d} chunk={chunk:5d}: CUDA OOM", flush=True)
            results.append(result)
            model.cpu()
            for parameter in model.parameters():
                parameter.grad = None
            gc.collect()
            torch.cuda.empty_cache()
            if result.get("oom"):
                break  # Larger batches with this chunk will not fit either.
    safe = [result for result in results if result.get("within_budget")]
    recommendation = max(safe, key=lambda result: result["tokens_per_second"]) if safe else None
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, {"args": vars(args), "environment": environment(), "memory_budget_gb": budget,
                     "results": results, "recommendation": recommendation})
    print(f"Recommendation (synthetic compute benchmark): {recommendation}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
