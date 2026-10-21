"""Command line: check (E0), unigram, train-lens, measure (E1-E3, E5 variants), analyze (E3, E4, E9)."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from depthprobe import __version__
from depthprobe.provenance import environment, model_fingerprint, write_json


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def add_model_args(parser):
    from depthprobe.model import DTYPES

    group = parser.add_argument_group("model")
    group.add_argument("--model", required=True, help="Local model directory (never downloaded)")
    group.add_argument("--model-revision", help="Commit/revision you downloaded; recorded only")
    group.add_argument("--dtype", choices=list(DTYPES), default="float32")
    group.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    group.add_argument("--attn", help="attn_implementation (e.g. eager, sdpa); default: library choice")
    group.add_argument("--init", choices=["pretrained", "random"], default="pretrained", help="random = E5 control")
    group.add_argument("--init-seed", type=int, default=0)
    group.add_argument("--shuffle", choices=["none", "all", "middle"], default="none",
                       help="Shuffle block order (E5 control); middle keeps first and last blocks")
    group.add_argument("--shuffle-seed", type=int, default=0)


def add_data_args(parser, split, *, max_tokens_flag=True):
    group = parser.add_argument_group("data")
    group.add_argument("--data", required=True, help="Local parquet directory/file or .txt file")
    group.add_argument("--split", default=split, help="Parquet file prefix inside a directory")
    if max_tokens_flag:
        group.add_argument("--max-tokens", type=int, help="Score at most this many tokens")
    group.add_argument("--joiner", default="", help="String joining lines of a document (default: none)")
    group.add_argument("--ctx", type=int, default=1024, help="Window length in tokens, including BOS if added")
    group.add_argument("--stride", type=int, default=512)
    group.add_argument("--prepend-bos", action="store_true", help="Start every window with the BOS token")
    group.add_argument("--batch-size", type=int, default=4)
    group.add_argument("--chunk", type=int, default=1024, help="Tokens per readout chunk; lower it if out of memory")


def prepare_dir(path, overwrite):
    out = Path(path)
    if out.exists() and any(out.iterdir()) and not overwrite:
        sys.exit(f"{out} is not empty; pass --overwrite to replace its results")
    out.mkdir(parents=True, exist_ok=True)
    return out


def variant(args):
    return {"init": args.init, "init_seed": args.init_seed if args.init == "random" else None,
            "shuffle": args.shuffle, "shuffle_seed": args.shuffle_seed if args.shuffle != "none" else None}


def setup(args):
    from depthprobe.model import (Readout, find_parts, load_model, load_tokenizer, max_context,
                                  permute_blocks, resolve_device, shuffle_permutation)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = resolve_device(args.device)
    tokenizer = load_tokenizer(args.model)
    model = load_model(args.model, dtype=args.dtype, init=args.init, init_seed=args.init_seed,
                       device=device, attn=args.attn)
    permutation = None
    if args.shuffle != "none":
        permutation = shuffle_permutation(len(find_parts(model).blocks), args.shuffle_seed, args.shuffle)
        permute_blocks(model, permutation)
    if args.ctx > max_context(model):
        sys.exit(f"--ctx {args.ctx} exceeds the model's {max_context(model)} positions")
    bos_id = None
    if args.prepend_bos:
        bos_id = tokenizer.bos_token_id
        if bos_id is None:
            sys.exit("Tokenizer has no BOS token")
    readout = Readout(model)
    meta = {"family": find_parts(model).family, "permutation": permutation, "variant": variant(args),
            "bos_id": bos_id, "device": device}
    return model, tokenizer, readout, bos_id, device, meta


def manifest(args, meta, **extra):
    return {"tool": "depthprobe", "version": __version__, "command": args.command, "args": vars(args),
            "environment": environment(), "model": model_fingerprint(args.model), **meta, **extra}


def windows_for(corpus, args, bos_id, stride):
    from depthprobe.data import make_windows

    return make_windows(len(corpus.ids), args.ctx - (bos_id is not None), stride)


def command_check(args):
    from depthprobe.data import load_corpus
    from depthprobe.measure import check

    out = prepare_dir(args.out, args.overwrite)
    model, tokenizer, readout, bos_id, device, meta = setup(args)
    corpus = load_corpus(tokenizer, args.data, args.split, max_tokens=args.max_tokens, joiner=args.joiner)
    windows = windows_for(corpus, args, bos_id, args.stride)
    log(f"E0 check: {len(corpus.ids)} tokens, {corpus.n_docs} documents, {len(windows)} windows")
    result = check(model, readout, corpus, windows, batch_size=args.batch_size, bos_id=bos_id,
                   chunk=args.chunk, device=device, log=log)
    write_json(out / "check.json", manifest(args, meta, corpus={"tokens": len(corpus.ids), "documents": corpus.n_docs,
                                                                "windows": len(windows)}, result=result))
    log(json.dumps(result, indent=2))


def command_unigram(args):
    from transformers import AutoConfig

    from depthprobe.data import load_corpus
    from depthprobe.measure import unigram_logp
    from depthprobe.model import load_tokenizer

    out = Path(args.out)
    if out.exists() and not args.overwrite:
        sys.exit(f"{out} exists; pass --overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args.model)
    vocab_size = AutoConfig.from_pretrained(args.model, local_files_only=True).vocab_size
    corpus = load_corpus(tokenizer, args.data, args.split, max_tokens=args.max_tokens, joiner=args.joiner,
                         shuffle_seed=args.seed)
    np.save(out, unigram_logp(corpus.ids, vocab_size, args.alpha))
    write_json(out.with_suffix(".json"), {"command": "unigram", "args": vars(args), "tokens": len(corpus.ids),
                                          "documents": corpus.n_docs, "vocab_size": vocab_size,
                                          "model": model_fingerprint(args.model)})
    log(f"unigram from {len(corpus.ids)} tokens -> {out}")


def command_train_lens(args):
    from depthprobe.data import load_corpus
    from depthprobe.lens import evaluate_lens, save_lens, train_lens
    from depthprobe.model import hidden_width, n_blocks

    out = prepare_dir(args.out, args.overwrite)
    model, tokenizer, readout, bos_id, device, meta = setup(args)
    torch.manual_seed(args.seed)
    train = load_corpus(tokenizer, args.data, args.split, max_tokens=args.train_tokens, joiner=args.joiner,
                        shuffle_seed=args.seed)
    train_windows = windows_for(train, args, bos_id, args.ctx - (bos_id is not None))
    order = np.random.default_rng(args.seed).permutation(len(train_windows))
    train_windows = [train_windows[i] for i in order]
    log(f"training lens on {len(train.ids)} tokens ({len(train_windows)} windows)")
    lens, history = train_lens(model, readout, train.ids, train_windows, batch_size=args.batch_size, bos_id=bos_id,
                               chunk=args.chunk, lr=args.lr, warmup_fraction=args.warmup, device=device, log=log)
    validation = load_corpus(tokenizer, args.data, args.val_split, max_tokens=args.val_tokens, joiner=args.joiner)
    val_windows = windows_for(validation, args, bos_id, args.ctx - (bos_id is not None))
    evaluation = evaluate_lens(model, readout, lens, validation.ids, val_windows, batch_size=args.batch_size,
                               bos_id=bos_id, chunk=args.chunk, device=device)
    lens_meta = {"n_states": n_blocks(model) + 1, "width": hidden_width(model), "variant": meta["variant"],
                 "permutation": meta["permutation"], "config_sha256": model_fingerprint(args.model)["config_sha256"],
                 "train_tokens": int(len(train.ids)), "validation": evaluation}
    save_lens(out / "lens.pt", lens, lens_meta)
    write_json(out / "lens.json", manifest(args, meta, lens=lens_meta, history=history))
    for layer, (raw, tuned) in enumerate(zip(evaluation["raw_kl"], evaluation["tuned_kl"])):
        log(f"state {layer:>3}: validation KL raw {raw:.4f}  tuned {tuned:.4f}")


def command_measure(args):
    from depthprobe.data import load_corpus
    from depthprobe.lens import load_lens
    from depthprobe.measure import measure

    out = prepare_dir(args.out, args.overwrite)
    model, tokenizer, readout, bos_id, device, meta = setup(args)
    lens, lens_meta = None, None
    if args.lens:
        lens, lens_meta = load_lens(args.lens, device)
        expected = {"variant": meta["variant"], "permutation": meta["permutation"],
                    "config_sha256": model_fingerprint(args.model)["config_sha256"]}
        found = {key: lens_meta.get(key) for key in expected}
        if found != expected:
            sys.exit(f"Lens was trained for {found}, but this run is {expected}")
    unigram = None
    if args.unigram:
        unigram = np.load(args.unigram)
        if unigram.size != readout.head.out_features:
            sys.exit(f"Unigram has {unigram.size} entries, the head has {readout.head.out_features}")
    corpus = load_corpus(tokenizer, args.data, args.split, max_tokens=args.max_tokens, joiner=args.joiner)
    windows = windows_for(corpus, args, bos_id, args.stride)
    log(f"measuring {len(corpus.ids)} tokens, {corpus.n_docs} documents, {len(windows)} windows")
    arrays, stats = measure(model, readout, corpus, windows, batch_size=args.batch_size, bos_id=bos_id,
                            chunk=args.chunk, device=device, lens=lens, unigram_logp=unigram,
                            max_final_nll_error=args.max_final_nll_error, log=log)
    np.savez(out / "metrics.npz", **arrays)
    write_json(out / "manifest.json", manifest(args, meta, stats=stats, lens=lens_meta,
                                               corpus={"tokens": len(corpus.ids), "documents": corpus.n_docs,
                                                       "windows": len(windows)}))
    log(json.dumps(stats, indent=2))


def command_analyze(args):
    from depthprobe.analysis import plot, summarize, write_csv

    out = prepare_dir(args.out, args.overwrite)
    summaries = [summarize(run, from_layer=args.from_layer, min_context=args.min_context, eps=args.eps,
                           n_boot=args.bootstrap, seed=args.seed, flat_tol=args.flat_tol) for run in args.runs]
    write_json(out / "summary.json", {"command": "analyze", "args": vars(args), "summaries": summaries})
    write_csv(out / "curves.csv", summaries)
    plot(out, summaries)
    for summary in summaries:
        log(f"{summary['label']}: {summary['tokens']} tokens, {summary['documents']} documents")
        for lens, stats in summary["e3"].items():
            parts = ", ".join(f"{key} {value[0]:.3f} [{value[1]:.3f}, {value[2]:.3f}]"
                              for key, value in stats.items() if key in ("monotone_nll", "monotone_kl", "correct_then_wrong"))
            log(f"  {lens}: {parts}")
        for lens, stats in summary["isotonic"].items():
            parts = ", ".join(f"{metric} isotonic R² {entry['r2_mean'][0]:.3f} [{entry['r2_mean'][1]:.3f}, {entry['r2_mean'][2]:.3f}]"
                              f" backtrack {entry['backtrack_mean'][0]:.3f} flat {entry['flat_fraction'][0]:.3f}"
                              for metric, entry in stats.items())
            log(f"  {lens}: {parts}")
    log(f"wrote {out}")


def command_finetune_segment(args):
    import copy

    from depthprobe.data import load_corpus
    from depthprobe.finetune import cache_training, evaluate_full, make_segment, token_count, train_segment, validate_range
    from depthprobe.model import find_parts, hidden_width, resolve_device

    if args.epochs < 1 or args.lr <= 0 or args.reg_strength < 0 or args.std_floor <= 0 or args.rms_eps <= 0:
        sys.exit("Need epochs >= 1, lr > 0, reg-strength >= 0, std-floor > 0, rms-eps > 0")
    kl_strength = args.reg_strength if args.kl_strength is None else args.kl_strength
    if kl_strength < 0:
        sys.exit("kl-strength must be nonnegative")
    if args.train_split == args.split:
        sys.exit("Training and evaluation splits must differ")
    if Path(args.data).is_file() and (not args.train_data or Path(args.train_data).resolve() == Path(args.data).resolve()):
        sys.exit("Single-file evaluation requires a separate --train-data file")
    if args.batch_size < 1 or args.chunk < 1 or (args.inference_batch_size is not None and args.inference_batch_size < 1):
        sys.exit("Batch sizes and chunk must be positive")
    if args.amp != "none" and args.dtype != "float32":
        sys.exit("Use --dtype float32 with --amp bfloat16 to keep FP32 optimizer/master weights")
    out = prepare_dir(args.out, args.overwrite)
    # Inference can run on CPU when the complete model does not fit on the training GPU.
    inference_args = copy.copy(args)
    inference_args.device = args.inference_device
    model, tokenizer, readout, bos_id, inference_device, meta = setup(inference_args)
    validate_range(model, args.first_block, args.last_block)
    partial_exit = args.last_block if args.partial_exit_block is None else args.partial_exit_block
    validate_range(model, 1, partial_exit)
    if args.train_head:
        if args.regularizer != "none":
            sys.exit("--train-head requires --regularizer none")
        if find_parts(model).head.weight is model.get_input_embeddings().weight:
            sys.exit("--train-head requires an untied output head to keep input embeddings frozen")
    device = resolve_device(args.device)
    if args.amp != "none" and (torch.device(device).type != "cuda" or not torch.cuda.is_bf16_supported()):
        sys.exit("--amp bfloat16 requires a BF16-capable CUDA device")
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    progress = not args.no_progress
    inference_batch_size = args.inference_batch_size or args.batch_size
    torch.manual_seed(args.seed)
    evaluation = load_corpus(tokenizer, args.data, args.split, max_tokens=args.max_tokens, joiner=args.joiner)
    eval_windows = windows_for(evaluation, args, bos_id, args.stride)
    inference_amp = args.amp if torch.device(inference_device).type == "cuda" else "none"
    eval_kwargs = dict(batch_size=inference_batch_size, bos_id=bos_id, device=inference_device,
                       chunk=args.chunk, progress=progress, amp=inference_amp)
    log("Evaluating untouched full model")
    before, before_arrays = evaluate_full(model, evaluation, eval_windows, description="Full model BEFORE", **eval_kwargs)
    np.savez(out / "before.npz", **before_arrays)
    partial_before, partial_before_arrays = evaluate_full(
        model, evaluation, eval_windows, last_block=partial_exit,
        description="Partial model BEFORE", **eval_kwargs)
    np.savez(out / "partial_before.npz", **partial_before_arrays)
    train = load_corpus(tokenizer, args.train_data or args.data, args.train_split,
                        max_tokens=args.train_tokens, joiner=args.joiner, shuffle_seed=args.seed)
    train_windows = windows_for(train, args, bos_id, args.ctx - (bos_id is not None))
    import shutil

    cache_dir = out / "cache"
    # Three hidden-state caches dominate disk use; leave room for weights/reports.
    cached_positions = sum(window.end - window.start for window in train_windows) + 2 * token_count(train_windows)
    estimate = cached_positions * hidden_width(model) * next(model.parameters()).element_size()
    free_disk = shutil.disk_usage(out).free
    log(f"Estimated activation cache: {estimate / 2**30:.1f} GiB; free disk: {free_disk / 2**30:.1f} GiB")
    if estimate > 0.9 * free_disk:
        sys.exit("Insufficient disk headroom for cache; reduce --train-tokens or choose --out on a larger disk")
    cache_meta = cache_training(model, train, train_windows, cache_dir, first=args.first_block,
                                last=args.last_block, batch_size=inference_batch_size, bos_id=bos_id,
                                device=inference_device, chunk=args.chunk, std_floor=args.std_floor, log=log,
                                progress=progress, amp=inference_amp, rms_eps=args.rms_eps)
    model.cpu()
    readout.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    segment = make_segment(model, args.first_block, args.last_block)
    history = train_segment(segment, readout, cache_dir, cache_meta, device=device, epochs=args.epochs,
                            lr=args.lr, strength=args.reg_strength, regularizer=args.regularizer,
                            chunk=args.chunk, seed=args.seed, log=log, batch_size=args.batch_size,
                            workers=args.workers, progress=progress, amp=args.amp,
                            lr_schedule=args.lr_schedule, warmup_fraction=args.warmup_fraction,
                            log_path=out / "training.csv", log_every=args.log_every,
                            l2_space=args.l2_space, rms_eps=args.rms_eps, kl_strength=kl_strength,
                            train_head=args.train_head)
    # Blocks are shared with the full model; the optional trained readout head is copied back.
    segment.cpu()
    readout.cpu()
    if args.train_head:
        find_parts(model).head.load_state_dict(readout.head.state_dict())
    blocks = torch.nn.ModuleList(list(find_parts(model).blocks)[args.first_block - 1:args.last_block])
    torch.save({"state_dict": blocks.state_dict(), "first_block": args.first_block,
                "last_block": args.last_block, "variant": meta["variant"], "permutation": meta["permutation"],
                "model": model_fingerprint(args.model), "train_head": args.train_head,
                "head_state_dict": readout.head.state_dict() if args.train_head else None}, out / "segment.pt")
    del segment, blocks
    model.requires_grad_(False).to(inference_device)
    log("Evaluating full model with original suffix restored")
    after, after_arrays = evaluate_full(model, evaluation, eval_windows, description="Full model AFTER", **eval_kwargs)
    if not np.array_equal(before_arrays["target_index"], after_arrays["target_index"]):
        raise RuntimeError("Before/after evaluation tokens do not align")
    np.savez(out / "after.npz", **after_arrays)
    partial_after, partial_after_arrays = evaluate_full(
        model, evaluation, eval_windows, last_block=partial_exit,
        description="Partial model AFTER", **eval_kwargs)
    for key in ("target_index", "target_id", "doc"):
        if not np.array_equal(partial_before_arrays[key], partial_after_arrays[key]):
            raise RuntimeError(f"Partial before/after {key} do not align")
        if not np.array_equal(before_arrays[key], partial_before_arrays[key]):
            raise RuntimeError(f"Full/partial evaluation {key} do not align")
    np.savez(out / "partial_after.npz", **partial_after_arrays)
    partial_comparison = {"before": partial_before, "after": partial_after,
                          "delta_after_minus_before": {
                              key: partial_after[key] - partial_before[key]
                              for key in ("cross_entropy", "perplexity", "accuracy")}}
    comparison = {"before": before, "after": after,
                  "delta_after_minus_before": {key: after[key] - before[key]
                                               for key in ("cross_entropy", "perplexity", "accuracy")},
                  "prediction_changed_fraction": float((before_arrays["prediction"] != after_arrays["prediction"]).mean())}
    write_json(out / "summary.json", manifest(args, meta, training_device=device,
                                               cache=cache_meta, history=history, evaluation=comparison,
                                               partial_evaluation=partial_comparison))
    if args.save_full_model:
        model.cpu().save_pretrained(out / "model")
        tokenizer.save_pretrained(out / "model")
    log(json.dumps(comparison, indent=2))
    from depthprobe.segment_report import report

    report(out)


def command_report_segment(args):
    from depthprobe.segment_report import report

    report(args.run_dir, out=args.out, show=args.show)


def parser():
    root = argparse.ArgumentParser(prog="depthprobe", description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check", help="E0: label alignment, final-layer readout vs native, native perplexity")
    add_model_args(check)
    add_data_args(check, "test")
    check.add_argument("--out", required=True)
    check.add_argument("--overwrite", action="store_true")
    check.set_defaults(run=command_check)

    unigram = commands.add_parser("unigram", help="Token-frequency baseline from a training split")
    unigram.add_argument("--model", required=True, help="Model directory (tokenizer and config only)")
    unigram.add_argument("--data", required=True)
    unigram.add_argument("--split", default="train")
    unigram.add_argument("--max-tokens", type=int)
    unigram.add_argument("--joiner", default="")
    unigram.add_argument("--alpha", type=float, default=1.0, help="Add-alpha smoothing")
    unigram.add_argument("--seed", type=int, default=0, help="Document shuffle seed")
    unigram.add_argument("--out", required=True, help="Output .npy file")
    unigram.add_argument("--overwrite", action="store_true")
    unigram.set_defaults(run=command_unigram)

    lens = commands.add_parser("train-lens", help="Train a tuned lens on a training split; evaluate on validation")
    add_model_args(lens)
    add_data_args(lens, "train", max_tokens_flag=False)
    lens.add_argument("--train-tokens", type=int, default=4_000_000)
    lens.add_argument("--val-split", default="validation")
    lens.add_argument("--val-tokens", type=int, default=200_000)
    lens.add_argument("--lr", type=float, default=1e-3)
    lens.add_argument("--warmup", type=float, default=0.1, help="Warmup as a fraction of steps")
    lens.add_argument("--seed", type=int, default=0)
    lens.add_argument("--out", required=True)
    lens.add_argument("--overwrite", action="store_true")
    lens.set_defaults(run=command_train_lens)

    measure = commands.add_parser("measure", help="Per-token metrics at every state (E1, E2, E3 inputs)")
    add_model_args(measure)
    add_data_args(measure, "test")
    measure.add_argument("--lens", help="lens.pt from train-lens for this exact model variant")
    measure.add_argument("--unigram", help=".npy from the unigram command")
    measure.add_argument("--max-final-nll-error", type=float, default=0.1,
                         help="Abort if readout(final state) differs from native NLL by more (nats)")
    measure.add_argument("--label", help="Name used in plots (default: output directory name)")
    measure.add_argument("--out", required=True)
    measure.add_argument("--overwrite", action="store_true")
    measure.set_defaults(run=command_measure)

    analyze = commands.add_parser("analyze", help="Curves with document bootstrap CIs, E3 monotonicity, plots")
    analyze.add_argument("runs", nargs="+", help="Output directories of measure")
    analyze.add_argument("--from-layer", type=int, default=1, help="First state included in E3 (0 = embeddings)")
    analyze.add_argument("--min-context", type=int, default=1, help="Drop tokens with less visible context")
    analyze.add_argument("--eps", type=float, default=1e-3, help="Increase (nats) counted as a reversal")
    analyze.add_argument("--flat-tol", type=float, default=1e-3,
                         help="Paths whose range (nats) is below this are 'flat'; isotonic scores skip them")
    analyze.add_argument("--bootstrap", type=int, default=1000)
    analyze.add_argument("--seed", type=int, default=0)
    analyze.add_argument("--out", required=True)
    analyze.add_argument("--overwrite", action="store_true")
    analyze.set_defaults(run=command_analyze)

    fine = commands.add_parser("finetune-segment", help="CE-train a block range, then evaluate the restored full model")
    add_model_args(fine)
    add_data_args(fine, "test")
    fine.add_argument("--first-block", required=True, type=int, help="First trained block, 1-based, inclusive")
    fine.add_argument("--last-block", required=True, type=int, help="Last trained block, 1-based, inclusive")
    fine.add_argument("--partial-exit-block", type=int,
                      help="Held-out partial-model exit (default: last trained block)")
    fine.add_argument("--train-data", help="Separate local training corpus (default: --data directory)")
    fine.add_argument("--train-split", default="train")
    fine.add_argument("--train-tokens", type=int, default=200_000)
    fine.add_argument("--inference-device", default="auto", help="Full-model cache/evaluation device; cpu saves GPU memory")
    fine.add_argument("--epochs", type=int, default=1)
    fine.add_argument("--train-head", action="store_true",
                      help="Also train the untied prediction head; requires --regularizer none")
    fine.add_argument("--lr", type=float, default=1e-5)
    fine.add_argument("--regularizer", choices=["l2", "kl", "both", "none"], default="l2")
    fine.add_argument("--reg-strength", type=float, default=1.0, help="L2 weight (and KL weight unless --kl-strength is set)")
    fine.add_argument("--kl-strength", type=float, default=None,
                      help="KL weight; defaults to --reg-strength. Used by --regularizer kl or both")
    fine.add_argument("--l2-space", choices=["residual", "rms"], default="residual",
                      help="L2 on raw residuals or parameter-free RMS-normalized activations, with matching reference STD")
    fine.add_argument("--rms-eps", type=float, default=1e-6, help="Epsilon for parameter-free RMS normalization (no learned gain/bias)")
    fine.add_argument("--std-floor", type=float, default=1e-4, help="Minimum per-feature activation STD for normalized L2")
    fine.add_argument("--inference-batch-size", type=int, help="Cache/evaluation windows per batch (default: --batch-size)")
    fine.add_argument("--amp", choices=["none", "bfloat16"], default="none", help="Mixed precision; FP32 master weights with BF16 compute")
    fine.add_argument("--tf32", action="store_true", help="Allow TF32 matmuls; recorded in provenance")
    fine.add_argument("--workers", type=int, default=0, help="Prefetch workers for memory-mapped activation shards")
    fine.add_argument("--lr-schedule", choices=["none", "cosine"], default="none")
    fine.add_argument("--warmup-fraction", type=float, default=0.05)
    fine.add_argument("--log-every", type=int, default=1, help="Progress-bar metric refresh interval, in optimizer steps")
    fine.add_argument("--no-progress", action="store_true", help="Disable tqdm bars (CSV logging remains enabled)")
    fine.add_argument("--seed", type=int, default=0)
    fine.add_argument("--save-full-model", action="store_true", help="Also export a full HF checkpoint (otherwise segment only)")
    fine.add_argument("--out", required=True)
    fine.add_argument("--overwrite", action="store_true")
    fine.set_defaults(run=command_finetune_segment)
    report = commands.add_parser("report-segment", help="Display/rebuild a segment fine-tuning report")
    report.add_argument("run_dir", help="finetune-segment output directory")
    report.add_argument("--out", help="Report directory (default: RUN/report)")
    report.add_argument("--show", action="store_true", help="Also open an interactive matplotlib window")
    report.set_defaults(run=command_report_segment)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
