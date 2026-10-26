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
    group.add_argument("--readout-dtype", choices=["float32", "model"], default="float32",
                       help="Precision of the final norm/head used to decode every state; 'model' repeats a "
                            "bfloat16 model's own head arithmetic, matching its logits up to kernel accumulation "
                            "differences (recommended with --dtype bfloat16)")
    group.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    group.add_argument("--attn", help="attn_implementation (e.g. eager, sdpa); default: library choice")
    group.add_argument("--init", choices=["pretrained", "random", "permuted"], default="pretrained",
                       help="random = untrained control; permuted = pretrained block weights shuffled entrywise "
                            "(matched-statistics control)")
    group.add_argument("--init-seed", type=int, default=0)
    group.add_argument("--shuffle", choices=["none", "all", "middle"], default="none",
                       help="Shuffle block order (E5 control); middle keeps first and last blocks")
    group.add_argument("--shuffle-seed", type=int, default=0)


def add_data_args(parser, split, *, max_tokens_flag=True):
    group = parser.add_argument_group("data")
    group.add_argument("--data", required=True, help="Local parquet directory/file, .txt file, or one JSON-lines shard "
                       "(.jsonl/.json[.gz], a 'text' field per line, e.g. C4)")
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
    return {"init": args.init, "init_seed": args.init_seed if args.init != "pretrained" else None,
            "shuffle": args.shuffle, "shuffle_seed": args.shuffle_seed if args.shuffle != "none" else None}


def setup(args):
    from depthprobe.model import (DTYPES, Readout, find_parts, load_model, load_tokenizer, max_context,
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
    readout_dtype = DTYPES[args.dtype] if args.readout_dtype == "model" else torch.float32
    readout = Readout(model, dtype=readout_dtype)
    meta = {"family": find_parts(model).family, "permutation": permutation, "variant": variant(args),
            "bos_id": bos_id, "device": device, "readout_dtype": str(readout_dtype).removeprefix("torch.")}
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
    # TF32 only while fitting, including the forward passes that produce the training states (speed for large
    # vocabularies); the lens's validation below and all measurement stay strict.
    previous_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    try:
        lens, history = train_lens(model, readout, train.ids, train_windows, batch_size=args.batch_size,
                                   bos_id=bos_id, chunk=args.chunk, lr=args.lr, warmup_fraction=args.warmup,
                                   device=device, log=log)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous_tf32
    validation = load_corpus(tokenizer, args.data, args.val_split, max_tokens=args.val_tokens, joiner=args.joiner)
    val_windows = windows_for(validation, args, bos_id, args.ctx - (bos_id is not None))
    evaluation = evaluate_lens(model, readout, lens, validation.ids, val_windows, batch_size=args.batch_size,
                               bos_id=bos_id, chunk=args.chunk, device=device)
    lens_meta = {"n_states": n_blocks(model) + 1, "width": hidden_width(model), "variant": meta["variant"],
                 "permutation": meta["permutation"], "config_sha256": model_fingerprint(args.model)["config_sha256"],
                 "train_tokens": int(len(train.ids)), "dtype": args.dtype, "readout_dtype": meta["readout_dtype"],
                 "tf32_during_fit": bool(args.tf32), "validation": evaluation}
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
                    "config_sha256": model_fingerprint(args.model)["config_sha256"],
                    "readout_dtype": meta["readout_dtype"]}
        if "dtype" in lens_meta:  # older lenses do not record the model dtype; it cannot be checked for them
            expected["dtype"] = args.dtype
        # Older lenses always used a float32 readout.
        found = {key: lens_meta.get(key, "float32" if key == "readout_dtype" else None) for key in expected}
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
    if args.readout_dtype == "model" and args.dtype != "float32":
        # Training runs through the readout; a bfloat16 head would round small optimizer updates away.
        sys.exit("finetune-segment trains through the readout; use --readout-dtype float32")
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


def command_perturb(args):
    from depthprobe.data import load_corpus
    from depthprobe.model import hidden_width, n_blocks
    from depthprobe.perturb import (FAMILIES, calibrate, load_calibration, load_sinks, parse_sites, run_perturbation,
                                    save_calibration)

    families = [part for part in args.families.split(",") if part]
    if sorted(set(families) - set(FAMILIES)):
        sys.exit(f"--families must be drawn from {','.join(FAMILIES)}")
    deltas = [float(part) for part in args.deltas.split(",") if part]
    if not deltas or min(deltas) <= 0:
        sys.exit("--deltas must be positive numbers")
    if args.every < 1 or args.replicas < 2 or args.replicas % 2:
        sys.exit("--every must be >= 1 and --replicas an even number >= 2")
    out = prepare_dir(args.out, args.overwrite)
    model, tokenizer, readout, bos_id, device, meta = setup(args)
    sites = parse_sites(args.sites, n_blocks(model))
    calibration_meta = {"variant": meta["variant"], "permutation": meta["permutation"], "dtype": args.dtype,
                        "n_states": n_blocks(model) + 1, "width": hidden_width(model),
                        "config_sha256": model_fingerprint(args.model)["config_sha256"],
                        "massive_factor": args.massive_factor, "keep_fraction": args.keep_fraction,
                        "split": args.cal_split, "tokens_requested": args.cal_tokens, "data": str(args.data),
                        "ctx": args.ctx, "stride": args.stride, "bos_id": bos_id}
    if args.calibration:
        calibration, found = load_calibration(args.calibration, device)
        mismatched = {key: (found.get(key), value) for key, value in calibration_meta.items() if found.get(key) != value}
        if mismatched:
            sys.exit(f"Calibration was made for a different model variant or settings (found, wanted): {mismatched}")
        calibration_meta = {**found, "source": str(Path(args.calibration).resolve())}
    else:
        cal = load_corpus(tokenizer, args.data, args.cal_split, max_tokens=args.cal_tokens, joiner=args.joiner)
        log(f"calibrating on {args.cal_split}: {len(cal.ids)} tokens")
        calibration = calibrate(model, readout, cal, windows_for(cal, args, bos_id, args.stride),
                                batch_size=args.batch_size, bos_id=bos_id, device=device,
                                massive_factor=args.massive_factor, keep_fraction=args.keep_fraction, log=log)
        calibration_meta = {**calibration_meta, "tokens": calibration["tokens"]}
        save_calibration(out / "calibration.pt", calibration, calibration_meta)
        calibration_meta = {**calibration_meta, "source": str((out / "calibration.pt").resolve())}
    fixed_sinks = None
    if args.sinks_from:
        fixed_sinks = load_sinks(Path(args.sinks_from) / "perturb.npz")
        log(f"using the measured tokens and sink positions of {args.sinks_from}")
    corpus = load_corpus(tokenizer, args.data, args.split, max_tokens=args.max_tokens, joiner=args.joiner)
    windows = windows_for(corpus, args, bos_id, args.stride)[::args.every]
    log(f"perturbing {len(windows)} windows from {args.split}; sites {sites}; families {families}; deltas {deltas}")
    arrays, result = run_perturbation(model, readout, corpus, windows, calibration, sites=sites, families=families,
                                      deltas=deltas, seed=args.seed, batch_size=args.batch_size, bos_id=bos_id,
                                      replicas=args.replicas, min_context=args.min_context,
                                      sink_factor=args.sink_factor, chunk=args.chunk, device=device,
                                      fixed_sinks=fixed_sinks, log=log)
    np.savez(out / "perturb.npz", **arrays)
    write_json(out / "manifest.json", manifest(args, meta, perturb=result, calibration=calibration_meta,
                                               corpus={"tokens": len(corpus.ids), "documents": corpus.n_docs,
                                                       "windows": len(windows)}))
    log(f"{result['measured_tokens']} measured tokens over {result['documents']} documents; "
        f"largest in-batch clean-replica deviation {result['floor_max']:.3g} (relative to the bulk norm)")


def command_analyze_perturb(args):
    from depthprobe.perturb_report import DocSums, contrasts, gate, load_perturb_run, plot, summarize_run, write_tables

    loaded = [load_perturb_run(path) for path in args.runs]
    labels = [run_manifest["args"].get("label") or Path(path).name for path, (run_manifest, _) in zip(args.runs, loaded)]
    if len(set(labels)) != len(labels):
        sys.exit(f"Run labels must be unique: {labels}")
    reference = args.reference or labels[0]
    if reference not in labels:
        sys.exit(f"Reference {reference!r} is not among the runs {labels}")
    first_manifest, first_arrays = loaded[labels.index(reference)]
    same_args = ("data", "split", "every", "ctx", "stride", "min_context", "max_tokens", "prepend_bos", "joiner")
    for label, (run_manifest, arrays) in zip(labels, loaded):
        differing = [key for key in same_args if run_manifest["args"].get(key) != first_manifest["args"].get(key)]
        if run_manifest["perturb"]["seed"] != first_manifest["perturb"]["seed"]:
            differing.append("seed")
        if run_manifest["perturb"]["conditions"] != first_manifest["perturb"]["conditions"]:
            differing.append("conditions (sites, families or deltas)")
        if arrays["counts"].shape != first_arrays["counts"].shape:
            differing.append("number of documents")
        elif not args.allow_different_tokens and not np.array_equal(arrays["counts"], first_arrays["counts"]):
            differing.append("measured tokens (run controls with --sinks-from the reference run)")
        if differing:
            sys.exit(f"Run {label!r} is not comparable with {reference!r}: {differing}")
        if args.delta not in run_manifest["perturb"]["deltas"]:
            sys.exit(f"--delta {args.delta} was not measured in {label!r} (deltas {run_manifest['perturb']['deltas']})")
    out = prepare_dir(args.out, args.overwrite)
    keep = np.any([arrays["counts"] > 0 for _, arrays in loaded], axis=0)  # one document set for every run
    runs = []
    for label, (run_manifest, arrays) in zip(labels, loaded):
        boot = DocSums(arrays["counts"], args.bootstrap, args.seed, keep=keep)
        summary, store = summarize_run(run_manifest, arrays, boot)
        runs.append({"label": label, "manifest": run_manifest, "summary": summary, "store": store})
    if not 1 <= args.min_sites <= len(first_manifest["perturb"]["sites"]):
        sys.exit(f"--min-sites must lie in 1..{len(first_manifest['perturb']['sites'])} (the measured sites)")
    if args.deciding == "auto":  # permuted weights and shuffled orders decide; random weights are reported only
        variants = {run["label"]: run["manifest"].get("variant") or {} for run in runs if run["label"] != reference}
        deciding = [label for label, v in variants.items()
                    if v.get("init") == "permuted" or v.get("shuffle", "none") != "none"]
        # The pre-registered set must be complete: a gate passed against a subset of it is not the gate.
        required = [("permuted weights", lambda v: v.get("init") == "permuted")] + [
            (f"shuffled middle seed {s}", lambda v, s=s: v.get("shuffle") == "middle" and v.get("shuffle_seed") == s)
            for s in (0, 1, 2)]
        deciding += [f"missing: {name}" for name, wanted in required if not any(map(wanted, variants.values()))]
    else:
        deciding = [label for label in args.deciding.split(",") if label]
        unknown = [label for label in deciding if label not in labels or label == reference]
        if unknown:
            sys.exit(f"--deciding names runs that are missing or the reference: {unknown}")
    contrast_rows = contrasts(runs, reference)
    decision = gate(contrast_rows, deciding, args.delta, args.margin, args.min_sites, args.rule)
    write_json(out / "summary.json", {"command": "analyze-perturb", "args": vars(args), "reference": reference,
                                      "runs": [{"label": r["label"], "variant": r["manifest"].get("variant"),
                                                "perturb_meta": {k: v for k, v in r["manifest"]["perturb"].items()
                                                                 if k != "conditions"},
                                                **r["summary"]} for r in runs],
                                      "contrasts": contrast_rows, "gate": decision})
    write_tables(out, runs, contrast_rows)
    plot(out, runs, args.delta)
    for control, definitions in decision["per_control"].items():
        details = ", ".join(f"{name} {entry['sites_passing']}/{entry['sites']}" for name, entry in definitions.items()
                            if isinstance(entry, dict))
        log(f"gate vs {control} ({'deciding' if definitions['deciding'] else 'reported only'}): {details} sites pass "
            f"-> {'PASS' if definitions['passes'] else 'fail'}")
    log(f"GATE ({args.rule} definitions, delta {args.delta}, margin {args.margin}, >= {args.min_sites} sites): "
        f"{'PASS' if decision['passes'] else 'FAIL'}" + (f" ({'; '.join(decision['reasons'])})" if decision["reasons"] else ""))
    log(f"wrote {out}")


def command_reproduce_absorption(args):
    from depthprobe.absorption import plot as plot_absorption
    from depthprobe.absorption import run_absorption, summarize as summarize_absorption
    from depthprobe.data import load_corpus
    from depthprobe.model import n_blocks
    from depthprobe.perturb import parse_sites

    etas = [float(part) for part in args.etas.split(",") if part]
    if not etas or min(etas) < 0 or len(set(etas)) != len(etas) or args.seeds < 1 or args.every < 1 \
            or args.replicas < 1:
        sys.exit("--etas must be distinct and >= 0 (0 measures the floor); --seeds, --every, --replicas at least 1")
    if args.real_swaps and not any(eta > 0 for eta in etas):
        sys.exit("--real-swaps needs at least one eta > 0")
    fixed_sinks = fixed_changed = None
    if args.sinks_from:
        from depthprobe.perturb import load_sinks

        reference = Path(args.sinks_from)
        if not (reference / "absorption.npz").exists():
            sys.exit(f"--sinks-from {reference}: no absorption.npz (finish the reference run first)")
        reference_args = json.loads((reference / "manifest.json").read_text())["args"]
        differing = [key for key in ("data", "split", "ctx", "stride", "every", "max_tokens", "prepend_bos", "joiner")
                     if reference_args.get(key) != vars(args).get(key)]
        if differing:
            sys.exit(f"--sinks-from {reference} was run with different {differing}")
        fixed_sinks = load_sinks(reference / "absorption.npz")
        if args.real_swaps:  # controls compare the trained model's prediction-changed / -preserved token sets
            with np.load(reference / "absorption.npz") as data:
                if "changed_tokens" not in data.files:
                    sys.exit(f"--sinks-from {reference} has no prediction-changed token sets; run it with --real-swaps")
                if data["changed_tokens"].shape[0] != args.seeds:
                    sys.exit(f"--sinks-from {reference} used {data['changed_tokens'].shape[0]} seeds, not {args.seeds}")
                changed = data["changed_tokens"]  # read once: indexing the npz re-reads the array every time
                fixed_changed = {int(start): changed[:, w] for w, start in enumerate(data["window_starts"])}
    out = prepare_dir(args.out, args.overwrite)
    model, tokenizer, readout, bos_id, device, meta = setup(args)
    sites = parse_sites(args.sites, n_blocks(model))
    corpus = load_corpus(tokenizer, args.data, args.split, max_tokens=args.max_tokens, joiner=args.joiner)
    windows = windows_for(corpus, args, bos_id, args.stride)[::args.every]
    conditions = len(etas) * args.seeds + (2 * sum(eta > 0 for eta in etas) * args.seeds if args.real_swaps else 0)
    passes = -(-len(windows) // args.batch_size) * len(sites) * -(-conditions // args.replicas)
    log(f"absorption probe: {len(windows)} windows, sites {sites}, etas {etas}, {args.seeds} seeds, "
        f"dtype {args.dtype}, scale {args.scale}, real swaps {'on' if args.real_swaps else 'off'}: "
        f"{passes} suffix passes of {args.replicas + 1} copies")
    use_readout = args.real_swaps or args.output_kl
    arrays, result = run_absorption(model, corpus, windows, sites=sites, etas=etas, seeds=args.seeds,
                                    batch_size=args.batch_size, bos_id=bos_id, device=device, scale=args.scale,
                                    exclude_sinks=args.exclude_sinks, sink_factor=args.sink_factor,
                                    replicas=args.replicas, readout=readout if use_readout else None,
                                    swaps=args.real_swaps, fixed_sinks=fixed_sinks, fixed_changed=fixed_changed,
                                    chunk=args.chunk, log=log)
    rows = summarize_absorption(arrays, result, n_boot=args.bootstrap)
    np.savez(out / "absorption.npz", **arrays)
    label = args.label or out.name
    write_json(out / "manifest.json", manifest(args, meta, absorption=result, summary=rows,
                                               corpus={"tokens": len(corpus.ids), "documents": corpus.n_docs,
                                                       "windows": len(windows)}))
    plot_absorption(out, rows, result, label)
    def show(value):
        return "-" if value is None else f"{value:.3g}"

    for row in rows:
        log(f"site {row['site']:>3} eta {row['eta']:g}: realised/eta {show(row['realised_over_eta'])}, "
            f"rounded away {show(row['rounded_away_fraction'])}, non-finite {show(row['nonfinite_fraction'])}, "
            f"R_L {show(row['R_last'])}, r = R_L/R_l {show(row['absorption'])} "
            f"(after final norm {show(row.get('absorption_final'))})")
    log(f"largest in-batch clean-replica deviation {result['floor_max']:.3g}; wrote {out}")


def command_analyze_absorption(args):
    from depthprobe.absorption import compare_runs, control_detail, control_verdict, plot_tests, summarize_tests

    runs, variants = [], {}
    for path in args.runs:
        path = Path(path)
        run_manifest = json.loads((path / "manifest.json").read_text())
        with np.load(path / "absorption.npz") as data:
            arrays = {key: data[key] for key in data.files}
        if "delta_norm" not in arrays:
            sys.exit(f"{path} was made before absolute sizes were recorded; rerun it")
        runs.append((run_manifest["args"].get("label") or path.name, arrays, run_manifest["absorption"]))
        variants[runs[-1][0]] = run_manifest.get("variant") or {}
    labels = [label for label, _, _ in runs]
    if len(set(labels)) != len(labels):
        sys.exit(f"Run labels must be unique: {labels}")
    reference_label = args.reference or labels[0]
    if reference_label not in labels:
        sys.exit(f"Reference {reference_label!r} is not among the runs {labels}")
    reference = runs[labels.index(reference_label)]
    if args.eta not in reference[2]["etas"] or args.eta == 0:
        sys.exit(f"--eta must be one of the measured etas > 0: {reference[2]['etas']}")
    if args.margin_ratio <= 1:
        sys.exit("--margin-ratio must be above 1")
    margin = float(np.log(args.margin_ratio))
    out = prepare_dir(args.out, args.overwrite)
    rows = summarize_tests(reference[1], reference[2], n_boot=args.bootstrap, seed=args.seed, margin=margin)
    try:
        comparisons = compare_runs(reference, [run for run in runs if run[0] != reference_label],
                                   n_boot=args.bootstrap, seed=args.seed, margin=margin)
    except ValueError as error:
        sys.exit(str(error))
    # Random weights are reported only (as in E6); permuted weights and shuffled layer orders decide.
    deciding = [label for label, _, _ in runs if label != reference_label and variants[label].get("init") != "random"]
    log(f"deciding controls: {deciding or 'none'}; reported only: "
        f"{[label for label, _, _ in runs if label != reference_label and label not in deciding] or 'none'}")
    answers = []
    for row in rows:
        if row["eta"] != args.eta:
            continue
        site = row["site"]

        def verdict(key):
            return {"answer": control_verdict(comparisons, site, args.eta, key, deciding),
                    "per_control": control_detail(comparisons, site, args.eta, key, deciding)}

        answer = {"site": site, "q1_noise_itself": row["q1"],
                  "q2_learned": verdict("trained_shrinks_more"),
                  "q2_relative_learned": verdict("trained_relative_lower"),
                  "q3_kind": row.get("q3", "not measured (run with --real-swaps)"),
                  "q3_statistic": row.get("q3_statistic"),
                  "q3_vs_noise_v1": row.get("q3_vs_noise", "not measured"),
                  "q3_without_massive_coordinates": row.get("q3_massive_check", "not measured"),
                  "q3_learned": verdict("selectivity_learned"),
                  "q4_noise_steering": row.get("q4", "not measured (needs the readout: --real-swaps or --output-kl)"),
                  "q4_trained_steers_away_more": verdict("trained_steers_away_more")}
        answers.append(answer)
        survival = np.exp(row["log_noise_survival"]["value"])
        growth = np.exp(row["log_signal_growth"]["value"])
        log(f"site {site:>3}: noise survival {survival:.3g}, clean signal growth {growth:.3g} "
            f"-> Q1 {answer['q1_noise_itself']}")
        log(f"           Q2 learned: relative {answer['q2_relative_learned']['answer']} "
            f"({answer['q2_relative_learned']['per_control']}); absolute (v1) {answer['q2_learned']['answer']}")
        if "log_selectivity_changed" in row:
            log(f"           Q3 {answer['q3_kind']} ({row['q3_statistic']}: changed "
                f"{np.exp(row['log_selectivity_changed']['value']):.3g}, preserved "
                f"{np.exp(row['log_selectivity_preserved']['value']):.3g}; prediction changed at "
                f"{row['changed_fraction']:.1%} of swapped tokens); real vs noise (v1): {answer['q3_vs_noise_v1']}")
            log(f"           Q3 massive coordinates: {answer['q3_without_massive_coordinates']}; Q3 learned: "
                f"{answer['q3_learned']['answer']} ({answer['q3_learned']['per_control']})")
        if "log_steering_noise" in row:
            post = row.get("log_steering_noise_post_norm")
            log(f"           Q4 {answer['q4_noise_steering']}: noise / size-matched random output effect "
                f"{np.exp(row['log_steering_noise']['value']):.3g}"
                + (f" (after the final norm {np.exp(post['value']):.3g})" if post else "")
                + (f", real / matched {np.exp(row['log_steering_real']['value']):.3g}"
                   if "log_steering_real" in row else "")
                + f"; vs controls: {answer['q4_trained_steers_away_more']['per_control'] or 'none'}")
        if "output_kl_noise" in row:
            log(f"           output KL from the clean prediction: noise {row['output_kl_noise']:.3g}, if carried "
                f"unchanged {row.get('output_kl_carry_noise', float('nan')):.3g}"
                + (f"; real {row['output_kl_real']:.3g}" if "output_kl_real" in row else ""))
    write_json(out / "verdict.json", {"command": "analyze-absorption", "args": vars(args), "reference": reference_label,
                                      "eta": args.eta, "margin_log": margin, "deciding_controls": deciding,
                                      "answers": answers, "rows": rows, "comparisons": comparisons})
    plot_tests(out, rows, comparisons, args.eta, reference_label, margin)
    log(f"wrote {out}")


def command_perturb_toys(args):
    from depthprobe.toys import toy_signatures

    out = prepare_dir(args.out, args.overwrite)
    results = toy_signatures(site=args.site, delta=args.delta, seed=args.seed)
    write_json(out / "toys.json", results)
    log(f"{'toy':<20} {'iso surv':>9} {'off surv':>9} {'on surv':>9} {'sel raw':>8} {'sel A':>8} {'sel Lam':>8} "
        f"{'expected off':>13}")
    for name, row in results.items():
        log(f"{name:<20} {row['iso']['log_survival']:>9.3f} {row['off']['log_survival']:>9.3f} "
            f"{row['on']['log_survival']:>9.3f} {row['selectivity']:>8.3f} {row['selectivity_A']:>8.3f} "
            f"{row['selectivity_Lambda']:>8.3f} {row['expected_off_survival']:>13.3f}")


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
    lens.add_argument("--tf32", action="store_true",
                      help="Allow TF32 matmuls while fitting the lens, including the forward passes that produce its "
                           "training states (faster on large vocabularies); validation and measurement stay strict")
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

    perturb = commands.add_parser("perturb", help="E6 denoise test: inject noise at each layer, measure what remains")
    add_model_args(perturb)
    add_data_args(perturb, "test")
    perturb.add_argument("--sites", default="all", help="Injection states, 'all' or e.g. 0,3,6,9")
    perturb.add_argument("--families", default="iso,off,on,read,unread,shared,signed",
                         help="Perturbation directions (see README)")
    perturb.add_argument("--deltas", default="0.03,0.1,0.3,1.0", help="Noise sizes relative to each token's bulk norm")
    perturb.add_argument("--seed", type=int, default=0, help="Noise seed (draws are shared across sites and variants)")
    perturb.add_argument("--every", type=int, default=4, help="Keep every Nth window")
    perturb.add_argument("--replicas", type=int, default=8, help="Perturbed copies per suffix pass (even)")
    perturb.add_argument("--min-context", type=int, default=64, help="Measure only tokens with this much context")
    perturb.add_argument("--sink-factor", type=float, default=10.0,
                         help="A token whose peak activation exceeds this multiple of the window median is a sink")
    perturb.add_argument("--calibration", help="Reuse calibration.pt from an earlier run of the same variant")
    perturb.add_argument("--sinks-from", help="Earlier perturb run whose sink positions (and so measured tokens) "
                                              "to reuse; pass the trained-model run for every control")
    perturb.add_argument("--cal-split", default="validation")
    perturb.add_argument("--cal-tokens", type=int, default=50_000)
    perturb.add_argument("--massive-factor", type=float, default=20.0,
                         help="A coordinate whose mean |value| exceeds this multiple of the median is massive")
    perturb.add_argument("--keep-fraction", type=float, default=0.9, help="Variance share of the on-manifold subspace")
    perturb.add_argument("--label", help="Name used in reports (default: output directory name)")
    perturb.add_argument("--out", required=True)
    perturb.add_argument("--overwrite", action="store_true")
    perturb.set_defaults(run=command_perturb)

    report_perturb = commands.add_parser("analyze-perturb", help="E6 statistics, controls contrasts and plots")
    report_perturb.add_argument("runs", nargs="+", help="Output directories of perturb (trained model first)")
    report_perturb.add_argument("--reference", help="Label of the trained-model run (default: the first run)")
    report_perturb.add_argument("--delta", type=float, default=0.1, help="Primary noise size for the gate and plots")
    report_perturb.add_argument("--margin", type=float, default=0.22,
                                help="Gate: trained-minus-control selectivity must exceed this (log units)")
    report_perturb.add_argument("--min-sites", type=int, default=3, help="Gate: sites that must pass, per control")
    report_perturb.add_argument("--rule", choices=["both", "either"], default="both",
                                help="Gate: selectivity must pass under both definitions (PCA and readout) or either")
    report_perturb.add_argument("--deciding", default="auto",
                                help="Comma-separated control labels that decide the gate; 'auto' = permuted and "
                                     "shuffled variants (random weights are reported only)")
    report_perturb.add_argument("--allow-different-tokens", action="store_true",
                                help="Compare runs whose measured tokens differ (not recommended)")
    report_perturb.add_argument("--bootstrap", type=int, default=1000)
    report_perturb.add_argument("--seed", type=int, default=0)
    report_perturb.add_argument("--out", required=True)
    report_perturb.add_argument("--overwrite", action="store_true")
    report_perturb.set_defaults(run=command_analyze_perturb)

    absorb = commands.add_parser("reproduce-absorption",
                                 help="Reproduce the perturbation-absorption probe of arXiv 2606.15161")
    add_model_args(absorb)
    add_data_args(absorb, "test")
    absorb.add_argument("--sites", default="all", help="Injection states, 'all' or e.g. 1,12,24")
    absorb.add_argument("--etas", default="0,1e-4,1e-3,1e-2,1e-1",
                        help="Noise sizes eta (the paper shows 1e-4, 1e-2); 0 measures the numerical floor")
    absorb.add_argument("--seeds", type=int, default=5, help="Noise directions per eta (the paper uses 5)")
    absorb.add_argument("--scale", choices=["frobenius", "token"], default="frobenius",
                        help="frobenius = the paper (eta x norm of the whole sequence tensor); token = per token")
    absorb.add_argument("--exclude-sinks", action="store_true",
                        help="Neither perturb nor measure attention-sink positions (not in the paper)")
    absorb.add_argument("--sink-factor", type=float, default=10.0)
    absorb.add_argument("--every", type=int, default=1, help="Keep every Nth window")
    absorb.add_argument("--replicas", type=int, default=4, help="Perturbed copies per suffix pass")
    absorb.add_argument("--real-swaps", action="store_true",
                        help="Also inject the difference made by swapping one earlier token (one swap per seed), "
                             "split by whether the swap changes the prediction; implies --output-kl")
    absorb.add_argument("--output-kl", action="store_true", help="Record the next-token KL of every perturbation")
    absorb.add_argument("--sinks-from", help="Reuse the sink positions of this earlier reproduce-absorption run "
                                             "(controls must exclude exactly the trained model's tokens)")
    absorb.add_argument("--bootstrap", type=int, default=1000)
    absorb.add_argument("--label")
    absorb.add_argument("--out", required=True)
    absorb.add_argument("--overwrite", action="store_true")
    absorb.set_defaults(run=command_reproduce_absorption)

    tests = commands.add_parser("analyze-absorption",
                                help="Q1 dilution, Q2 learned (vs controls), Q3 denoising/convergence/compression")
    tests.add_argument("runs", nargs="+", help="reproduce-absorption output directories (trained model first)")
    tests.add_argument("--reference", help="Label of the trained run (default: the first)")
    tests.add_argument("--eta", type=float, default=1e-2,
                       help="The pre-registered noise size the answers use (v2: 1e-2; v1 used 1e-3)")
    tests.add_argument("--margin-ratio", type=float, default=1.25,
                       help="'Clearly different' means a ratio beyond this, with the 95%% interval excluding 1")
    tests.add_argument("--bootstrap", type=int, default=1000)
    tests.add_argument("--seed", type=int, default=0)
    tests.add_argument("--out", required=True)
    tests.add_argument("--overwrite", action="store_true")
    tests.set_defaults(run=command_analyze_absorption)

    toys = commands.add_parser("perturb-toys", help="Run the E6 metrics on analytic stacks with known answers")
    toys.add_argument("--site", type=int, default=3)
    toys.add_argument("--delta", type=float, default=0.05)
    toys.add_argument("--seed", type=int, default=0)
    toys.add_argument("--out", required=True)
    toys.add_argument("--overwrite", action="store_true")
    toys.set_defaults(run=command_perturb_toys)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
