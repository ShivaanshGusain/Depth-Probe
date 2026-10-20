# depth-probe

A fresh HF/PyTorch pipeline for `../research_plan_depth_as_diffusion.md`. This is **step 1**: a
trustworthy measurement tool and baseline on GPT-2 small, before any noise experiments.

| Plan item | What this code does |
|---|---|
| **E0** audit | `check`: scoring positions vs the library's own loss; final-norm + head readout vs native logits; native perplexity |
| **E1** lens curves | `measure`: raw logit lens and tuned lens at every state (NLL, perplexity, top-1, entropy) |
| **E2** convergence to self | exact KL(p_L ‖ p_l), top-1 agreement with the final layer, cos(h_l, h_L), ‖h_L − h_l‖/‖h_L‖, ‖h_l‖ |
| **E3** per-token monotonicity | `analyze`: fraction of monotone tokens (NLL and KL), reversal counts and sizes, correct-then-wrong, prediction depth |
| **E4** statistics | 95% bootstrap CIs resampling **documents**, not tokens |
| **E5** controls | `--init random`, `--shuffle all|middle`, unigram-frequency baseline (`unigram`) |
| **E9** gain profile | per-block improvement in NLL and KL |

Not here yet: E6/E7 (noise injection, noise↔depth), E8, looped models, massive-activation handling.
The residual hooks in `depthprobe/model.py` are where E6/E7 will plug in.

## Pilot vs confirmatory (keeps the test honest)

GPT-2 small is the **exploratory** model. We use it to debug the pipeline, see effect sizes, and
write `PREREGISTRATION.md`. Only runs on models that the thresholds were **not** tuned on
(Llama, Mistral, Qwen, Pythia, Ouro, Huginn) count as confirmatory evidence.

## Rules the code follows

- **Never downloads.** `HF_HUB_OFFLINE=1` and `local_files_only=True`. Every model and dataset is a local path you pass in.
- **No `output_hidden_states`.** States come from our own hooks on each block (state 0 = embeddings,
  state l = output of block l, before the final norm). Every state is decoded with frozen float32
  copies of the model's own final norm and head.
- **Hard stops.** `measure` aborts if decoding the last state differs from the model's own NLL by more than 0.1 nat.
  A tuned lens refuses to load on a different model variant (pretrained / random / shuffled).
- **Reproducibility.** Every output has a manifest with the arguments, library versions, device, a
  hash of this package's source, and the model file list with `config.json` hash. TF32 is turned off.

## Setup with uv

**GX10 pod (container already has a CUDA build of PyTorch).** Reuse the container's torch; don't install another:

```bash
cd depth-probe
uv venv --system-site-packages .venv
source .venv/bin/activate
uv pip install -e ".[dev]"            # base deps + pytest; torch comes from the container
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name())"
```

Check that `torch.__file__` still points at the container's install. Use the activated venv or
`.venv/bin/python`, not `uv run`: `uv run` syncs the environment from the lock file.

**Without the container**, pick one torch build:

```bash
uv sync --extra cu130 --extra dev     # CUDA 13 (GX10 bare metal)
uv sync --extra cu128 --extra dev     # CUDA 12.8
uv sync --extra cpu --extra dev       # CPU only
```

Unverified: that PyTorch publishes aarch64 wheels on the cu130 index for the version uv resolves. If
it does not, use the container route.

Tests use tiny models built from configs, so they need no downloads: `python -m pytest`.

## Downloads (you run these; nothing here downloads)

Choose `MODELS` and `DATA` directories on the GX10. Record the revision you download.

```bash
# GPT-2 small (pilot model)
hf download openai-community/gpt2 --include "*.json" "*.safetensors" "*.txt" --local-dir "$MODELS/gpt2"

# WikiText-103 (raw), parquet files: test / validation / train
hf download Salesforce/wikitext --repo-type dataset --include "wikitext-103-raw-v1/*" --local-dir "$DATA/wikitext"
```

The loader expects `{split}-*.parquet` files with a `text` column in the directory you pass. If the
dataset repo's layout differs, point `--data` at the directory holding those files. A single `.txt`
file also works. Older `huggingface_hub` versions call the CLI `huggingface-cli download` instead of `hf download`.

Later (confirmatory): base models such as `meta-llama/Llama-3.1-8B`, `mistralai/Mistral-7B-v0.3`,
a Qwen base model, and the `EleutherAI/pythia-*` suites with `--revision stepN` checkpoints.

## Running the pilot

```bash
MODELS=/path/models DATA=/path/data OUT=/path/runs bash scripts/pilot_gpt2.sh
```

Optional variables: `DEVICE` (default `auto`), `BATCH` (8), `TRAIN_TOKENS` (4,000,000),
`EVAL_TOKENS` (default: the whole test split), `PY` (python executable). The script runs:

1. `check`: the **E0 gate**. Inspect `check/check.json` before trusting anything else.
2. `unigram`: token-frequency baseline from the training split.
3. For each of `pretrained`, `random`, `shuffle_all`, `shuffle_middle`: `train-lens` (training split,
   validated on the validation split), then `measure` on the test split with both lenses.
4. `analyze`: `report/summary.json`, `report/curves.csv`, and PNG plots.

Individual commands: `python -m depthprobe {check,unigram,train-lens,measure,analyze} --help`.

## Local middle-block fine-tuning

### Ready-to-run RTX 4090 pilot

```bash
bash scripts/train_gpt2_4090.sh
```

This finds already-cached local GPT-2/WikiText-103 snapshots (or accepts `MODEL`
and `DATA` paths). Defaults: blocks 4–8, one million training tokens, three epochs,
full validation evaluation, batch 32, inference batch 16, readout chunk 1024,
BF16 autocast with **FP32 master weights**, SDPA attention, fused AdamW, TF32,
5% warmup and cosine decay. The script now defaults to **parameter-free RMS-space
L2** (`L2_SPACE=rms`, strength 1.0). It downloads nothing. Reserve test for final evaluation.

```bash
BATCH=16 TRAIN_TOKENS=2000000 EPOCHS=3 OUT=runs/my-run \
  bash scripts/train_gpt2_4090.sh
python -m depthprobe report-segment runs/my-run --show
```

Live tqdm bars show cache/evaluation/training progress, token throughput, ETA,
CE, accuracy, unweighted regularizer, maximum L2 deviation in reference-STD units,
LR and peak allocated GPU memory. Every
optimizer step is also flushed to `training.csv`. The final report is generated
automatically: `report/results.png`, `report/results.md`, and paired 95%
document-bootstrap changes in `report/paired_changes.json`. `--show` opens the plot.
The image separates **truncated training-path** curves from **restored full-model**
held-out results. The script exports a reusable full checkpoint to `model/`.

Training and inference batch sizes are independent: cached windows are regrouped
without padding by a memory-mapped, prefetched DataLoader. `--batch-size` controls
the actual optimizer batch; `--inference-batch-size` controls caching/evaluation;
`--chunk` bounds vocabulary-head working memory, **not** the optimizer batch.
There is no gradient accumulation by default. Use `--no-progress` for quiet runs.

On the local 4090, a short synthetic GPT-2 blocks 4–8 benchmark at context 1024
measured roughly 78–82k scored tokens/sec for batches 8–32; batch 64 OOMed.
Batch 32 used about 12.7 GiB peak allocated memory. These are compute-only results,
not end-to-end guarantees; `training.csv` records actual throughput. Larger batches
reduce the number of updates at a fixed token budget, so batch size is also an
experimental hyperparameter. Benchmark a different range with:

```bash
python scripts/benchmark_segment.py --model /local/gpt2 \
  --first-block 4 --last-block 8 --batch-sizes 8,16,32,48 --chunks 1024,2048
```

One million tokens require approximately 9 GiB for these float32 hidden-state
caches; four million require roughly 35 GiB. Disk headroom is checked before
caching. The original continuous-stream boundary convention remains unchanged:
documents may attend across boundaries inside a window; batch rows remain isolated
and attention remains causal. No EOS separators or document-boundary masks are added.

`finetune-segment` trains **actual Transformer blocks**, not lens translators. The
training path is cached prefix activations → selected blocks → frozen original
final norm/head, supervised by ordinary next-token cross-entropy. The suffix is
omitted **only during training**. Before/after evaluation uses the complete model,
with the untouched suffix receiving the modified segment's outputs.

```bash
python -m depthprobe finetune-segment \
  --model "$MODELS/gpt2" --data "$DATA/wikitext/wikitext-103-raw-v1" \
  --first-block 4 --last-block 8 \
  --train-split train --train-tokens 200000 \
  --split validation --max-tokens 20000 \
  --ctx 1024 --stride 512 --batch-size 2 --chunk 256 \
  --device cuda --inference-device auto \
  --epochs 1 --lr 1e-5 --regularizer l2 --reg-strength 1.0 --l2-space rms \
  --out "$OUT/gpt2-blocks-4-8"
```

Block numbers are **1-based and inclusive**: `4..8` trains five blocks. Both ends
are parameters so range/length grids can use the same command with distinct output
directories. Each invocation starts from `--model`, not the preceding run. Use
validation for initial runs/grid selection and reserve the test split for final
evaluation. These settings are starting points, not validated hyperparameters.
For individual text/Parquet files, provide a separate `--train-data` file.

The L2 regularizer constrains **output activations, not weights**. With `--l2-space rms`:

```text
loss = mean_token CE(next_token, original_head(original_final_norm(new_exit)))
     + reg_strength * mean_token,feature ((rms(new_exit) - rms(original_exit)) / scale)^2
rms(h) = h / sqrt(mean_feature(h^2) + rms_eps)
scale[feature] = max(sample_STD(rms(original_exit)[:, feature]), std_floor)
```

This RMS transform has **no learned parameters**, gain, bias, or mean subtraction;
it does not use the model's LayerNorm/RMSNorm weights. `--rms-eps` defaults to
`1e-6`, and the same epsilon is used for reference statistics and training. The
model's original frozen final normalization remains in the CE/readout path only.
RMS normalization removes overall per-token magnitude, so this penalty constrains
normalized coordinates, not the raw residual-stream norm.

The per-feature sample standard deviations are computed over scored training
tokens during the initial untouched-model caching pass (`--std-floor`, default
`1e-4`, protects near-constant features). Supervision and regularization apply at
the **segment exit**, not every internal block. Reference moments are frozen and
saved under `rms` in `cache/activation_stats.pt`. Original residual-space moments
are also retained. The CLI default `--l2-space residual` reproduces the earlier raw
activation penalty; the 4090 script defaults to `rms`. Use `L2_SPACE=residual` to
compare against the earlier script behavior. **L2 is a soft mean penalty, not a
hard bound on every activation**; `max_dev_std` in tqdm and
`max_l2_deviation_std` in the CSV expose large individual deviations. Alternatives:

The max-STD diagnostic is only measured for L2 runs: it is omitted from the
KL/CE-only progress bars and stored as JSON `null` / a blank CSV cell, not zero.
KL progress instead labels its measured regularization term `KL`. Historical
KL runs written before this fix may contain placeholder zeros, not measured drift.

- `--regularizer kl`: KL(original untuned exit-lens distribution ‖ new exit-lens
  distribution). This is a regularizer; targets remain the real next tokens.
  `--kl-strength` sets its weight and defaults to `--reg-strength`.
- `--regularizer both`: L2 plus KL:
  `loss = CE + reg_strength * L2 + kl_strength * KL`. The two weights are
  independent, and the CSV records `l2_penalty` and `kl_penalty` separately.
- `--regularizer none`: CE only.

To run both with the RMS-space L2:

```bash
REGULARIZER=both REG_STRENGTH=1.0 KL_STRENGTH=1.0 OUT=runs/gpt2-both \
  bash scripts/train_gpt2_4090.sh
```

Equal weights do not mean equal constraint strength: the two penalties have
different units and gradients. The progress bar shows `L2` and `KL` separately,
and the report plots both unweighted terms.

On this workstation, experiment storage resides at
`/media/user/archive/Depth-Probe/runs` on the larger hard drive, with the local
`runs` path retained as a symlink. An absolute `OUT` on that drive
also works directly. Each invocation generates its own fresh activation cache;
there is no cross-run cache-reuse option.

Cached disk shards contain prefix activations, original segment-exit and final
pre-norm states, native full-model predictions/NLL, labels, and document/token
indices. Reference states reconstruct original lens probabilities through the
frozen readout without storing vocabulary-sized distributions. Only the selected
blocks and frozen norm/head reside on the training GPU; optimizer state belongs
only to the selected blocks. Readout backward passes are chunked by `--chunk`.
The complete model remains in CPU RAM during segment training. Initial caching
and full-model evaluation are ordinary inference; use `--inference-device cpu`
if the complete model cannot fit on GPU (slower). This is not yet a streaming or
multi-device inference implementation. Dropout is disabled during this experiment
to keep the activation references deterministic; gradients still train the blocks.

Outputs:

- `cache/batch-*.pt`, `cache/activation_stats.pt`, and `cache/cache.json` (per-run caches, freshly generated).
- `segment.pt`: trained block weights, inclusive range, and source-model metadata;
  selected modules share parameters with the full model during the run.
- `before.npz` / `after.npz`: paired native full-model per-token NLL, correctness,
  predictions, target IDs/indices, and document IDs.
- `summary.json`: full-model CE, perplexity, next-token accuracy, after-minus-before
  changes, prediction-change fraction, training-path history, and provenance.
- With `--save-full-model`, `model/` is a directly loadable HF model/tokenizer.

The command verifies that the extracted segment reproduces its original cached
outputs **before making any updates**. Improved truncated-path training metrics
do not imply improved full-model metrics; either outcome is an experimental result.

### What to check in the outputs

- `check.json`:
  - `max_abs_library_loss_gap` should be ≈ 0. This proves the label shift and the scored positions are right.
  - `max_abs_final_nll_error` should be ≈ 0 in float32.
  - `native_ppl` should be plausible for GPT-2 small. WikiText numbers depend on protocol (detokenization, joining, context, stride), so compare against a source whose protocol you can state.
- `lens/lens.json` → `validation`: tuned KL should be **below** raw KL at every non-final state. If not, the lens is undertrained; raise `TRAIN_TOKENS`.
- Controls: the trained model's E3 numbers only count if they are clearly above the random-init and shuffled runs.

## Output files

`measure/metrics.npz`: arrays with the token axis last, so `[states, tokens]` or `[tokens]`.

- `raw_*`, `tuned_*`: `nll`, `kl`, `top1`, `agree`, `rank`, `entropy`.
- `cos`, `rel_dist`, `norm`.
- `native_nll`, `unigram_nll`, `unigram_kl`.
- `target_id`, `target_index`, `doc`, `context`.

`measure/manifest.json` records how the file was produced. For GPT-2 small on the full WikiText-103 test
split, expect roughly a couple of hundred MB per measure run; cap with `EVAL_TOKENS` if needed.

## Known limitations

- WikiText-103 test has only a few dozen articles (see `documents` in the manifest), so the document bootstrap has few units. Wider CIs are honest, not a bug.
- Documents are tokenized separately and concatenated; windows can span article boundaries, as in standard WikiText evaluation.
- The shuffled control requires `use_cache=False` (always used here). Attention layers keep their original indices.
- bfloat16 models: the readout runs in float32 while the model runs in bf16. The final-layer check will show a small nonzero error; `--max-final-nll-error` sets the abort threshold.
- Ouro needs `transformers<4.56` and Huginn needs custom loops. They will need their own adapters and environment.
