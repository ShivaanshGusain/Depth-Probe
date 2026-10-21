# depth-probe

**Do Transformers denoise through depth?**

Each layer of a decoder-only Transformer adds an update to the residual stream. That has the same
form as one step of a diffusion model's reverse (denoising) process. This project tests whether the
**learned dynamics** of pretrained language models actually behave like denoising across depth,
or merely like generic iterative refinement. If they do, existing models, especially looped ones,
could be fine-tuned one block or loop at a time, DiffusionBlocks-style, with far less memory.

Predictions getting better with depth *on average* is already known (logit lens, tuned lens).
The diffusion view makes stronger, testable predictions:

| | Prediction | Generic refinement | Denoising |
|---|---|---|---|
| H1a | Intermediate predictions converge to the final one | yes | yes |
| H1b | Convergence holds per token, not just on average | not necessarily | mostly |
| H1c | Noise injected at layer *l* is removed by later layers | not necessarily | yes |
| H1d | "Layer *l* + noise δ" behaves like an earlier layer *l − k* | no | **yes** |
| H1e | What remains (h_l − h_L) is noise-like and shrinks | no | yes |

This repository is the measurement pipeline. Research plan: `../research_plan_depth_as_diffusion.md`.
Literature: `../literature_review.md`.

## What it measures

For every token of a held-out corpus, the residual stream is captured **at every depth** and decoded
with the model's own final normalization and output head:

1. **Raw logit lens**: decode the state directly.
2. **Tuned lens**: first apply a per-layer affine map trained to match the final layer
   (Belrose et al., 2023). This removes the "not yet in the head's format" artefact of the raw lens.

At each depth it records:

- **Against the data:** NLL of the true next token, top-1 accuracy, rank, entropy.
- **Against the model's own final answer:** exact KL(p_L ‖ p_l), top-1 agreement.
- **Hidden geometry:** cos(h_l, h_L), ‖h_L − h_l‖ / ‖h_L‖, ‖h_l‖.

Every result is compared against **controls**, so that a pattern counts only if the trained model
shows it and the controls do not:

- the same architecture with **random weights**;
- the trained model with its **layers shuffled** (all, or all but first and last);
- a **token-frequency (unigram)** baseline.

Statistics are per token. Confidence intervals resample **documents**, not tokens.

## Experiment map

| Plan | Command | Question |
|---|---|---|
| E0 | `check` | Is the pipeline correct? Scoring matches the library's own loss; decoding the last state reproduces the native logits; native perplexity |
| E1 | `train-lens`, `measure` | Raw vs tuned lens curves across depth |
| E2 | `measure` | Convergence to the model's own final distribution |
| E3 | `analyze` | Per-token monotonicity, reversals, correct-then-wrong, prediction depth |
| E4 | `analyze` | Document-level bootstrap 95% CIs |
| E5 | `measure --init random`, `--shuffle`, `unigram` | Controls |
| E9 | `analyze` | Which layers contribute the most improvement |

Next stages: noise injection and noise↔depth equivalence (E6/E7, the decisive tests), residual
structure (E8), looped models (Ouro, Huginn).

## Protocol

- **Pilot vs confirmatory.** GPT-2 small is the exploratory model: it is used to develop the
  pipeline and set the refutation thresholds in `PREREGISTRATION.md`. Only models the thresholds were
  not tuned on count as evidence (Llama, Mistral, Qwen, Pythia, Ouro, Huginn).
- **No quantization.** float32 for small models; bfloat16 weights for large ones, with all
  probabilities computed in float32.
- **Offline.** Nothing is downloaded by the code. Models and data are local paths.
- **Provenance.** Every output has a manifest: arguments, library versions, device, source-code
  hash, and model file list.
- **Hard stops.** A run aborts if decoding the final state does not reproduce the model's own NLL.
  A tuned lens refuses to load on a different model variant.

## Setup (uv)

GX10 pod, reusing the container's CUDA PyTorch:

```bash
cd depth-probe
uv venv --system-site-packages .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
python -m pytest
```

Outside the container, choose a torch build:

```bash
uv sync --extra cu130 --extra dev    # or cu128, or cpu
```

## Data and models

Downloaded manually into directories of your choice:

```bash
hf download openai-community/gpt2 --include "*.json" "*.safetensors" "*.txt" --local-dir "$MODELS/gpt2"
hf download Salesforce/wikitext --repo-type dataset --include "wikitext-103-raw-v1/*" --local-dir "$DATA/wikitext"
```

`--data` takes a directory of `{split}-*.parquet` files with a `text` column, or a single `.txt` file.

## Run

Full GPT-2 pilot: E0 check, unigram baseline, then for each of pretrained / random / shuffle_all /
shuffle_middle a tuned lens and a measurement, then the report:

```bash
MODELS=/path/models DATA=/path/data OUT=/path/runs bash scripts/pilot_gpt2.sh
```

Optional variables: `DEVICE`, `BATCH`, `TRAIN_TOKENS`, `EVAL_TOKENS`, `PY`.
Single steps:

```bash
python -m depthprobe check      --model M --data D --out runs/check
python -m depthprobe unigram    --model M --data D --out runs/unigram.npy
python -m depthprobe train-lens --model M --data D --out runs/lens
python -m depthprobe measure    --model M --data D --lens runs/lens/lens.pt --unigram runs/unigram.npy --out runs/measure
python -m depthprobe analyze    runs/measure [more runs ...] --out runs/report
```

## Local middle-block fine-tuning

For the memory-constrained staged-training goal and the Pythia/GPT-NeoX pilot,
see [PYTHIA_EXPERIMENT_SUMMARY.md](PYTHIA_EXPERIMENT_SUMMARY.md).
`scripts/train_pythia_410m_4090.sh` trains two late blocks;
`scripts/compare_pythia_last_head_4090.sh` compares one-epoch last-block/head
adaptation from original versus staged weights. Both default to archive storage.
`--train-head` enables an untied trainable prediction head with `--regularizer none`;
the final norm stays frozen. `--partial-exit-block` selects the held-out partial
readout depth (default: last trained block); reports include before/after partial
metrics on the same evaluation tokens as the full model.

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
Every command has `--help`.

## Outputs

| File | Contents |
|---|---|
| `check/check.json` | `max_abs_library_loss_gap`, `max_abs_final_nll_error` (both should be ≈ 0), native perplexity |
| `lens/lens.pt`, `lens/lens.json` | Tuned lens, plus validation KL per layer (tuned should be below raw) |
| `measure/metrics.npz` | Per-token arrays, token axis last. `raw_*` and `tuned_*` (`nll`, `kl`, `top1`, `agree`, `rank`, `entropy`), `cos`, `rel_dist`, `norm`, `native_nll`, `unigram_*`, `target_id`, `doc`, `context` |
| `measure/manifest.json` | How the run was produced |
| `report/` | `summary.json`, `curves.csv`, and plots of every curve with CIs (tuned solid, raw dashed) plus E3 monotonicity |

State indexing: state 0 = embeddings, state *l* = output of block *l* (before the final norm),
state L = last block.

## Layout

```
depthprobe/
  model.py       loading, residual-stream hooks, readout, random-init and shuffle controls
  data.py        corpora, document boundaries, sliding-window scoring
  lens.py        tuned lens: training and evaluation
  measure.py     E0 check and per-token metrics
  analysis.py    bootstrap CIs, monotonicity, gain profile, plots
  provenance.py  manifests
scripts/pilot_gpt2.sh
PREREGISTRATION.md   thresholds, fixed before confirmatory runs
```
