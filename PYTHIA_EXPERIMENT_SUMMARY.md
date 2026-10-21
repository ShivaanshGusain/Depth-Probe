AI GENERATED

# Memory-constrained staged Pythia fine-tuning: preliminary experiment summary

This is an AI-generated account of saved experiment artifacts, not a human-reviewed
research conclusion. Results come from `summary.json`, `report/paired_changes.json`,
and the direct-versus-staged paired comparison saved alongside the runs.

## Goal: improve the full model without training it all on GPU

The intended use case is a model whose complete weights do not fit on the training
GPU. Instead of updating only its prediction head, train small block segments in
separate stages, hoping their improvements compose into a better full model.
Cached frozen-prefix activations let each optimizer stage load only the selected
blocks and final readout onto the GPU. Earlier stages need not backpropagate
through the full model or retain the downstream suffix during training.

**This pilot supports the staged-training idea, but does not yet demonstrate the
complete oversized-model use case.** Pythia-410M fits on our RTX 4090, and these
runs used full-model GPU inference for caching and validation. The implementation
can instead perform that inference on CPU, but does not stream/offload inference
layer by layer. It also keeps the full model in host RAM and writes activation
caches to disk. Nor have we tested a true head-only baseline yet: the direct
baseline below trains the **last block and head together**.

## Shared setup

- Model: `EleutherAI/pythia-410m`, GPT-NeoX architecture, 24 blocks, width 1,024.
  Original revision: `9879c9b5f8bea9051dcb0e68dff21493d67e9d4f`.
  All block numbers are 1-based and inclusive.
- Data: local WikiText-103. Training documents shuffled with seed 0; a nominal
  million-token budget gives 999,024 scored targets per epoch. Stages reuse the
  same training selection; extra epochs do not add new unique training data.
- Validation: 250,010 scored targets across 60 documents, context 1,024, stride
  512; each target is scored once. The test split remains unused.
- AdamW, no weight decay, LR 1e-5, 5% warmup and cosine decay, batch 32,
  inference batch 8, readout chunk 1,024; no gradient accumulation.
- RTX 4090; BF16 autocast with FP32 master parameters, SDPA, TF32, dropout off.
- Continuous token stream, no added EOS separators or document masks;
  cross-document attention is allowed within a window, never across batch rows
  or into future tokens.
- Checkpoint, caches, exports and reports reside under
  `/media/user/archive/Depth-Probe`; repository `runs/` is a symlink there.

## Stage 1: two late blocks, frozen readout

Train blocks **21–22** through the original frozen final normalization and head.
Blocks 1–20 supply cached prefix activations; blocks 23–24 are omitted during
training and restored for full-model validation. All other weights stay frozen.

```text
loss = next-token CE + 1.0 * RMS_L2 + KL_strength * KL
```

RMS_L2 compares original and modified segment-exit activations after parameter-free
per-token RMS normalization (`eps=1e-6`), scaled by frozen original per-feature
sample STDs (`std_floor=1e-4`). It is a soft mean penalty, not a hard bound.
KL is `KL(original exit-lens distribution || modified exit-lens distribution)`,
not KL against the full-model teacher. Equal coefficients do not imply equal
constraint strength.

Each run starts afresh from pretrained weights. The five-epoch run uses its own
five-epoch LR schedule; it is not two additional epochs resumed after the
three-epoch schedule reached zero.

### Restored full-model validation

| Setting | CE ↓ | Perplexity ↓ | Accuracy ↑ |
|---|---:|---:|---:|
| Original | 2.804394 | 16.5171 | 44.8898% |
| KL 1, RMS-L2 1; 3 epochs | 2.815611 | 16.7034 | 44.4506% |
| KL 5, RMS-L2 1; 3 epochs | 2.789391 | 16.2711 | 44.8630% |
| KL 5, RMS-L2 1; 5 epochs | 2.787545 | 16.2411 | 44.8450% |

Increasing KL from 1 to 5 changed the observed result from degradation to improved
full-model CE/perplexity. Five epochs added a modest further improvement, not
evidence that convergence was reached. Accuracy did not improve in stage 1.

### Held-out partial model: blocks 1–22 → frozen final norm/head

These use the **same validation tokens** as the full model, not epoch-average
training metrics. The initial KL-1 run predates partial validation measurement;
no partial before/after result is available for it.

| Setting | CE ↓ | Perplexity ↓ | Accuracy ↑ |
|---|---:|---:|---:|
| Original partial model | 3.625301 | 37.5360 | 37.6169% |
| KL 5; 3 epochs | 3.410918 | 30.2930 | 39.0896% |
| KL 5; 5 epochs | 3.390265 | 29.6738 | 39.1836% |

## Stage 2: one epoch on the last block and prediction head

Two matched adaptation runs train **block 24 + the untied prediction head** for
one epoch of ordinary next-token CE, with no KL or L2. The final normalization
and all other blocks remain frozen. Each gets a fresh optimizer and one-epoch
LR schedule, identical data order, LR and batch settings.

- **Direct:** start from the original pretrained model.
- **Staged:** start from the five-epoch KL-5/RMS-L2-1 stage-1 checkpoint.

### Full-model before/after

| Starting point | CE before → after ↓ | Perplexity before → after ↓ | Accuracy before → after ↑ |
|---|---:|---:|---:|
| Direct | 2.804394 → 2.757813 | 16.5171 → 15.7653 | 44.8898% → 45.4882% |
| Staged | 2.787545 → **2.737648** | 16.2411 → **15.4506** | 44.8450% → **45.6038%** |

### Partial model before/after, still exiting at block 22

The partial readout uses the current head, which changes during stage 2; block 24
is not on this partial path. Consequently its stage-2 changes reflect head
adaptation, not any additional updates to blocks 1–22.

| Starting point | CE before → after ↓ | Perplexity before → after ↓ | Accuracy before → after ↑ |
|---|---:|---:|---:|
| Direct | 3.625301 → 3.622790 | 37.5360 → 37.4419 | 37.6169% → 37.6513% |
| Staged | 3.390265 → 3.389541 | 29.6738 → 29.6523 | 39.1836% → 39.2056% |

## Paired uncertainty and interpretation

Intervals are paired 95% document-bootstrap intervals, 1,000 resamples, seed 0,
60 validation documents. Accuracy changes below are absolute percentage points.

| Comparison | CE change [95% interval] | Accuracy change, pp [95% interval] |
|---|---|---|
| Stage 1, KL 5 / 3 epochs minus original | -0.01500 [-0.01794, -0.01190] | -0.0268 [-0.1437, +0.0760] |
| Stage 1, KL 5 / 5 epochs minus original | -0.01685 [-0.02060, -0.01287] | -0.0448 [-0.1712, +0.0747] |
| Direct adaptation minus its starting point | -0.04658 [-0.04932, -0.04416] | +0.5984 [+0.4999, +0.6947] |
| Staged adaptation minus its starting point | -0.04990 [-0.05317, -0.04701] | +0.7588 [+0.6740, +0.8437] |
| Final staged minus final direct | -0.02016 [-0.02219, -0.01815] | +0.1156 [+0.0297, +0.2004] |

**The staged model beat direct one-epoch last-block/head adaptation in this pilot.**
Its final perplexity was lower by 0.3147 and accuracy higher by 0.1156 pp.
Most of its CE advantage already existed before stage 2; the observed stage-2 CE
reduction was only about 0.00332 larger than the direct arm's reduction. This does
not isolate a special interaction between the stages.

These are within-validation uncertainty estimates, not across-seed uncertainty.
Cross-document context limits strict independent-document interpretation.
Repeated validation-guided choices of KL and epochs also make this exploratory,
not a preregistered confirmatory result on Pythia.

## Memory, runtime and limitations

- Logged training peak allocated GPU memory: approximately **4.86 GiB** for
  two-block stage 1 and **3.50 GiB** for last-block/head adaptation. These figures
  cover the measured training loop, not end-to-end caching/validation peaks,
  total reserved VRAM, or all GPU processes.
- Observed training-loop throughput: approximately 57–65k scored tokens/sec for
  stage 1, 110–112k for stage 2; not end-to-end or controlled speed comparisons.
- Each million-token activation cache is approximately 11.4 GiB. Caches are fresh
  per run, not shared. CPU RAM, disk capacity and cache I/O are additional costs.
- The staged arm uses five extra passes over the same data: 160 stage-1 updates
  plus 32 stage-2 updates versus 32 direct updates. This is not an equal-compute,
  equal-epoch or equal-token-exposure comparison.
- One seed, one model, one late block pair, one stage-2 epoch. No true head-only,
  six-epoch direct, full-model fine-tuning, or matched-budget control was run.
- No claim that the partial model matches full-model quality, that more epochs
  always help, or that these settings transfer to larger models.
- The next decisive controls are head-only adaptation, matched-budget direct
  adaptation, repeated seeds, and an actual model that exceeds GPU weight memory
  with CPU or streamed inference. Reserve test evaluation for a selected protocol.

## Artifacts and reproduction

Under `runs/` (large artifacts remain outside Git):

- `pythia-410m-both-21-22-20261021`: KL 1, 3 epochs.
- `pythia-410m-both-21-22-kl5-20261021`: KL 5, 3 epochs.
- `pythia-410m-both-21-22-kl5-5epochs-20261021`: KL 5, 5 epochs.
- `pythia-last-head-1epoch-20261021/{direct,staged}`: stage-2 comparison.
- `pythia-last-head-1epoch-20261021/paired_staged_minus_direct.json`: paired
  final-arm comparison, after verifying document, target ID and target index alignment.

Runs contain manifests, training CSVs, full and (where measured) partial per-token
validation arrays, block checkpoints, reusable full-model exports and reports.
The local checkpoint must already exist; training scripts never download.

```bash
KL_STRENGTH=5 EPOCHS=5 OUT=/media/user/archive/Depth-Probe/runs/my-stage1 \
  bash scripts/train_pythia_410m_4090.sh \
  --model-revision 9879c9b5f8bea9051dcb0e68dff21493d67e9d4f
STAGED_MODEL=/media/user/archive/Depth-Probe/runs/my-stage1/model \
  bash scripts/compare_pythia_last_head_4090.sh
```

The implementation passed 124 offline tiny-model tests, including partial-exit
readout equivalence and non-mutation checks, head/segment updates with frozen
normalization, and rejection of a trainable head with a moving KL teacher.
