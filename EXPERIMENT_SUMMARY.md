AI GENERATED

# Middle-block GPT-2 fine-tuning: preliminary experiment summary

This is an AI-generated account of the saved experiment artifacts, not a
human-reviewed research conclusion. Numerical results below were read from each
run's `summary.json` and `report/paired_changes.json`. No training code or existing
run artifacts were changed to prepare this document.

## Question and setup

Can a small middle segment learn better next-token predictions through the
original frozen final normalization and vocabulary head, while remaining useful
when the original downstream blocks are restored?

- Model: pretrained GPT-2, 12 Transformer blocks. Block numbers are 1-based and
  inclusive. Each experiment starts from the original pretrained model.
- Training: selected actual Transformer blocks only. The original prefix is
  cached/frozen; the suffix is omitted from the training forward path, then
  restored for full-model validation. The final normalization and head are frozen.
- Data: local WikiText-103, training documents shuffled with seed 0 before taking
  a nominal 1,000,000-token budget; 999,024 scored next-token targets per epoch.
  Three epochs, 96 optimizer updates total.
- Optimizer/settings: AdamW, learning rate 1e-5, 5% warmup and cosine decay,
  batch size 32, context 1,024, seed 0, no gradient accumulation.
- Compute: RTX 4090, BF16 autocast with FP32 master parameters, SDPA and TF32.
- Validation: 247,288 scored targets, overlapping windows with stride 512;
  targets are scored once. These are validation results, not final test results.
- Boundary convention: continuous token stream, no inserted EOS separators;
  cross-document attention is allowed within a window, but not across batch rows.

## Regularizers tried separately and together

Single-regularizer experiments used `loss = next-token CE + 1.0 * regularizer`.
The combined experiment used `loss = CE + 1.0 * RMS_L2 + 1.0 * KL`.
Equal coefficients do not imply equally strong constraints across different regularizers.

1. **Residual-space L2:** mean squared segment-exit activation change, divided
   per feature by the original sample STD before squaring.
2. **Parameter-free RMS-space L2:** apply
   `rms(h) = h / sqrt(mean_feature(h^2) + 1e-6)` independently to each token's
   original and modified segment-exit vectors, then compare them using the
   original per-feature sample STDs measured in that same normalized space.
   This transform has no learned gain, bias, or mean subtraction; it does not use
   GPT-2's learned final LayerNorm parameters. The prediction path still uses
   the model's original frozen final norm/head.
3. **KL:** `KL(original exit-lens distribution || modified exit-lens distribution)`.
   Both distributions use the original frozen final norm/head directly at the
   segment exit. The teacher is the original **exit lens**, not the original
   full-model output after the suffix.

L2 reference STDs are per-feature sample STDs over original scored training
activations, frozen after caching and floored at 1e-4. L2 is a soft average
penalty, not a hard per-activation limit. RMS-space L2 constrains normalized
coordinates, not the overall raw per-token activation magnitude.

## Restored full-model validation results

Lower CE/perplexity and higher accuracy are better. Accuracy changes are absolute
percentage points relative to the original model.

| Run | Trainable blocks | Regularizer | CE | Perplexity | Accuracy | Accuracy change |
|---|---|---|---:|---:|---:|---:|
| Original baseline | None | None | 3.269051 | 26.2864 | 39.6542% | — |
| `runs/gpt2-serious` | 4–8 (5/12) | Residual L2 | 3.688903 | 40.0009 | 35.4550% | -4.1992 pp |
| `runs/gpt2-rms` | 4–8 (5/12) | RMS L2 | 3.713771 | 41.0081 | 35.2007% | -4.4535 pp |
| `runs/gpt2-kl` | 4–8 (5/12) | KL | 3.523997 | 33.9197 | 37.6132% | -2.0409 pp |
| `runs/gpt2-kl-6-7` | 6–7 (2/12) | KL | 3.318205 | 27.6108 | 39.4285% | -0.2256 pp |
| `runs/gpt2-both-6-7` | 6–7 (2/12) | RMS L2 + KL | 3.312638 | 27.4575 | 39.4714% | -0.1828 pp |

**No completed experiment improved the restored full model over its original
baseline.** The two-block combined run preserved it best among the variants tried,
with a small observed advantage over two-block KL alone.

### Paired validation uncertainty

Saved paired 95% document-bootstrap intervals use 60 validation documents.
Positive CE change and negative accuracy change mean degradation.

| Run | CE change [95% interval] | Accuracy change, pp [95% interval] |
|---|---|---|
| Residual L2, 4–8 | +0.41985 [+0.41099, +0.42938] | -4.1992 [-4.3993, -4.0005] |
| RMS L2, 4–8 | +0.44472 [+0.43562, +0.45470] | -4.4535 [-4.6364, -4.2624] |
| KL, 4–8 | +0.25495 [+0.24850, +0.26171] | -2.0409 [-2.1921, -1.8814] |
| KL, 6–7 | +0.04915 [+0.04644, +0.05199] | -0.2256 [-0.3145, -0.1208] |
| RMS L2 + KL, 6–7 | +0.04359 [+0.04105, +0.04616] | -0.1828 [-0.2759, -0.0752] |

These are within-validation resampling intervals, not uncertainty across training
seeds. Continuous-stream cross-document context also limits a strict
independent-document interpretation.

## What we observed

- **The local training objective improves without improving the restored model.**
  Epoch-average truncated-path CE fell from 6.603 to 5.040 for residual L2,
  6.603 to 5.030 for RMS L2, 6.644 to 5.332 for five-block KL, and 8.069 to 6.732
  for two-block KL, and 8.069 to 6.740 for two-block RMS L2 + KL.
  These are training-path metrics, not full-model validation
  CE, and the two block ranges have different exit depths.
- **KL preserved downstream performance better than either L2 variant at the
  tested coefficient and block range.** This does not establish a general
  advantage over tuned L2 or a combination of the two penalties.
- **Parameter-free RMS L2 did not improve these validation metrics over residual
  L2 at coefficient 1.** Their scales and invariances differ; only one setting
  and one seed were tried.
- **The smaller KL run degraded the full model much less.** Accuracy loss fell
  from 2.0409 to 0.2256 percentage points, and CE increase from 0.25495 to 0.04915.
  However, this changed both block count and block positions/exit depth, so we
  cannot attribute the difference to block count alone.
- **Mean L2 does not prevent large individual changes.** In the RMS-L2 run,
  the maximum observed normalized coordinate deviation reached about 5.61
  reference STDs during epoch 3, despite an epoch-average penalty of 0.2125.
  The maximum is over training forwards, not a final held-out drift measurement.
- **Combining RMS L2 with KL gave a small observed improvement over KL alone at
  the same two-block range.** CE was lower by 0.00557, perplexity by 0.1533, and
  accuracy higher by 0.0429 percentage points. Both still degraded the original
  model. The intervals above compare each run to baseline, not directly to each
  other; no significance claim for the combined-versus-KL difference is made.
  There is only one training seed and no two-block L2-only control yet.
- In the combined run's third epoch, the mean unweighted RMS-L2 penalty was
  0.04444 and KL was 0.31198; the maximum observed L2 coordinate deviation was
  3.08 reference STDs. The terms' magnitudes alone do not determine their
  relative gradient influence.

A plausible interpretation is that learning to predict directly from a middle
exit changes representations in ways the frozen suffix does not fully tolerate.
That is a hypothesis consistent with the results, not an isolated causal finding.

## Runtime and diagnostics

Measured training-loop throughput (including the training data-loading loop,
but excluding initial caching, validation and export):

| Configuration | Approximate scored tokens/sec | Peak allocated GPU memory |
|---|---:|---:|
| Five-block residual L2 | 77–80k | 12.68 GiB |
| Five-block RMS L2 | 80–81k | 12.68 GiB |
| Five-block KL | 47–48k | 12.87 GiB |
| Two-block KL | 60k | 6.48 GiB |
| Two-block RMS L2 + KL | 64k | 6.48 GiB |

KL adds vocabulary-distribution work; fewer trainable blocks did not eliminate
that cost. These are observed runs, not controlled hardware benchmarks.

The historical `runs/gpt2-kl` max-STD values of zero were **unmeasured
placeholders**, not evidence of zero activation drift. The UI/logging fix hides
that L2-only diagnostic for KL/CE-only runs and writes null/blank instead. The
two-block KL run uses this fix. Small measured L2 deviations also receive more
display precision. Combined runs display and save both raw penalties separately.
The implementation passed 109 tests after removal of cross-run cache reuse,
including direct combined-gradient checks and zero-coefficient controls across
tiny GPT-2, Llama and GPT-NeoX models, cache-rebatch token/reference alignment,
and chunked evaluation equivalence to native full-model logits. A real-4090
combined-path smoke run and the full combined experiment also completed.

## Limits and next experiments

- One full combined L2+KL run is now included, at blocks 6–7 and both coefficients
  set to 1. It has not established improvement over the original model.
- No coefficient sweep, repeated-seed study, or matched-exit block-count sweep
  has been completed. No full-scale CE-only control is included.
- Combined penalties should have separately recorded coefficients and raw
  components; matching the numerical coefficient alone does not match strength.
- Useful controls include stronger KL, an RMS-L2/KL coefficient grid, and a
  smaller segment with the same exit depth as the five-block run. These are
  proposals, not completed experiments.
- Keep using validation for selection and reserve the test split for the chosen
  configuration. Repeated validation comparisons can overfit selection.
- A million-token activation cache occupies about 8.6 GiB regardless of the
  number of trained blocks here. Runs were moved off the SSD to
  `/media/user/archive/Depth-Probe/runs`, with local `runs/` retained as a symlink.
  New runs still generate their own caches; cross-run cache reuse was removed.
  Existing experiment artifacts were preserved and remain outside Git.

Each run directory contains its own `summary.json`, `training.csv`, before/after
evaluation arrays, and `report/` outputs. To view a saved report, for example:

```bash
python -m depthprobe report-segment runs/gpt2-kl-6-7 --show
```
