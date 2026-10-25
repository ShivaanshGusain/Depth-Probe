# Pre-registration (draft, fill in before confirmatory runs)

Fill this in **after** the GPT-2 small pilot and **before** any run on confirmatory models.
Commit it with a timestamp. Once committed, changes go in an amendment log below, never silently.

- Exploratory (pilot) models, excluded from confirmatory claims: GPT-2 small
- Confirmatory models: _list_
- Primary lens: _tuned / raw_
- Data, context length, stride, token count per model: _..._
- Token filter (`--min-context`), reversal tolerance (`--eps`), first state in E3 (`--from-layer`): _..._

| Sub-hypothesis | Metric (exact definition) | Counts as support | Counts as refutation |
|---|---|---|---|
| H1a convergence | tuned-lens mean KL(p_L ‖ p_l) per state, document-bootstrap 95% CI | _..._ | _..._ |
| H1b per-token monotonicity | fraction of monotone tokens vs random-init and shuffled controls | _..._ | _..._ |
| H1c noise recovery (denoising, E6 v1) | see "E6 v1" below | _..._ | _..._ |
| H1d noise ↔ depth | _define once E7 exists_ | _..._ | _..._ |
| H1e residual structure | _..._ | _..._ | _..._ |

Required number of confirmatory models that must meet each criterion: _..._

## E6 v1: denoise test (draft for review; fill the blanks, then commit before any confirmatory run)

**What v1 can and cannot show.** v1 tests whether later layers *denoise* (H1c): remove noise-like
perturbations while keeping content. A manifold attractor also passes this test (see `perturb-toys`), so v1
cannot establish *diffusion*. The diffusion-specific claims come from E7 (noise maps onto earlier depth) and
from the Jacobian-spectrum signature.

**Coordinates and the noise law (amends the proposal).** A state is read in normalized coordinates (the
direction the next layer norm sees). Write it as signal plus noise with 1/SNR = sigma^2. Adding independent,
isotropic noise whose size is delta relative to the token's total (bulk) size gives
`1 + sigma_k^2 = (1 + sigma_l^2)(1 + delta^2)`, i.e. `log(1 + sigma^2)` adds across injections. The
proposal's `sigma_k^2 = sigma_l^2 + delta^2` is the same law with delta measured against the signal size;
this study uses the variance-preserving form above. Assumptions: the injected noise is independent of the
state and isotropic within the bulk coordinates (massive coordinates excluded), and delta is relative to the
token's bulk norm.

**Exploratory model:** GPT-2 small, float32, WikiText-103 test split, every 4th window, positions with at
least 64 tokens of context, sink positions excluded. Calibration (massive coordinates, PCA and readout
subspaces) on the validation split.

**Primary statistic (fixed now):** selectivity at the last state,
`log A(content) - log A(noise-like)` at matched size, under both definitions:
PCA (`on` = top principal directions carrying 90% of variance, capped at half the bulk width, `off` =
low-variance half; the realised variance share is recorded per state, and the cap can bind in near-isotropic
controls) and readout (`read` / `unread` split of the final head's sensitivity). Metrics compare noisy rows
with the in-batch clean replica; every control measures the trained run's tokens (`--sinks-from`).
Primary size: delta = _0.1_.

**Controls it must beat (fixed now):** shuffled middle layers (seeds 0, 1, 2) and the matched-statistics
control (block weights permuted entrywise). Random weights are reported but do not decide.

**Gate (form fixed now; numbers filled after the GPT-2 run, before confirmatory models):**
trained-minus-control selectivity > _margin_ (log units) with the 95% document-bootstrap interval above
zero, at >= _N_ injection sites, against every deciding control, under _both / either_ definition(s)
(`analyze-perturb --margin --min-sites --rule`; the overall gate fails if any deciding control is missing).
Supporting, not required: a depth trend in log A or log Lambda; noise removed (raw survival < 1) and signal
added (content growth) reported separately; the dilution null A0 reported with every absorption number;
repaired damage measured against the output-space dilution null (the injected perturbation carried unchanged).

**Must hold before any result counts:** `perturb-toys` reproduces the known signatures (exact contraction,
sigma_L/sigma_b for off-manifold noise in the diffusion toys, no selectivity for rotation and point
attractor); zero-noise segments reproduce the full model; the in-batch clean replica matches the clean pass.

## Amendment log

| Date | Change | Reason |
|---|---|---|
