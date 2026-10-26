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

## Absorption tests v2 (fixed 2026-10-26, before the v2 run; revised the same day after an independent review)

v1 (`storage/runs/absorption-tests`, Qwen3-8B, C4 validation shard 00000, eta 1e-3) found: noise grows in absolute
terms while the clean signal grows ~3x faster; real swap directions survive ~2x more than noise; random-init
amplifies relative noise and permuted weights explode it. The relative comparison with the controls and the
learned status of the selectivity were not pre-registered, so v2 tests them on fresh data with stronger controls.

**Fixed settings.** `scripts/absorption_tests.sh` defaults: float32, per-token noise, sinks excluded, sites 1, 12,
24, 32 windows of 2048 tokens, 3 seeds, eta = 1e-2, replicas 5. Data: C4 validation shard **00001** (fresh; v1
used 00000). Model: Qwen3-8B; a second model may be added only after its name is written here before its run.
eta = 1e-2 only: in v1 the trained model and the random-init control were linear between 1e-3 and 1e-2; the
permuted control was not (R_L 0.68 -> 1.13 at site 1, 0.075 -> 0.56 at site 12), which the saturation rule
below handles.

**Perturbations.** noise (independent Gaussian per token); real (the difference a swapped early token makes at the
injection state, shared across later tokens); rotated (the real difference times a fixed random orthogonal matrix
per seed: same per-token sizes and token-to-token correlations, random feature directions). All are scaled to eta
times each token's norm.

**Controls.** Deciding: shuffled middle layers (seed 0; seeds 1 and 2 if run, as in E6) and permuted block
weights. Reported only: random-init weights. Every control runs with token swaps and reuses the trained run's
sinks, swaps and prediction-changed token sets (`--sinks-from`; the analysis refuses otherwise). A comparison
against a control whose noise reaches R_L > 0.1 at the last state (window mean at eta) is "not applicable
(control saturated)" and does not decide; the report names the controls that did decide.

**Rule for "clearly".** A log ratio beyond log 1.25 with its 95% window-bootstrap interval on the same side of 0;
"about equal" = the whole interval within +/- log 1.25; otherwise "unclear". Decided per site. For control
comparisons: `yes` only if every applicable deciding control gives a clearly positive difference; `no` if any
applicable deciding control's interval lies below +log 1.25 (rules out a 1.25x effect); otherwise `unclear`.

| Question | Statistic | Answer |
|---|---|---|
| Q1 (replication) | absolute noise survival | `shrinks` / `carried unchanged (dilution)` / `grows` |
| Q2 relative (primary for "learned") | log(control relative noise / trained relative noise), relative = survival / clean norm growth | control rule above |
| Q3 | real / rotated survival on prediction-changed and -preserved tokens (feature direction only) | denoising (both clearly > 1) / convergence (preserved about 1, changed clearly > 1) / compression (both about 1) / unclear |
| Q3 v1 replicate (reported) | real / noise survival, same token sets | same labels; real/noise vs rotated/noise shows how much came from correlation across tokens |
| Q3 learned | trained Q3 statistic minus the control's, both token sets (the trained run's token sets) | control rule above, required on both sets |
| Q3 without massive coordinates | Q3 with coordinates whose mean abs value exceeds 20x the median left out of the measurement | same verdict as Q3 = robust; different = depends on massive coordinates; "not applicable" if no such coordinate exists at the site or last state. A measurement-level check only: it does not remove them from the injection |
| Q4 steering | KL of the propagated noise at the last state over KL of a random direction with exactly the same per-token size there (all kept tokens) | `steered away` if clearly below 1, `not steered` if about equal, `steered toward` if clearly above |
| Q4 vs controls (reported) | control Q4 statistic minus trained, control rule above | does the trained model steer away more than the controls? |

Also reported, not deciding: Q4 per unit of size after the final norm (separates directions the final norm
discards from steering), Q4 for real and rotated perturbations, the fixed-size random reference used in the first
v2 draft, massive-coordinate energy shares, and output KL against the carried-unchanged perturbation.

Q4 prediction recorded before the run, from rough v1 aggregates (noise after the layers over a random direction
at the last layer, per unit size): about 0.4 at site 1 (steered away), 1.3 at site 12 (borderline), 1.7 at
site 24 (steered toward). The teammate's hypothesis (noise ends up nearly orthogonal to what the prediction
reads) predicts clearly below 1 at every site. v1-based expectation for Q2 relative: `yes` against permuted
where it is not saturated; the shuffled-middle control is new, so no expectation.

## Amendment log

| Date | Change | Reason |
|---|---|---|
| 2026-10-26 | Absorption tests v2 revised before any v2 run: Q3 against rotated real perturbations; shuffled-middle and permuted controls decide, random-init reported only; controls reuse the trained token sets; saturation rule; Q4 against a size-matched random direction | An independent review found that the first v2 draft confounded Q3 with cross-token correlation, Q4 with size and token weighting, and used controls that are degenerate or saturated |
