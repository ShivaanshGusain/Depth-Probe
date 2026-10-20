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
