#!/usr/bin/env bash
# Model-agnostic pilot (E0 check, E1/E2/E3 with E5 controls) for any supported decoder:
# GPT-2, Pythia, Llama, Mistral, Qwen2/3, and mixture-of-experts models such as Qwen3-MoE, Mixtral, OLMoE.
# Downloads nothing. Steps whose outputs already exist are skipped, so an interrupted run can be restarted.
#
#   MODEL=/path/models/qwen3-8b-base NAME=qwen3-8b OUT=/path/runs \
#   WIKI=/path/data/wikitext/wikitext-103-raw-v1 bash scripts/pilot.sh
#
# bfloat16 models (e.g. 30B MoE): DTYPE=bfloat16 READOUT_DTYPE=model
set -euo pipefail

: "${MODEL:?set MODEL to a local model directory}"
: "${NAME:?set NAME (output directory and plot labels)}"
: "${OUT:?set OUT to the directory for results}"
: "${WIKI:?set WIKI to the local wikitext-103-raw-v1 parquet directory}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-float32}"
READOUT_DTYPE="${READOUT_DTYPE:-float32}"
CTX="${CTX:-1024}"
STRIDE="${STRIDE:-$((CTX / 2))}"
BATCH="${BATCH:-4}"
CHUNK="${CHUNK:-512}"
TRAIN_TOKENS="${TRAIN_TOKENS:-4000000}"
VAL_TOKENS="${VAL_TOKENS:-200000}"
VARIANTS="${VARIANTS:-pretrained random shuffle_all shuffle_middle}"
LENS_TF32="${LENS_TF32:-0}"
PREPEND_BOS="${PREPEND_BOS:-0}"
# Final-layer agreement between the readout and the model's own logits. A systematic mismatch raises the
# mean; isolated rounding only the maximum. One bfloat16 step is 0.125 nats for logits in [16, 32) and
# 0.25 in [32, 64), so bfloat16 runs get a wide maximum and rely on the mean.
if [[ "$DTYPE" == float32 ]]; then
    MAX_FINAL_NLL_ERROR="${MAX_FINAL_NLL_ERROR:-0.1}"
    MAX_MEAN_FINAL_NLL_ERROR="${MAX_MEAN_FINAL_NLL_ERROR:-0.001}"
else
    MAX_FINAL_NLL_ERROR="${MAX_FINAL_NLL_ERROR:-0.5}"
    MAX_MEAN_FINAL_NLL_ERROR="${MAX_MEAN_FINAL_NLL_ERROR:-0.01}"
fi
PY="${PY:-python}"
RUN="$OUT/$NAME"

# Every result-affecting setting is stamped; a rerun with different settings refuses to mix outputs.
mkdir -p "$RUN"
stamp="MODEL=$MODEL
MODEL_REVISION=${MODEL_REVISION:-}
WIKI=$WIKI
DEVICE=$DEVICE
DTYPE=$DTYPE
READOUT_DTYPE=$READOUT_DTYPE
CTX=$CTX
STRIDE=$STRIDE
PREPEND_BOS=$PREPEND_BOS
EVAL_TOKENS=${EVAL_TOKENS:-}
TRAIN_TOKENS=$TRAIN_TOKENS
VAL_TOKENS=$VAL_TOKENS
LENS_TF32=$LENS_TF32"
if [[ -f "$RUN/settings.env" ]]; then
    if [[ "$(cat "$RUN/settings.env")" != "$stamp" ]]; then
        echo "Settings differ from the earlier run in $RUN:" >&2
        diff <(cat "$RUN/settings.env") <(printf '%s\n' "$stamp") >&2 || true
        echo "Use a new NAME, or delete $RUN to start over." >&2
        exit 1
    fi
elif [[ -n "$(ls -A "$RUN")" ]]; then
    echo "$RUN already holds outputs from a run without a settings stamp; their settings are unknown." >&2
    echo "Use a new NAME, or write $RUN/settings.env yourself after confirming the earlier settings." >&2
    exit 1
else
    printf '%s\n' "$stamp" > "$RUN/settings.env"
fi

common=(--model "$MODEL" --device "$DEVICE" --dtype "$DTYPE" --readout-dtype "$READOUT_DTYPE"
        --ctx "$CTX" --batch-size "$BATCH" --chunk "$CHUNK")
if [[ "$PREPEND_BOS" == 1 ]]; then
    common+=(--prepend-bos)
fi
if [[ -n "${MODEL_REVISION:-}" ]]; then
    common+=(--model-revision "$MODEL_REVISION")
fi
eval_limit=()
if [[ -n "${EVAL_TOKENS:-}" ]]; then
    eval_limit=(--max-tokens "$EVAL_TOKENS")
fi
lens_extra=()
if [[ "$LENS_TF32" == 1 ]]; then
    lens_extra=(--tf32)
fi

step() { echo "[pilot] $NAME: $*"; }

if [[ -f "$RUN/check/check.json" ]]; then
    step "check done, skipping"
else
    step "E0 check"
    "$PY" -m depthprobe check "${common[@]}" --data "$WIKI" --split test --stride "$STRIDE" \
        ${eval_limit[@]+"${eval_limit[@]}"} --out "$RUN/check" --overwrite
fi

# E0 gate: stop before the expensive lens training if scoring or the final-layer readout is wrong.
if ! "$PY" - "$RUN/check/check.json" "$MAX_FINAL_NLL_ERROR" "$MAX_MEAN_FINAL_NLL_ERROR" "$READOUT_DTYPE" "$DTYPE" <<'PYCHECK'
import json, sys
result = json.load(open(sys.argv[1]))["result"]
max_limit, mean_limit, readout_dtype, dtype = float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5]
worst, mean, gap = result["max_abs_final_nll_error"], result["mean_abs_final_nll_error"], result["max_abs_library_loss_gap"]
print(f"[pilot] E0: final-layer NLL error max {worst:.3g} (limit {max_limit}), mean {mean:.3g} (limit {mean_limit}); "
      f"library loss gap {gap:.3g}")
failed = False
if gap >= 1e-3:
    print("[pilot] E0 FAILED: scoring does not match the library's own loss (label alignment or an extra loss term).")
    failed = True
if worst > max_limit or mean > mean_limit:
    print("[pilot] E0 FAILED: the readout of the last state does not reproduce the model's own logits.")
    if dtype != "float32" and readout_dtype != "model":
        print("[pilot] For a bfloat16 model set READOUT_DTYPE=model, with a new NAME.")
    failed = True
sys.exit(1 if failed else 0)
PYCHECK
then
    exit 1
fi

if [[ -f "$RUN/unigram.json" ]]; then
    step "unigram done, skipping"
else
    step "unigram baseline"
    "$PY" -m depthprobe unigram --model "$MODEL" --data "$WIKI" --split train \
        --max-tokens "$TRAIN_TOKENS" --out "$RUN/unigram.npy" --overwrite
fi

measured=()
for variant in $VARIANTS; do
    case "$variant" in
        pretrained) control=() ;;
        random) control=(--init random --init-seed 0) ;;
        shuffle_all) control=(--shuffle all --shuffle-seed 0) ;;
        shuffle_middle) control=(--shuffle middle --shuffle-seed 0) ;;
        *) echo "Unknown variant: $variant (use pretrained, random, shuffle_all, shuffle_middle)" >&2; exit 1 ;;
    esac
    # lens.json is written atomically and only after lens.pt is complete.
    if [[ -f "$RUN/$variant/lens/lens.json" ]]; then
        step "$variant lens done, skipping"
    else
        step "$variant: training tuned lens"
        "$PY" -m depthprobe train-lens "${common[@]}" ${control[@]+"${control[@]}"} ${lens_extra[@]+"${lens_extra[@]}"} \
            --data "$WIKI" --train-tokens "$TRAIN_TOKENS" --val-tokens "$VAL_TOKENS" \
            --out "$RUN/$variant/lens" --overwrite
    fi
    if [[ -f "$RUN/$variant/measure/manifest.json" ]]; then
        step "$variant measure done, skipping"
    else
        step "$variant: measuring"
        "$PY" -m depthprobe measure "${common[@]}" ${control[@]+"${control[@]}"} --data "$WIKI" --split test \
            --stride "$STRIDE" ${eval_limit[@]+"${eval_limit[@]}"} --lens "$RUN/$variant/lens/lens.pt" \
            --unigram "$RUN/unigram.npy" --max-final-nll-error "$MAX_FINAL_NLL_ERROR" \
            --label "$NAME-$variant" --out "$RUN/$variant/measure" --overwrite
    fi
    measured+=("$RUN/$variant/measure")
done

step "report"
"$PY" -m depthprobe analyze "${measured[@]}" --out "$RUN/report" --overwrite
step "done: $RUN/report"
