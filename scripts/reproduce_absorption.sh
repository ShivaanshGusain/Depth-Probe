#!/usr/bin/env bash
# Reproduce the perturbation-absorption probe of Jing et al. (arXiv 2606.15161), then vary the two choices
# that could create its result: FP16 precision and whole-tensor (Frobenius) noise scaling.
#   paper            float16, frobenius   (their recipe)
#   fp32             float32, frobenius   (is small-eta amplification an FP16 rounding effect?)
#   fp32-nosink      float32, frobenius, sinks neither perturbed nor measured (the sink effect alone)
#   fp32-token       float32, token       (does scaling against the whole tensor, i.e. the sink tokens, matter?)
#   fp32-token-nosink  float32, token, sinks neither perturbed nor measured
# Downloads nothing. Finished configurations are skipped; settings are stamped.
# The defaults are a quick first pass (3 sites, 16 sequences of 2048, 3 seeds); each run logs its number of
# suffix passes and a time estimate. For the full sweep: SITES=all MAX_TOKENS=262144 SEEDS=5 ETAS=0,1e-4,1e-3,1e-2,1e-1
# SITES are state indices (output of block l); 1,12,24 suits 32-36 layer models (Qwen3-8B, LLaMA-2-7B), not GPT-2.
# Windows are cut from one token stream, so no window starts with BOS unless PREPEND_BOS=1; LLaMA-2 expects it
# (set PREPEND_BOS=1), Qwen3 has no BOS token (leave 0).
#
#   MODEL=/path/models/qwen3-8b DATA=/path/c4-validation.00000-of-00008.json.gz OUT=/path/runs \
#     bash scripts/reproduce_absorption.sh
set -euo pipefail

: "${MODEL:?set MODEL to a local model directory (the paper uses LLaMA-2-7B, Qwen3-8B and others)}"
: "${DATA:?set DATA to a local corpus: a C4 .json.gz shard (the paper calibrates on C4) or a WikiText parquet dir}"
: "${OUT:?set OUT to the directory for results}"
NAME="${NAME:-absorption}"
DEVICE="${DEVICE:-auto}"
SPLIT="${SPLIT:-validation}"
CTX="${CTX:-2048}"            # the paper's sequence length; GPT-2 needs CTX=1024
STRIDE="${STRIDE:-$CTX}"      # non-overlapping sequences
MAX_TOKENS="${MAX_TOKENS:-32768}"
BATCH="${BATCH:-1}"
REPLICAS="${REPLICAS:-4}"
SITES="${SITES:-1,12,24}"
ETAS="${ETAS:-0,1e-4,1e-2}"
SEEDS="${SEEDS:-3}"
PREPEND_BOS="${PREPEND_BOS:-0}"
CONFIGS="${CONFIGS:-paper fp32 fp32-nosink fp32-token fp32-token-nosink}"
PY="${PY:-python}"
RUN="$OUT/$NAME"

mkdir -p "$RUN"
stamp="MODEL=$MODEL
DATA=$DATA
DEVICE=$DEVICE
SPLIT=$SPLIT
CTX=$CTX
STRIDE=$STRIDE
MAX_TOKENS=$MAX_TOKENS
SITES=$SITES
ETAS=$ETAS
SEEDS=$SEEDS
PREPEND_BOS=$PREPEND_BOS"
if [[ -f "$RUN/settings.env" ]]; then
    if [[ "$(cat "$RUN/settings.env")" != "$stamp" ]]; then
        echo "Settings differ from the earlier run in $RUN:" >&2
        diff <(cat "$RUN/settings.env") <(printf '%s\n' "$stamp") >&2 || true
        echo "Use a new NAME, or delete $RUN to start over." >&2
        exit 1
    fi
elif [[ -n "$(ls -A "$RUN")" ]]; then
    echo "$RUN holds outputs without a settings stamp; use a new NAME." >&2
    exit 1
else
    printf '%s\n' "$stamp" > "$RUN/settings.env"
fi

common=(--model "$MODEL" --device "$DEVICE" --data "$DATA" --split "$SPLIT" --ctx "$CTX" --stride "$STRIDE"
        --max-tokens "$MAX_TOKENS" --batch-size "$BATCH" --replicas "$REPLICAS" --sites "$SITES"
        --etas "$ETAS" --seeds "$SEEDS")
if [[ "$PREPEND_BOS" == 1 ]]; then
    common+=(--prepend-bos)
fi

for config in $CONFIGS; do
    case "$config" in
        paper) options=(--dtype float16 --scale frobenius) ;;
        fp32) options=(--dtype float32 --scale frobenius) ;;
        fp32-nosink) options=(--dtype float32 --scale frobenius --exclude-sinks) ;;
        fp32-token) options=(--dtype float32 --scale token) ;;
        fp32-token-nosink) options=(--dtype float32 --scale token --exclude-sinks) ;;
        *) echo "Unknown configuration: $config" >&2; exit 1 ;;
    esac
    if [[ -f "$RUN/$config/manifest.json" ]]; then
        echo "[absorption] $config done, skipping"
    else
        echo "[absorption] $config"
        "$PY" -m depthprobe reproduce-absorption "${common[@]}" "${options[@]}" --label "$config" \
            --out "$RUN/$config" --overwrite
    fi
done
echo "[absorption] done: compare $RUN/*/absorption.png and the r_l values in each manifest.json"
