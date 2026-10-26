#!/usr/bin/env bash
# Answer three questions about perturbation absorption on one model, with rules fixed before the run:
#   Q1  Does the noise itself shrink, or is it carried unchanged while the clean signal grows (dilution)?
#   Q2  Is the shrinking learned: clearly more than in a random-init and a permuted-weight copy of the model?
#   Q3  Denoising, convergence or compression: does a real perturbation (the difference a swapped earlier token
#       makes) survive better than noise of the same size on the same tokens, for tokens whose prediction the
#       swap changes and for tokens whose prediction it preserves?
# Settings follow the cleanest reproduction configuration: float32, per-token noise, sink tokens excluded
# (the controls reuse the trained model's sink positions). Answers use PRIMARY_ETA; "clearly" means a ratio
# beyond MARGIN_RATIO with the 95% window-bootstrap interval on the same side. Downloads nothing; finished
# runs are skipped; settings are stamped and cannot change between attempts.
#
#   MODEL=/workspace/models/qwen3-8b DATA=/workspace/data/c4/en/c4-validation.00000-of-00008.json.gz \
#     OUT=$PWD/storage/runs bash scripts/absorption_tests.sh
#
# Time on the GX10 for Qwen3-8B (from the float32 reproduction, ~15 s per suffix pass): the trained run with
# swaps ~75 min, each control ~50 min, about 3 hours in all. The log prints the pass count and a time estimate.
set -euo pipefail

: "${MODEL:?set MODEL to a local model directory}"
: "${DATA:?set DATA to a local corpus (e.g. a C4 .json.gz shard)}"
: "${OUT:?set OUT to the directory for results}"
NAME="${NAME:-absorption-tests}"
DEVICE="${DEVICE:-auto}"
SPLIT="${SPLIT:-validation}"
CTX="${CTX:-2048}"
STRIDE="${STRIDE:-$CTX}"
MAX_TOKENS="${MAX_TOKENS:-65536}"   # 32 sequences of 2048
BATCH="${BATCH:-1}"
REPLICAS="${REPLICAS:-4}"
SITES="${SITES:-1,12,24}"
ETAS="${ETAS:-1e-3,1e-2}"           # float32 is linear here; the eta = 0 floor was ~1e-5 in the reproduction
SEEDS="${SEEDS:-3}"
PREPEND_BOS="${PREPEND_BOS:-0}"
PRIMARY_ETA="${PRIMARY_ETA:-1e-3}"
MARGIN_RATIO="${MARGIN_RATIO:-1.25}"
CONTROLS="${CONTROLS:-random permuted}"
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
PREPEND_BOS=$PREPEND_BOS
PRIMARY_ETA=$PRIMARY_ETA
MARGIN_RATIO=$MARGIN_RATIO"
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
        --etas "$ETAS" --seeds "$SEEDS" --dtype float32 --scale token --exclude-sinks)
if [[ "$PREPEND_BOS" == 1 ]]; then
    common+=(--prepend-bos)
fi

if [[ -f "$RUN/trained/manifest.json" ]]; then
    echo "[tests] trained done, skipping"
else
    echo "[tests] trained model, with real token swaps"
    "$PY" -m depthprobe reproduce-absorption "${common[@]}" --real-swaps --label trained \
        --out "$RUN/trained" --overwrite
fi

runs=("$RUN/trained")
for control in $CONTROLS; do
    case "$control" in
        random) options=(--init random --init-seed 0) ;;
        permuted) options=(--init permuted --init-seed 0) ;;
        *) echo "Unknown control: $control" >&2; exit 1 ;;
    esac
    if [[ -f "$RUN/$control/manifest.json" ]]; then
        echo "[tests] $control done, skipping"
    else
        echo "[tests] control: $control"
        "$PY" -m depthprobe reproduce-absorption "${common[@]}" "${options[@]}" --sinks-from "$RUN/trained" \
            --label "$control" --out "$RUN/$control" --overwrite
    fi
    runs+=("$RUN/$control")
done

"$PY" -m depthprobe analyze-absorption "${runs[@]}" --reference trained --eta "$PRIMARY_ETA" \
    --margin-ratio "$MARGIN_RATIO" --out "$RUN/report" --overwrite
echo "[tests] done: answers in $RUN/report/verdict.json, plot in $RUN/report/tests.png"
