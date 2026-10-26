#!/usr/bin/env bash
# Answer three questions about perturbation absorption on one model, with rules fixed before the run:
#   Q1  Does the noise itself shrink, or is it carried unchanged while the clean signal grows (dilution)?
#   Q2  Is the shrinking learned: clearly more than in a random-init and a permuted-weight copy of the model?
#   Q3  Denoising, convergence or compression: does a real perturbation (the difference a swapped earlier token
#       makes) survive better than noise of the same size on the same tokens, for tokens whose prediction the
#       swap changes and for tokens whose prediction it preserves?
# v2 (rules in PREREGISTRATION.md, "Absorption tests v2"):
#   - Q3 compares the real perturbation with the same perturbation rotated to random feature directions, so the
#     correlation of the real perturbation across tokens cannot pose as a preference for data directions;
#   - controls: shuffled middle layers and permuted weights decide, random weights are reported only; every
#     control reuses the trained run's sinks, swaps and prediction-changed token sets; comparisons against a
#     control whose noise saturates (R_L > 0.1) are "not applicable";
#   - Q2 relative (noise relative to clean-signal growth), Q3 learned, Q3 without massive coordinates, and
#   - Q4  Does the model steer noise out of the directions the prediction reads? KL of the propagated noise vs
#       KL of a random direction with the same per-token size at the last layer.
# Settings follow the cleanest reproduction configuration: float32, per-token noise, sink tokens excluded
# (the controls reuse the trained model's sink positions). Answers use PRIMARY_ETA; "clearly" means a ratio
# beyond MARGIN_RATIO with the 95% window-bootstrap interval on the same side. Downloads nothing; finished
# runs are skipped; settings are stamped and cannot change between attempts.
#
#   MODEL=/workspace/models/qwen3-8b DATA=/workspace/data/c4/en/c4-validation.00001-of-00008.json.gz \
#     OUT=$PWD/storage/runs bash scripts/absorption_tests.sh
#
# Time on the GX10 for Qwen3-8B: v1 took ~30 s per suffix pass of 5 copies (2.3 h trained, 1.6 h per control).
# v2 runs 2 passes of 6 copies per site and window (9 conditions, REPLICAS=5) plus extra readouts: expect about
# 2-2.5 h per run, 8-10 h for the trained model and three controls. The deciding controls run first; drop
# "random" from CONTROLS to save ~2 h. The log prints the pass count and a time estimate after the first window.
set -euo pipefail

: "${MODEL:?set MODEL to a local model directory}"
: "${DATA:?set DATA to a local corpus (e.g. a C4 .json.gz shard)}"
: "${OUT:?set OUT to the directory for results}"
NAME="${NAME:-absorption-tests-v2}"
DEVICE="${DEVICE:-auto}"
SPLIT="${SPLIT:-validation}"
CTX="${CTX:-2048}"
STRIDE="${STRIDE:-$CTX}"
MAX_TOKENS="${MAX_TOKENS:-65536}"   # 32 sequences of 2048
BATCH="${BATCH:-1}"
REPLICAS="${REPLICAS:-5}"        # 9 conditions (noise, real, rotated x 3 seeds) in 2 passes of 6 copies
SITES="${SITES:-1,12,24}"
ETAS="${ETAS:-1e-2}"                # v1: survival identical at 1e-3 and 1e-2 (linear), KL better resolved at 1e-2
SEEDS="${SEEDS:-3}"
PREPEND_BOS="${PREPEND_BOS:-0}"
PRIMARY_ETA="${PRIMARY_ETA:-1e-2}"
MARGIN_RATIO="${MARGIN_RATIO:-1.25}"
CONTROLS="${CONTROLS:-shuffle_middle_0 permuted random}"
CONTROL_SWAPS="${CONTROL_SWAPS:-1}"  # 1: controls also get token swaps, so Q3-learned can be answered
PY="${PY:-python}"
RUN="$OUT/$NAME"

if [[ "$DATA" == *"c4-validation.00000-of-00008"* ]]; then
    echo "[tests] WARNING: $DATA is the shard v1 used; the v2 pre-registration asks for fresh data (shard 00001)" >&2
fi
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
MARGIN_RATIO=$MARGIN_RATIO
CONTROLS=$CONTROLS
CONTROL_SWAPS=$CONTROL_SWAPS"
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
control_extra=()
if [[ "$CONTROL_SWAPS" == 1 ]]; then
    control_extra=(--real-swaps)
fi
for control in $CONTROLS; do
    case "$control" in
        random) options=(--init random --init-seed 0) ;;
        permuted) options=(--init permuted --init-seed 0) ;;
        shuffle_middle_*) options=(--shuffle middle --shuffle-seed "${control##*_}") ;;
        *) echo "Unknown control: $control" >&2; exit 1 ;;
    esac
    if [[ -f "$RUN/$control/manifest.json" ]]; then
        echo "[tests] $control done, skipping"
    else
        echo "[tests] control: $control"
        "$PY" -m depthprobe reproduce-absorption "${common[@]}" "${options[@]}" ${control_extra[@]+"${control_extra[@]}"} \
            --sinks-from "$RUN/trained" --label "$control" --out "$RUN/$control" --overwrite
    fi
    runs+=("$RUN/$control")
done

"$PY" -m depthprobe analyze-absorption "${runs[@]}" --reference trained --eta "$PRIMARY_ETA" \
    --margin-ratio "$MARGIN_RATIO" --out "$RUN/report" --overwrite
echo "[tests] done: answers in $RUN/report/verdict.json, plot in $RUN/report/tests.png"
