#!/usr/bin/env bash
# E6 denoise test v1 on GPT-2 small (exploratory): analytic toys, then the trained model and its controls
# (random weights, entrywise-permuted block weights, three shuffled-middle orders), then the analysis.
# Downloads nothing. Finished steps are skipped; settings are stamped so reruns never mix configurations.
#
#   MODEL=/path/models/gpt2 WIKI=/path/data/wikitext/wikitext-103-raw-v1 OUT=/path/runs bash scripts/perturb_gpt2.sh
set -euo pipefail

: "${MODEL:?set MODEL to the local GPT-2 directory}"
: "${WIKI:?set WIKI to the local wikitext-103-raw-v1 parquet directory}"
: "${OUT:?set OUT to the directory for results}"
NAME="${NAME:-gpt2-e6}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-float32}"
CTX="${CTX:-1024}"
STRIDE="${STRIDE:-512}"
BATCH="${BATCH:-4}"
CHUNK="${CHUNK:-1024}"
EVERY="${EVERY:-4}"
REPLICAS="${REPLICAS:-8}"
SITES="${SITES:-all}"
FAMILIES="${FAMILIES:-iso,off,on,read,unread,shared,signed}"
DELTAS="${DELTAS:-0.03,0.1,0.3,1.0}"
SEED="${SEED:-0}"
MIN_CONTEXT="${MIN_CONTEXT:-64}"
CAL_TOKENS="${CAL_TOKENS:-50000}"
PRIMARY_DELTA="${PRIMARY_DELTA:-0.1}"
VARIANTS="${VARIANTS:-pretrained random permuted shuffle_middle_0 shuffle_middle_1 shuffle_middle_2}"
PY="${PY:-python}"
RUN="$OUT/$NAME"

mkdir -p "$RUN"
stamp="MODEL=$MODEL
WIKI=$WIKI
DEVICE=$DEVICE
DTYPE=$DTYPE
CTX=$CTX
STRIDE=$STRIDE
EVERY=$EVERY
SITES=$SITES
FAMILIES=$FAMILIES
DELTAS=$DELTAS
SEED=$SEED
MIN_CONTEXT=$MIN_CONTEXT
CAL_TOKENS=$CAL_TOKENS"
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

step() { echo "[e6] $*"; }

if [[ -f "$RUN/toys/toys.json" ]]; then
    step "toys done, skipping"
else
    step "analytic toys (the metrics must separate these before any model run counts)"
    "$PY" -m depthprobe perturb-toys --out "$RUN/toys" --overwrite
fi

common=(--model "$MODEL" --device "$DEVICE" --dtype "$DTYPE" --ctx "$CTX" --stride "$STRIDE"
        --batch-size "$BATCH" --chunk "$CHUNK" --data "$WIKI" --split test --every "$EVERY"
        --replicas "$REPLICAS" --sites "$SITES" --families "$FAMILIES" --deltas "$DELTAS" --seed "$SEED"
        --min-context "$MIN_CONTEXT" --cal-tokens "$CAL_TOKENS")
run_variant() {
    local variant=$1
    local control=()
    case "$variant" in
        pretrained) ;;
        random) control=(--init random --init-seed 0) ;;
        permuted) control=(--init permuted --init-seed 0) ;;
        shuffle_middle_*) control=(--shuffle middle --shuffle-seed "${variant##*_}") ;;
        *) echo "Unknown variant: $variant" >&2; exit 1 ;;
    esac
    if [[ "$variant" != pretrained ]]; then
        control+=(--sinks-from "$RUN/pretrained")  # every control measures exactly the trained model's tokens
    fi
    if [[ -f "$RUN/$variant/manifest.json" ]]; then
        step "$variant done, skipping"
    else
        step "$variant: calibrating and perturbing"
        "$PY" -m depthprobe perturb "${common[@]}" ${control[@]+"${control[@]}"} --label "$variant" \
            --out "$RUN/$variant" --overwrite
    fi
}

run_variant pretrained  # the reference, and the source of the shared measured tokens
for variant in $VARIANTS; do
    [[ "$variant" == pretrained ]] || run_variant "$variant"
done

# Analyse every finished variant, the trained model as reference, whatever subset this invocation ran.
runs=("$RUN/pretrained")
for dir in "$RUN"/*/; do
    name=$(basename "$dir")
    if [[ "$name" != pretrained && -f "$dir/manifest.json" && -f "$dir/perturb.npz" ]]; then
        runs+=("${dir%/}")
    fi
done
step "analysis of ${#runs[@]} runs (reference: pretrained)"
"$PY" -m depthprobe analyze-perturb "${runs[@]}" --reference pretrained --delta "$PRIMARY_DELTA" \
    --out "$RUN/report" --overwrite
step "done: $RUN/report"
