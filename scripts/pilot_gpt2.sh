#!/usr/bin/env bash
# Exploratory pilot on GPT-2 small: E0 audit, then E1/E2/E3 with E5 controls.
# Downloads nothing. Run from the project root after downloading the files listed in README.md.
#
#   MODELS=/path/models DATA=/path/data OUT=/path/runs bash scripts/pilot_gpt2.sh
set -euo pipefail

: "${MODELS:?set MODELS to the directory holding downloaded models}"
: "${DATA:?set DATA to the directory holding downloaded datasets}"
: "${OUT:?set OUT to the directory for results}"
MODEL="${MODEL:-$MODELS/gpt2}"
WIKI="${WIKI:-$DATA/wikitext/wikitext-103-raw-v1}"
DEVICE="${DEVICE:-auto}"
TRAIN_TOKENS="${TRAIN_TOKENS:-4000000}"
BATCH="${BATCH:-8}"
NAME="${NAME:-gpt2}"
RUN="$OUT/$NAME"
PY="${PY:-python}"

common=(--model "$MODEL" --device "$DEVICE" --dtype float32 --ctx 1024 --batch-size "$BATCH")
eval_limit=()
if [[ -n "${EVAL_TOKENS:-}" ]]; then
    eval_limit=(--max-tokens "$EVAL_TOKENS")
fi

"$PY" -m depthprobe check "${common[@]}" --data "$WIKI" --split test --stride 512 \
    ${eval_limit[@]+"${eval_limit[@]}"} --out "$RUN/check"

"$PY" -m depthprobe unigram --model "$MODEL" --data "$WIKI" --split train \
    --max-tokens "$TRAIN_TOKENS" --out "$RUN/unigram.npy"

for variant in pretrained random shuffle_all shuffle_middle; do
    case "$variant" in
        pretrained) control=() ;;
        random) control=(--init random --init-seed 0) ;;
        shuffle_all) control=(--shuffle all --shuffle-seed 0) ;;
        shuffle_middle) control=(--shuffle middle --shuffle-seed 0) ;;
    esac
    "$PY" -m depthprobe train-lens "${common[@]}" ${control[@]+"${control[@]}"} --data "$WIKI" \
        --train-tokens "$TRAIN_TOKENS" --out "$RUN/$variant/lens"
    "$PY" -m depthprobe measure "${common[@]}" ${control[@]+"${control[@]}"} --data "$WIKI" --split test \
        --stride 512 ${eval_limit[@]+"${eval_limit[@]}"} --lens "$RUN/$variant/lens/lens.pt" \
        --unigram "$RUN/unigram.npy" --label "$NAME-$variant" --out "$RUN/$variant/measure"
done

"$PY" -m depthprobe analyze "$RUN"/{pretrained,random,shuffle_all,shuffle_middle}/measure --out "$RUN/report"
