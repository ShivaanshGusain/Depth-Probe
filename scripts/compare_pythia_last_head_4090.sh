#!/usr/bin/env bash
# Matched one-epoch CE adaptation: original vs five-epoch segment checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="${ROOT:-/media/user/archive/Depth-Probe}"
BASE_MODEL="${BASE_MODEL:-$ROOT/models/pythia-410m}"
STAGED_MODEL="${STAGED_MODEL:-$ROOT/runs/pythia-410m-both-21-22-kl5-5epochs-20261021/model}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/pythia-last-head-1epoch-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT_ROOT"
export FIRST_BLOCK=24 LAST_BLOCK=24 EPOCHS=1 REGULARIZER=none
export REG_STRENGTH=0 KL_STRENGTH=0
export PYTHONUNBUFFERED=1
for arm in direct staged; do
    if [[ "$arm" == direct ]]; then model="$BASE_MODEL"; else model="$STAGED_MODEL"; fi
    MODEL="$model" OUT="$OUT_ROOT/$arm" bash scripts/train_pythia_410m_4090.sh \
        --train-head --partial-exit-block 22 "$@" > "$OUT_ROOT/$arm.log" 2>&1
done
echo "Comparison runs complete: $OUT_ROOT/{direct,staged}/report/results.md"
