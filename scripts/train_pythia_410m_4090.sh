#!/usr/bin/env bash
# Two late GPT-NeoX blocks with RMS-space L2 + KL; local files only.
# Download the checkpoint separately; all new artifacts default to the archive disk.
set -euo pipefail
cd "$(dirname "$0")/.."
export MODEL="${MODEL:-/media/user/archive/Depth-Probe/models/pythia-410m}"
export OUT="${OUT:-/media/user/archive/Depth-Probe/runs/pythia-410m-both-21-22-$(date +%Y%m%d-%H%M%S)}"
export FIRST_BLOCK="${FIRST_BLOCK:-21}"
export LAST_BLOCK="${LAST_BLOCK:-22}"
export REGULARIZER="${REGULARIZER:-both}"
export REG_STRENGTH="${REG_STRENGTH:-1.0}"
export KL_STRENGTH="${KL_STRENGTH:-1.0}"
export L2_SPACE="${L2_SPACE:-rms}"
export INFERENCE_BATCH="${INFERENCE_BATCH:-8}"
bash scripts/train_gpt2_4090.sh "$@"
