#!/usr/bin/env bash
# Download what scripts/reproduce_absorption.sh needs: one model and one C4 validation shard.
# Run it yourself on the pod, inside the activated venv (hf comes with huggingface_hub, a transformers
# dependency). Already-downloaded files are skipped, so it can be re-run after an interruption.
#
#   MODELS=/workspace/models DATA=/workspace/data bash scripts/download_absorption.sh
#   MODEL_REPO=meta-llama/Llama-2-7b-hf MODEL_NAME=llama-2-7b MODELS=... DATA=... bash scripts/download_absorption.sh
#
# Gated models (LLaMA-2) need an accepted licence on huggingface.co and `hf auth login` first; never put the
# token in this script or in a shared file.
set -euo pipefail

: "${MODELS:?set MODELS to the directory for model weights}"
: "${DATA:?set DATA to the directory for datasets}"
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3-8B}"
MODEL_NAME="${MODEL_NAME:-qwen3-8b}"
C4_SHARD="${C4_SHARD:-en/c4-validation.00000-of-00008.json.gz}"

PATTERNS=("*.json" "*.safetensors" "*.txt" "*.model" "*.tiktoken")
# The two CLIs parse --include differently: hf wants the flag once per pattern (extra words would become
# file names), huggingface-cli takes all patterns after one flag (a repeated flag keeps only the last).
if command -v hf >/dev/null; then
    HF=(hf download)
    INCLUDE=()
    for pattern in "${PATTERNS[@]}"; do
        INCLUDE+=(--include "$pattern")
    done
elif command -v huggingface-cli >/dev/null; then  # huggingface_hub before 0.34
    HF=(huggingface-cli download)
    INCLUDE=(--include "${PATTERNS[@]}")
else
    echo "Neither hf nor huggingface-cli found: activate the venv (source .venv/bin/activate) first." >&2
    exit 1
fi

echo "[download] $MODEL_REPO -> $MODELS/$MODEL_NAME"
# Weights, config and tokenizer only (no .bin / .pth duplicates or other formats).
"${HF[@]}" "$MODEL_REPO" "${INCLUDE[@]}" --local-dir "$MODELS/$MODEL_NAME"
for needed in config.json tokenizer_config.json; do
    if [[ ! -s "$MODELS/$MODEL_NAME/$needed" ]]; then
        echo "[download] $needed is missing in $MODELS/$MODEL_NAME: the download is incomplete" >&2
        exit 1
    fi
done
if [[ ! -s "$MODELS/$MODEL_NAME/tokenizer.json" && ! -s "$MODELS/$MODEL_NAME/tokenizer.model" ]]; then
    echo "[download] no tokenizer.json or tokenizer.model in $MODELS/$MODEL_NAME: the download is incomplete" >&2
    exit 1
fi
ls "$MODELS/$MODEL_NAME"/*.safetensors >/dev/null 2>&1 || { echo "[download] no .safetensors weights" >&2; exit 1; }

echo "[download] allenai/c4 $C4_SHARD -> $DATA/c4"
"${HF[@]}" allenai/c4 "$C4_SHARD" --repo-type dataset --local-dir "$DATA/c4"

du -sh "$MODELS/$MODEL_NAME" "$DATA/c4/$C4_SHARD"
echo "[download] done. Next:"
echo "  MODEL=$MODELS/$MODEL_NAME DATA=$DATA/c4/$C4_SHARD OUT=/path/runs bash scripts/reproduce_absorption.sh"
