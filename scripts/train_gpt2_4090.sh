#!/usr/bin/env bash
# Dense 1024-token GPT-2 pilot; never downloads. Override any setting via environment.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python}"
export TOKENIZERS_PARALLELISM=false

if [[ -z "${MODEL:-}" ]]; then
    MODEL="$($PY - <<'PY'
import os
from pathlib import Path
root = Path(os.environ.get('HF_HUB_CACHE', Path(os.environ.get('HF_HOME', Path.home() / '.cache/huggingface')) / 'hub'))
for name in ('models--gpt2', 'models--openai-community--gpt2'):
    for path in sorted((root / name / 'snapshots').glob('*')):
        if (path / 'config.json').is_file() and (path / 'model.safetensors').is_file():
            print(path)
            raise SystemExit(0)
raise SystemExit('No local GPT-2 snapshot found; set MODEL=/path/to/gpt2')
PY
)"
fi
if [[ -z "${DATA:-}" ]]; then
    DATA="$($PY - <<'PY'
import os
from pathlib import Path
root = Path(os.environ.get('HF_HUB_CACHE', Path(os.environ.get('HF_HOME', Path.home() / '.cache/huggingface')) / 'hub'))
for snapshot in sorted((root / 'datasets--Salesforce--wikitext' / 'snapshots').glob('*')):
    path = snapshot / 'wikitext-103-raw-v1'
    if list(path.glob('train-*.parquet')):
        print(path)
        raise SystemExit(0)
raise SystemExit('No local WikiText-103 parquet found; set DATA=/path/to/wikitext-103-raw-v1')
PY
)"
fi

OUT="${OUT:-runs/gpt2-4090-$(date +%Y%m%d-%H%M%S)}"
limit=()
if [[ -n "${EVAL_TOKENS:-}" ]]; then limit=(--max-tokens "$EVAL_TOKENS"); fi
kl=()
if [[ -n "${KL_STRENGTH:-}" ]]; then kl=(--kl-strength "$KL_STRENGTH"); fi
"$PY" -m depthprobe finetune-segment \
  --model "$MODEL" --data "$DATA" \
  --first-block "${FIRST_BLOCK:-4}" --last-block "${LAST_BLOCK:-8}" \
  --train-tokens "${TRAIN_TOKENS:-1000000}" --split "${SPLIT:-validation}" \
  --epochs "${EPOCHS:-3}" --lr "${LR:-1e-5}" --lr-schedule cosine --warmup-fraction 0.05 \
  --regularizer "${REGULARIZER:-l2}" --reg-strength "${REG_STRENGTH:-1.0}" "${kl[@]}" \
  --l2-space "${L2_SPACE:-rms}" --rms-eps "${RMS_EPS:-1e-6}" \
  --ctx 1024 --stride 512 --batch-size "${BATCH:-32}" \
  --inference-batch-size "${INFERENCE_BATCH:-16}" --chunk "${CHUNK:-1024}" \
  --device cuda --inference-device cuda --dtype float32 --amp bfloat16 --tf32 --attn sdpa \
  --workers "${WORKERS:-2}" --save-full-model --out "$OUT" "${limit[@]}" "$@"
echo "Report: $OUT/report/results.png"
echo "Reopen: $PY -m depthprobe report-segment '$OUT' --show"
