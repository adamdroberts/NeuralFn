#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
conda run -n NeuralFn python scripts/train_jepa_semantic.py \
  --run-id jepa-10min \
  --device cuda \
  --output "$HOME/NeuralFn/artifacts/jepa_semantic_hybrid_10min.pt" \
  --max-steps 20000 \
  --train-seq-len 1024 \
  --batch-size 64 \
  --train-batch-tokens 524288 \
  --eval-batches 20 \
  --eval-batch-size 64 \
  --num-layers 5 \
  --model-dim 320 \
  --num-heads 5 \
  --num-kv-heads 5 \
  --mlp-mult 2.0 \
  --multiple-of 64 \
  --experts 8 \
  --top-k 2 \
  --qk-gain-init 1.5 \
  --rope-base 10000 \
  --logit-softcap 30 \
  --optimizer-profile adamw \
  --learning-rate 0.0006 \
  --weight-decay 0.1 \
  --warmup-steps 60 \
  --warmdown-fraction 0.0 \
  --beta1 0.9 \
  --beta2 0.95 \
  --adam-eps 1e-8 \
  --grad-clip-norm 1.0 \
  --ar-loss-coef 1.0 \
  --jepa-loss-coef 0.25 \
  --semantic-align-loss-coef 0.5
