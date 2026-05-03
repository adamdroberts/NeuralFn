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
  --max-steps 2500 \
  --max-wallclock-seconds 600 \
  --train-seq-len 192 \
  --batch-size 8 \
  --train-batch-tokens 24576 \
  --eval-batches 4 \
  --eval-batch-size 8 \
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
  --optimizer-profile parameter_golf \
  --embed-lr 0.02 \
  --head-lr 0.005 \
  --tied-embed-lr 0.01 \
  --matrix-lr 0.008 \
  --scalar-lr 0.004 \
  --warmup-steps 20 \
  --warmdown-fraction 0.1 \
  --muon-momentum 0.95 \
  --muon-backend-steps 5 \
  --muon-momentum-warmup-start 0.85 \
  --muon-momentum-warmup-steps 128 \
  --beta1 0.9 \
  --beta2 0.95 \
  --adam-eps 1e-8 \
  --grad-clip-norm 1.0 \
  --ar-loss-coef 1.0 \
  --jepa-loss-coef 0.25 \
  --semantic-align-loss-coef 0.5
