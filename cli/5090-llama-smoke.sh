#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
conda run -n NeuralFn python scripts/train_llama_fast.py \
  --run-id llama-smoke \
  --device cuda \
  --output "$HOME/NeuralFn/artifacts/llama_fast_smoke.pt" \
  --max-steps 2500 \
  --max-wallclock-seconds 900 \
  --train-seq-len 192 \
  --batch-size 8 \
  --train-batch-tokens 24576 \
  --eval-batches 8 \
  --eval-batch-size 8 \
  --num-layers 5 \
  --model-dim 320 \
  --num-heads 5 \
  --num-kv-heads 5 \
  --mlp-mult 2.0 \
  --multiple-of 64 \
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
  --grad-clip-norm 1.0

PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
conda run -n NeuralFn python scripts/eval_llama_fast.py \
  --device cuda \
  --graph "$HOME/NeuralFn/artifacts/llama_fast_smoke.json" \
  --weights "$HOME/NeuralFn/artifacts/llama_fast_smoke.pt" \
  --report-path "$HOME/NeuralFn/artifacts/llama_fast_smoke.eval.json"
