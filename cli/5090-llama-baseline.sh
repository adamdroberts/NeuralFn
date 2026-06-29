#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
conda run -n NeuralFn python scripts/train_llama_fast.py \
  --run-id llama-baseline \
  --device cuda \
  --output "$HOME/NeuralFn/artifacts/llama_fast_baseline.pt" \
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
  --qk-gain-init 1.5 \
  --rope-base 10000 \
  --logit-softcap 30 \
  --optimizer-profile adamw \
  --learning-rate 0.0006 \
  --weight-decay 0.1 \
  --warmup-steps 600 \
  --warmdown-fraction 0.0 \
  --beta1 0.9 \
  --beta2 0.95 \
  --adam-eps 1e-8 \
  --grad-clip-norm 1.0

PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
conda run -n NeuralFn python scripts/eval_llama_fast.py \
  --device cuda \
  --graph "$HOME/NeuralFn/artifacts/llama_fast_baseline.json" \
  --weights "$HOME/NeuralFn/artifacts/llama_fast_baseline.pt" \
  --report-path "$HOME/NeuralFn/artifacts/llama_fast_baseline.eval.json"
