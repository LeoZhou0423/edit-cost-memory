#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/root/autodl-tmp/transformer-capacity}"
mkdir -p "$ROOT"

for m in 8 16 32; do
  tag="transformer_pos_m${m}_s0_lr1e-4"
  python experiments/benchmark_mqar_fair.py \
    --which transformer --m "$m" --steps 10000 \
    --batch-size 16 --eval-batch-size 64 --eval-repeats 16 --eval-every 500 \
    --dm 512 --lr 1e-4 --seed 0 --data-seed 1000 --test-data-seed 5000 \
    --output "$ROOT/${tag}.jsonl" --checkpoint "$ROOT/${tag}.pt" \
    > "$ROOT/${tag}.log" 2>&1
done
