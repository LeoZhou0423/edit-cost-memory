#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/root/autodl-tmp/transformer-tune}"
mkdir -p "$ROOT"

for lr in 1e-4 3e-4 1e-3; do
  tag="transformer_pos_m64_s0_lr${lr}"
  python experiments/benchmark_mqar_fair.py \
    --which transformer --m 64 --steps 10000 \
    --batch-size 16 --eval-batch-size 64 --eval-repeats 16 --eval-every 500 \
    --dm 512 --lr "$lr" --seed 0 --data-seed 1000 --test-data-seed 5000 \
    --output "$ROOT/${tag}.jsonl" --checkpoint "$ROOT/${tag}.pt" \
    > "$ROOT/${tag}.log" 2>&1
done

