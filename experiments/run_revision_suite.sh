#!/usr/bin/env bash
set -euo pipefail

# First publication-oriented MQAR suite.  Each run uses a fixed 1,024-example
# validation set per distance and a disjoint fixed test stream.  The queue is
# intentionally conservative; learning-rate sweeps are scheduled after these
# pilots reveal the convergence range of each family.

ROOT="${1:-/root/autodl-tmp/revision-runs-query-only}"
mkdir -p "$ROOT"

run_one() {
  local model="$1"
  local seed="$2"
  local steps="$3"
  local lr="$4"
  local tag="${model}_m64_s${seed}_lr${lr}"
  python experiments/benchmark_mqar_fair.py \
    --which "$model" --m 64 --steps "$steps" \
    --batch-size 16 --eval-batch-size 64 --eval-repeats 16 --eval-every 250 \
    --dm 512 --dk 256 --dv 128 --nl 4 --n-ssm 2 --state 16 \
    --lr "$lr" --seed "$seed" --data-seed "$((1000 + seed * 10))" \
    --test-data-seed "$((5000 + seed))" \
    --output "$ROOT/${tag}.jsonl" --checkpoint "$ROOT/${tag}.pt" \
    > "$ROOT/${tag}.log" 2>&1
}

for seed in 0 1 2; do
  run_one memory "$seed" 1500 3e-4
done

for model in transformer mamba2 linatt; do
  for seed in 0 1 2; do
    run_one "$model" "$seed" 5000 3e-4
  done
done
