#!/usr/bin/env bash
# Dense-objective pilot for the revision MQAR comparison.
#
# The historical single-query protocol puts exactly one supervised retrieval
# answer in a 2m+2-token window.  Every baseline stayed at the 1/m candidate-set
# chance level under it, at every m from 8 to 64, so that protocol cannot
# separate "the baseline cannot bind" from "the baseline was given one training
# signal per 130 tokens".  This queue repeats the comparison with m query blocks
# per sequence (objective=multi), which is the dense MQAR objective, and then
# isolates how much of the main model's gap comes from its orthonormal frozen key
# prior instead of from the recurrent cell itself.
#
# Usage: bash experiments/run_revision_dense.sh [ROOT] [PYTHON]

set -uo pipefail

ROOT="${1:-/root/autodl-tmp/dense-pilot}"
PY="${2:-/root/miniconda3/bin/python}"
mkdir -p "$ROOT"
cd "$(dirname "$0")/.."

run_one() {
  local model="$1" m="$2" steps="$3" lr="$4" bs="$5" extra="$6" tag="$7"
  echo "[$(date '+%F %T')] START ${tag}"
  $PY experiments/benchmark_mqar_fair.py \
    --which "$model" --m "$m" --steps "$steps" --objective multi \
    --batch-size "$bs" --eval-batch-size 64 --multi-eval-batch-size 16 \
    --eval-repeats 8 --eval-every 500 \
    --dm 512 --dk 256 --dv 128 --nl 4 --n-ssm 2 --state 16 \
    --lr "$lr" --seed 0 --data-seed 1000 --test-data-seed 5000 \
    $extra \
    --output "$ROOT/${tag}.jsonl" --checkpoint "$ROOT/${tag}.pt" \
    > "$ROOT/${tag}.log" 2>&1 \
    && echo "[$(date '+%F %T')] DONE  ${tag}" \
    || echo "[$(date '+%F %T')] FAIL  ${tag} (see ${tag}.log)"
}

run_one transformer 8  10000 3e-4 16 ""                  transformer_m8_multi_lr3e-4
run_one transformer 64 10000 3e-4 16 ""                  transformer_m64_multi_lr3e-4
run_one transformer 64 10000 1e-4 16 ""                  transformer_m64_multi_lr1e-4
run_one mamba2      64 10000 3e-4 16 ""                  mamba2_m64_multi_lr3e-4
run_one memory      64  1500 3e-4 16 "--key-mode orth"    memory_m64_multi_orth
run_one memory      64  1500 3e-4 16 "--key-mode learned" memory_m64_multi_learned
run_one linatt      64 10000 3e-4  8 ""                  linatt_m64_multi_lr3e-4

echo "DENSE_PILOT_FINISHED"
