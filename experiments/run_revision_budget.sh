#!/usr/bin/env bash
# Budget probe: is the Transformer baseline's MQAR failure a plateau or a schedule?
#
# The dense-objective pilot still had the Transformer at the 1/m candidate-set
# chance level after 3,000 steps at m=8.  Either it needs a much larger budget
# (in which case the earlier "chance level" table is not a ceiling and must not be
# reported as one), or it does not acquire the binding at all.  This queue keeps
# the identical task and evaluation and only raises the number of examples seen.
#
# Usage: bash experiments/run_revision_budget.sh [ROOT] [PYTHON]

set -uo pipefail

ROOT="${1:-/root/autodl-tmp/dense-budget}"
PY="${2:-/root/miniconda3/bin/python}"
mkdir -p "$ROOT"
cd "$(dirname "$0")/.."

run_one() {
  local model="$1" m="$2" steps="$3" lr="$4" bs="$5" extra="$6" tag="$7"
  echo "[$(date '+%F %T')] START ${tag}"
  $PY experiments/benchmark_mqar_fair.py \
    --which "$model" --m "$m" --steps "$steps" --objective multi \
    --batch-size "$bs" --eval-batch-size 64 --multi-eval-batch-size 16 \
    --eval-repeats 8 --eval-every 1000 \
    --dm 512 --dk 256 --dv 128 --nl 4 --n-ssm 2 --state 16 \
    --lr "$lr" --seed 0 --data-seed 1000 --test-data-seed 5000 \
    $extra \
    --output "$ROOT/${tag}.jsonl" --checkpoint "$ROOT/${tag}.pt" \
    > "$ROOT/${tag}.log" 2>&1 \
    && echo "[$(date '+%F %T')] DONE  ${tag}" \
    || echo "[$(date '+%F %T')] FAIL  ${tag} (see ${tag}.log)"
}

run_one transformer 8  20000 3e-4 64 "" "transformer_m8_multi_b64_lr3e-4_20k"
run_one transformer 8  20000 1e-3 64 "" "transformer_m8_multi_b64_lr1e-3_20k"
run_one transformer 64 20000 3e-4 64 "" "transformer_m64_multi_b64_lr3e-4_20k"

echo "DENSE_BUDGET_FINISHED"
