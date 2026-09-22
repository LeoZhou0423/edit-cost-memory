#!/usr/bin/env bash
# Fair MQAR comparison after the two protocol repairs.
#
# Two defects invalidated the earlier baseline numbers:
#   1. --objective query leaves one supervised answer in a 2m+2-token window;
#   2. the baselines were built with PyTorch's default initialisation, which puts
#      the tied output head at logit scale ~O(dm) and the initial retrieval loss
#      at ~298 against ln(V)=10.8, so the softmax starts saturated.
# Both are fixed here.  Runs are ordered so the decisive control lands first: a
# Transformer on the canonical Zoology layout at a load where published attention
# reaches 1.000.  If that control succeeds, no claim of architectural inability
# survives; if it fails at a plateau, the budget and the protocol are recorded
# with the number.
#
# Usage: bash experiments/run_revision_fair.sh [ROOT] [PYTHON]

set -uo pipefail

ROOT="${1:-/root/autodl-tmp/fair-runs}"
PY="${2:-/root/miniconda3/bin/python}"
mkdir -p "$ROOT"
cd "$(dirname "$0")/.."

decay="--lr-schedule cosine --warmup-frac 0.1"

run_one() {
  local model="$1" task="$2" m="$3" steps="$4" lr="$5" bs="$6" extra="$7" tag="$8"
  echo "[$(date '+%F %T')] START ${tag}"
  $PY experiments/benchmark_mqar_fair.py \
    --which "$model" --task "$task" --m "$m" --steps "$steps" \
    --batch-size "$bs" --eval-batch-size 64 --multi-eval-batch-size 16 \
    --eval-repeats 8 --eval-every 500 \
    --dm 512 --dk 256 --dv 128 --nl 4 --n-ssm 2 --state 16 \
    --lr "$lr" --seed 0 --data-seed 1000 --test-data-seed 5000 --tag "$tag" \
    $extra \
    --output "$ROOT/${tag}.jsonl" --checkpoint "$ROOT/${tag}.pt" \
    > "$ROOT/${tag}.log" 2>&1 \
    && echo "[$(date '+%F %T')] DONE  ${tag}" \
    || echo "[$(date '+%F %T')] FAIL  ${tag} (see ${tag}.log)"
}

# --- A. canonical control: can a Transformer learn the binding at all? --------
run_one transformer canonical 8  8000 3e-4 64 "$decay --seq-len 64"  transformer_canon_m8_T64
run_one transformer canonical 64 8000 3e-4 32 "$decay --seq-len 512" transformer_canon_m64_T512
run_one memory      canonical 8  1500 3e-4 64 "$decay --seq-len 64"  memory_canon_m8_T64_orth
run_one mamba2      canonical 8  4000 3e-4 64 "$decay --seq-len 64"  mamba2_canon_m8_T64
run_one linatt      canonical 8  4000 3e-4 32 "$decay --seq-len 64"  linatt_canon_m8_T64

# --- B. load-one layout with the dense objective ------------------------------
run_one memory      loadone 64 1500 3e-4 16 "--objective multi --key-mode orth"    memory_m64_multi_orth
run_one memory      loadone 64 1500 3e-4 16 "--objective multi --key-mode learned" memory_m64_multi_learned
run_one transformer loadone 64 8000 3e-4 16 "--objective multi" transformer_m64_multi
run_one mamba2      loadone 64 8000 3e-4 16 "--objective multi" mamba2_m64_multi
run_one linatt      loadone 64 8000 3e-4  8 "--objective multi" linatt_m64_multi

echo "FAIR_RUNS_FINISHED"
