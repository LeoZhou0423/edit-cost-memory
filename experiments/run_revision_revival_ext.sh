#!/usr/bin/env bash
# Budget extension for the revival match.
#
# The m=8 revival run showed that the two linear cells at dk=16 (2,048 state
# numbers) had NOT converged within 2,000 steps: their validation recall was still
# rising steeply at the final evaluation (0.318 at step 1400, 0.574 at 1600, 0.802
# at 1800, 0.878 at 2000).  The three cells with 32,768 state numbers had all
# converged by step 600.  The matched-address-dimension comparison is therefore
# confounded by budget at 2,000 steps, and this queue removes that confound.
#
# Usage: bash experiments/run_revision_revival_ext.sh [ROOT] [PYTHON]

set -uo pipefail

ROOT="${1:-/root/autodl-tmp/revival-ext}"
PY="${2:-/root/miniconda3/bin/python}"
mkdir -p "$ROOT"
cd "$(dirname "$0")/.."

SCHED="--lr-schedule cosine --warmup-frac 0.1"
STEPS=8000

run_one() {
  local tag="$1" m="$2" cell="$3" dk="$4" T="$5"
  if [ -f "$ROOT/${tag}.jsonl" ] && grep -q '"event": "test"' "$ROOT/${tag}.jsonl"; then
    echo "[$(date '+%F %T')] SKIP  ${tag}"
    return
  fi
  echo "[$(date '+%F %T')] START ${tag}"
  $PY experiments/benchmark_mqar_fair.py \
    --which memory --cell "$cell" --dk "$dk" --dv 128 \
    --task canonical --seq-len "$T" --m "$m" \
    --steps "$STEPS" --batch-size 16 --eval-batch-size 16 --eval-repeats 8 \
    --eval-every 250 --dm 512 --nl 4 --n-ssm 2 --state 16 \
    --lr 3e-4 $SCHED --emb-init gpt --key-mode orth \
    --seed 0 --data-seed 1000 --test-data-seed 5000 --tag "$tag" \
    --output "$ROOT/${tag}.jsonl" --checkpoint "$ROOT/${tag}.pt" \
    > "$ROOT/${tag}.log" 2>&1 \
    && echo "[$(date '+%F %T')] DONE  ${tag}" \
    || echo "[$(date '+%F %T')] FAIL  ${tag} (see ${tag}.log)"
}

# the two cells that had not converged at 2,000 steps, extended to 8,000
run_one "ext_dk16_stable_m8"   8 stable   16 64
run_one "ext_dk16_delta_m8"    8 delta    16 64
# matched-state companion, extended to the same budget for a like-for-like point
run_one "ext_dk16_tensor3d_m8" 8 tensor3d 16 64
run_one "ext_dk256_stable_m8"  8 stable   256 64

echo "REVIVAL_EXT_FINISHED"
