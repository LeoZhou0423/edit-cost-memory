#!/usr/bin/env bash
# Mop-up queue for runs that failed or were not reached in the main queues.
#
# The linear-attention baseline at m=8/T=64 with base-state-dim 512 and 25 layers
# was killed by the OOM killer at batch 32 (empty log, no partial record).  It is
# re-run at batch 8 with a smaller evaluation batch.  Sequence length follows the
# reference runs (T = 8m) so it stays comparable.
#
# Usage: bash experiments/run_revision_mopup.sh [ROOT] [PYTHON]

set -uo pipefail

ROOT="${1:-/root/autodl-tmp/mopup}"
PY="${2:-/root/miniconda3/bin/python}"
mkdir -p "$ROOT"
cd "$(dirname "$0")/.."

SCHED="--lr-schedule cosine --warmup-frac 0.1"

run_one() {
  local which="$1" task="$2" m="$3" steps="$4" bs="$5" extra="$6" tag="$7"
  if [ -f "$ROOT/${tag}.jsonl" ] && grep -q '"event": "test"' "$ROOT/${tag}.jsonl"; then
    echo "[$(date '+%F %T')] SKIP  ${tag}"
    return
  fi
  echo "[$(date '+%F %T')] START ${tag}"
  $PY experiments/benchmark_mqar_fair.py \
    --which "$which" --task "$task" --m "$m" --steps "$steps" \
    --batch-size "$bs" --eval-batch-size 16 --multi-eval-batch-size 16 \
    --eval-repeats 8 --eval-every 500 --dm 512 --nl 4 --n-ssm 2 --state 16 \
    --lr 3e-4 $SCHED --seed 0 --data-seed 1000 --test-data-seed 5000 --tag "$tag" \
    $extra \
    --output "$ROOT/${tag}.jsonl" --checkpoint "$ROOT/${tag}.pt" \
    > "$ROOT/${tag}.log" 2>&1 \
    && echo "[$(date '+%F %T')] DONE  ${tag}" \
    || echo "[$(date '+%F %T')] FAIL  ${tag} (see ${tag}.log)"
}

# linear attention, canonical, batch reduced from 32 to 8 after the OOM kill
run_one linatt canonical 8 4000 8 "$SCHED --seq-len 64 --base-state-dim 512" linatt_canon_m8_T64_b8
# same family at the smaller state that its own parameter rule implies, as a
# check that the failure above was memory and not the family
run_one linatt canonical 8 4000 8 "$SCHED --seq-len 64 --base-state-dim 64" linatt_canon_m8_T64_s64

echo "MOPUP_FINISHED"
