#!/usr/bin/env bash
# Matched-budget, multi-seed sweep of the gated EAR cell.
#
# Why this queue exists.  The v11 paper's trained table (Table IX) and the
# deletion result came from ONE run per dk, each evaluated at its own
# validation-best step (dk32: 2250, dk64: 1250, dk256: 1000).  That leaves two
# defects, and this driver removes both.
#
#   1. dk confounded with training budget.  Fixed by --select-step 4000: every
#      arm is saved and tested at the SAME step, so a cross-dk difference in the
#      gate alignment g cannot be a budget effect.
#   2. one seed per dk.  Fixed by running seeds 0..4.
#
# Everything else is held at the configuration of the archived gated_fixed runs
# (cell edit_gated, m=8, seq_len=72, dv=128, batch 32, --orth, gpt init,
# lr 3e-4 cosine, warmup 0.1, gate-bias 2,2,6, steps 4000), so the new arms are
# directly comparable to those and to each other.
set -u
ROOT=${1:-/root/autodl-tmp/ear-seeds}
PY=${2:-/root/miniconda3/bin/python}
EXP=/root/autodl-tmp/dcgr3d-revision/experiments
mkdir -p "$ROOT"

run_one() {
  local tag="$1" dk="$2" seed="$3"
  if [ -f "$ROOT/${tag}.jsonl" ] && grep -q '"event": "test"' "$ROOT/${tag}.jsonl"; then
    echo "[$(date '+%F %T')] SKIP  ${tag}"; return
  fi
  echo "[$(date '+%F %T')] START ${tag}"
  "$PY" "$EXP/benchmark_ear.py" \
      --cell edit_gated --dk "$dk" --dv 128 --m 8 --n-edit 4 --seq-len 72 \
      --batch-size 32 --steps 4000 --lr 3e-4 --lr-schedule cosine --warmup-frac 0.1 \
      --emb-init gpt --orth --seed "$seed" --select-step 4000 --eval-every 250 \
      --checkpoint "$ROOT/${tag}.pt" \
      --output "$ROOT/${tag}.jsonl" > "$ROOT/${tag}.log" 2>&1
  echo "[$(date '+%F %T')] DONE  ${tag}"
}

for dk in 32 64 256; do
  for seed in 0 1 2 3 4; do
    run_one "gated_s${seed}_dk${dk}" "$dk" "$seed"
  done
done

echo "[$(date '+%F %T')] ANALYSIS"
bash "$EXP/analyze_ear_seeds.sh" "$ROOT" "$PY"
echo "GATED_SEEDS_FINISHED"
