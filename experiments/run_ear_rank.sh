#!/usr/bin/env bash
# Two questions the m = 8 arms cannot answer.
#
# (1) Is the trained failure of the gated arm about gating, or about the gate's
#     initialisation?  The gated cell starts with sigmoid(2) = 0.881 on the erase
#     gate, which leaves a 12% erase residual before training has done anything,
#     where the ungated cell starts at g = 1 exactly.  This sweeps the gate bias
#     between the default and a near pass-through.
#
# (2) Does the rank condition m <= dk bind in the trained setting?  At m = 8 every
#     arm satisfies it, so the m = 8 arms test the learned-code question only.
#     These arms place m on both sides of dk at matched sequence shape.
#
# Sequence length is 6m + 2e here rather than the 8m + 2e of the m = 8 arms,
# because the per-position python loop makes the sequence length the dominant
# cost and a longer one would not finish.  Both arms of a pair share it.
set -u
ROOT=${1:-/root/autodl-tmp/ear-rank}
PY=${2:-python}
mkdir -p "$ROOT"

run_one() {
  local tag="$1" cell="$2" dk="$3" m="$4" T="$5" bs="$6" steps="$7"; shift 7
  if [ -f "$ROOT/${tag}.jsonl" ] && grep -q '"event": "test"' "$ROOT/${tag}.jsonl"; then
    echo "[$(date '+%F %T')] SKIP  ${tag}"
    return
  fi
  echo "[$(date '+%F %T')] START ${tag}"
  "$PY" experiments/benchmark_ear.py \
    --cell "$cell" --dk "$dk" --dv 128 --m "$m" --n-edit "$((m / 2))" \
    --seq-len "$T" --dm 512 --nl 4 --n-ssm 2 --state 16 \
    --steps "$steps" --batch-size "$bs" \
    --lr 3e-4 --lr-schedule cosine --warmup-frac 0.1 \
    --emb-init gpt --eval-batch-size 8 --eval-repeats 6 --eval-every 250 \
    --seed 0 --data-seed 1000 --test-data-seed 5000 \
    --checkpoint "$ROOT/${tag}.pt" --output "$ROOT/${tag}.jsonl" \
    "$@" > "$ROOT/${tag}.log" 2>&1
  echo "[$(date '+%F %T')] DONE  ${tag}  (exit $?)"
}

# --- (1) gate initialisation --------------------------------------------------
run_one "gated_dk32_pass"    edit_gated 32 8 72 32 4000 --orth --gate-bias 6,6,6
run_one "gated_dk32_default" edit_gated 32 8 72 32 4000 --orth --gate-bias 2,2,6

# --- (2) the rank condition ---------------------------------------------------
run_one "m32_dk16_add"  edit_add 16 32 224 16 2500 --orth
run_one "m32_dk256_add" edit_add 256 32 224 16 2500 --orth

echo "EAR_RANK_FINISHED"
