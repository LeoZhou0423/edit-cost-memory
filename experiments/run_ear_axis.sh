#!/usr/bin/env bash
# The coherence x update-rule matrix: the paper's main trained figure.
#
# Erasing the binding at address k writes -k (x) (e^T S) into the state, so the
# change it causes at another address a_p is -(a_p . k)(e^T S).  The first factor
# is a property of the addressing code; the second is what any gate can change.
# If that decomposition is right, then with a coherent code the erase residual and
# the collateral damage cannot both be small under any update rule, and the
# locality floor is the code's mu -- which dk sets through the Welch bound
# mu >= sqrt((K - dk) / (dk (K - 1))) for K = 256 addresses.
#
# The additive arms at these dk live in run_ear.sh (tags orth_dk*, identical
# settings), so this script carries the delta and gated arms plus the dk = 256
# gated control.  Both scripts write into the same root, and the skip check below
# means a run already produced by the other one is not repeated.
set -u
ROOT=${1:-/root/autodl-tmp/ear-axis}
PY=${2:-python}
mkdir -p "$ROOT"

run_one() {
  local tag="$1" cell="$2" dk="$3" seed="$4"; shift 4
  if [ -f "$ROOT/${tag}.jsonl" ] && grep -q '"event": "test"' "$ROOT/${tag}.jsonl"; then
    echo "[$(date '+%F %T')] SKIP  ${tag}"
    return
  fi
  echo "[$(date '+%F %T')] START ${tag}"
  "$PY" experiments/benchmark_ear.py \
    --cell "$cell" --dk "$dk" --dv 128 --m 8 --n-edit 4 \
    --dm 512 --nl 4 --n-ssm 2 --state 16 \
    --steps 4000 --batch-size 32 \
    --lr 3e-4 --lr-schedule cosine --warmup-frac 0.1 \
    --emb-init gpt --eval-batch-size 16 --eval-repeats 8 --eval-every 250 \
    --seed "$seed" --data-seed 1000 --test-data-seed 5000 \
    --checkpoint "$ROOT/${tag}.pt" --output "$ROOT/${tag}.jsonl" \
    "$@" > "$ROOT/${tag}.log" 2>&1
  echo "[$(date '+%F %T')] DONE  ${tag}  (exit $?)"
}

# --- the decisive arm: does a channel-wise gate move the locality off mu? -----
for dk in 32 64 128; do
  run_one "gated_dk${dk}" edit_gated "$dk" 0 --orth
done
# --- the write rule at the same code -----------------------------------------
for dk in 128 64 32; do
  run_one "delta_dk${dk}" edit_delta "$dk" 0 --orth
done
# --- the orthonormal control: everything should be exact here ----------------
run_one "gated_dk256" edit_gated 256 0 --orth

echo "EAR_AXIS_FINISHED"
