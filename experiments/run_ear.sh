#!/usr/bin/env bash
# Edited associative recall: does a fixed state support edits, and what governs
# whether it does?
#
# Core prediction.  Erasing the binding at address k writes -e (x) (a_j^T S) into
# the state, so the change it causes at another address a_p is
# -(a_p . e_j)(a_j^T S).  A perfect edit needs an erase direction with
# a_j . e_j = 1 and a_p . e_j = 0 for every other p, which is the dual vector of
# the addressing code, and that is a property of the code rather than of the
# update rule.  dk sets it: with K = 256 addresses in R^dk the coherence cannot
# fall below the Welch bound sqrt((K - dk) / (dk (K - 1))), so dk >= K removes the
# floor entirely while dk < K leaves one that no gate can move.
#
# Order matters here.  The constrained arms (small dk) come first because they
# are the decisive ones: they are the only place where the update rule has
# anything to prove, and they are the control for the gated and delta arms
# running in run_ear_axis.sh at the same dk.
#
# Fixed core: m = 8, T = 8m + 2e, e = m//2 (half the keys get exactly one edit,
# half of those a delete and half an overwrite), cosine with 10% warmup, GPT-2
# init, separate train/val/test streams, test once at the validation-selected
# step.  Batch is 32 rather than the 64 used by the canonical MQAR runs because
# the 50,257-way readout is materialised at every one of ~72 positions; every arm
# here shares it, so the within-task comparisons are unaffected.
set -u
ROOT=${1:-/root/autodl-tmp/ear}
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

# --- decisive first: the constrained address dimension ------------------------
run_one "orth_dk32_s0"    edit_add 32  0 --orth
run_one "learned_dk32_s0" edit_add 32  0
run_one "orth_dk64_s0"    edit_add 64  0 --orth
run_one "orth_dk128_s0"   edit_add 128 0 --orth
# --- the orthonormal control, where the algebra says nothing should be lost ---
run_one "orth_dk256_s0"   edit_add 256 0 --orth
run_one "orth_dk256_s1"   edit_add 256 1 --orth
run_one "orth_dk256_s2"   edit_add 256 2 --orth
# --- does training find a better code on its own? ----------------------------
run_one "learned_dk256_s0" edit_add 256 0
run_one "learned_dk256_s1" edit_add 256 1
run_one "learned_dk256_s2" edit_add 256 2
# --- seeds on the constrained arm -------------------------------------------
run_one "orth_dk32_s1"    edit_add 32 1 --orth
run_one "orth_dk32_s2"    edit_add 32 2 --orth
# --- controls ---------------------------------------------------------------
run_one "orth_dk256_noerase_s0" edit_add 256 0 --orth --ablate-erase
run_one "bilinear_dk16_s0"      edit_add_bi 16 0 --orth

echo "EAR_FINISHED"
