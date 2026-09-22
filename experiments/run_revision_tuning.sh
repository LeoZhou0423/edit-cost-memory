#!/usr/bin/env bash
# Hyperparameter protocol for the fair comparison.
#
# Rules this script follows, and that any table must state:
#
#  1. Selection is on validation.  Test is evaluated once, after selection.  No
#     number in this script reads test recall.
#  2. Every family gets the same fixed core (task, layout, m, T, batch size,
#     evaluation stream) and its own swept axes (learning rate, schedule, depth,
#     state width).
#  3. A run that has not passed its phase transition is not a result.  The
#     selection metric is steps-to-threshold on the validation curve, not final
#     accuracy, because a model can report chance right up to the step before it
#     solves the task.
#  4. The parameter-matching rule that picks 17 Transformer blocks is a
#     convention, not a validated choice, so depth is swept like any other knob.
#
# Stages (pass as $3, comma separated):
#   lr        learning rate x decay schedule per family
#   depth     depth sweep at the reference learning rate
#   seeds     three seeds of each family at the reference configuration
#   frontier  state-width sweep at fixed m (the state-capacity frontier)
#
# Usage: bash experiments/run_revision_tuning.sh [ROOT] [PYTHON] [STAGES]

set -uo pipefail

ROOT="${1:-/root/autodl-tmp/tuning}"
PY="${2:-/root/miniconda3/bin/python}"
STAGES="${3:-lr,depth,seeds}"
mkdir -p "$ROOT"
cd "$(dirname "$0")/.."

# Fixed core shared by every run in every stage.
CORE="--task canonical --seq-len 64 --eval-batch-size 64 --eval-repeats 8 \
      --eval-every 500 --dm 512 --dk 256 --dv 128 --nl 4 --n-ssm 2 --state 16 \
      --data-seed 1000 --test-data-seed 5000"
REF_LR=3e-4
REF_STEPS=6000

run_one() {
  local tag="$1" steps="$2" extra="$3"
  if [ -f "$ROOT/${tag}.jsonl" ] && grep -q '"event": "test"' "$ROOT/${tag}.jsonl"; then
    echo "[$(date '+%F %T')] SKIP  ${tag} (already has a test record)"
    return
  fi
  echo "[$(date '+%F %T')] START ${tag}"
  $PY experiments/benchmark_mqar_fair.py \
    $CORE --steps "$steps" --seed 0 --tag "$tag" $extra \
    --output "$ROOT/${tag}.jsonl" --checkpoint "$ROOT/${tag}.pt" \
    > "$ROOT/${tag}.log" 2>&1 \
    && echo "[$(date '+%F %T')] DONE  ${tag}" \
    || echo "[$(date '+%F %T')] FAIL  ${tag} (see ${tag}.log)"
}

sched="--lr-schedule cosine --warmup-frac 0.1"

if [[ ",$STAGES," == *",lr,"* ]]; then
  for lr in 1e-4 3e-4 1e-3; do
    run_one "tune_lr_transformer_${lr}" $REF_STEPS \
      "$sched --lr $lr --which transformer" || true
    run_one "tune_lr_mamba2_${lr}" $REF_STEPS \
      "$sched --lr $lr --which mamba2" || true
    run_one "tune_lr_linatt_${lr}" $REF_STEPS \
      "$sched --lr $lr --which linatt --batch-size 32" || true
  done
  for lr in 3e-4 1e-3; do
    run_one "tune_lr_memory_${lr}" 1500 \
      "$sched --lr $lr --which memory --batch-size 64" || true
  done
fi

if [[ ",$STAGES," == *",depth,"* ]]; then
  for d in 2 4 8 17; do
    run_one "tune_depth_transformer_${d}" $REF_STEPS \
      "$sched --lr $REF_LR --which transformer --batch-size 64 --base-depth $d" || true
  done
  for d in 2 8 25; do
    run_one "tune_depth_linatt_${d}" $REF_STEPS \
      "$sched --lr $REF_LR --which linatt --batch-size 32 --base-depth $d" || true
    run_one "tune_depth_mamba2_${d}" $REF_STEPS \
      "$sched --lr $REF_LR --which mamba2 --batch-size 64 --base-depth $d" || true
  done
fi

if [[ ",$STAGES," == *",seeds,"* ]]; then
  for s in 0 1 2; do
    for fam in transformer mamba2 linatt; do
      bs=64; [ "$fam" = linatt ] && bs=32
      run_one "seed_${fam}_s${s}" $REF_STEPS \
        "$sched --lr $REF_LR --which $fam --batch-size $bs --seed $s" || true
    done
    run_one "seed_memory_s${s}" 1500 \
      "$sched --lr $REF_LR --which memory --batch-size 64 --seed $s" || true
  done
fi

if [[ ",$STAGES," == *",frontier,"* ]]; then
  # Fixed-state frontier: how much load can a given state carry, per write rule.
  # The main model's state is dk x dv, so both are swept together.
  for m in 16 32 64 128; do
    T=$((4 * m))
    run_one "frontier_memory_m${m}_dk256_dv128" 2000 \
      "$sched --lr $REF_LR --which memory --batch-size 32 --m $m --seq-len $T --dk 256 --dv 128" || true
    run_one "frontier_memory_m${m}_dk128_dv128" 2000 \
      "$sched --lr $REF_LR --which memory --batch-size 32 --m $m --seq-len $T --dk 128 --dv 128" || true
    for sd in 64 128 512; do
      run_one "frontier_linatt_m${m}_s${sd}" 8000 \
        "$sched --lr $REF_LR --which linatt --batch-size 16 --m $m --seq-len $T --base-state-dim $sd" || true
    done
    run_one "frontier_transformer_m${m}" 12000 \
      "$sched --lr $REF_LR --which transformer --batch-size 16 --m $m --seq-len $T" || true
  done
fi

echo "TUNING_FINISHED"
