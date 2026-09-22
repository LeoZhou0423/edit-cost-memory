#!/usr/bin/env bash
# The deletion-scored arm: the same fifteen runs, but the objective now SCORES
# deletion.
#
# Why this queue exists.  Appendix A concludes that the erase gate is not
# mis-learned but *unlearnable*, because a deleted key has no correct token and
# the deletion probes therefore contribute no term to the training loss -- so
# the only route by which the gate can reach the objective is collateral damage
# to the pairs that were not erased, which can only ever penalise erasing.  That
# argument has a hole: it is a statement about THIS task's objective as much as
# about the cell.  If the gate only looks unlearnable because deletion is never
# scored, the claim is an artefact of the protocol and does not transfer.
#
# The fix is one flag.  --del-target noise supervises a query on a deleted key
# with the reserved marker token: the correct answer to "what is the value of a
# key that was deleted" becomes "nothing" (NOISE is never a key and never a
# value, so it is unambiguous).  Deletion then enters the loss, and the erase
# gate gets a real gradient toward g = 1.
#
# Everything else is byte-identical to run_gated_seeds.sh (cell edit_gated,
# m=8, seq_len=72, dv=128, batch 32, --orth, gpt init, lr 3e-4 cosine,
# warmup 0.1, gate-bias 2,2,6, steps 4000, --select-step 4000), so the two
# fifteen-run sweeps differ in exactly one variable.
set -u
ROOT=${1:-/root/autodl-tmp/ear-delscored}
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
      --del-target noise \
      --checkpoint "$ROOT/${tag}.pt" \
      --output "$ROOT/${tag}.jsonl" > "$ROOT/${tag}.log" 2>&1
  echo "[$(date '+%F %T')] DONE  ${tag}"
}

for dk in 32 64 256; do
  for seed in 0 1 2 3 4; do
    run_one "del_s${seed}_dk${dk}" "$dk" "$seed"
  done
done

echo "[$(date '+%F %T')] ANALYSIS"
bash "$EXP/analyze_ear_seeds.sh" "$ROOT" "$PY" del
echo "DEL_SCORED_FINISHED"
